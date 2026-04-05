import os
import sys
import glob
import torch
import shutil
import numpy as np
import pandas as pd
import argparse
import pyarrow as pa
import pyarrow.parquet as pq


# Fix path import issues
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))  # Adjust according to your directory structure
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from structRFM.infer import structRFM_infer


# =========================================================
# Reverse Complement
# =========================================================
def reverse_complement(seq: str) -> str:
    trans = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(trans)[::-1]


# =========================================================
# FASTA Reader
# Prioritize pysam, fallback to pyfaidx if failed
# =========================================================
class FastaReader:
    def __init__(self, fasta_path: str):
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"Reference genome file does not exist: {fasta_path}")

        self.backend = None
        self.ref = None

        try:
            import pysam
            self.ref = pysam.FastaFile(fasta_path)
            self.backend = "pysam"
            print(f"[INFO] Reading FASTA using pysam: {fasta_path}")
            return
        except Exception as e:
            print(f"[WARN] Failed to load pysam: {e}")

        try:
            from pyfaidx import Fasta
            self.ref = Fasta(fasta_path, rebuild=False)
            self.backend = "pyfaidx"
            print(f"[INFO] Reading FASTA using pyfaidx: {fasta_path}")
            return
        except Exception as e:
            print(f"[WARN] Failed to load pyfaidx: {e}")

        raise RuntimeError(
            "Cannot open reference genome FASTA. Please install pysam or pyfaidx, and ensure the fasta and index files are valid."
        )

    def get_chrom_length(self, chrom: str) -> int:
        if self.backend == "pysam":
            return self.ref.get_reference_length(chrom)
        elif self.backend == "pyfaidx":
            return len(self.ref[chrom])
        else:
            raise RuntimeError("Unknown FASTA backend")

    def fetch(self, chrom: str, start0: int, end0: int) -> str:
        # 0-based, half-open [start0, end0)
        if self.backend == "pysam":
            return self.ref.fetch(chrom, start0, end0)
        elif self.backend == "pyfaidx":
            return str(self.ref[chrom][start0:end0])
        else:
            raise RuntimeError("Unknown FASTA backend")

    def close(self):
        try:
            if self.backend == "pysam":
                self.ref.close()
        except Exception:
            pass


# =========================================================
# Chromosome name compatibility
# Compatible with formats like chr1 / 1
# =========================================================
def resolve_chrom_name(fasta_reader: FastaReader, chrom: str) -> str:
    chrom = str(chrom).strip()

    try:
        fasta_reader.get_chrom_length(chrom)
        return chrom
    except Exception:
        pass

    if chrom.startswith("chr"):
        alt = chrom[3:]
        try:
            fasta_reader.get_chrom_length(alt)
            return alt
        except Exception:
            pass
    else:
        alt = "chr" + chrom
        try:
            fasta_reader.get_chrom_length(alt)
            return alt
        except Exception:
            pass

    raise KeyError(f"Chromosome not found in the reference genome: {chrom}")


# =========================================================
# Extract undirected centered sequence (forward strand first)
# =========================================================
def extract_centered_sequence_raw(
    fasta_reader: FastaReader,
    chrom: str,
    site_1based: int,
    flank: int = 1000
) -> str:
    chrom_len = fasta_reader.get_chrom_length(chrom)

    center0 = site_1based - 1
    wanted_start0 = center0 - flank
    wanted_end0 = center0 + flank + 1

    fetch_start0 = max(0, wanted_start0)
    fetch_end0 = min(chrom_len, wanted_end0)

    seq = fasta_reader.fetch(chrom, fetch_start0, fetch_end0).upper()

    left_pad = max(0, -wanted_start0)
    right_pad = max(0, wanted_end0 - chrom_len)

    seq = ("N" * left_pad) + seq + ("N" * right_pad)

    expected_len = 2 * flank + 1
    if len(seq) != expected_len:
        raise ValueError(
            f"Sequence length error after extraction: {chrom}:{site_1based}, got {len(seq)}, expected {expected_len}"
        )

    return seq


# =========================================================
# Determine forward/reverse strand based on center base A/T
# - If forward center is A: use directly
# - If forward center is T: use reverse complement
# The final returned seq guarantees the center position is A
# =========================================================
def extract_centered_sequence_by_center_AT(
    fasta_reader: FastaReader,
    chrom: str,
    site_1based: int,
    flank: int = 1000
) -> str:
    raw_seq = extract_centered_sequence_raw(
        fasta_reader=fasta_reader,
        chrom=chrom,
        site_1based=site_1based,
        flank=flank
    )

    center_idx = flank
    raw_center_base = raw_seq[center_idx].upper()

    if raw_center_base == "A":
        return raw_seq

    if raw_center_base == "T":
        rc_seq = reverse_complement(raw_seq)
        rc_center_base = rc_seq[center_idx].upper()
        if rc_center_base != "A":
            raise ValueError(
                f"The center base is not A after reverse complement: {chrom}:{site_1based}, "
                f"raw_center={raw_center_base}, rc_center={rc_center_base}"
            )
        return rc_seq

    raise ValueError(
        f"The center base is not A/T and cannot be treated as an A site: {chrom}:{site_1based}, "
        f"center base={raw_center_base}"
    )


# =========================================================
# Generate 4-column records directly from the genome site file:
# chrom, pos, kmer, seq
# Intentionally keeping the exact same structure as the old version
# =========================================================
def parse_genome_site_and_extract_seq(file_path, fasta_reader, flank=1000, failed_rows=None):
    data = []

    with open(file_path, 'r') as f:
        for row_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 3:
                if failed_rows is not None:
                    failed_rows.append({
                        "row_index_0based": row_index,
                        "Chr": None,
                        "Sites": None,
                        "5mer": None,
                        "error": f"Less than 3 columns: {line}"
                    })
                continue

            chrom_raw = parts[0]
            pos_raw = parts[1]
            kmer_raw = parts[2]

            try:
                pos_int = int(pos_raw)
                chrom = resolve_chrom_name(fasta_reader, chrom_raw)
                seq = extract_centered_sequence_by_center_AT(
                    fasta_reader=fasta_reader,
                    chrom=chrom,
                    site_1based=pos_int,
                    flank=flank
                )

                # Keep raw strings for pos to mimic the old output style
                data.append([chrom_raw, pos_raw, kmer_raw, seq])

            except Exception as e:
                if failed_rows is not None:
                    failed_rows.append({
                        "row_index_0based": row_index,
                        "Chr": chrom_raw,
                        "Sites": pos_raw,
                        "5mer": kmer_raw,
                        "error": str(e)
                    })

    return data


def save_chunk_to_parquet(results, output_file):
    """Save the current batch directly as an independent Parquet file."""
    df = pd.DataFrame(results)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_file, compression='snappy')


def main():
    parser = argparse.ArgumentParser(description='structRFM Sequence Embedding Extraction Tool (Genome Version)')

    parser.add_argument('--model_path', type=str, default='/home/zj/111NanoFM/structRFM_checkpoint')
    parser.add_argument(
        '--input',
        type=str,
    )
    parser.add_argument(
        '--reference_fasta',
        type=str,
    )
    parser.add_argument(
        '--output',
        type=str,
        help="The final output path for the single Parquet file (e.g., res.parquet)"
    )

    parser.add_argument('--target_seq_len', type=int, default=501,
                        help='Actual sequence length for the model. Extracted from the center of the total sequence.')

    parser.add_argument('--chunk_size', type=int, default=5000, help='Save chunk to disk every N records')
    parser.add_argument('--flank', type=int, default=500, help='Flanking length on each side (default 500 makes total 1001)')
    args = parser.parse_args()

    # Validate target_seq_len
    total_seq_len = 2 * args.flank + 1
    if args.target_seq_len % 2 == 0 or args.target_seq_len < 5:
        raise ValueError(f"Error: target_seq_len must be an odd number >= 5! Current input: {args.target_seq_len}")
    if args.target_seq_len > total_seq_len:
        raise ValueError(f"Error: target_seq_len cannot be greater than total sequence length {total_seq_len}! Current: {args.target_seq_len}")

    # =========================================================
    # Logic to dynamically determine max_length
    # =========================================================
    if args.target_seq_len <= 512:
        actual_max_len = 514
        print(f"target_seq_len is {args.target_seq_len} (<=512), using the model's native optimal max_length = 514")
    else:
        actual_max_len = 2048
        print(f"target_seq_len is {args.target_seq_len} (>512), enabling long sequence extrapolation support, using max_length = 2048")

    # Initialize model
    model_paras = dict(max_length=actual_max_len, dim=768, layer=12, num_attention_heads=12)
    print(f"Loading model: {args.model_path} ...")
    model = structRFM_infer(from_pretrained=args.model_path, **model_paras)
    model.model.eval()

    # =========================================================
    # First extract 4-column records matching the old script
    # =========================================================
    print(f"Extracting {total_seq_len}bp sequences from genome site: {args.input} ...")
    failed_rows = []
    fasta_reader = FastaReader(args.reference_fasta)

    records = parse_genome_site_and_extract_seq(
        file_path=args.input,
        fasta_reader=fasta_reader,
        flank=args.flank,
        failed_rows=failed_rows
    )
    fasta_reader.close()

    total_count = len(records)
    print(f"Parsing complete. {total_count} available records. Will crop the central {args.target_seq_len} bp.")

    fail_path = os.path.splitext(args.output)[0] + ".failed_rows.tsv"
    if failed_rows:
        fail_df = pd.DataFrame(failed_rows)
        fail_df.to_csv(fail_path, sep="\t", index=False)
        print(f"[WARN] {len(failed_rows)} records failed to extract, saved to: {fail_path}")

    # =========================================================
    # Initialization of chunked storage logic
    # =========================================================
    final_output_path = args.output
    temp_chunk_dir = final_output_path.replace('.', '_') + "_temp_chunks"

    if os.path.exists(final_output_path):
        print(f"✔ Final file {final_output_path} already exists, skipping task.")
        return

    os.makedirs(temp_chunk_dir, exist_ok=True)

    start_idx = 0
    existing_files = sorted(glob.glob(os.path.join(temp_chunk_dir, "chunk_*.parquet")))
    if existing_files:
        for f in existing_files:
            start_idx += pq.read_metadata(f).num_rows
        print(f"⏭ Detected {start_idx} processed records, resuming from breakpoint...")

    if start_idx < total_count:
        results = []
        # Pre-calculate coordinates
        crop_start = (total_seq_len - args.target_seq_len) // 2
        crop_end = crop_start + args.target_seq_len
        center_start = (args.target_seq_len - 5) // 2
        center_end = center_start + 5
        center_idx = args.target_seq_len // 2

        with torch.no_grad():
            for i, record in enumerate(records[start_idx:], start=start_idx):
                chrom, pos, kmer, seq = record
                seq = seq.upper().replace('T', 'U')
                kmer = kmer.upper().replace('T', 'U')

                if len(seq) != total_seq_len:
                    continue

                sub_seq = seq[crop_start:crop_end]

                # Only require the center position of the model input to remain 'A'
                if sub_seq[center_idx] != 'A':
                    raise ValueError(
                        f"Validation failed: {chrom}:{pos}. The center base of the model input is not A, got {sub_seq[center_idx]}"
                    )

                # Extract features
                features = model.extract_raw_feature(sub_seq, return_all=False, output_attentions=False)

                seq_level_emb = features[0].cpu().numpy()
                sub_seq_emb = features[1:-1].cpu().numpy()
                center_5mer_emb = sub_seq_emb[center_start:center_end]

                # Strictly keep the exact same output fields as the old version
                results.append({
                    "chr": chrom,
                    "pos": pos,
                    "5mer": kmer,
                    "center_5mer_emb": center_5mer_emb.astype(np.float16).tolist(),
                    "seq_level_emb": seq_level_emb.astype(np.float16).tolist()
                })

                if len(results) >= args.chunk_size or (i + 1) == total_count:
                    chunk_name = f"chunk_{i + 1 - len(results)}_to_{i + 1}.parquet"
                    save_chunk_to_parquet(results, os.path.join(temp_chunk_dir, chunk_name))
                    print(f"Progress: {i + 1}/{total_count}, saved chunk: {chunk_name}")
                    results = []

    # =========================================================
    # Automatically merge into a single file and clean up
    # =========================================================
    print(f"\nAll chunks processed, rapidly merging into a single file: {final_output_path} ...")

    try:
        df_final = pd.read_parquet(temp_chunk_dir)
        df_final.to_parquet(final_output_path, compression='snappy', index=False)
        print(f"Merge successful! Total rows: {len(df_final)}")

        print(f"Cleaning up temporary chunk directory: {temp_chunk_dir} ...")
        shutil.rmtree(temp_chunk_dir)
        print("Workspace cleanup complete.")

    except Exception as e:
        print(f"Merge failed: {e}")
        print(f"⚠ Warning: Chunked data remains in {temp_chunk_dir}, you can try to merge manually.")

    print(f"Task executed successfully! Results saved to: {final_output_path}\n")


if __name__ == '__main__':
    main()