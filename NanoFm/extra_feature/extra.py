from __future__ import absolute_import
import argparse
import os
import sys
import re
import h5py
import glob
import multiprocessing
import traceback
import tempfile

import numpy as np
from scipy import interpolate
from statsmodels import robust
from tqdm import tqdm


# --- Helper functions and data extraction functions (Same as previous version, no modification needed) ---

def parse_cigar(cigar_string):
    if cigar_string == '*': return []
    return [(int(length), op) for length, op in re.findall(r'(\d+)([MIDNSH])', cigar_string)]


def convert_base_name(base_name):
    merge_bases = {'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T', 'M': '[AC]', 'V': '[ACG]', 'R': '[AG]', 'H': '[ACT]',
                   'W': '[AT]', 'D': '[AGT]', 'S': '[CG]', 'B': '[CGT]', 'Y': '[CT]', 'N': '[ACGT]', 'K': '[GT]'}
    pattern = ''
    for base in base_name: pattern += merge_bases.get(base, base)
    return pattern


def process_signal_segment(segment, target_len=65):
    segment_len = len(segment)
    if segment_len == 0: return np.zeros(target_len)
    if segment_len < target_len:
        padding_needed = target_len - segment_len
        pad_left = padding_needed // 2
        pad_right = padding_needed - pad_left
        return np.pad(segment, (pad_left, pad_right), 'constant', constant_values=0)
    else:
        indices = np.sort(np.random.choice(segment_len, target_len, replace=False))
        return segment[indices]


def get_base_quality(reference, sam_file_path):
    # Input is now the path to the SAM file
    base_quality_dict, reference_dict = {}, {}
    with open(reference) as f:
        for line in f:
            if line.startswith('>'):
                contig = line.strip().split()[0][1:]
                reference_dict[contig] = ""
            else:
                reference_dict[contig] += line.strip()

    with open(sam_file_path) as f:
        for line in f:
            if line.startswith('@'): continue
            try:
                items = line.strip().split("\t")
                read_id, flag, rname, pos, cigar, seq, qual = items[0], int(items[1]), items[2], int(items[3]), items[
                    5], items[9], items[10]
                if rname == "*" or (flag & 4): continue

                alignment_map, cigar_ops = [], parse_cigar(cigar)
                read_pos, ref_pos = 0, pos - 1
                for length, op in cigar_ops:
                    if op == 'M':
                        for _ in range(length): alignment_map.append(
                            {'type': 'M', 'read_idx': read_pos, 'ref_idx': ref_pos}); read_pos += 1; ref_pos += 1
                    elif op == 'D':
                        for _ in range(length): alignment_map.append({'type': 'D', 'ref_idx': ref_pos}); ref_pos += 1
                    elif op == 'I':
                        for _ in range(length): alignment_map.append({'type': 'I', 'read_idx': read_pos}); read_pos += 1
                    elif op == 'S':
                        read_pos += length

                if not alignment_map: continue
                mismatches, insertions, deletions, qualities = [], [], [], []
                for i, event in enumerate(alignment_map):
                    if event['type'] == 'I': continue
                    adj_to_ins = (i > 0 and alignment_map[i - 1]['type'] == 'I') or (
                                i < len(alignment_map) - 1 and alignment_map[i + 1]['type'] == 'I')
                    insertions.append(1 if adj_to_ins else 0)
                    if event['type'] == 'M':
                        mismatches.append(1 if seq[event['read_idx']] != reference_dict[rname][event['ref_idx']] else 0)
                        deletions.append(0)
                        qualities.append(ord(qual[event['read_idx']]) - 33 if qual != '*' else 0)
                    elif event['type'] == 'D':
                        mismatches.append(0);
                        deletions.append(1);
                        qualities.append(0)

                if not mismatches: continue
                aligned_ref_len = len(mismatches)
                ref_end = pos - 1 + aligned_ref_len
                if ref_end > len(reference_dict[rname]): continue
                aligned_ref_seq = reference_dict[rname][pos - 1: ref_end]

                if flag == 0:
                    base_quality_dict[read_id] = [rname, pos, aligned_ref_seq, qualities, mismatches, insertions,
                                                  deletions]
            except (IndexError, KeyError):
                continue
    return base_quality_dict


def get_signal_from_fast5(fast5_path, basecall_group, basecall_subgroup):
    try:
        with h5py.File(fast5_path, 'r') as fast5_data:
            raw_signal = list(fast5_data['/Raw/Reads/'].values())[0]['Signal'][()]
            events_path = f'/Analyses/{basecall_group}/{basecall_subgroup}/Events'
            events = fast5_data[events_path][()]
            corr_start_rel_to_raw = fast5_data[events_path].attrs['read_start_rel_to_raw']
            event_starts = events['start'] + corr_start_rel_to_raw
            event_lengths = events['length']
            signal_segments = [raw_signal[start: start + length] for start, length in zip(event_starts, event_lengths)]
            read_id = os.path.basename(fast5_path).split('.')[0]
            return read_id, signal_segments
    except Exception:
        return None, None


def process_read_to_5mer_features(read_id, alignment_data, signal_segments, args):
    output_lines = []
    try:
        chr_name, start, reference_sequence, base_quality_list, mismatch_list, insertion_list, deletion_list = alignment_data
        kmer_filter = convert_base_name(args.motif)
        full_length_signal = np.array([val for seg in signal_segments for val in seg], dtype=int)
        if len(full_length_signal) == 0: return []

        full_length_signal_uniq = np.unique(full_length_signal)
        median_val, mad_val = np.median(full_length_signal_uniq), robust.mad(full_length_signal_uniq)
        if mad_val == 0: return []

        for index in range(args.clip, len(reference_sequence) - args.clip):
            kmer_ref_sequence = reference_sequence[index - 2:index + 3]
            if len(kmer_ref_sequence) < 5 or not re.search(kmer_filter, kmer_ref_sequence): continue

            kmer_raw_signal_float = [np.array(s, dtype=float) for s in signal_segments[index - 2:index + 3]]
            scaled_kmer_signal = [(s - median_val) / mad_val for s in kmer_raw_signal_float]

            mean = [np.round(np.mean(s), 3) for s in scaled_kmer_signal]
            std = [np.round(np.std(s), 3) for s in scaled_kmer_signal]
            median = [np.round(np.median(s), 3) for s in scaled_kmer_signal]
            length = [len(s) for s in scaled_kmer_signal]
            processed_signals = [process_signal_segment(s, target_len=65) for s in scaled_kmer_signal]

            kmer_base_quality = "|".join(map(str, base_quality_list[index - 2:index + 3]))
            kmer_mismatch = "|".join(map(str, mismatch_list[index - 2:index + 3]))
            kmer_insertion = "|".join(map(str, insertion_list[index - 2:index + 3]))
            kmer_deletion = "|".join(map(str, deletion_list[index - 2:index + 3]))

            line = "%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
                read_id, chr_name, start + index, kmer_ref_sequence,
                "|".join(map(str, mean)), "|".join(map(str, std)), "|".join(map(str, median)),
                "|".join(map(str, length)), kmer_base_quality,
                "|".join(map(lambda x: f"{x:.4f}", processed_signals[0])),
                "|".join(map(lambda x: f"{x:.4f}", processed_signals[1])),
                "|".join(map(lambda x: f"{x:.4f}", processed_signals[2])),
                "|".join(map(lambda x: f"{x:.4f}", processed_signals[3])),
                "|".join(map(lambda x: f"{x:.4f}", processed_signals[4])),
                kmer_mismatch, kmer_insertion, kmer_deletion
            )
            output_lines.append(line)
    except Exception:
        pass
    return output_lines


def worker_process_fast5(args_tuple):
    f5_path, base_quality_dict, args = args_tuple
    read_id, signal_segments = get_signal_from_fast5(f5_path, args.basecall_group, args.basecall_subgroup)
    if read_id and read_id in base_quality_dict:
        alignment_data = base_quality_dict[read_id]
        return process_read_to_5mer_features(read_id, alignment_data, signal_segments, args)
    return []


# --- Core workflow modification section ---

# MODIFIED: Added tqdm to this function to implement a progress bar within the batch
def run_batch_processing(args, fast5_files, sam_header, batch_sam_records, output_handle):
    """
    Process a batch of data (a small group of FAST5 files and their corresponding SAM records).
    """
    # Create a temporary, small SAM file
    with tempfile.NamedTemporaryFile(mode='w', delete=True, suffix=".sam") as tmp_sam_file:
        tmp_sam_file.write(sam_header)
        for record in batch_sam_records:
            tmp_sam_file.write(record)
        tmp_sam_file.flush()  # Ensure all content is written to disk

        # 1. Load features from this small SAM file
        base_quality_dict = get_base_quality(args.reference, tmp_sam_file.name)

        # 2. Multi-process the FAST5 files in this batch
        tasks = [(f5_path, base_quality_dict, args) for f5_path in fast5_files]
        with multiprocessing.Pool(processes=args.process) as pool:

            # --- NEW: Add a tqdm progress bar for processing FAST5 files within the batch ---
            # Use the leave=False parameter to automatically clear the inner progress bar upon completion, keeping the interface clean
            pbar_batch = tqdm(pool.imap_unordered(worker_process_fast5, tasks),
                              total=len(tasks),
                              desc=f"Batch {os.path.basename(os.path.dirname(fast5_files[0]))}",
                              leave=False)

            for result_lines in pbar_batch:
                if result_lines:
                    for line in result_lines:
                        output_handle.write(line)
            # --- End of NEW section ---

def main(args):
    """
    New main function implementing a 'divide and conquer' strategy.
    """
    print("Step 1: Finding all FAST5 subdirectories...")
    # Get all subdirectories
    subdirs = [d.path for d in os.scandir(args.fast5) if d.is_dir()]
    if not subdirs:
        print(f"Warning: No subdirectories found in {args.fast5}. Treating it as a single directory.")
        subdirs = [args.fast5]
    print(f"Found {len(subdirs)} subdirectories to process as batches.")

    print("Step 2: Reading SAM header...")
    sam_header = ""
    all_sam_records = {}
    with open(args.sam, 'r') as f:
        for line in f:
            if line.startswith('@'):
                sam_header += line
            else:
                read_id = line.split('\t')[0]
                if read_id not in all_sam_records:
                    all_sam_records[read_id] = []
                all_sam_records[read_id].append(line)
    print(f"Done. Read header and {len(all_sam_records)} unique read IDs from SAM file.")

    # Prepare to write to the final output file
    output_dir = os.path.dirname(args.output)
    if output_dir: os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w") as out_f:
        # Create an overall progress bar
        pbar = tqdm(subdirs, total=len(subdirs), desc="Overall Progress")
        for subdir_path in pbar:
            pbar.set_description(f"Processing {os.path.basename(subdir_path)}")

            # a. Get all read_ids in the directory
            fast5_in_batch = glob.glob(os.path.join(subdir_path, '*.fast5'))
            if not fast5_in_batch:
                continue
            read_ids_in_batch = {os.path.basename(f).split('.')[0] for f in fast5_in_batch}

            # b. Filter corresponding SAM records from memory
            sam_records_for_batch = []
            for read_id in read_ids_in_batch:
                if read_id in all_sam_records:
                    sam_records_for_batch.extend(all_sam_records[read_id])

            if not sam_records_for_batch:
                continue

            # c. Call the batch processing function
            run_batch_processing(args, fast5_in_batch, sam_header, sam_records_for_batch, out_f)

    print("All tasks completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract 5-mer features directly from FAST5 and SAM files using a batch-processing strategy.')

    parser.add_argument('-o', '--output', required=True, help="Final output feature file.")
    parser.add_argument('--fast5', required=True, help='Top-level directory containing FAST5 subdirectories.')
    parser.add_argument('-r', '--reference', required=True, help='Reference transcripts FASTA file.')
    parser.add_argument('--sam', required=True, help='The complete SAM file from alignment.')
    parser.add_argument('--motif', required=True, help="Sequence motif to filter (e.g., RRACH).")
    parser.add_argument('--clip', type=int, default=10, help='Bases to clip at both ends of a read.')
    parser.add_argument('-p', '--process', type=int, default=1, help='Number of processes to use FOR EACH BATCH.')
    parser.add_argument('--basecall_group', default="RawGenomeCorrected_000", help='Basecall group in FAST5 file.')
    parser.add_argument('--basecall_subgroup', default='BaseCalled_template', help='Basecall subgroup in FAST5 file.')

    args = parser.parse_args()
    main(args)