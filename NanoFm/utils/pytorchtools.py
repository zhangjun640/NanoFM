import numpy as np
import torch
import os  # Ensure the os module is imported


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_acc, model, optimizer):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_acc, model, optimizer):
        '''Saves model when validation loss decrease.'''
        # --- [Core modification] ---
        # Before saving the model, ensure that the target directory exists
        # os.path.dirname(self.path) gets the directory path of the file
        # exist_ok=True ensures that no error is raised if the directory already exists
        if self.path:
            try:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
            except OSError as e:
                print(f"Error creating directory {os.path.dirname(self.path)}: {e}")
                # You can choose to raise an exception or perform other error handling here
                return  # If directory creation fails, do not continue saving

        state = {'net': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        torch.save(state, self.path)
        self.val_loss_min = val_acc

    def recount(self):
        self.counter = 0