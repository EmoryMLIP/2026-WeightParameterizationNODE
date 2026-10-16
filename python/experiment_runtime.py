"""
Runtime utilities shared across experiment scripts."""

import os
import builtins
import torch


def print(*args, **kwargs):
    """Print with flush=True by default, so SLURM/nohup logs update live
    rather than buffering until the process exits."""
    kwargs.setdefault('flush', True)
    return builtins.print(*args, **kwargs)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def relative_error(pred, true, eps=1e-12):
    """
    Relative error over a trajectory segment: ||pred - true||_2 / ||true||_2,
    matching the MATLAB convention (relErrTrain/relErrVal/relErrTest in
    runExperiment.m), so the two pipelines report comparably.
    """
    num = torch.sqrt(torch.sum((pred - true) ** 2))
    den = torch.sqrt(torch.sum(true ** 2)) + eps
    return (num / den).item()


class RunningAverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
