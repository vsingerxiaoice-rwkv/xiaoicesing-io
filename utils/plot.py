import matplotlib.pyplot as plt
import numpy as np
import torch

LINE_COLORS = ['w', 'r', 'y', 'cyan', 'm', 'b', 'lime']


def spec_to_figure(spec, vmin=None, vmax=None):
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    fig = plt.figure(figsize=(12, 9))
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    plt.tight_layout()
    return fig


def spec_f0_to_figure(spec, f0s, figsize=None):
    max_y = spec.shape[1]
    if isinstance(spec, torch.Tensor):
        spec = spec.detach().cpu().numpy()
        f0s = {k: f0.detach().cpu().numpy() for k, f0 in f0s.items()}
    f0s = {k: f0 / 10 for k, f0 in f0s.items()}
    fig = plt.figure(figsize=(12, 6) if figsize is None else figsize)
    plt.pcolor(spec.T)
    for i, (k, f0) in enumerate(f0s.items()):
        plt.plot(f0.clip(0, max_y), label=k, c=LINE_COLORS[i], linewidth=1, alpha=0.8)
    plt.legend()
    return fig


def dur_to_figure(dur_gt, dur_pred, txt):
    if isinstance(dur_gt, torch.Tensor):
        dur_gt = dur_gt.cpu().numpy()
    if isinstance(dur_pred, torch.Tensor):
        dur_pred = dur_pred.cpu().numpy()
    dur_gt = dur_gt.astype(np.int64)
    dur_pred = dur_pred.astype(np.int64)
    dur_gt = np.cumsum(dur_gt)
    dur_pred = np.cumsum(dur_pred)
    width = max(12, min(48, len(txt) // 2))
    fig = plt.figure(figsize=(width, 8))
    plt.vlines(dur_pred, 12, 22, colors='r', label='pred')
    plt.vlines(dur_gt, 0, 10, colors='b', label='gt')
    for i in range(len(txt)):
        shift = (i % 8) + 1
        plt.text((dur_pred[i-1] + dur_pred[i]) / 2 if i > 0 else dur_pred[i] / 2, 12 + shift, txt[i],
                 size=16, horizontalalignment='center')
        plt.text((dur_gt[i-1] + dur_gt[i]) / 2 if i > 0 else dur_gt[i] / 2, shift, txt[i],
                 size=16, horizontalalignment='center')
        plt.plot([dur_pred[i], dur_gt[i]], [12, 10], color='black', linewidth=2, linestyle=':')
    plt.yticks([])
    plt.xlim(0, max(dur_pred[-1], dur_gt[-1]))
    fig.legend()
    fig.tight_layout()
    return fig


def f0_to_figure(f0_gt, f0_pred=None):
    if isinstance(f0_gt, torch.Tensor):
        f0_gt = f0_gt.cpu().numpy()
    if isinstance(f0_pred, torch.Tensor):
        f0_pred = f0_pred.cpu().numpy()
    fig = plt.figure()
    if f0_pred is not None:
        plt.plot(f0_pred, color='green', label='pred')
    plt.plot(f0_gt, color='r', label='gt')
    plt.legend()
    plt.tight_layout()
    return fig
