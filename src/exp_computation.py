import csv
import os
import subprocess
import sys

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# (E, B) pairs sorted by u 
eb_pairs = [(1, 600), (10, 600), (1, 50), (20, 600), (1, 10),
            (10, 50), (20, 50), (10, 10), (20, 10)]
iid_values = [1, 0]

c = 0.1
lr = 0.01
epochs = 500
target = 0.95
seed = 1
client_size = 600  # images per client

results_dir = 'save/results'
figures_dir = 'save/figures'


# u = E * n / (K * B), the local updates per client per round
def updates(e, b):
    return e * client_size // b


def batch_label(b):
    return '∞' if b == client_size else str(b)


def result_path(iid, e, b):
    return os.path.join(results_dir, 'mlp_iid{}_C{}_B{}_E{}_lr{}.csv'.format(
        iid, c, b, e, lr))


def run(iid, e, b):
    cmd = [sys.executable, 'src/federated_main.py',
           '--model=mlp', '--dataset=mnist',
           f'--iid={iid}', f'--frac={c}', f'--local_bs={b}',
           f'--local_ep={e}', f'--lr={lr}', f'--epochs={epochs}',
           f'--target={target}', f'--seed={seed}', '--verbose=0']
    if torch.cuda.is_available():
        cmd.append('--gpu=cuda:0')
    subprocess.run(cmd, check=True)


def run_all():
    for iid in iid_values:
        for e, b in eb_pairs:
            if os.path.exists(result_path(iid, e, b)):
                print(f'skip (already done): iid={iid} E={e} B={b}')
                continue
            print(f'run: iid={iid} E={e} B={b} u={updates(e, b)}')
            run(iid, e, b)


def dist_label(iid):
    return 'IID' if iid == 1 else 'Non-IID'


def load_curve(iid, e, b):
    path = result_path(iid, e, b)
    if not os.path.exists(path):
        return None
    rounds, acc = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            rounds.append(int(row['round']))
            acc.append(float(row['test_accuracy']))
    return np.array(rounds), np.maximum.accumulate(np.array(acc))


def rounds_to_target(curve):
    if curve is None:
        return None
    _, best = curve
    hit = np.where(best >= target)[0]
    if len(hit) == 0:
        return None
    i = hit[0]
    if i == 0:
        return 1.0
    y0, y1 = best[i - 1], best[i]
    return i + (target - y0) / (y1 - y0)


def cell(iid, e, b):
    curve = load_curve(iid, e, b)
    if curve is None:
        return 'missing'
    r = rounds_to_target(curve)
    return 'N/R' if r is None else f'{r:.0f}'


def save_table():
    header = ['E', 'B', 'u'] + [dist_label(iid) for iid in iid_values]
    rows = [[e, batch_label(b), updates(e, b)] + [cell(iid, e, b) for iid in iid_values]
            for e, b in eb_pairs]

    with open(os.path.join(figures_dir, 'table_q2.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print('\nRounds to reach {:.0f}% test accuracy (C = {})'.format(100 * target, c))
    print(' | '.join(str(h) for h in header))
    for row in rows:
        print(' | '.join(str(v) for v in row))


def plot_curves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, iid in zip(axes, iid_values):
        for e, b in eb_pairs:
            curve = load_curve(iid, e, b)
            if curve is None:
                continue
            rounds, best = curve
            ax.plot(rounds, 100 * best,
                    label=f'E = {e}, B = {batch_label(b)} (u = {updates(e, b)})')
        ax.axhline(100 * target, color='gray', linestyle='--', linewidth=1)
        ax.set_title(dist_label(iid))
        ax.set_xlabel('Communication rounds')
        ax.set_ylabel('Test accuracy (%)')
        ax.set_ylim(top=100)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'q2_fig1_accuracy_vs_rounds.png'), dpi=200)
    plt.close(fig)


def plot_bars():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(eb_pairs))
    labels = [f'{updates(e, b)}\n(E{e}, B{batch_label(b)})' for e, b in eb_pairs]
    for ax, iid in zip(axes, iid_values):
        heights = [rounds_to_target(load_curve(iid, e, b)) for e, b in eb_pairs]
        reached = [h for h in heights if h is not None]
        top = max(reached) if reached else 1
        ax.bar(x, [h if h is not None else 0 for h in heights], color='steelblue')
        for xi, h in zip(x, heights):
            if h is None:
                ax.text(xi, top * 0.02, 'N/R', ha='center', va='bottom')
            else:
                ax.text(xi, h, f'{h:.0f}', ha='center', va='bottom')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, top * 1.15)
        ax.set_title(dist_label(iid))
        ax.set_xlabel('Local updates per round, u')
        ax.set_ylabel('Rounds to reach {:.0f}%'.format(100 * target))
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'q2_fig2_rounds_vs_u.png'), dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(figures_dir, exist_ok=True)
    run_all()
    save_table()
    plot_curves()
    plot_bars()
    print(f'\nTable and figures saved to {figures_dir}')