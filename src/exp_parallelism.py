import csv
import os
import subprocess
import sys

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

c_values = [0.0, 0.1, 0.2, 0.5, 1.0]
b_values = [600, 10]
iid_values = [1, 0]

lr = 0.01
epochs = 200
target = 0.95
local_ep = 1
seed = 1

results_dir = 'save/results'
figures_dir = 'save/figures'


def result_path(iid, c, b):
    return os.path.join(results_dir, 'mlp_iid{}_C{}_B{}_E{}_lr{}.csv'.format(
        iid, c, b, local_ep, lr))


def run(iid, c, b):
    cmd = [sys.executable, 'src/federated_main.py',
           '--model=mlp', '--dataset=mnist',
           f'--iid={iid}', f'--frac={c}', f'--local_bs={b}',
           f'--local_ep={local_ep}', f'--lr={lr}', f'--epochs={epochs}',
           f'--target={target}', f'--seed={seed}', '--verbose=0']
    if torch.cuda.is_available():
        cmd.append('--gpu=cuda:0')
    subprocess.run(cmd, check=True)


def run_all():
    for iid in iid_values:
        for b in b_values:
            for c in c_values:
                if os.path.exists(result_path(iid, c, b)):
                    print(f'skip (already done): iid={iid} B={b} C={c}')
                    continue
                print(f'run: iid={iid} B={b} C={c}')
                run(iid, c, b)


def panels():
    return [(iid, b) for iid in iid_values for b in b_values]


def panel_label(iid, b):
    dist = 'IID' if iid == 1 else 'Non-IID'
    batch = 'B = ∞' if b == 600 else f'B = {b}'
    return f'{dist}, {batch}'


def load_curve(iid, c, b):
    path = result_path(iid, c, b)
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


def save_table():
    header = ['C'] + [panel_label(iid, b) for iid, b in panels()]
    rows = []
    for c in c_values:
        row = [c]
        for iid, b in panels():
            curve = load_curve(iid, c, b)
            r = rounds_to_target(curve)
            if curve is None:
                row.append('missing')
            elif r is None:
                row.append('N/R')
            else:
                row.append(f'{r:.1f}')
        rows.append(row)

    with open(os.path.join(figures_dir, 'table_q1.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print('\nRounds to reach {:.0f}% test accuracy'.format(100 * target))
    print(' | '.join(str(h) for h in header))
    for row in rows:
        print(' | '.join(str(v) for v in row))


def plot_curves():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (iid, b) in zip(axes.flat, panels()):
        for c in c_values:
            curve = load_curve(iid, c, b)
            if curve is None:
                continue
            rounds, best = curve
            ax.plot(rounds, 100 * best, label=f'C = {c}')
        ax.axhline(100 * target, color='gray', linestyle='--', linewidth=1)
        ax.set_title(panel_label(iid, b))
        ax.set_xlabel('Communication rounds')
        ax.set_ylabel('Test accuracy (%)')
        ax.set_ylim(top=100)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'fig1_accuracy_vs_rounds.png'), dpi=200)
    plt.close(fig)


def plot_bars():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    x = np.arange(len(c_values))
    for ax, (iid, b) in zip(axes.flat, panels()):
        heights = [rounds_to_target(load_curve(iid, c, b)) for c in c_values]
        reached = [h for h in heights if h is not None]
        top = max(reached) if reached else 1
        ax.bar(x, [h if h is not None else 0 for h in heights], color='steelblue')
        for xi, h in zip(x, heights):
            if h is None:
                ax.text(xi, top * 0.02, 'N/R', ha='center', va='bottom')
            else:
                ax.text(xi, h, f'{h:.0f}', ha='center', va='bottom')
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in c_values])
        ax.set_ylim(0, top * 1.15)
        ax.set_title(panel_label(iid, b))
        ax.set_xlabel('Client fraction C')
        ax.set_ylabel('Rounds to reach {:.0f}%'.format(100 * target))
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'fig2_rounds_to_target.png'), dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(figures_dir, exist_ok=True)
    run_all()
    save_table()
    plot_curves()
    plot_bars()
    print(f'\nTable and figures saved to {figures_dir}')