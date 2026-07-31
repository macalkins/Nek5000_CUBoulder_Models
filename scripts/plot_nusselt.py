#!/usr/bin/env python3
"""
Plot Nusselt number time series from Nek5000 CSV output.

Reads nusselt.csv (columns: time, Nu_in, Nu_out) and produces:
  - Time series of inner and outer Nusselt numbers
  - Cumulative running average
  - Statistics (mean, RMS fluctuation) over a selected time window

Usage:
    python plot_nusselt.py                          # default: nusselt.csv
    python plot_nusselt.py -f mycase.csv            # custom file
    python plot_nusselt.py --start 0.01             # skip spinup before t=0.01
    python plot_nusselt.py --start 0.01 --end 0.05  # specific time window
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys


def load_csv(filename):
    """Load Nek5000 CSV, handling Fortran line-wrapping.

    Fortran writes may split a single data row across multiple lines.
    We join all non-header lines, split by commas, and reshape based
    on the number of header columns.
    """
    with open(filename, 'r') as f:
        header = f.readline().strip()
        body = f.read()
    col_names = [c.strip() for c in header.split(',')]
    ncols = len(col_names)

    # Join all lines, split by comma, parse as floats
    values = [float(v) for v in body.replace('\n', ',').split(',') if v.strip()]
    nrows = len(values) // ncols
    data = np.array(values[:nrows * ncols]).reshape(nrows, ncols)
    return col_names, data


def main():
    parser = argparse.ArgumentParser(description='Plot Nusselt number time series')
    parser.add_argument('-f', '--file', default='nusselt.csv', help='CSV file')
    parser.add_argument('--start', type=float, default=None,
                        help='Start time for averaging (skip spinup)')
    parser.add_argument('--end', type=float, default=None, help='End time')
    parser.add_argument('-o', '--output', default='nusselt_vs_time.png',
                        help='Output plot filename')
    args = parser.parse_args()

    col_names, data = load_csv(args.file)
    time = data[:, 0]
    Nu_in = data[:, 1]
    Nu_out = data[:, 2] if data.shape[1] > 2 else None

    # Time window for statistics
    t0 = args.start if args.start is not None else time[0]
    t1 = args.end if args.end is not None else time[-1]
    mask = (time >= t0) & (time <= t1)

    # Cumulative running average (from t0 onward)
    Nu_in_cumavg = np.cumsum(Nu_in[mask]) / np.arange(1, np.sum(mask) + 1)
    if Nu_out is not None:
        Nu_out_cumavg = np.cumsum(Nu_out[mask]) / np.arange(1, np.sum(mask) + 1)

    # Statistics
    Nu_in_mean = np.mean(Nu_in[mask])
    Nu_in_rms = np.std(Nu_in[mask])
    print(f"Time window: [{t0:.6e}, {t1:.6e}] ({np.sum(mask)} samples)")
    print(f"  Nu_in:  mean = {Nu_in_mean:.6f},  rms = {Nu_in_rms:.4e}")
    if Nu_out is not None:
        Nu_out_mean = np.mean(Nu_out[mask])
        Nu_out_rms = np.std(Nu_out[mask])
        print(f"  Nu_out: mean = {Nu_out_mean:.6f},  rms = {Nu_out_rms:.4e}")
        print(f"  (Nu_in + Nu_out) / 2 = {0.5*(Nu_in_mean + Nu_out_mean):.6f}")

    # Plot: single figure
    fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)

    ax.plot(time, Nu_in, 'b-', linewidth=1.0, label=r'$Nu_{in}$')
    if Nu_out is not None:
        ax.plot(time, Nu_out, 'r-', linewidth=1.0, label=r'$Nu_{out}$')
    if args.start is not None:
        ax.axvline(t0, color='k', linestyle='--', alpha=0.3, label='avg start')

    # Title with averages
    title = f'$Nu_{{in}}$ = {Nu_in_mean:.4f}'
    if Nu_out is not None:
        title += f',  $Nu_{{out}}$ = {Nu_out_mean:.4f}'
    ax.set_title(title)

    ax.set_xlabel('time')
    ax.set_ylabel('Nu')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == '__main__':
    main()
