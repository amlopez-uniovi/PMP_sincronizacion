"""
sync_signals.py

Utility script to estimate an affine time-mapping between two time series
and remap the target signal to the reference timeline using cross-correlation
maximization (grid search + optional refinement).

Usage (from project root):
  python sync_signals.py --ref data/PMP1050_W1_C.csv --tgt data/PMP1050_W1_M.csv

CSV format expected: two columns: timestamp, value (header optional). Timestamp can be in seconds or milliseconds.

Produces printed a,b estimates and saves two PNGs:
 - sync_overlay.png  (overlay reference and warped target)
 - sync_xcorr.png    (cross-correlation after warping)

Dependencies: numpy, scipy, matplotlib, pandas (for csv reading)

Author: generated helper
"""

import argparse
import numpy as np
import os

try:
    from scipy.signal import fftconvolve
    from scipy.interpolate import interp1d
    from scipy.optimize import differential_evolution, minimize
except Exception as e:
    raise ImportError("This script requires scipy (signal, interpolate, optimize). Install it in your environment.")

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import pandas as pd
except Exception:
    pd = None


def to_seconds_if_ms(ts):
    ts = np.asarray(ts, dtype=float)
    return ts / 1000.0 if ts.max() > 1e9 else ts


def read_csv_timeseries(path, ts_col=0, val_col=1, skip_header=True):
    if pd is None:
        # fallback to numpy
        data = np.genfromtxt(path, delimiter=',', skip_header=1 if skip_header else 0)
        return data[:, ts_col], data[:, val_col]
    df = pd.read_csv(path, header=0 if skip_header else None)
    cols = df.columns.tolist()
    ts = df.iloc[:, ts_col].to_numpy(dtype=float)
    val = df.iloc[:, val_col].to_numpy(dtype=float)
    return ts, val


def interp_to_ref(t_tgt, x_tgt, t_ref, a, b):
    t_mapped = a * np.asarray(t_tgt, dtype=float) + float(b)
    f = interp1d(t_mapped, x_tgt, bounds_error=False, fill_value=np.nan)
    return f(t_ref)


def normalized_xcorr_peak(y_ref, y_tgt):
    mask = ~np.isnan(y_ref) & ~np.isnan(y_tgt)
    if mask.sum() < 10:
        return np.nan, None, None
    y1 = y_ref[mask] - np.mean(y_ref[mask])
    y2 = y_tgt[mask] - np.mean(y_tgt[mask])
    s1 = np.std(y1)
    s2 = np.std(y2)
    if s1 == 0 or s2 == 0:
        return np.nan, None, None
    y1n = y1 / s1
    y2n = y2 / s2
    corr = fftconvolve(y2n, y1n[::-1], mode='full')
    peak_idx = np.argmax(np.abs(corr))
    peak_val = corr[peak_idx]
    lag = peak_idx - (len(y1n) - 1)
    return peak_val, lag, corr


def objective_neg_peak(params, t_ref, x_ref, t_tgt, x_tgt):
    a, b = params
    y_tgt_interp = interp_to_ref(t_tgt, x_tgt, t_ref, a, b)
    peak_val, _, _ = normalized_xcorr_peak(x_ref, y_tgt_interp)
    if np.isnan(peak_val):
        return 1e6
    return -abs(peak_val)


def estimate_affine_by_xcorr(t_ref, x_ref, t_tgt, x_tgt,
                             a_grid=(0.98, 1.02, 41), b_grid=None,
                             refine=True, subsample=10):
    t_ref_s = to_seconds_if_ms(t_ref)
    t_tgt_s = to_seconds_if_ms(t_tgt)

    t0 = max(t_ref_s[0], t_tgt_s[0])
    t1 = min(t_ref_s[-1], t_tgt_s[-1])
    if t1 <= t0:
        raise ValueError("No temporal overlap between series")

    idx_ref = np.arange(0, len(t_ref_s), max(1, int(subsample)))
    t_ref_sub = t_ref_s[idx_ref]
    x_ref_sub = x_ref[idx_ref]

    a_min, a_max, a_n = a_grid
    a_vals = np.linspace(a_min, a_max, int(a_n))

    total_range = t_ref_s[-1] - t_ref_s[0] if (t_ref_s[-1] - t_ref_s[0]) > 0 else 1.0
    if b_grid is None:
        b_min = -0.05 * total_range
        b_max = 0.05 * total_range
        b_n = 41
    else:
        b_min, b_max, b_n = b_grid
    b_vals = np.linspace(b_min, b_max, int(b_n))

    best = {'score': -np.inf, 'a': None, 'b': None, 'peak': None}
    for a in a_vals:
        for b in b_vals:
            y_tgt_interp = interp_to_ref(t_tgt_s, x_tgt, t_ref_sub, a, b)
            peak_val, _, _ = normalized_xcorr_peak(x_ref_sub, y_tgt_interp)
            if np.isnan(peak_val):
                continue
            score = abs(peak_val)
            if score > best['score']:
                best.update(score=score, a=a, b=b, peak=peak_val)

    a0, b0 = best['a'], best['b']
    res = None
    if refine and a0 is not None:
        bounds = [(a_min, a_max), (b_min, b_max)]
        try:
            res = minimize(lambda p: objective_neg_peak(p, t_ref_s, x_ref, t_tgt_s, x_tgt),
                           x0=[a0, b0], bounds=bounds, method='L-BFGS-B', options={'maxiter': 200})
            if res.success:
                a_hat, b_hat = res.x
                y_tgt_interp = interp_to_ref(t_tgt_s, x_tgt, t_ref_s, a_hat, b_hat)
                peak_val, lag, corr = normalized_xcorr_peak(x_ref, y_tgt_interp)
                return a_hat, b_hat, peak_val, lag, res
        except Exception:
            # fall back to grid result
            pass
    return best['a'], best['b'], best['peak'], None, res


def resample_with_affine_mapping(t_tgt, x_tgt, a, b, t_ref):
    return interp_to_ref(t_tgt, x_tgt, t_ref, a, b)


def plot_results(t_ref, x_ref, t_tgt, x_tgt_warped, corr_after, out_prefix='sync'):
    if plt is None:
        print('matplotlib not available; skipping plots')
        return
    # overlay
    plt.figure(figsize=(10, 4))
    plt.plot(to_seconds_if_ms(t_ref), x_ref, label='reference', alpha=0.8)
    plt.plot(to_seconds_if_ms(t_ref), x_tgt_warped, label='target (warped)', alpha=0.8)
    plt.legend()
    plt.title('Overlay reference vs warped target')
    plt.xlabel('time (s)')
    plt.grid(True)
    out1 = f'{out_prefix}_overlay.png'
    plt.savefig(out1, bbox_inches='tight')
    print('Saved', out1)

    # xcorr plot
    peak_val, lag, corr = normalized_xcorr_peak(x_ref, x_tgt_warped)
    if corr is not None:
        lags = np.arange(- (len(corr) // 2), (len(corr) // 2) + 1)[:len(corr)]
        # approximate lag axis in seconds by median dt
        dt = np.median(np.diff(to_seconds_if_ms(t_ref)))
        lags_s = lags * dt
        plt.figure(figsize=(10,4))
        plt.plot(lags_s, corr)
        plt.axvline(lag * dt if lag is not None else 0, color='C1', linestyle='--')
        plt.title(f'Cross-correlation after warp (peak={peak_val:.3f})')
        plt.xlabel('lag (s)')
        plt.grid(True)
        out2 = f'{out_prefix}_xcorr.png'
        plt.savefig(out2, bbox_inches='tight')
        print('Saved', out2)


def main():
    p = argparse.ArgumentParser(description='Estimate affine time mapping between two time series using xcorr')
    p.add_argument('--ref', required=True, help='CSV file for reference series (timestamp,value)')
    p.add_argument('--tgt', required=True, help='CSV file for target series (timestamp,value)')
    p.add_argument('--a_min', type=float, default=0.98)
    p.add_argument('--a_max', type=float, default=1.02)
    p.add_argument('--a_n', type=int, default=41)
    p.add_argument('--b_min', type=float, default=None, help='min offset (s); default ±5% of ref range')
    p.add_argument('--b_max', type=float, default=None)
    p.add_argument('--subsample', type=int, default=10, help='decimation for grid search')
    p.add_argument('--no-refine', dest='refine', action='store_false')
    p.add_argument('--out-prefix', default='sync')
    args = p.parse_args()

    ts_ref, x_ref = read_csv_timeseries(args.ref)
    ts_tgt, x_tgt = read_csv_timeseries(args.tgt)

    ts_ref_s = to_seconds_if_ms(ts_ref)
    total_range = ts_ref_s[-1] - ts_ref_s[0] if ts_ref_s[-1] - ts_ref_s[0] > 0 else 1.0
    if args.b_min is None:
        b_min = -0.05 * total_range
        b_max = 0.05 * total_range
        b_n = 41
    else:
        b_min = args.b_min
        b_max = args.b_max if args.b_max is not None else -b_min
        b_n = 41

    print('Running grid+refine search...')
    a_grid = (args.a_min, args.a_max, args.a_n)
    b_grid = (b_min, b_max, b_n)

    a_hat, b_hat, peak, lag, res = estimate_affine_by_xcorr(
        ts_ref, x_ref, ts_tgt, x_tgt,
        a_grid=a_grid, b_grid=b_grid, refine=args.refine, subsample=args.subsample)

    print('Estimate result: a=', a_hat, ' b=', b_hat, ' peak=', peak)
    if res is not None:
        print('Optimizer success:', res.success, 'message:', res.message)

    # produce warped target on reference times
    t_ref_s_full = to_seconds_if_ms(ts_ref)
    x_tgt_warped = resample_with_affine_mapping(ts_tgt, x_tgt, a_hat, b_hat, t_ref_s_full)

    # compute correlation after
    mask = ~np.isnan(x_tgt_warped)
    if mask.sum() >= 10:
        corr_after = np.corrcoef(x_ref[mask], x_tgt_warped[mask])[0,1]
        print('Correlation after warp (Pearson):', corr_after)
    else:
        corr_after = np.nan
        print('Too few valid samples after warp')

    plot_results(ts_ref, x_ref, ts_tgt, x_tgt_warped, corr_after, out_prefix=args.out_prefix)


if __name__ == '__main__':
    main()
