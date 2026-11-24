import sys
import numpy as np
from astropy.time import Time
from orbitize import planet_detection

BINARY_PARAMS = {
    'sma_as1': 17.4930,
    'inc1': np.deg2rad(79.2430),
    'aop1': np.deg2rad(231.519),
    'pan1': np.deg2rad(205.073),
    'P1': 79.7620,
    'ecc1': 0.51460,
    'epp1': Time(1955.564, format='decimalyear').mjd,
    'plx': 750.81,
    'M0': 1.0788,
    'M1': 0.9092,
}
PA_AXIS = np.deg2rad(110.0)
SEP_ERR_MAS = 0.0038
TIME_OFFSET_MJD = 60467.0
MIN_PERIOD_YR = 0.05


def load_dataset(data_file, time_file, row):
    with open(time_file) as f:
        t = np.array([float(v) for v in f.readline().split()]) + TIME_OFFSET_MJD
    with open(data_file) as f:
        for i, line in enumerate(f):
            if i == row:
                y = np.array([float(v) for v in line.split()]) * 1000
                break
    yerr = np.full_like(t, SEP_ERR_MAS)
    return t, y, yerr


def main(data_file, time_file, row, fap=1e-3):
    t, y, yerr = load_dataset(data_file, time_file, row)
    baseline_yr = (t.max() - t.min()) / 365.25
    min_period = MIN_PERIOD_YR
    max_period = max(0.2, 2.5 * baseline_yr)
    tau_ref_epoch = 0.5 * (t.min() + t.max())

    result = planet_detection.search_planets(
        t, y, yerr, PA_AXIS, min_period, max_period, tau_ref_epoch, BINARY_PARAMS,
        fap=fap, max_planets=5, polish_width_frac=0.01, polish_npts=9)

    print(f'{len(result["planets"])} planet(s) found at FAP <= {fap:g}')
    for p, p_fap in zip(result['planets'], result['faps']):
        ecc = p['sesw'] ** 2 + p['secw'] ** 2
        print(f'  P = {p["P"]:.4f} yr   mass = {p["mass"]:.3f} M_earth   '
              f'ecc = {ecc:.3f}   tau_pa = {p["tau_pa"]:.4f}   FAP = {p_fap:.3g}')
    return result


if __name__ == '__main__':
    args = sys.argv[1:]
    if '--help' in args or len(args) < 3:
        print('Usage: python mara_planet_detection.py data_file time_file row')
        print('Optional parameters: --fap X')
        sys.exit()
    fap = 1e-3 if '--fap' not in args else float(args[args.index('--fap') + 1])
    main(args[0], args[1], int(args[2]), fap=fap)
