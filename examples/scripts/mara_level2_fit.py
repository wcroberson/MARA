import os
import sys
import time
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import orbitize
from orbitize import priors, results, system, sampler, read_input
from scipy.stats import truncnorm

RAYLEIGH_ECC_SIGMA = 0.06

LOAD_POOL_FACTOR = 10

# Reference epoch for tau (MJD): 2026-01-01, near the middle of the simulated
# data (2024.43-2027.42). The Level 3 script uses the same one.
TAU_REF_EPOCH = 61041


def set_priors(sys_obj, param_file, swapAB=False):
    """

    :param sys_obj: system object for running MCMC
    :param param_file: csv file with stellar (and optionally planetary) parameters
    :param  swapAB: a bool to determine which star the planet is orbiting, the primary (default) or the secondary
    :return: nothing, but modifies the priors of sys_obj
    """
    with open(param_file, 'r') as f:
        lines = f.readlines()
        data = lines[1].split(',')
        sys_obj.sys_priors[sys_obj.param_idx['plx']] = priors.GaussianPrior(float(data[17]), float(data[18]))
        sys_obj.sys_priors[sys_obj.param_idx['ra_com_ref']] = priors.GaussianPrior(float(data[19]), float(data[20]))
        sys_obj.sys_priors[sys_obj.param_idx['dec_com_ref']] = priors.GaussianPrior(float(data[21]), float(data[22]),
                                                                                    no_negatives=False)
        sys_obj.sys_priors[sys_obj.param_idx['pmra_com_ref']] = priors.GaussianPrior(float(data[23]), float(data[24]),
                                                                                     no_negatives=False)
        sys_obj.sys_priors[sys_obj.param_idx['pmdec_com_ref']] = priors.GaussianPrior(float(data[25]), float(data[26]),
                                                                                      no_negatives=False)
        sys_obj.sys_priors[sys_obj.param_idx['rv_com_ref']] = priors.GaussianPrior(float(data[27]), float(data[28]),
                                                                                   no_negatives=False)
        if swapAB:
            data = lines[2].split(',')
        sys_obj.sys_priors[sys_obj.param_idx['M0']] = priors.GaussianPrior(float(data[13]), float(data[14]))
        data = lines[2].split(',')
        if swapAB:
            data = lines[1].split(',')
        sys_obj.sys_priors[sys_obj.param_idx['sma_as1']] = priors.GaussianPrior(float(data[1]), float(data[2]))
        sys_obj.sys_priors[sys_obj.param_idx['ecc1']] = priors.TruncatedGaussianPrior(float(data[3]), float(data[4]),
                                                                                      0, 1)
        sys_obj.sys_priors[sys_obj.param_idx['inc1']] = priors.GaussianPrior(float(data[5]) * np.pi/180 % (2*np.pi),
                                                                             float(data[6]) * np.pi/180 % (2*np.pi))
        sys_obj.sys_priors[sys_obj.param_idx['aop1']] = priors.GaussianPrior(float(data[7]) * np.pi/180 % (2*np.pi),
                                                                             float(data[8]) * np.pi/180 % (2*np.pi))
        sys_obj.sys_priors[sys_obj.param_idx['pan1']] = priors.GaussianPrior(float(data[9]) * np.pi/180 % (2*np.pi),
                                                                             float(data[10]) * np.pi/180 % (2*np.pi))
        sys_obj.sys_priors[sys_obj.param_idx['epp1']] = priors.GaussianPrior(float(data[11]), float(data[12]))
        sys_obj.sys_priors[sys_obj.param_idx['P1']] = priors.GaussianPrior(float(data[15]), float(data[16]))

        # Planet masses are in Earth masses in this basis
        for i in range(2, len(lines) - 2):
            sys_obj.sys_priors[sys_obj.param_idx['sma' + str(i)]] = priors.LogUniformPrior(.1, 10)
            sys_obj.sys_priors[sys_obj.param_idx['m' + str(i)]] = priors.LogUniformPrior(.01, 100)
            sys_obj.sys_priors[sys_obj.param_idx['ecc' + str(i)]] = priors.RayleighPrior(RAYLEIGH_ECC_SIGMA)
    return


def set_rand_pos(sampler_obj, param_file, check_priors=True, swapAB=False):
    """
    Draws walker positions around the param file values, in the order of the system labels. A value of -999 leaves
    the parameter where orbitize's own draw from the prior put it. Planet parameters come from a truncated normal
    bounded by that draw, so no walker starts outside the prior.

    :param sampler_obj: sampler to generate positions for
    :param param_file: csv file with stellar (and optionally planetary) parameters
    :param check_priors: a bool to print each parameter's prior and the range of its walkers
    :param  swapAB: a bool to determine which star the planet is orbiting, the primary (default) or the secondary
    :return: array of walker positions, the same shape as sampler_obj.curr_pos
    """
    labels = sampler_obj.system.labels
    curr_pos = sampler_obj.curr_pos
    walker_centers = []
    walker_1sigmas = []
    with open(param_file, 'r') as f:
        lines = f.readlines()

    for line in [lines[1] if swapAB else lines[2]] + lines[4:]:
        data = line.split(',')
        walker_centers += [float(data[i]) for i in (1, 3, 5, 7, 9, 11)]
        walker_1sigmas += [float(data[i]) for i in (2, 4, 6, 8, 10, 12)]

    data = lines[1].split(',')
    walker_centers += [float(data[i]) for i in (17, 19, 21, 23, 25, 27)]
    walker_1sigmas += [float(data[i]) for i in (18, 20, 22, 24, 26, 28)]

    data = lines[2].split(',')
    walker_centers += [float(data[15])]
    walker_1sigmas += [float(data[16])]
    for line in lines[4:]:
        data = line.split(',')
        walker_centers += [float(data[13])]
        walker_1sigmas += [float(data[14])]
    data = lines[2 if swapAB else 1].split(',')
    walker_centers += [float(data[13])]
    walker_1sigmas += [float(data[14])]

    walker_centers = np.array(walker_centers)
    walker_1sigmas = np.array(walker_1sigmas)
    unchanged = np.abs(walker_centers + 999) < 1e-5
    print('Unchanged Categories:', [lab for lab, keep in zip(labels, unchanged) if keep])

    angles = np.array([lab[:3] in ('inc', 'aop', 'pan') for lab in labels])
    walker_centers[angles] = walker_centers[angles] * np.pi/180 % (2*np.pi)
    walker_1sigmas[angles] = walker_1sigmas[angles] * np.pi/180

    new_pos = np.random.standard_normal(curr_pos.shape) * walker_1sigmas + walker_centers

    # Six orbital params per planet follow the binary's six; the planet masses sit just before M0
    num_planets = len(lines) - 4
    planet_idxs = list(range(6, 6 + num_planets * 6)) + [len(labels) - i - 2 for i in range(num_planets)]
    for idx in planet_idxs:
        if unchanged[idx]:
            continue
        a = (np.min(curr_pos[:, :, idx]) - walker_centers[idx]) / walker_1sigmas[idx]
        b = (np.max(curr_pos[:, :, idx]) - walker_centers[idx]) / walker_1sigmas[idx]
        new_pos[:, :, idx] = truncnorm.rvs(a, b, loc=walker_centers[idx], scale=walker_1sigmas[idx],
                                           size=curr_pos.shape[:2])

    new_pos[:, :, unchanged] = curr_pos[:, :, unchanged]
    sma_idxs = np.array([sampler_obj.system.param_idx['sma' + str(i)] for i in range(2, len(lines) - 2)])
    new_pos[:, :, sma_idxs] = np.sort(new_pos[:, :, sma_idxs], axis=-1)
    print('Position shape:', curr_pos.shape)

    if check_priors:
        for i in range(len(labels)):
            print([labels[i]])
            print(vars(sampler_obj.system.sys_priors[i]))
            print(np.min(new_pos[:, :, i]), np.max(new_pos[:, :, i]))
    return new_pos


def seed_from_loaded_posterior(sampler_obj, load_file, last_pos=False):
    """
    Sets walker positions from a completed fit: either where its walkers finished, or the top lnlike samples, one
    sample per walker per temperature. Top lnlike samples are unique across walkers unless the loaded posterior is too
    small to fill every slot.

    :param sampler_obj: sampler to generate positions for
    :param load_file: results file of the completed fit
    :param last_pos: a bool to continue from the last walker positions instead of the top lnlike samples
    :return: array of walker positions, the same shape as sampler_obj.curr_pos
    """
    lr = results.Results()
    lr.load_results(load_file)

    if last_pos:
        print('Using last walker positions from', load_file)
        return np.copy(lr.curr_pos)

    print('Using top lnlike positions from', load_file)
    num_temps, num_walkers = sampler_obj.curr_pos.shape[:2]
    num_needed = num_temps * num_walkers
    pool_size = min(LOAD_POOL_FACTOR * num_needed, len(lr.lnlike))
    top_params = lr.post[np.argsort(lr.lnlike)[-pool_size:]]

    chosen = np.random.choice(pool_size, size=num_needed, replace=pool_size < num_needed)
    print('  reseeded', num_temps, 'temperatures x', num_walkers, 'walkers from the top', pool_size, 'of',
          len(lr.lnlike), 'samples by lnlike')
    return top_params[chosen].reshape(num_temps, num_walkers, -1)


def save_run_details(sampler_obj, details_filename, filename, param_file, load_file, last_pos, steps,
                     burn_steps, thin, swapAB, run_name, run_duration_hours, hdf5_filename, corner_filename):
    sys_obj = sampler_obj.system
    with open(details_filename, 'w') as f:
        f.write("Run Details:\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Filename: {filename}\n")
        f.write(f"Parameter File: {param_file}\n")
        f.write(f"Tau Reference Epoch: {sys_obj.tau_ref_epoch}\n")
        f.write(f"Number of Temperatures: {sampler_obj.num_temps}\n")
        f.write(f"Number of Walkers: {sampler_obj.num_walkers}\n")
        f.write(f"Number of Threads: {sampler_obj.num_threads}\n")
        f.write(f"Load File: {load_file}\n")
        f.write(f"Load Last Position: {last_pos}\n")
        f.write(f"Steps: {steps}\n")
        f.write(f"Burn Steps: {burn_steps}\n")
        f.write(f"Thin: {thin}\n")
        f.write(f"SwapAB: {swapAB}\n")
        f.write(f"Run Name: {run_name}\n")
        f.write(f"Run Duration: {run_duration_hours:.2f} hours\n")
        f.write("----------------------Misc----------------------\n")
        f.write(f"Results File: {hdf5_filename}\n")
        f.write(f"Corner File: {corner_filename}\n")
        f.write(f"Fitting Basis: {sys_obj.fitting_basis}\n")
        f.write(f"PA: {sys_obj.PA}\n")
        f.write("----------------------Posterior Summary----------------------\n")
        sampler_obj.results.print_results(file=f)


def run_mcmc(filename, param_file, num_temps=10, num_walkers=32, num_threads=mp.cpu_count(),
             load_file=None, last_pos=False, steps=10000, burn_steps=1000, thin=1, swapAB=False, save_name=None,
             coplanar=False, PA=110 * np.pi / 180, tau_ref_epoch=TAU_REF_EPOCH):

    """

    :param  filename: name of the csv file which contains the separation data
    :param param_file: name of the csv file which contains stellar (and optionally planetary) parameters
    :param  num_temps: number of temperatures
    :param  num_walkers: number of walkers per temperature
    :param  num_threads: number of threads to use for multithreading
    :param  load_file: save file to load walker positions from; defaults to None, which will cause walkers to be
                        randomly distributed around the values in param_file
    :param  last_pos: a bool to continue from the last walker positions in load_file instead of its top lnlike samples
    :param  steps: number of steps for MCMC
    :param  burn_steps: number of steps for the burn-in
    :param  thin: thinning parameter for MCMC
    :param  swapAB: a bool to determine which star the planet is orbiting, the primary (default) or the secondary
    :param save_name: Directory and beginning of common names for this run. Defaults to using the same directory as the
                      data file and the common name "mara_run_"
    :param coplanar: a bool to fit the planets in the plane of the binary
    :param PA: position angle the planet phase is defined at, in radians
    :param tau_ref_epoch: reference epoch for tau, in MJD
    :return: No return, but results and corner plot are saved
    """
    start_time = time.time()
    # system parameters
    with open(param_file, 'r') as f:
        lines = f.readlines()
        num_secondary_bodies = len(lines) - 3
        data = lines[1].split(',')
        plx = float(data[17])  # [mas]
        plx_err = float(data[18])  # [mas]
        if swapAB:
            print('Planets orbiting B instead of A...')
            data = lines[2].split(',')
        total_mass = float(data[13])  # [Msol]
        mass_err = float(data[14])  # [Msol]

    fit_secondary_mass = True
    data_table = orbitize.read_input.read_file(filename)
    fitting_basis = 'MARA_level2_coplanar' if coplanar else 'MARA_level2'
    sys_obj = system.System(
        num_secondary_bodies,
        data_table,
        total_mass,
        plx,
        fit_secondary_mass=fit_secondary_mass,
        mass_err=mass_err,
        plx_err=plx_err,
        tau_ref_epoch=tau_ref_epoch,
        fitting_basis=fitting_basis,
        PA=PA
    )
    print(sys_obj.fitting_basis)

    print('Modifying priors and starting positions...')
    set_priors(sys_obj, param_file, swapAB=swapAB)

    assert sys_obj.fit_secondary_mass
    assert sys_obj.track_planet_perturbs

    sampler_obj = sampler.MCMC(
        sys_obj,
        num_threads=num_threads,
        num_temps=num_temps,
        num_walkers=num_walkers,
    )

    if load_file is None:
        sampler_obj.curr_pos = set_rand_pos(sampler_obj, param_file, swapAB=swapAB)
    else:
        sampler_obj.curr_pos = seed_from_loaded_posterior(sampler_obj, load_file, last_pos=last_pos)

    print('Starting MCMC')
    print('Running with', num_temps, 'temperatures and', num_walkers, 'walkers for', steps, 'steps')
    # Run MCMC
    total_orbits = steps * num_walkers

    sampler_obj.run_sampler(total_orbits, burn_steps=0, thin=thin)

    end_time = time.time()
    run_duration_hours = (end_time - start_time) / 3600

    # Save results
    if save_name is None:
        run_name = filename.split('.')[0]
        iteration = 0
        hdf5_filename = run_name + '_mara_run_' + str(steps) + '_steps_' + str(iteration) + '.hdf5'
        while os.path.isfile(hdf5_filename):
            iteration += 1
            hdf5_filename = run_name + '_mara_run_' + str(steps) + '_steps_' + str(iteration) + '.hdf5'
        corner_filename = run_name + '_corner_plot_' + str(steps) + '_steps_' + str(iteration) + '.png'
    else:
        run_name = save_name
        save_dir = save_name[:save_name.rfind('/') + 1]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        iteration = 0
        hdf5_filename = run_name + '_results_' + str(iteration) + '.hdf5'
        while os.path.isfile(hdf5_filename):
            iteration += 1
            hdf5_filename = run_name + '_results_' + str(iteration) + '.hdf5'
        corner_filename = run_name + '_corner_' + str(iteration) + '.png'

    details_filename = run_name + 'run_details_' + str(iteration) + '.txt'
    save_run_details(sampler_obj, details_filename, filename, param_file, load_file, last_pos, steps, burn_steps,
                     thin, swapAB, run_name, run_duration_hours, hdf5_filename, corner_filename)

    figs = sampler_obj.examine_chains()
    for i, fig in enumerate(figs):
        fig.savefig(run_name + sys_obj.labels[i] + '_chain_plot_' + str(steps) + '.png')

    sampler_obj.chop_chains(burn_steps)

    print('MCMC completed, saving results to', hdf5_filename)
    sampler_obj.results.save_results(hdf5_filename)

    corner_plot = sampler_obj.results.plot_corner(bins=100)
    plt.savefig(corner_filename)


def main():
    args = sys.argv[1:]
    if '--help' in args or len(args) == 0:
        print('Required parameters: data_filename parameter_filename')
        print('Optional parameters: --num_temps X --num_walkers X --num_threads X --load_file X --last_pos')
        print('                     --num_steps X --burn_steps X --thin X --swapAB --save_name X')
        print('                     --coplanar --PA X --tau_ref_epoch X')
        return 0
    filename = args[0]
    param_file = args[1]
    print('Data file:', filename)
    print('Parameter file:', param_file)

    # Assign optional parameters
    num_temps = 5 if '--num_temps' not in args else int(args[args.index('--num_temps') + 1])
    num_walkers = 32 if '--num_walkers' not in args else int(args[args.index('--num_walkers') + 1])
    num_threads = mp.cpu_count() if '--num_threads' not in args else int(args[args.index('--num_threads') + 1])
    steps = 7500 if '--num_steps' not in args else int(args[args.index('--num_steps') + 1])
    load_file = None if '--load_file' not in args else args[args.index('--load_file') + 1]
    last_pos = False if '--last_pos' not in args else True
    burn_steps = 1500 if '--burn_steps' not in args else int(args[args.index('--burn_steps') + 1])
    thin = 1 if '--thin' not in args else int(args[args.index('--thin') + 1])
    swapAB = False if '--swapAB' not in args else True
    save_name = None if '--save_name' not in args else args[args.index('--save_name') + 1]
    coplanar = False if '--coplanar' not in args else True
    PA = 110 * np.pi / 180 if '--PA' not in args else float(args[args.index('--PA') + 1]) * np.pi / 180
    tau_ref_epoch = TAU_REF_EPOCH if '--tau_ref_epoch' not in args else float(args[args.index('--tau_ref_epoch') + 1])

    run_mcmc(filename, param_file, num_temps=num_temps, num_walkers=num_walkers, num_threads=num_threads,
             load_file=load_file, last_pos=last_pos, steps=steps, burn_steps=burn_steps, thin=thin, swapAB=swapAB,
             save_name=save_name, coplanar=coplanar, PA=PA, tau_ref_epoch=tau_ref_epoch)
    return 0


if __name__ == "__main__":
    main()
