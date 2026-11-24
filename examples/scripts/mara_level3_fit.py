import os
import sys
import time
import multiprocessing as mp
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import orbitize
from orbitize import driver
from astropy import constants as const
from astropy import units as u
from astropy.time import Time
from orbitize import priors, results, system, sampler, read_input
from copy import deepcopy
from scipy.stats import truncnorm

RAYLEIGH_ECC_SIGMA = 0.06

LOAD_POOL_FACTOR = 10


def set_priors(sys_obj, param_file, swapAB=False, restrict_sma=None):
    """

    :param sys_obj: system object for running MCMC
    :param param_file: csv file with stellar (and optionally planetary) parameters
    :param  swapAB: a bool to determine which star the planet is orbiting, the primary (default) or the secondary
    :param restrict_sma: optional bounds for the planet sma priors
    :return: nothing, but modifies the priors of sys_obj
    """
    with open(param_file, 'r') as f:
        lines = f.readlines()
        data = lines[1].split(',')
        sys_obj.sys_priors[sys_obj.param_idx['plx']] = priors.GaussianPrior(float(data[17]), float(data[18]))
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

        for i in range(2, len(lines) - 2):
            if restrict_sma is None:
                sys_obj.sys_priors[sys_obj.param_idx['sma' + str(i)]] = priors.LogUniformPrior(.1, 10)
            else:
                sys_obj.sys_priors[sys_obj.param_idx['sma' + str(i)]] = priors.LogUniformPrior(restrict_sma[i-2], restrict_sma[i-1])
            sys_obj.sys_priors[sys_obj.param_idx['m' + str(i)]] = priors.LogUniformPrior(.01, 100)
            sys_obj.sys_priors[sys_obj.param_idx['ecc' + str(i)]] = priors.RayleighPrior(RAYLEIGH_ECC_SIGMA)
    return


def set_rand_pos(sampler_obj, param_file, check_priors=True, swapAB=False):
    walker_centers = []
    uniform_check = []
    walker_1sigmas = []
    with open(param_file, 'r') as f:
        lines = f.readlines()
        data = lines[2].split(',')
        if swapAB:
            data = lines[1].split(',')
        walker_centers += [float(data[1]), float(data[3]), float(data[5]) * np.pi/180%(2*np.pi),
                           float(data[7]) * np.pi/180%(2*np.pi), float(data[9]) * np.pi/180%(2*np.pi), float(data[11])]
        uniform_check += [float(data[1]), float(data[3]), float(data[5]), float(data[7]), float(data[9]), float(data[11])]
        walker_1sigmas += [float(data[2]), float(data[4]), float(data[6]) * np.pi/180,
                           float(data[8]) * np.pi/180, float(data[10]) * np.pi/180, float(data[12])]
        for line in lines[4:]:
            data = line.split(',')
            walker_centers += [float(data[1]), float(data[3]), float(data[5]) * np.pi/180%(2*np.pi),
                               float(data[7]) * np.pi/180%(2*np.pi), float(data[9]) * np.pi/180%(2*np.pi), float(data[11])]
            uniform_check += [float(data[1]), float(data[3]), float(data[5]), float(data[7]), float(data[9]), float(data[11])]
            walker_1sigmas += [float(data[2]), float(data[4]), float(data[6]) * np.pi/180,
                               float(data[8]) * np.pi/180, float(data[10]) * np.pi/180, float(data[12])]

        data = lines[1].split(',')
        walker_centers += [float(data[17])]
        uniform_check += [float(data[17])]
        walker_1sigmas += [float(data[18])]

        data = lines[2].split(',')
        walker_centers += [float(data[15])]
        uniform_check += [float(data[15])]
        walker_1sigmas += [float(data[16])]
        for line in lines[4:]:
            data = line.split(',')
            walker_centers += [float(data[13])]
            uniform_check += [float(data[13])]
            walker_1sigmas += [float(data[14])]
        data = lines[1].split(',')
        if swapAB:
            data = lines[2].split(',')
        walker_centers += [float(data[13])]
        uniform_check += [float(data[13])]
        walker_1sigmas += [float(data[14])]

        walker_centers = np.array(walker_centers)
        uniform_check = np.array(uniform_check)
        walker_1sigmas = np.array(walker_1sigmas)

        curr_pos_shape = sampler_obj.curr_pos.shape

        new_pos = np.random.standard_normal(curr_pos_shape) * walker_1sigmas + walker_centers

        sma_idxs = np.array([sampler_obj.system.param_idx['sma' + str(i)] for i in range(2, len(lines) - 2)])
        new_pos[:, :, sma_idxs] = np.sort(new_pos[:, :, sma_idxs], axis=-1)
    
    unchanged_categories = []
    for i, lab in enumerate(sampler_obj.system.labels):
        if np.abs(uniform_check[i] + 999) < 1e-5:
            unchanged_categories.append(lab)
    print('Unchanged Categories:', unchanged_categories)
    for cat in unchanged_categories:
        cat_ind = sampler_obj.system.param_idx[cat]
        new_pos[:, :, cat_ind] = np.copy(sampler_obj.curr_pos[:, :, cat_ind])
        
    uniform_categories = []
    for i, lab in enumerate(sampler_obj.system.labels):
        if np.abs(uniform_check[i] + 111) < 1e-5:
            uniform_categories.append(lab)
    print('Uniform Categories:', uniform_categories)
    for cat in uniform_categories:
        cat_ind = sampler_obj.system.param_idx[cat]
        low = {'inc': 0, 'pan': 0, 'aop': 0, 'm': 0, 'tau': 0, 'sma': 0}[cat[:-1]]
        high = {'inc': np.pi, 'pan': 2*np.pi, 'aop': 2*np.pi, 'm': 0, 'tau': 0, 'sma': 0}[cat[:-1]]
        new_pos[:, :, cat_ind] = np.random.uniform(low=low, high=high, size=new_pos[:, :, cat_ind].shape)

    # Ensure starting planet params are within priors
    # Use truncated normal for planet parameters to keep walkers within prior bounds
    num_planets = len(lines) - 4  # lines[4:] gives one line per planet
    planet_orbital_start = 6  # first 6 params are binary orbital params
    
    # Orbital params: 6 per planet starting at index 6
    # Mass params: one per planet at indices -2, -3, ... (index -1 is binary primary, skip)
    planet_param_idxs = list(range(planet_orbital_start, planet_orbital_start + num_planets * 6))
    planet_param_idxs += [-(i + 2) for i in range(num_planets)]  # -2, -3, ...
    
    for idx in planet_param_idxs:
        label = sampler_obj.system.labels[idx]
        if label in unchanged_categories or label in uniform_categories:
            continue
        param_idx = sampler_obj.system.param_idx[label]
        low = np.min(sampler_obj.curr_pos[:, :, param_idx])
        high = np.max(sampler_obj.curr_pos[:, :, param_idx])
        a = (low - walker_centers[idx]) / walker_1sigmas[idx]
        b = (high - walker_centers[idx]) / walker_1sigmas[idx]
        new_pos[:, :, param_idx] = truncnorm.rvs(a, b, loc=walker_centers[idx],
                                                  scale=walker_1sigmas[idx],
                                                  size=curr_pos_shape[:2])
    sma_idxs = np.array([sampler_obj.system.param_idx['sma' + str(i)] for i in range(2, len(lines) - 2)])
    new_pos[:, :, sma_idxs] = np.sort(new_pos[:, :, sma_idxs], axis=-1)
    print('Position shape:', curr_pos_shape)

    if check_priors:
        for i in range(len(walker_centers)):
            print([sampler_obj.system.labels[i]])
            print(vars(sampler_obj.system.sys_priors[i]))
            print(np.min(new_pos[:, :, i]), np.max(new_pos[:, :, i]))
    return new_pos


def rayleigh_ecc_weight(e_samples, sigma=RAYLEIGH_ECC_SIGMA):
    """The Rayleigh acceptance weight, normalized to its own maximum."""
    w = (e_samples / sigma ** 2) * np.exp(-e_samples ** 2 / (2 * sigma ** 2))
    return w / w.max()


def seed_from_loaded_posterior(sampler_obj, load_file, num_secondary_bodies,
                               apply_ecc_filter=True):
    """Set walker positions from the top lnlike samples of a completed fit, after
    reject-sampling its eccentricity down to the Rayleigh prior.

    `apply_ecc_filter=False` for a fit that already carried the Rayleigh prior
    in its chain, so the prior is not applied twice.
    """
    print('Using top lnlike positions from', load_file)
    lr = results.Results()
    lr.load_results(load_file)

    keep = np.ones(len(lr.lnlike), dtype=bool)
    if apply_ecc_filter:
        for i in range(2, num_secondary_bodies + 1):
            ecc = lr.post[:, sampler_obj.system.param_idx['ecc' + str(i)]]
            keep &= np.random.random(len(ecc)) < rayleigh_ecc_weight(ecc)
        print('Rayleigh eccentricity filter kept', int(keep.sum()), 'of',
              len(keep), 'posterior samples')
    else:
        print('Loaded run already carried the Rayleigh prior -- no filter applied,',
              len(keep), 'posterior samples')

    load_n_temps = sampler_obj.curr_pos.shape[0]
    load_n_walkers = sampler_obj.curr_pos.shape[1]
    load_n_needed = load_n_temps * load_n_walkers

    kept_idx = np.flatnonzero(keep)
    load_n_available = kept_idx.size
    load_pool_size = min(LOAD_POOL_FACTOR * load_n_needed, load_n_available)

    order = kept_idx[np.argsort(lr.lnlike[kept_idx])]
    top_params = lr.post[order[-load_pool_size:]]

    # Without replacement unless the survivors are too few to fill every slot.
    load_replace = load_pool_size < load_n_needed
    if load_replace:
        print('  only', load_n_available, 'survivors for', load_n_needed,
              'walker slots -- drawing with replacement')
    chosen = np.random.choice(load_pool_size, size=load_n_needed, replace=load_replace)
    sampler_obj.curr_pos = np.copy(
        top_params[chosen].reshape(load_n_temps, load_n_walkers, -1))

    print('  reseeded', load_n_temps, 'temperatures x', load_n_walkers,
          'walkers from the top', load_pool_size, 'of', load_n_available,
          'samples by lnlike')


def run_mcmc(filename, param_file, num_temps=10, num_walkers=32, num_threads=mp.cpu_count(), load_file=None, steps=10000,
             burn_steps=1000, thin=1, swapAB=False, save_name=None, coplanar=True, command=None, PA=110 * np.pi / 180,
             num_plot_bins=0, tau_ref_epoch=61041, restrict_sma=None, loaded_rayleigh=False, last_pos=False):

    """

    :param  filename: name of the csv file which contains delta sep information
    :param param_file: name of the csv file which contains stellar (and optionally planetary) parameters
    :param  num_temps: number of temperatures
    :param  num_walkers: number of walkers per temperature
    :param  num_threads: number of threads to use for multithreading
    :param  load_file: save file to load walker positions from; defaults to None, which will cause walkers to be
                        randomly distributed around the values in param_file
    :param  loaded_rayleigh: True if the run named by load_file already carried the Rayleigh eccentricity prior in
                        its chain, in which case its posterior is not reweighted again before seeding
    :param  last_pos: a bool to continue from the last walker positions in load_file instead of its top lnlike samples
    :param  steps: number of steps for MCMC
    :param  burn_steps: number of steps for the burn-in
    :param  thin: thinning parameter for MCMC
    :param  swapAB: a bool to determine which star the planet is orbiting, the primary (default) or the secondary
    :param save_name: Directory and beginning of common names for this run. Defaults to using the same directory as the
                      data file and the common name "mara_run_"
    :return: No return, but results, corner plot and delta rho plot are saved
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
    fitting_basis = 'SHERAEM_coplanar' if coplanar else 'SHERAEM'
    data_table = orbitize.read_input.read_file(filename)
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

    print('Modifying priors and starting positions...')
    # Set priors and starting params
    set_priors(sys_obj, param_file, swapAB=swapAB, restrict_sma=restrict_sma)

    assert sys_obj.fit_secondary_mass
    assert sys_obj.track_planet_perturbs

    sampler_obj = sampler.MCMC(
        sys_obj,
        num_threads=num_threads,
        num_temps=num_temps,
        num_walkers=num_walkers,
    )

    new_pos = set_rand_pos(sampler_obj, param_file, swapAB=swapAB)

    sampler_obj.curr_pos = np.copy(new_pos)

    if load_file is not None:
        if last_pos:
            print('Using walker positions from', load_file)
            lr = results.Results()
            lr.load_results(load_file)
            sampler_obj.curr_pos = np.copy(lr.curr_pos)
        else:
            seed_from_loaded_posterior(sampler_obj, load_file, num_secondary_bodies,
                                       apply_ecc_filter=not loaded_rayleigh)


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
        delta_rho_filename = run_name + '_delta_rho_plot_' + str(iteration) + '.png'
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
        delta_rho_filename = run_name + '_delta_rho_plot_' + str(iteration) + '.png'

    with open(run_name + 'run_details_' + str(iteration) + '.txt', 'w') as f:
        f.write("Run Details:\n\n")
        f.write(f"Bash Command: {command}\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Filename: {filename}\n")
        f.write(f"Parameter File: {param_file}\n")
        f.write(f"Tau Reference Epoch: {tau_ref_epoch}\n")
        f.write(f"Number of Temperatures: {num_temps}\n")
        f.write(f"Number of Walkers: {num_walkers}\n")
        f.write(f"Number of Threads: {num_threads}\n")
        f.write(f"Load File: {load_file}\n")
        f.write(f"Load Last Position: {last_pos}\n")
        f.write(f"MARA: {os.path.dirname(orbitize.__file__)}\n")
        f.write(f"Steps: {steps}\n")
        f.write(f"Burn Steps: {burn_steps}\n")
        f.write(f"Thin: {thin}\n")
        f.write(f"SwapAB: {swapAB}\n")
        f.write(f"Run Name: {run_name}\n")
        f.write(f"Run Duration: {run_duration_hours:.2f} hours\n")
        f.write("----------------------Misc----------------------\n")
        f.write(f"Results File: {hdf5_filename}\n")
        f.write(f"Corner File: {corner_filename}\n")
        f.write(f"Delta Rho File: {delta_rho_filename}\n")
        f.write(f"Fitting Basis: {fitting_basis}\n")
        f.write(f"PA: {PA}\n")
        f.write(f"Number of Bins for Plotting: {num_plot_bins}\n")
        f.write("----------------------Posterior Summary----------------------\n")
        sampler_obj.results.print_results(file=f)
        

    figs = sampler_obj.examine_chains()
    for i, fig in enumerate(figs):
        filename = run_name + '_' + sys_obj.labels[i] + '_chain_plot_' + str(steps) + '.png'
        fig.savefig(filename)

    sampler_obj.chop_chains(burn_steps)

    print('MCMC completed, saving results to', hdf5_filename)
    sampler_obj.results.save_results(hdf5_filename)

    corner_plot = sampler_obj.results.plot_corner(bins=100)
    plt.savefig(corner_filename)

    plt.figure()
    post = sampler_obj.results.post
    actual_data_table = deepcopy(sampler_obj.results.system.data_table)
    data_table = actual_data_table.copy(copy_data=False)
    data_table.remove_rows(range(0, len(data_table)))
    for i in np.linspace(np.min(actual_data_table['epoch']), np.max(actual_data_table['epoch']), 1000):
        data_table.add_row([i, 1, 0.0, 0.0, 0.0, 0.0, np.nan, 'seppa', 'defsp'])
    sampler_obj.results.system.data_table = data_table

    for i in range(100):
        rand_idx = np.random.randint(0, post.shape[0])
        params = post[rand_idx, :]
        model, _ = sampler_obj.results.system.compute_model(params)
        model = model[:, 0]
        plt.plot(Time(sampler_obj.results.system.data_table['epoch'], format='mjd').decimalyear, model,
                 color='cornflowerblue', alpha=0.5)
    plt.grid(True)
    plt.errorbar(Time(actual_data_table['epoch'], format='mjd').decimalyear, actual_data_table['quant1'],
                 actual_data_table['quant1_err'], ls='', color='red')

    if num_plot_bins > 0:
        binned_epochs = np.array([np.mean(x) for x in np.array_split(Time(actual_data_table['epoch'],
                                                                        format='mjd').decimalyear, num_plot_bins)])
        binned_seps = np.array([np.mean(x) for x in np.array_split(actual_data_table['quant1'], num_plot_bins)])
        binned_errs = np.array([np.mean(x) for x in np.array_split(actual_data_table['quant1_err'], num_plot_bins)])
        binned_errs /= np.sqrt(len(actual_data_table['quant1']) / num_plot_bins)
        plt.scatter(binned_epochs, binned_seps, binned_errs, ls='', color='purple', marker='*')

    params = post[np.argmax(sampler_obj.results.lnlike), :]
    model, _ = sampler_obj.results.system.compute_model(params)
    model = model[:, 0]
    plt.plot(Time(sampler_obj.results.system.data_table['epoch'], format='mjd').decimalyear, model,
             color='black')

    plt.savefig(delta_rho_filename)

    return


def main():
    args = sys.argv[1:]
    if '--help' in args or len(args) == 0:
        print('Required parameters: data_filename parameter_filename')
        print('Optional parameters: --num_temps X --num_walkers X --num_threads X --load_file X --last_pos')
        print('                     --num_steps X --burn_steps X --thin X --swapAB --save_name X')
        print('                     --loaded_rayleigh  (the --load_file run already had the Rayleigh ecc prior)')
        return 0
    filename = args[0]
    param_file = args[1]
    command = ' '.join(sys.argv).strip()
    print('Data file:', filename)
    print('Parameter file:', param_file)

    # Assign optional parameters
    num_temps = 5 if '--num_temps' not in args else int(args[args.index('--num_temps') + 1])
    num_walkers = 32 if '--num_walkers' not in args else int(args[args.index('--num_walkers') + 1])
    num_threads = mp.cpu_count() if '--num_threads' not in args else int(args[args.index('--num_threads') + 1])
    steps = 7500 if '--num_steps' not in args else int(args[args.index('--num_steps') + 1])
    load_file = None if '--load_file' not in args else args[args.index('--load_file') + 1]
    burn_steps = 1500 if '--burn_steps' not in args else int(args[args.index('--burn_steps') + 1])
    thin = 1 if '--thin' not in args else int(args[args.index('--thin') + 1])
    swapAB = False if '--swapAB' not in args else True
    save_name = None if '--save_name' not in args else args[args.index('--save_name') + 1]
    coplanar = False if '--coplanar' not in args else True
    PA = 110 * np.pi / 180 if '--PA' not in args else float(args[args.index('--PA') + 1]) * np.pi / 180
    num_plot_bins = 0 if '--bins' not in args else int(args[args.index('--bins') + 1])
    tau_ref_epoch = 61041 if '--tau_ref_epoch' not in args else float(args[args.index('--tau_ref_epoch') + 1])
    restrict_sma = None if '--restrict_sma' not in args else [float(x) for x in args[args.index('--restrict_sma') + 1].split(',')]
    loaded_rayleigh = False if '--loaded_rayleigh' not in args else True
    last_pos = False if '--last_pos' not in args else True

    run_mcmc(filename, param_file, num_temps=num_temps, num_walkers=num_walkers, num_threads=num_threads,
             load_file=load_file, steps=steps, burn_steps=burn_steps, thin=thin, swapAB=swapAB, save_name=save_name,
             coplanar=coplanar, command=command, PA=PA, num_plot_bins=num_plot_bins, tau_ref_epoch=tau_ref_epoch,
             restrict_sma=restrict_sma, loaded_rayleigh=loaded_rayleigh, last_pos=last_pos)
    return 0


if __name__ == "__main__":
    main()
