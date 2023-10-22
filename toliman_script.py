# This script runs a modified orbitize! MCMC process to find planetary
# parameters based on micro-arcsecond delta separations for a binary
# star system. Currently parameters are hardcoded for Alpha Centauri,
# but this will be changed in future work.

# Import statements (orbitize import must the modified version)
import os
import sys
import multiprocessing as mp
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import orbitize
from orbitize import driver
from astropy import constants as const
from astropy import units as u
from astropy.time import Time
from orbitize import priors, results


def set_priors(my_driver, param_file):
    """
    Currently hard coded for Alpha Cen
    :param my_driver: driver for running MCMC
    :param param_file: csv file with stellar (and optionally planetary) parameters
    :return: nothing, but modifies the parameters for my_driver
    """
    with open(param_file, 'r') as f:
        lines = f.readlines()
        data = lines[1].split(',')
        my_driver.sampler.priors[my_driver.system.param_idx['m0']] = priors.GaussianPrior(float(data[13]),
                                                                                          float(data[14]))
        my_driver.sampler.priors[my_driver.system.param_idx['plx']] = priors.GaussianPrior(float(data[17]),
                                                                                           float(data[18]))
        data = lines[2].split(',')
        my_driver.sampler.priors[my_driver.system.param_idx['sma1']] = priors.GaussianPrior(float(data[1]),
                                                                                            float(data[2]))
        my_driver.sampler.priors[my_driver.system.param_idx['ecc1']] = priors.GaussianPrior(float(data[3]),
                                                                                            float(data[4]))
        my_driver.sampler.priors[my_driver.system.param_idx['inc1']] = priors.GaussianPrior(float(data[5]) * np.pi/180,
                                                                                            float(data[6]) * np.pi/180)
        my_driver.sampler.priors[my_driver.system.param_idx['aop1']] = priors.GaussianPrior(float(data[7]) * np.pi/180,
                                                                                            float(data[8]) * np.pi/180)
        my_driver.sampler.priors[my_driver.system.param_idx['pan1']] = priors.GaussianPrior(float(data[9]) * np.pi/180,
                                                                                            float(data[10]) * np.pi/180)
        my_driver.sampler.priors[my_driver.system.param_idx['tau1']] = priors.GaussianPrior(float(data[11]),
                                                                                            float(data[12]))
        my_driver.sampler.priors[my_driver.system.param_idx['m1']] = priors.GaussianPrior(float(data[13]),
                                                                                          float(data[14]))
        for i in range(2, len(lines) - 2):
            my_driver.sampler.priors[my_driver.system.param_idx['sma' + str(i)]] = priors.UniformPrior(.1, 10)
            my_driver.sampler.priors[my_driver.system.param_idx['m' + str(i)]] = priors.UniformPrior(.000000001, .1)
    return


def set_rand_pos(my_driver, param_file):
    walker_centers = []
    walker_1sigmas = []
    with open(param_file, 'r') as f:
        lines = f.readlines()
        data = lines[2].split(',')
        walker_centers += [float(data[1]), float(data[3]), float(data[5]) * np.pi/180,
                           float(data[7]) * np.pi/180, float(data[9]) * np.pi/180, float(data[11])]
        walker_1sigmas += [float(data[2]), float(data[4]), float(data[6]) * np.pi/180,
                           float(data[8]) * np.pi/180, float(data[10]) * np.pi/180, float(data[12])]
        for line in lines[4:]:
            data = line.split(',')
            walker_centers += [float(data[1]), float(data[3]), float(data[5]) * np.pi/180,
                               float(data[7]) * np.pi/180, float(data[9]) * np.pi/180, float(data[11])]
            walker_1sigmas += [float(data[2]), float(data[4]), float(data[6]) * np.pi/180,
                               float(data[8]) * np.pi/180, float(data[10]) * np.pi/180, float(data[12])]

        walker_centers += [float(lines[1].split(',')[17])]
        walker_1sigmas += [float(lines[1].split(',')[18])]
        data = lines[2].split(',')
        walker_centers += [float(data[15])]
        walker_1sigmas += [float(data[16])]
        for line in lines[4:]:
            data = line.split(',')
            walker_centers += [float(data[13])]
            walker_1sigmas += [float(data[14])]
        walker_centers += [float(lines[1].split(',')[13])]
        walker_1sigmas += [float(lines[1].split(',')[14])]

        walker_centers = np.array(walker_centers)
        walker_1sigmas = np.array(walker_1sigmas)

        curr_pos_shape = my_driver.sampler.curr_pos.shape
        print(walker_1sigmas)

        new_pos = np.random.standard_normal(curr_pos_shape) * walker_1sigmas + walker_centers

        sma_idxs = np.array([my_driver.system.param_idx['sma' + str(i)] for i in range(2, len(lines) - 2)])
        new_pos[:, :, sma_idxs] = np.sort(new_pos[:, :, sma_idxs], axis=-1)

    unchanged_categories = []
    for i in range(2, len(lines) - 2):
        unchanged_categories += ['ecc' + str(i), 'tau' + str(i)]

    for cat in unchanged_categories:
        cat_ind = my_driver.system.param_idx[cat]
        new_pos[:, :, cat_ind] = np.copy(my_driver.sampler.curr_pos[:, :, cat_ind])

    print('Position shape:', curr_pos_shape)
    # Below can be uncommented to check spread of the initial points for finding prior errors
    '''for i in range(len(walker_centers)):
        print([my_driver.system.labels[i]])
        print(np.min(new_pos[:, :, i]), np.max(new_pos[:, :, i]))'''
    return new_pos


def run_mcmc(filename, param_file, num_temps=10, num_walkers=32, num_threads=mp.cpu_count(), load_file=None, steps=10000,
             burn_steps=1000, thin=1):

    """

    :param  filename: name of the csv file which contains delta sep information; will add function to convert
                        txt file to properly formatted csv
    :param param_file: name of the csv file which contains stellar (and optionally planetary) parameters
    :param  num_temps: number of temperatures
    :param  num_walkers: number of walkers per temperature
    :param  num_threads: number of threads to use for multithreading
    :param  load_file: save file to load walker positions from; defaults to None, which will cause walkers to be
                        randomly distributed (currently according to a hardcoded set of params)
    :param  steps: number of steps for MCMC
    :param  burn_steps: number of steps for the burn-in
    :param  thin: thinning parameter for MCMC
    :return: No return, but results and corner plot are saved (will save orbit plot eventually)
    """
    # system parameters
    with open(param_file, 'r') as f:
        lines = f.readlines()
        num_secondary_bodies = len(lines) - 3
        data = lines[1].split(',')
        total_mass = float(data[13])  # [Msol]
        plx = float(data[17])  # [mas]
        mass_err = float(data[14])  # [Msol]
        plx_err = float(data[18])  # [mas]

    print('Creating Driver...')
    my_driver = driver.Driver(
        filename, 'MCMC', num_secondary_bodies, total_mass, plx, mass_err=mass_err, plx_err=plx_err,
        system_kwargs={'fit_secondary_mass': True, 'tau_ref_epoch': 62100},
        mcmc_kwargs={'num_temps': num_temps, 'num_walkers': num_walkers, 'num_threads': num_threads}
    )

    print('Modifying priors and starting positions...')
    # Set priors and starting params
    # Will change to no longer be hard-coded eventually
    set_priors(my_driver, param_file)

    new_pos = set_rand_pos(my_driver, param_file)

    my_driver.sampler.curr_pos = np.copy(new_pos)

    if load_file is not None:
        print('Using walker positions from', load_file)
        lr = results.Results()
        lr.load_results(load_file)
        my_driver.sampler.curr_pos = np.copy(lr.curr_pos)

    print('Starting MCMC')
    print('Running with', num_temps, 'temperatures and', num_walkers, 'walkers for', steps, 'steps')
    # Run MCMC
    total_orbits = steps * num_walkers

    my_driver.sampler.run_sampler(total_orbits, burn_steps=burn_steps, thin=thin)

    # Save results
    run_name = filename.split('.')[0]
    iteration = 0
    hdf5_filename = run_name + '_toliman_run_' + str(steps) + '_steps_' + str(iteration) + '.hdf5'
    while os.path.isfile(hdf5_filename):
        iteration += 1
        hdf5_filename = run_name + '_toliman_run_' + str(steps) + '_steps_' + str(iteration) + '.hdf5'

    print('MCMC completed, saving results to', hdf5_filename)
    my_driver.sampler.results.save_results(hdf5_filename)

    corner_filename = run_name + '_toliman_corner_plot_' + str(steps) + '_steps_' + str(iteration) + '.png'
    corner_plot = my_driver.sampler.results.plot_corner(bins=100)
    plt.savefig(corner_filename)


def main():
    args = sys.argv[1:]
    if '--help' in args or len(args) == 0:
        print('Required parameters: data_filename parameter_filename')
        print('Optional parameters: --num_temps X --num_walkers X --num_threads X --load_file X')
        print('                     --num_steps X --burn_steps X --thin X')
        return 0
    filename = args[0]
    param_file = args[1]
    print('Data file:', filename)
    print('Parameter file:', param_file)

    # Assign optional parameters
    num_temps = 5 if '--num_temps' not in args else int(args[args.index('--num_temps') + 1])
    num_walkers = 64 if '--num_walkers' not in args else int(args[args.index('--num_walkers') + 1])
    num_threads = mp.cpu_count() if '--num_threads' not in args else int(args[args.index('--num_threads') + 1])
    steps = 7500 if '--num_steps' not in args else int(args[args.index('--num_steps') + 1])
    load_file = None if '--load_file' not in args else args[args.index('--load_file') + 1]
    burn_steps = 1500 if '--burn_steps' not in args else int(args[args.index('--burn_steps') + 1])
    thin = 1 if '--thin' not in args else int(args[args.index('--thin') + 1])

    run_mcmc(filename, param_file, num_temps=num_temps, num_walkers=num_walkers, num_threads=num_threads,
             load_file=load_file, steps=steps, burn_steps=burn_steps, thin=thin)
    return 0


if __name__ == "__main__":
    main()
