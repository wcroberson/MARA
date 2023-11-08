# Orbitize4TOLIMAN

These instructions are written for a bash terminal, using Python 3. The commands included assume a unix-based shell is used. I recommend using miniconda virtual environments, one of which can be created using a .yml file included in the script_and_data directory of the repo.

Getting Started:

1. Clone github repo into the directory you will be working from
    > git clone https://github.com/wcroberson/orbitize4toliman.git

2. Copy everything from the script_and_data directory into the working directory
    > cp orbitize4toliman/script_and_data/* .
3. Make a new conda environment (can ignore if not using conda venvs)
    > conda create -n toliman python=3.9
   
    > conda activate toliman
   
    > conda install numpy
4. Install orbitize4toliman in editable mode
   > pip install -e ./orbitize4toliman
5. You're now ready to run the script. The script has a number of required and optional command line arguments. Begin by running the following command
    > python3 toliman_script.py --help
6. To run on a sample of simulated data, you can run the following command
    > python3 toliman_script.py measured_focus.csv alpha_cen_1planet_params.csv 
7. This will run a basic MCMC with the default parameters.

Command Line Paramters:
Required:
* data_filename: name (or filepath) of the file containing the observed (or simulated) data. Specifically, it should be a csv file, formatted the same as the measured_focus.csv file included in the script_and_data directory.

* paramter_filename: name (or filepath) of the csv file containg paramters for the binary system, as well as starting guesses for the planetary paramters. It should be formatted like one of the param files contained in the script_and_data directory. The MCMC will run assuming a number of planets equal to the number included in this file.

Optional:

* --num_temps X: sets the number of temperatures used in the parallel tempered MCMC process. Results come only from the lowest temperature, so more temperatures means a slower run time. However, more temperatures will allow the MCMC to cross low probability regions more quickly, so for posteriors with multiple peaks will be mapped more quickly. I've found that 5-10 temperatures is a good range if there is a decent starting guess, however higher temperatures could be better for worse starting postions.

* --num_walkers X: sets the number of walkers per temperature used in the MCMC process. 32 is the default, however the number must be at least twice the number of paramters. This means it must be increased for multi planet fits. I use 64 for two planet fits (one planet adds 7 parameters, so at least 14 walkers must be added per additional planets).

* --num_steps X: sets the number of steps that each walker will take. Total orbits calculated will be num_walkers * num_temps * num_steps, and total orbits in the results will be num_walkers * num_steps. For good starting positions, I've found that over a million total orbits (num_walkers * num_steps) is necessary for convergance, however this can vary greatly, so the corner plots should be used to check for convergance.

* --num_threads X: sets the number of threads used to muli-thread the MCMC process. If this is not included, all computer cores will be used for multi_threading.

* --thing X: sets how many steps will be saved in the results of the MCMC. A fraction of 1/X steps are saved. The default is 1, which is typically a good option.

* --burn_steps X: sets the number of steps used in the burn in of the MCMC. For a good starting position only a small burn in is required (about 2500 - 75000 steps). For worse starting positions (or if you are unsure of how good the starting positions are) a longer burn in is a good idea. If starting from a previous MCMC run that has already found the general shape of the posteriors, no burn in is required.

* --load_file X: X should be the filename (or filepath) of the results (.hdf5) file from which you want to take walker positions. The number of temperatures and number of walkers must be the same between the current run and the load run. The final positions of the walkers in the load run will become the starting positions of the walkers in the new run. For this reason, it can be useful for breaking long runs into several shorter runs. Note that to get full benefits from doing this, the results must be concatonated together (i.e. 10 10,000 step runs are not as good as a single 100,000 run, but they will be the same if the results from the shorter runs are concatonated together).

* --swapAB: If this flag is included, than the star listed as B in the parameter file will be treated as the star the planets are orbiting. Otherwise, the planets will orbit star A.