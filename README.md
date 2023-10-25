# Orbitize4TOLIMAN

These instructions are written for a bash terminal, using Python 3. The commands included assume a unix-based shell is used. I recommend using miniconda virtual environments, one of which can be created using a .yml file included in the script_and_data directory of the repo.

Getting Started:

1. Clone github repo into the directory you will be working from
	> git clone https://github.com/wcroberson/orbitize4toliman.git
2. Move into the orbitize4toliman/ directory and run the setup.py file
	> cd orbitize4toliman/
	> python3 setup.py install
3. Navigate back to the original directory which contains orbitize4toliman and move or copy everything from the script_and_data directory into the working directory
	> cd ..
	> cp orbitize4toliman/scripts_and_data/* .
4. Make a new conda environment (can ignore if not using conda venvs)
	> conda env create -f toliman_environment.yml
	> conda activate toliman
5. You're now ready to run the script. The script has a number of required and optional command line arguments. Begin by running the following command
	> python3 toliman_script.py --help
6. To run on a sample of simulated data, you can run the following command
	> python3 toliman_script.py measured_focus.csv alpha_cen_1planet_params.csv
   This will run a basic MCMC with the default parameters.