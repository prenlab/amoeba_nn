# Copper Ion Model 

In this folder, all required files for tranining the Copper model are listed:
1. **data.h5** caontains atomic number, coordinates, energy and force for both QM and MM.
2. **force_tartget.h5** contains the force target, in this model it is defined as difference between QM force and MM force.
3. **config_cv.yaml** contains training parameters.
4. To run the training, run ``` python ../../run.py -ip config_cv.yaml cv ```
5. Trained model, force field parameter file and energy prediction results are stored in [results](results/)
6. Details for running energy calculation, geometrty minimization and molecular dynamics simulation by calling Tinker9 could be found in [Tinker9nn_py](docs/tinker9nn_py.md) and [Tinker9nn_cuda](docs/tinker9nn_cuda.md). 
