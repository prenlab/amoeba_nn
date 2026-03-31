# Copper Ion Model 

In this folder, all required files for tranining the Copper model are listed:
1. **data.h5** caontains atomic number, coordinates, energy and force for both QM and MM.
2. **force_tartget.h5** contains the force target, in this model it is defined as difference between QM force and MM force.
3. **config_cv.yaml** contains training parameters.
4.  Trained model, force field parameter file and energy prediction results are stored in [results](results/)
5. To run the training, run ``` python ../../run.py -ip config_cv.yaml cv ```
6. To run the energy prediction with our model, go to ```results/prediction/``` folder and run ```python ../../../run.py -ip config_predict.yaml predict``` , if you ever moved the .pt file, make sure there is a 'config.yaml' in the parent directory of .pt file
7. To run energy minimization, make sure  ```tinker9nn.yaml``` exists in the folder.
8. Details for running energy calculation, geometrty minimization and molecular dynamics simulation by calling Tinker9 could be found in [Tinker9nn_py](docs/tinker9nn_py.md) and [Tinker9nn_cuda](docs/tinker9nn_cuda.md). 
9. In the data.h5 file, each group has 6 datasets, ```amoeba.energy```, ```amoeba.force```,```atmoic_number```,```mp2.energy```,```mp2.force```,```coordinates```. If the group names contains 'mc', then it is from Monte-Carlo sampling; if it contains 'opt', then it is from geomerty optimization from high energy structures; otherwise samples are extracrted from MD simulations. a***n***b***o*** in the group name refers to a ammonia and b water in the system. 
