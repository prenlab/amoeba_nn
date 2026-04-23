## Running AMOEBA+NN with Python/Tinker-GPU  Interface

This section details how to run simulations with AMOEBA+NN hybrid model with the Python/Tinker-GPU interface. This interface acts as a bridge, allowing the Python-based Neural Network to communicate directly with the C++/CUDA-based Tinker-GPU engine. The Python-side logic resides in `utils/tinker9_interface.py`, while the Tinker-GPU-side logic can be found https://github.com/prenlab/tinker-gpu/tree/nn_py.


> **⚠️ Note:** This interface is primarily intended for development, prototyping, and convenient testing of new network architectures. It does NOT support Periodic Boundary Condition properly.

### 1. Prerequisites & Compilation

You must compile the version of Tinker-GPU specific for this interface.

1.  **Environment Setup:**

    Download the current repository if you have not. 

    Ensure you are in the conda environment with the correct PyTorch and CUDA versions (e.g., `cuda11`).
    ```bash
    conda activate cuda11
    ```

2.  **Compilation:**
    ```bash
    git clone -b nn_py https://github.com/prenlab/tinker-gpu.git
    cd tinker-gpu
    mkdir build
    cd build
    bash compile.sh
    ```
    For Ren lab cluster users, the `compile.sh` is 
    ```bash
    #!/bin/bash

    # build on bme-sugar
    # Chengwen Liu

    rm -fr src* *Make* cmake-* tinker9
    rm -rf cmake* all.tests doxygen-awesome.css ext/ test/ *commands* catch2/

    export CUDAHOME=/usr/local/cuda-11.8
    export CUDACXX=$CUDAHOME/bin/nvcc
    export FC=/usr/bin/gfortran
    export CXX=/usr/bin/g++
    export ACC=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/compilers/bin/nvc++
    export opt=release
    export host=0
    export prec=d
    export compute_capability=60,70,75,80,86
    export cuda_dir=$CUDAHOME
    export CMAKEHOME=/home/liuchw/shared/cmake3.21/bin/

    $CMAKEHOME/cmake ..

    make -j 20
    ``` 
    *   Ensure the CUDA version matches your PyTorch CUDA version (e.g., 11.8).
    *   Compile using the double precision.


### 2. Execution

You can run Tinker-GPU executables you obtained in last step (like `dynamic9`, `analyze9`, or `testgrad9`) while passing the Python-side NN configurations via the `nn.key.yaml` file in your working directory.

* **Path Configuration:**
    You must export the path to the `amoeba_nn` repository, so the model code can be successfully imported in Tinker-GPU C++ code.
    ```bash
    export PYTHONPATH=$PYTHONPATH:/path/to/amoeba_nn/
    ```
* **`tinker9nn.yaml`** The interface relies on a specific YAML configuration file to define how the Neural Network is applied to the molecular system. The file has to be named this way and placed in the working directory. It's hardcoded. An example file is available as `config_example/tinker9nn.yaml` with the following content.
    ```yaml
    # Path to the trained PyTorch model (.pt file)
    nn_model: "/path/to/saved/models/best_model.pt"

    # Hardware device
    device: "cuda"

    # Hybridization Mode
    # Set True for Covalent NN (Intramolecular)
    # Set False for Metal Ion Correction (Intermolecular)
    is_bonded: True

    # Atom Selection
    # List of atom indices (1-based index) that are treated by the NN.
    # For Valence mode: The indices of the specific molecule(s).
    # For Metal mode: The indices of the metal ions.
    nn_atoms: [
        [1, 2, 3, 4, 5, 6],
    ]
    ```

**Command Line Syntax:** The normal way of using Tinker-GPU.
```bash
/path/to/tinker-gpu/build/dynamic9 <xyz_file> -k <key_file> ...
```
