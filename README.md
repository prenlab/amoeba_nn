# AMOEBA+NN: A Hybrid Neural Network and Polarizable Force Field

**amoeba_nn** is the PyTorch-based repository dedicated to the **Neural Network (NN)** components of the **AMOEBA+NN** hybrid force field, designed to bridge the gap between quantum mechanical (QM) accuracy and molecular mechanics (MM) efficiency.

## ⚠️ Scope of This Repository

**Please Note:** This repository contains **only the Python code for the Neural Network term** in AMOEBA+NN. It handles:
*   Model training (learning the difference between QM and AMOEBA).
*   Model testing/inference on datasets.
*   An interface to bridge with Tinker9 for development purposes.

**This code does not perform AMOEBA calculations** (energies, forces). **Tinker software** (Tinker, Tinker9, or Tinker-HP) is required to compute the classical AMOEBA terms in AMOEBA+NN.

## Getting Started

Download this repository and create the Conda environment: 

```bash
# Create the environment
conda create -f amoeba_nn_conda.yaml

# Activate
conda activate cuda11
```

The main entry to the program is `amoeba_nn/run.py`. To print help message: 

```bash
python run.py -h

usage: run.py [-h] -ip INPUT_PARAMS [--device DEVICE] [--debug] [--nolog] task

AMOEBA+NN

positional arguments:
  task                  what task to run. options: train, cv, predict.

optional arguments:
  -h, --help            show this help message and exit
  -ip INPUT_PARAMS, --input_params INPUT_PARAMS
                        input yaml file
  --device DEVICE       device. e.g., cpu, cuda:0.
  --debug               logging with debug level
  --nolog               disable logging to file
```

For detailed instructions on model training / testing, see [AMOEBA+NN: Model Training and Testing](docs/nn_training.md) and [AMOEBA+NN: Metal Ion Model](data/copper/README.md)

For the installation of Tinker software, refer to its official page.


## Software Ecosystem & Implementations

To run geometry minimization or molecular dynamics (MD) simulations with the AMOEBA+NN model, the NN and AMOEBA must work together. We support three modes of integration:

1.  **Python-Tinker9 Interface:**
    *   🔴 Requires this repository.
    *   Setup Guide: [Running AMOEBA+NN with Python-Tinker9 Interface](docs/tinker9nn_py.md)
    *   This mode allows the Python-based NN to communicate with the C++/CUDA-based Tinker9 engine.
    *   *Best for:* Quick prototyping, testing, and development.

2.  **Native Tinker9 Implementation:**
    *   🟢 Does NOT require this repository.
    *   Setup Guide: [Running AMOEBA+NN with Native NN in Tinker9](docs/tinker9nn_cuda.md)
    *   The NN is implemented directly within the **Tinker9** software using C++/CUDA.
    *   This provides a more cohesive and efficient integration, eliminating the overhead of Python-C++ communication.
    *   *Best for:* Production MD simulations on GPUs.

3.  **Deep-HP (Tinker-HP):**
    *   🟢 Does NOT require this repository.
    *   Setup Guide: Refer to the official repository (https://github.com/TinkerTools/tinker-hp)
    *   The NN is integrated into **Tinker-HP** to utilize Massively Parallel Processing (MPI) for High-Performance Computing (HPC).
    *   *Best for:* Large-scale simulations requiring multi-GPU/multi-node setups.


## Methodological Details

This repository has been used in the development of the following two hybridization strategies:

### 1. Organic Molecules / Proteins (Intramolecular)
For elements H, C, N, and O, the NNP replaces the classical bonded terms (bond, angle, torsion).
*   **Total Energy:** $U_{total} = U_{NN(bonded)} + U_{AMOEBA(non-bonded)}$
*   **Training Target:** $U_{NN} \approx U_{QM} - U_{AMOEBA\_non-bonded}$

### 2. Metal Ions (Intermolecular Correction)
For metal ions (e.g., $Cu^{2+}$), the NNP acts as a correction term to capture complex many-body effects (like Jahn-Teller distortion).  
*   **Total Energy:** $U_{total} = U_{AMOEBA} + U_{NN(correction)}$
*   **Training Target:** $U_{NN} \approx U_{QM} - U_{AMOEBA}$


## Citation

If you use this work, please cite the associated publications:

> Yanxing Wang, Théo Jaffrelot Inizan, Chengwen Liu, Jean-Philip Piquemal, and Pengyu Ren. "Incorporating Neural Networks into the AMOEBA Polarizable Force Field." *Journal of Physical Chemistry B*, 128(10), 2381–2388, 2024. [DOI: 10.1021/acs.jpcb.3c08166](https://doi.org/10.1021/acs.jpcb.3c08166)

## Acknowledgements

This repository is a migrated and refined version of a private development branch. Special thanks to Zhi Wang and Zhecheng He for their insightful feedback and contributions that helped shape this program.

## License

MIT License. See `LICENSE` for details.
