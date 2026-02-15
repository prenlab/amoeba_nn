## Running AMOEBA+NN with Native NN in Tinker9

For production simulations with valence NN or metal NN, AMOEBA+NN has been implemented natively in C++/CUDA within **Tinker9**. This implementation reads the Neural Network weights and biases directly from a special `.prm` file, bypassing the Python interface entirely.

### 1. Structure of the Parameter File (`.prm`)

Unlike the Python interface which loads a `.pt` file, the native implementation requires the trained model parameters to be written explicitly into the force field parameter file. A script `pt2prm.py` has been provided to facilitate the conversion process.

The structure is roughly as the example file below `amoeba09-nn-cu.prm`. The `nnp` block contains the parameters for neural network terms. 

```
      #################################
      ##                             ##
      ##  Neural Network Parameters  ##
      ##                             ##
      #################################


nnp metal
    aev 0.9  3.2  16  16.0  0.9  3.2  8  8.0  32.0  10  0
        8  7
    nn 29
       linear    -8.4831838607788086e+00    -3.9254033565521240e+00    -6.4422231912612915e-01     2.0902694761753082e-01
                 -4.6414364129304886e-02     3.3583706617355347e-01     1.6728377342224121e+00     1.9034616947174072e+00
       ...
       celu 1.000000
       linear    -1.9842497110366821e+00     2.1480055153369904e-01    -3.2890509814023972e-02    -2.5892978534102440e-02
                 -1.1370308399200439e+00    -6.1559015512466431e-01    -1.2513109445571899e+00    -6.5065503120422363e-01...
       ...
```

*   **`nnp metal`**: This Neural Network Potential is named `metal`.
*   **`aev ...`**: Defines the Atomic Environment Vector hyperparameters matching your training configuration. The order of the hyperparameters is 
    ```
    R_m_0 R_m_c R_m_d eta_m R_q_0 R_q_c R_q_d eta_q zeta_p theta_p_d topo_cutoff
    list of supported atomic numbers
    ```
*   **`nn <Z>`**: Specifies that the following network parameters belong to the element with atomic number `<Z>` (e.g., `29` for Copper).
*   **`linear ... celu ...`**: Contains the flattened arrays of weights and biases extracted from the trained PyTorch model for each layer.

### 2. Configuration (`.key` file)

To enable the NN term for a simulation, you must add specific keywords to your Tinker `.key` file.

```tinker
# Load the parameter file containing the embedded NN weights
parameters       amoeba09-nn-cu.prm

# ---------------------------------------------
# AMOEBA+NN Specific Keywords
# ---------------------------------------------

# Define atom groups in the regular way in tinker
group 1 1
# Activate the Metal NN Term, Syntax: nnterm <nnp name> <group id for atoms with NN>
nnterm metal 1
```

*   **`nnterm metal 1`**: This flag tells the Tinker9 engine to calculate the Neural Network energy/forces for atom group `1` with the nnp named `metal` using the parameters found in the `.prm` file.

### 3. Compilation and Execution

Please ensure you are using the `nn_cuda` branch from the repository (https://github.com/CanisW/tinker9/tree/nn_cuda). From there, you can proceed with the standard build and execution process as you would with any regular Tinker9 distribution.
