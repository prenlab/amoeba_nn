## AMOEBA+NN: Model Training and Testing

The core functionality of this repository is executed via the `amoeba_nn/run.py` script. This script handles training, cross-validation, and inference based on the arguments and configuration files provided.

### 1. Data Preparation

The program requires two types of data files: a **CSV Index** (metadata and energies) and **HDF5 Databases** (heavy matrices/tensors). Most of the column names in your csv files are not hardcoded. You define the mapping in your YAML configuration file (see Section 3).

#### A. CSV Index File
This file controls which molecules and conformations are used for training/validation.
*   **Structure:**
    ```csv
    ID,CONF_ID,dft_energy,amoeba_energy,label_e,sample_weight,division
    md6n,0,-371.8382907330203,-384.5017,12.663409266979727,1,0
    md6n,1,-375.0891351534926,-387.4825,12.393364846507438,1,4
    md6n,2,-383.4926932250219,-395.5316,12.038906774978102,1,0
    md6n,3,-385.6007126684773,-398.0587,12.457987331522702,1,6
    md6n,4,-376.32142548620374,-389.3464,13.02497451379628,1,3
    ...
    ```
*   **Important Columns:**
    *   **Molecule ID (`ID`):** (e.g., `md6n`) Must match the group name in your HDF5 file.
    *   **Conformation ID (`CONF_ID`):** (e.g., `0`, `1`) Integer index of the conformation.
    *   **Target Label (`label_e`):** The energy value to learn. The column name is not hardcoded. Indicate which column you would like to use in the YAML file. Prepare this column using whatever formula you need.
        *   *Organic:* $U_{QM} - U_{AMOEBA(nonbonded)}$
        *   *Metal:* $U_{QM} - U_{AMOEBA(total)}$
    *   **Split ID (`division`):** Integer (0-9) used to assign data to training or validation folds. Also not hardcoded.
    *   **Sample Weight (`sample_weight`):** (Optional) Weights for loss calculation. Also not hardcoded.

#### B. HDF5 Data Files
These files store the atomic numbers, coordinates, and forces. Usually we have two files: one for atomic numbers and coordinates, the other for forces. 

*   **`data.h5`:**
    *   `/Group_Name/atomic_numbers`: Shape `(N_atoms,)`.
    *   `/Group_Name/coordinates`: Shape `(N_confs, N_atoms, 3)`.
*   **`forces.h5`:**
    *   `/Group_Name/forces`: (Optional) Shape `(N_confs, N_atoms, 3)`. Required if training includes atomic forces in the targets.

---

### 2. Program Tasks

Run the program using: `python amoeba_nn/run.py <TASK> --input_params <CONFIG_FILE> --device <DEVICE>`

#### `train` (Single Split)
Trains a model on a fixed training set and validates on a fixed validation set defined in the config.
*   **Usage:** Prototyping architectures or training final production models on all data.
*   **Logic:** Uses the specific indices listed in `split: train` and `split: val` in the YAML file to filter the CSV.

#### `cv` (Cross-Validation)
Performs k-fold cross-validation to test model transferability. This is the one used most often.
*   **Usage:** Research validation and generating error bars.
*   **Logic:** It rotates through the `split: train` IDs defined in the YAML. If `num_folds: 5`, it automatically partitions the provided IDs into 5 groups, iteratively holding one out for validation.

#### `predict` (Inference)
Loads a pre-trained model and predicts energies for a dataset.
*   **Output:** Generates a new CSV file (e.g., `test_pred.csv`) with a new column containing predicted energies.
*   **Note:** This task **only outputs energies** to the CSV. It does not currently calculate predicted force vectors.

---

### 3. Configuration Guide

The YAML file acts as the input configuration for your task. There are example files in `config_example/`. Below are the explanations of some important keywords.

#### General Settings

*   **`exp_name`**: **Experiment Name**
    *   *Usage:* This string is used to name the output directory for this specific run.
    *   *Behavior:* The program automatically appends a timestamp to this name to prevent overwriting previous work.
    *   *Example:* If you set `exp_name: 'copper_test'`, the program creates a folder named something like `copper_test_20260215-103000` inside your `save_path`.

*   **`save_path`**: **Root Output Directory**
    *   *Usage:* The path to the folder where you want to store all your experiment results.
    *   *Behavior:* The specific experiment folder (defined above) will be created *inside* this directory. This folder will eventually contain:
        *   `models/`: Saved `.pt` checkpoints.
        *   `tb/`: TensorBoard log files for visualizing loss curves.
        *   `config.yaml`: A copy of the configuration used for reproducibility.
        *   `summary.csv`: A table of the best validation metrics.
        *   `source_code.tar.gz`: A backup of the current version of code used for this run, for reproducibility. 

*   **`remarks`**: **Metadata/Notes**
    *   *Usage:* A free-text string for your own reference.
    *   *Behavior:* This does not affect training logic. It is saved into the `config.yaml` copy in the output folder.

*   **`device`**: **Hardware Accelerator**
    *   *Usage:* Specifies which hardware (CPU or GPU) PyTorch should use for computation.
    *   *Options:*
        *   `'cpu'`: Runs on the CPU (slow, useful for debugging).
        *   `'cuda'`: Uses the default NVIDIA GPU (usually `cuda:0`).
        *   `'cuda:N'`: Uses a specific GPU index (e.g., `'cuda:1'` for the second GPU).
    *   *Note:* This can be overridden by the command line argument `--device`. For example, `python run.py ... --device cuda:1` will ignore the value in the YAML file.

#### Data Mapping
You can name your CSV columns whatever you want, provided you update these fields:
```yaml
train:
  # 1. Dataset Loader Class
  # "Metal": For Cu/Ions (MetalDatabaseH5)
  # "SPICE": For SPICE dataset (SPICEDatabaseH5)
  # "ANI1": Standard ANI format (ANINetworkDataset)
  dataset_name: "Metal"

  # 2. File Paths
  csv_path: '/path/to/data.csv'
  h5_files: ['/path/to/data.h5']
  h5_force: '/path/to/force.h5' # Optional: Leave empty "" if not training forces

  # 3. Column Mapping
  # The code looks for this column name in your CSV to find the training target
  label_column: 'label_e' 
  
  # The code looks for this column to perform Train/Val splits
  split:
    column: 'division' 
    train: # IDs included in this run
  
  # number of threads for data loading
  loading_workers: 0
```

#### Training Logic
```yaml
  # If true, model trains on (E_i - E_j). Essential for valence NN term.
  relative_training: true 
  
  # Force Training Settings
  loss_force_weight: 0.1  # 0.0 = Energy only. >0.0 = Energy + Force
  
  # Loss Function for Forces:
  # "CartMSE": Standard Mean Squared Error (XYZ components).
  # "CosMag":  Cosine Similarity + Magnitude.
  loss_fn_force: "CartMSE" 
```

#### Model Architecture
```yaml
model:
  # "ANINetwork": Standard organic molecules
  # "ANINetwork_Relative": Training with relative energies
  # "ANINetwork_Metal": Special architecture for ions
  arch: 'ANINetwork_Metal'
  
  # Required only if arch is ANINetwork_Metal
  metal: "Cu" 
  
  # Atomic Environment Vector (AEV) Hyperparameters
  aev:
    radial_cutoff: 3.5  
    ...
```

---

### 4. Monitoring and Model Selection

#### TensorBoard
Training metrics are logged automatically. View them by pointing TensorBoard to the save directory:
```bash
tensorboard --logdir <save_path>
```
You will see various metrics that has been recorded along the training process.

#### Picking the Best Model
Checkpoints are saved in `<save_path>/<exp_name>/models/`.
1.  **Do not assume the last epoch is best.**
2.  Check the `summary.csv` file generated in the experiment folder. It contains a table of `Best Validation` metrics.
3.  Identify the epoch with the lowest validation loss in `summary.csv` and use the corresponding `.pt` file (e.g., `Checkpoint_Epoch378_...pt`) for your Tinker simulations.
4.  However, a model with superior validation metrics does not necessarily translate to robust performance in MD simulations. In practice, models from earlier training stages often exhibit greater MD stability, even if their static energy and force accuracy are slightly lower. Finding the balance is non-trivial but important.
