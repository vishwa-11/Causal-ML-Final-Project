# Causal-ML-Final-Project
This project is based on a critical replication of Vietch's paper [Counterfactual Invariance to Spurious Correlations: Why and How to Pass Stress Tests](https://arxiv.org/abs/2106.00545).

## Instructions for setting up project:

### 1. Dataset and models are stored as git-lfs in a compressed form. You need to extract them.
- Skip this step if you are downloading data (see [section](#computed-resources))
- Run `$ git lfs fetch origin main` to pull it.
- Check that you have `data\Clothing_Shoes_and_Jewelry_5.json.gz`
- Check that you have `pretrained_models\bert_en_uncased_L-12_H-768_A-12_4.tar.gz`
- Check that you have `pretrained_models\bert_en_uncased_preprocess_3.tar.gz`

### 2. Dataset needs to be generated from original dataset
- Skip this step if you are downloading data (see [section](#computed-resources))
- Run script `generate_synthetic_dataset.ipynb` to generate synthetic dataset.
- Run scripts `generate_natural_dataset.ipynb` and `generate_natural_dataset2.ipynb` got generate natural dataset. (Descirption to be updated)

### 3. Pretrained models need to be extracted to be loaded
- Skip this step if you are downloading data (see [section](#computed-resources))
- Run `pretrained_model\extract_models.sh` to extract models into respective directory.

### 4. See example on how to run training to familiarize with code
- Read and run `train_bert_pipeline_example.ipynb` to familiarize with the current setup
- (`original_bert_example.ipynb` is the original example the code was based off but with different changes for MMD loss calculation function and for training on great lakes offline)

## Instructions for training network on your computer

### 1. Ensure you have necessary tensorflow requirements
- Install tensorflow based on the instructions on [https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip)
- Instal tensorflow-text based on instructions on [https://www.tensorflow.org/text/tutorials/classify_text_with_bert](https://www.tensorflow.org/text/tutorials/classify_text_with_bert)
- NOTE: specify the version you want, both tensorflow and tensorflow-text MUST have the same versions

### 2. Run train.py
- For available argument options, run `$ python train.py --help`

## Instructions for training network on your Umich Great Lakes

### 1. Create environment to run tensorflow stuff
1. Log in to great lakes CLI (for linux `$ ssh <uniqname>@greatlakes.arc-ts.umich.edu`)
2. Prior to any changes ensure your modules are purged first (`$ module purge`) this will ensure clean working environment and ensure the right modules are loaded later
3. Load tensorflow `$ module load tensorflow`
4. Change directory into the location you want to store your environment (lets call this )
    - For `trainJob_synthetic.sh` and `trainJob_natural.sh` we have used directory `envs`
    - `$ mkdir envs`
    - `$ cd envs`
5. Create a virtual environment (`$ python -m venv <env>` where is name of environment)
    - For `trainJob.sh` we named env `tf_207_bert_309`
    - `$ python -m venv tf_207_bert_309`
    - when done you will see a new directory `envs/tf_207_bert_309`
7. Load virtual environment ($ source <dir>/<env>/bin/activate or just $ source <env>/bin/acivate if you are in <env>)
    - `$ cd ../` (go back to main directory since thats where we will be running `trainJob.sh`)
    - `$ source envs/tf_207_bert_309/bin/activate`
8. You will notice a bracket with env name in front of the commands now (eg. (env) [...]$ ...)
9. Install all packages using pip (`$ pip install <packages>...`)
    - DO THIS IN THE VIRTUAL ENVIRONMENT -> I DO NOT KNOW WHAT HAPPENS IF YOU DON`T BY IT MIGHT MESS UP YOUR WORKSPACE
    - exact command I used for this project:
        - `$ pip install --upgrade pip` (ensure pip is updated)
        - `$ pip install -U "tensorflow-text==2.7.*"` (use 2.7 since the tensorflow version is 2.7)
10. Deactivate environment ($ deactivate)
11. Optionally: purge modules ($ module purge)

### 2. Ensure the necessary folders and files are available
- You should have `pretrained_models` directory with the following:
    - `bert_en_uncased_L-12_H-768_A-12_4` directory
    - `bert_en_uncased_preprocess_3` directory
- You should have `data` directory with the following:
    - `syn_train_large.npz`
    - `syn_val_large.npz`
    - `nat_train_large.npz`
    - `nat_val_large.npz`
- You should have `checkpoints` directory
- You should have `trained_model_weights` directory

### 3. Submit training job
- Call `sbatch` with `trainJob_synthetic.sh` for synthetic data or  `trainJob_natural.sh` for natural data
    - `$ sbatch trainJob_synthetic.sh`

### 4. (Additional) Starting training from a previous checkpoint
- Specify addition `-r` flag after `train.py` in the `trainJob_snthetic.sh` or `trainJob_natural.sh`

## Computed Resources
For cases where you might encounter difficulty with any of the above steps, here are some of the necessary files that have already been processed can be found here: [https://drive.google.com/drive/folders/1IIYvxUDeH-wH54ePF1kOMz1S9P4KzGpi?usp=sharing](https://drive.google.com/drive/folders/1IIYvxUDeH-wH54ePF1kOMz1S9P4KzGpi?usp=sharing)

Descriptions of the files are as follows:
1. `\pretrained_models` contains weights for BERT preprocessor and encoder required for this project.
    - `bert_en_uncased_preprocess_3.tar.gz` for BERT preprocessor to tokenize words
    - `bert_en_uncased_L-12_H-768_A-12_4.tar.gz` for BERT encoder, actual bi-directional transformer for encoding the text
    - For both files, paste them in directory `\pretrained_models` and fully unzip them: `$ tar -xvzf xxx.tar.gz`
2. `\data` contains all the processed data used for this project.
    - Unzip `data.zip` into `\data` folder: `$ unzip data.zip`
  
## Default folder structure

    <root>
    ├── checkpoints             # Store checkpoints generated during training (IMPORTANT for great lakes, max training time only 8 hours!)
    │   ├── ...
    ├── data                    # All the data files
    │   ├── Clothing_Shoes_and_Jewelry_5.json.gz          # Stored as git-LFS
    │   ├── nat_test_large.npz                            # Test set for natural dataset (not required for training)
    │   ├── nat_train_large.npz                           # Train set for natural dataset
    │   ├── nat_val_large.npz                             # Validation set for natural dataset (used for evaluating training stopping condition)
    │   ├── ...
    │   └── syn_train_large.npz
    ├── pretrained_models       # All pretrained models
    │   ├── bert_en_uncased_L-12_H-768_A-12_4             # BERT transformer model and weights
    │   |   ├── ...
    │   ├── bert_en_uncased_preprocess_3                  # Tokenizer (preprocessor) model and weights
    │   |   ├── ...
    ├── trained_model_weights   # Stores all models that hit stopping condition
    │   ├── ...
    ├── train.py                # Training script (`--help` for arguments)
    ├── trainJob_natural.sh     # Sbatch file for submitting training job to great lakes (update accordingly)
    ├── trainJob_synthetic.sh   # Sbatch file for submitting training job to great lakes (update accordingly)
    ├── ...
    └── README.md               # This readme file

- Submit a `sbatch trainJob_natural.sh` from `<root>` when running in Great Lakes for minimal changes required.
