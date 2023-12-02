# Causal-ML-Final-Project

## Instructions for setting up project:

### 1. Dataset and models are stored as git-lfs in a compressed form. You need to extract them.
- Run `$ git lfs fetch origin main` to pull it.
- Check that you have `data\Clothing_Shoes_and_Jewelry_5.json.gz`
- Check that you have `pretrained_models\bert_en_uncased_L-12_H-768_A-12_4.tar.gz`
- Check that you have `pretrained_models\bert_en_uncased_preprocess_3.tar.gz`


### 2. Dataset needs to be generated from original dataset
- For now only synthetic dataset are available.
- Run script `generate_synthetic_dataset.ipynb` to generate training and test dataset.

### 3. Pretrained models need to be extracted to be loaded
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
