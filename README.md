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
- (`original_bert_example.ipynb` is the original example the code was based off but with different changes for MMD
    loss calculation function and for training on great lakes offline)