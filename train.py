import os
import argparse

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text # required to load from hub (even though pylance doesnt detect it)
# from official.nlp import optiimzation  # to create AdamW optimizer

from train_methods import *

# path related args
data_dir = 'data'
train_ds_filename = 'syn_train_large.npz'
val_ds_filename = 'syn_val_large.npz'
pretrained_dir = 'pretrained_models'
preprocessor_filename = 'bert_en_uncased_preprocess_3'
bert_filename = 'bert_en_uncased_L-12_H-768_A-12_4'
checkpoint_dir = 'checkpoints'
trained_dir = 'trained_model_weights'
trained_weights_filename = 'BERT_synthetic_noMMD'
# training related args
batch_size = 1024
num_epochs = 500
learning_rate = 1024e-5
patience = 10

if __name__ == '__main__':
    # args parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', default=batch_size, type=int,
                        help='Training (and validation) batch size')
    parser.add_argument('-e', '--num-epochs', default=num_epochs, type=int,
                        help='Number of epoch to train')
    parser.add_argument('-lr', '--learning-rate', default=learning_rate, type=float,
                        help='Learning rate for training')
    parser.add_argument('-p', '--patience', default=patience, type=int,
                        help='Patience for early stopping')
    parser.add_argument('--data-dir', default=data_dir, type=str,
                        help='Dataset directory')
    parser.add_argument('--train-ds-filename', default=train_ds_filename, type=str,
                        help='Training dataset file name')
    parser.add_argument('--val-ds-filename', default=val_ds_filename, type=str,
                        help='Validation dataset file name')
    parser.add_argument('--pretrained-dir', default=pretrained_dir, type=str,
                        help='Pretrained directory path')
    parser.add_argument('--preprocessor-filename', default=preprocessor_filename, type=str,
                        help='Preprocesser file name')
    parser.add_argument('--bert-filename', default=bert_filename, type=str,
                        help='BERT model file name')
    parser.add_argument('--checkpoint-dir', default=checkpoint_dir, type=str,
                        help='Checkpoint directory path')
    parser.add_argument('--trained-dir', default=trained_dir, type=str,
                        help='Trained model weights directory path')
    parser.add_argument('--trained-weights-filename', default=trained_weights_filename, type=str,
                        help='Trained model weights filename')

    args = parser.parse_args()

    # load dataset
    train_ds = load_dataset(os.path.join(args.data_dir, args.train_ds_filename),
                            text_key='syntheticText', confounder_key='syntheticType',
                            label_key='above3Stars')
    val_ds = load_dataset(os.path.join(args.data_dir,args.val_ds_filename),
                          text_key='syntheticText', confounder_key='syntheticType',
                          label_key='above3Stars')
    train_ds = train_ds.batch(args.batch_size)
    val_ds = val_ds.batch(args.batch_size)

    # load text preprocessor
    tfhub_handle_preprocess = hub.load(os.path.join(args.pretrained_dir, args.preprocessor_filename))
    bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')

    # load bert encoder
    tfhub_handle_encoder = hub.load(os.path.join(args.pretrained_dir, args.bert_filename))
    bert_model = hub.KerasLayer(tfhub_handle_encoder, name='BERT_encoder')

    # build augmented model with MMD loss functions
    model = build_augmented_model(bert_preprocess_model, bert_model) # no MMD loss for now

    # define optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate)

    # additional callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=args.patience,
                                                      mode='max', restore_best_weights=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(args.checkpoint_dir, args.trained_weights_filename),
                                                          save_best_only=True, save_weights_only=True)

    # compile model with optimizers and loss functions (excluding MMD)
    model.compile(optimizer=optimizer,
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics = tf.metrics.SparseCategoricalAccuracy())

    # training of model
    tf.get_logger().setLevel('ERROR')
    history = model.fit(x=train_ds,
                        epochs=args.num_epochs,
                        validation_data=val_ds,
                        callbacks=[early_stopping, model_checkpoint])

    # save weights on completion
    model.save_weights(os.path.join(args.trained_dir, args.trained_weights_filename))