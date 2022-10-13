"""
Perform the training of the model for the unmixing problem.
"""

import os
from typing import Dict

import numpy as np
import tensorflow as tf

from mbcnn import enums, models
from mbcnn import io, transforms
from mbcnn import time_metrics
from mbcnn.models import get_index_of_named_layer, RepeatVector5D
from mbcnn.performance_metrics import UNMIXING_LOSSES, \
    UNMIXING_TRAIN_METRICS, cnn_rmse, overall_rms_abundance_angle_distance, \
    sum_per_class_rmse
from mbcnn.transforms import UNMIXING_TRANSFORMS


def train(data: Dict[str, np.ndarray],
          model_name: str,
          dest_path: str,
          sample_size: int,
          n_classes: int,
          neighborhood_size: int,
          lr: float,
          batch_size: int,
          epochs: int,
          verbose: int,
          shuffle: bool,
          patience: int,
          endmembers_path: str,
          seed: int):
    """
    Function for running experiments on various unmixing models,
    given a set of hyper parameters.
    :param data: Either path to the input data or the data dict itself.
        First dimension of the dataset should be the number of samples.
    :param model_name: Name of the model, it serves as a key in the
        dictionary holding all functions returning models.
    :param dest_path: Path to where all experiment runs will be saved as
        subdirectories in this directory.
    :param sample_size: Spectral size of the input sample.
    :param n_classes: Number of classes.
    :param neighborhood_size: Size of the spatial patch.
    :param lr: Learning rate for the model i.e., it regulates
        the size of the step in the gradient descent process.
    :param batch_size: Size of the batch used in training phase,
        it is the number of samples per gradient step.
    :param epochs: Number of epochs for the model to train.
    :param verbose: Verbosity mode used in training, (0, 1 or 2).
    :param shuffle: Boolean indicating whether to shuffle the dataset.
    :param patience: Number of epochs without improvement in order to
        stop the training phase.
    :param endmembers_path: Path to the endmembers file
        containing the average reflectances for each class
        i.e., the pure spectra. Used only when use_unmixing is set to True.
    :param seed: Seed for experiment reproducibility.
    """
    # Reproducibility:
    tf.reset_default_graph()
    tf.set_random_seed(seed=seed)
    np.random.seed(seed=seed)
    min_, max_ = data[enums.DataStats.MIN], data[enums.DataStats.MAX]
    np.savetxt(os.path.join(dest_path,
                            'min-max.csv'), np.array([min_, max_]),
               delimiter=',', fmt='%f')
    if model_name == 'unmixing_linear':
        return
    model = models.get_model(
        model_key=model_name,
        **{'input_size': sample_size,
           'n_classes': n_classes,
           'neighborhood_size': neighborhood_size,
           'endmembers': np.load(
               endmembers_path) if endmembers_path is not None else None,
           'dest_path': dest_path})
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        loss=UNMIXING_LOSSES[model_name],
        metrics=UNMIXING_TRAIN_METRICS.get(
            model_name,
            [cnn_rmse,
             overall_rms_abundance_angle_distance,
             sum_per_class_rmse]))
    time_history = time_metrics.TimeHistory()

    mcp_save = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(dest_path, model_name),
        save_best_only=True,
        monitor='val_loss',
        mode='min')

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min')
    callbacks = [time_history, mcp_save, early_stopping]
    train_dict = data[enums.Dataset.TRAIN]
    val_dict = data[enums.Dataset.VAL]
    transformations = [] if data[enums.Dataset.NAME] == 'samson' else \
        [transforms.MinMaxNormalize(min_=min_, max_=max_)]
    transformations += [t(**{'neighborhood_size': neighborhood_size}) for t
                        in UNMIXING_TRANSFORMS[model_name]]
    print(f'Applying train and val transformations: {transformations}')
    train_dict = transforms.apply_transformations(train_dict, transformations)
    val_dict = transforms.apply_transformations(val_dict, transformations)

    if model_name == 'multiple_output_feature_learning_model':
        y_train = (train_dict[enums.Dataset.LABELS],
                   train_dict[enums.Dataset.LABELS],
                   train_dict[enums.Dataset.LABELS])
        y_val = (val_dict[enums.Dataset.LABELS],
                 val_dict[enums.Dataset.LABELS],
                 val_dict[enums.Dataset.LABELS])
    else:
        y_train = train_dict[enums.Dataset.LABELS]
        y_val = val_dict[enums.Dataset.LABELS]
    history = model.fit(x=train_dict[enums.Dataset.DATA],
                        y=y_train, epochs=epochs, verbose=verbose,
                        shuffle=shuffle,
                        validation_data=(val_dict[enums.Dataset.DATA], y_val),
                        callbacks=callbacks, batch_size=batch_size)
    if model_name != 'unmixing_svr':
        history.history[time_metrics.TimeHistory.__name__] = \
            time_history.average

    io.save_metrics(
        dest_path=dest_path, file_name='training_metrics.csv',
        metrics=history if model_name == 'unmixing_svr' else history.history)

    if model_name == 'multiple_output_feature_learning_model':
        # If the model is the multi-output variation,
        # train the parallel modules together:
        custom_objects = {metric.__name__: metric for metric in
                          UNMIXING_TRAIN_METRICS.get(
                              model_name,
                              [cnn_rmse,
                               overall_rms_abundance_angle_distance,
                               sum_per_class_rmse])}
        custom_objects['RepeatVector5D'] = RepeatVector5D
        model = tf.keras.models.load_model(
            os.path.join(dest_path, model_name),
            compile=True,
            custom_objects=custom_objects)
        input = model.layers[get_index_of_named_layer(model,
                                                      'common_input')].output
        first_block_out_idx = get_index_of_named_layer(
            model, 'first_block_out')
        second_block_out_idx = get_index_of_named_layer(
            model, 'second_block_out')
        third_block_out_idx = get_index_of_named_layer(
            model, 'third_block_out')
        first_block_out = model.layers[first_block_out_idx].output
        second_block_out = model.layers[second_block_out_idx].output
        third_block_out = model.layers[third_block_out_idx].output
        x = tf.keras.layers.Concatenate()(
            [first_block_out, second_block_out, third_block_out])
        x = tf.keras.layers.Dense(512, activation='relu', name='classifier-1')(
            x)
        x = tf.keras.layers.Dense(64, activation='relu', name='classifier-2')(
            x)
        x = tf.keras.layers.Dense(model.output_shape[0][1],
                                  activation='softmax', name='classifier-3')(x)
        new_model = tf.keras.models.Model(inputs=input,
                                          outputs=x)
        for layer in new_model.layers:
            if 'classifier' not in layer.name:
                layer.trainable = True
            print(f'Layer: {layer.name}, trainable: {layer.trainable}')
        new_model.summary()
        new_model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=lr),
            # Take first loss, since now only one is sufficient,
            # (after concatenation, there is only one output):
            loss=UNMIXING_LOSSES[model_name][0],
            metrics=UNMIXING_TRAIN_METRICS.get(
                model_name,
                [cnn_rmse,
                 overall_rms_abundance_angle_distance,
                 sum_per_class_rmse]))

        time_history = time_metrics.TimeHistory()

        mcp_save = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(dest_path, model_name + '_after_concatenation'),
            save_best_only=True,
            monitor='val_loss',
            mode='min')

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min')

        callbacks = [time_history, mcp_save, early_stopping]

        history = new_model.fit(
            x=train_dict[enums.Dataset.DATA],
            y=train_dict[enums.Dataset.LABELS],
            epochs=epochs,
            verbose=verbose,
            shuffle=shuffle,
            validation_data=(val_dict[enums.Dataset.DATA],
                             val_dict[enums.Dataset.LABELS]),
            callbacks=callbacks,
            batch_size=batch_size)

        history.history[
            time_metrics.TimeHistory.__name__] = time_history.average

        io.save_metrics(dest_path=dest_path,
                        file_name='training_metrics_after_concatenation.csv',
                        metrics=history.history)
