"""
Run experiments given set of hyperparameters.
"""

import os

import clize
import tensorflow as tf
from clize.parameters import multi

from mbcnn import enums
from mbcnn import evaluate_unmixing, train_unmixing
from mbcnn import prepare_data, artifacts_reporter
from mbcnn.models import unmixing_pixel_based_dcae, unmixing_pixel_based_cnn, \
    unmixing_cube_based_cnn, unmixing_cube_based_dcae
from mbcnn.utils import parse_train_size, subsample_test_set

# Literature hyperparameters settings:
NEIGHBORHOOD_SIZES = {
    unmixing_cube_based_dcae.__name__: 5,
    unmixing_cube_based_cnn.__name__: 3
}

LEARNING_RATES = {
    unmixing_pixel_based_dcae.__name__: 0.001,
    unmixing_cube_based_dcae.__name__: 0.0005,

    unmixing_pixel_based_cnn.__name__: 0.01,
    unmixing_cube_based_cnn.__name__: 0.001
}


def run_experiments(*,
                    data_file_path: str,
                    ground_truth_path: str = None,
                    train_size: ('train_size', multi(min=0)),
                    val_size: float = 0.1,
                    sub_test_size: int = None,
                    spatial_overlap: bool = True,
                    channels_idx: int = -1,
                    neighborhood_size: int = None,
                    n_runs: int = 1,
                    model_name: str,
                    save_data: bool = 0,
                    dest_path: str,
                    sample_size: int,
                    n_classes: int,
                    lr: float = 0.001,
                    batch_size: int = 256,
                    epochs: int = 100,
                    verbose: int = 2,
                    shuffle: bool = True,
                    patience: int = 15,
                    endmembers_path: str = None):
    """
    Function for running experiments given a set of hyper parameters.
    :param data_file_path: Path to the data file. Supported types are: .npy
    :param ground_truth_path: Path to the ground-truth data file.
    :param train_size: If float, should be between 0.0 and 1.0,
        if int, it represents number of samples to draw from data.
    :param val_size: Should be between 0.0 and 1.0. Represents the
        percentage of samples to extract from the training set.
    :param sub_test_size: Number of pixels to subsample the test set
        instead of performing the inference on the entire subset.
    :param spatial_overlap: Boolean indicating whether to create non
        overlapping patches of training and test subsets.
    :param channels_idx: Index specifying the channels
        position in the provided data.
    :param neighborhood_size: Size of the spatial patch.
    :param save_data: Boolean indicating whether to save the prepared dataset.
    :param n_runs: Number of total experiment runs.
    :param model_name: Name of the model, it serves as a key in the
        dictionary holding all functions returning models.
    :param dest_path: Path to the directory where all experiment runs
        will be saved as subdirectories.
    :param sample_size: Spectral size of the input sample.
    :param n_classes: Number of classes.
    :param lr: Learning rate for the model i.e., it regulates
        the size of the step in the gradient descent process.
    :param batch_size: Size of the batch used in training phase,
        it is the number of samples to utilize per single gradient step.
    :param epochs: Total number of epochs for model to train.
    :param verbose: Verbosity mode used in training, (0, 1 or 2).
    :param shuffle: Boolean indicating whether to shuffle dataset.
    :param patience: Number of epochs without improvement in order to
        stop the training phase.
    :param use_mlflow: Boolean indicating whether to log metrics
        and artifacts to mlflow.
    :param endmembers_path: Path to the endmembers file containing
        the average reflectances for each class. Used only when
        'use_unmixing' is set to True.
    :param experiment_name: Name of the experiment. Used only if
        'use_mlflow' is set to True.
    :param run_name: Name of the run. Used only if 'use_mlflow' is set to True.
    """
    for experiment_id in range(n_runs):
        experiment_dest_path = os.path.join(
            dest_path,
            '{}_{}'.format(enums.Experiment.EXPERIMENT, str(experiment_id)))

        os.makedirs(experiment_dest_path, exist_ok=True)

        # Apply default literature hyperparameters:
        if neighborhood_size is None and model_name in NEIGHBORHOOD_SIZES:
            neighborhood_size = NEIGHBORHOOD_SIZES[model_name]
        if lr is None and model_name in LEARNING_RATES:
            lr = LEARNING_RATES[model_name]

        data = prepare_data.main(data_file_path=data_file_path,
                                 ground_truth_path=ground_truth_path,
                                 train_size=parse_train_size(train_size),
                                 val_size=val_size,
                                 stratified=False,
                                 background_label=-1,
                                 channels_idx=channels_idx,
                                 neighborhood_size=neighborhood_size,
                                 save_data=save_data,
                                 seed=experiment_id,
                                 use_unmixing=True,
                                 spatial_overlap=spatial_overlap,
                                 output_path=experiment_dest_path)

        if sub_test_size is not None:
            subsample_test_set(data[enums.Dataset.TEST], sub_test_size)

        train_unmixing.train(model_name=model_name,
                             dest_path=experiment_dest_path,
                             data=data,
                             sample_size=sample_size,
                             neighborhood_size=neighborhood_size,
                             n_classes=n_classes,
                             lr=lr,
                             batch_size=batch_size,
                             epochs=epochs,
                             verbose=verbose,
                             shuffle=shuffle,
                             patience=patience,
                             endmembers_path=endmembers_path,
                             seed=experiment_id)

        evaluate_unmixing.evaluate(
            model_path=os.path.join(experiment_dest_path, model_name),
            data=data,
            dest_path=experiment_dest_path,
            neighborhood_size=neighborhood_size,
            batch_size=batch_size,
            endmembers_path=endmembers_path)

        tf.keras.backend.clear_session()

    artifacts_reporter.collect_artifacts_report(
        experiments_path=dest_path,
        dest_path=dest_path)


if __name__ == '__main__':
    clize.run(run_experiments)
