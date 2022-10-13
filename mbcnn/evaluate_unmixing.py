"""
Perform the inference of the unmixing model on the test dataset.
"""

import os

import joblib
import numpy as np
import tensorflow as tf

from mbcnn import enums
from mbcnn import io, transforms
from mbcnn.models import RepeatVector5D, unmixing_linear
from mbcnn.performance_metrics import \
    calculate_unmixing_metrics, UNMIXING_TRAIN_METRICS, cnn_rmse, \
    overall_rms_abundance_angle_distance, sum_per_class_rmse
from mbcnn.time_metrics import timeit
from mbcnn.transforms import UNMIXING_TRANSFORMS
from mbcnn.utils import get_central_pixel_spectrum


def evaluate(data,
             model_path: str,
             dest_path: str,
             neighborhood_size: int,
             batch_size: int,
             endmembers_path: str):
    """
    Function for evaluating the trained model for the unmixing problem.

    :param model_path: Path to the model.
    :param data: Either path to the input data or the data dict.
    :param dest_path: Path to the directory to store the calculated metrics.
    :param neighborhood_size: Size of the spatial patch.
    :param batch_size: Size of the batch for inference.
    :param endmembers_path: Path to the endmembers file containing
        average reflectances for each class.
        Used only when use_unmixing is set to True.
    """
    model_name = os.path.basename(model_path)
    custom_objects = {metric.__name__: metric for metric in
                      UNMIXING_TRAIN_METRICS.get(
                          model_name,
                          [cnn_rmse,
                           overall_rms_abundance_angle_distance,
                           sum_per_class_rmse])}
    if model_name == 'multiple_feature_learning_model_rezidual' or \
            model_name == 'multiple_feature_learning_model_feature_extraction_rezidual' or \
            model_name == 'multiple_output_feature_learning_model':
        custom_objects['RepeatVector5D'] = RepeatVector5D
    if model_name != 'unmixing_linear' and model_name != 'unmixing_svr':
        model = tf.keras.models.load_model(
            model_path + '_after_concatenation'
            if model_name == 'multiple_output_feature_learning_model'
            else model_path,
            compile=True,
            custom_objects=custom_objects)
    elif model_name == 'unmixing_linear':
        model = unmixing_linear(np.load(endmembers_path).transpose())
    elif model_name == 'unmixing_svr':
        model = joblib.load(os.path.join(
            os.path.dirname(model_path), 'svr.pkl'))
    else:
        raise ValueError('Model path does not exist.')
    test_dict = data[enums.Dataset.TEST]
    min_, max_ = io.read_min_max(os.path.join(
        os.path.dirname(model_path), 'min-max.csv'))
    transformations = [] if data[enums.Dataset.NAME] == 'samson' else \
        [transforms.MinMaxNormalize(min_=min_, max_=max_)]
    transformations += [t(**{'neighborhood_size': neighborhood_size}) for t
                        in UNMIXING_TRANSFORMS[model_name]]
    print(f'Applying test transformations: {transformations}')
    test_dict_transformed = transforms.apply_transformations(
        test_dict.copy(), transformations)
    if 'dcae' in model_name:
        model.pop()
    predict = timeit(model.predict)
    if model_name == 'unmixing_svr':
        y_pred, inference_time = predict(
            test_dict_transformed[enums.Dataset.DATA])
    else:
        y_pred, inference_time = predict(
            test_dict_transformed[enums.Dataset.DATA],
            batch_size=batch_size)
    np.savetxt(os.path.join(dest_path, 'y_pred.txt'),
               y_pred, fmt='%f', delimiter=',')
    model_metrics = calculate_unmixing_metrics(**{
        'endmembers': np.load(endmembers_path)
        if endmembers_path is not None else None,
        'y_pred': y_pred,
        'y_true': test_dict[enums.Dataset.LABELS],
        'x_true': get_central_pixel_spectrum(
            test_dict[enums.Dataset.DATA],
            neighborhood_size)
    })

    model_metrics['inference_time'] = [inference_time]
    io.save_metrics(dest_path=dest_path,
                    file_name=enums.Experiment.INFERENCE_METRICS,
                    metrics=model_metrics)
