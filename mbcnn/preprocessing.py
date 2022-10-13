import functools
from itertools import product
from typing import Tuple, Union, List

import cv2
import numpy as np

from mbcnn.enums import Sample
from mbcnn.transforms import PerBandMinMaxNormalization
from mbcnn.utils import shuffle_arrays_together, \
    get_label_indices_per_class

HEIGHT = 0
WIDTH = 1
DEPTH = 2


def normalize_labels(labels: np.ndarray) -> np.ndarray:
    """
    Normalize labels so that they always start from 0
    :param labels: labels to normalize
    :return: Normalized labels
    """
    for label_index, label in enumerate(np.unique(labels)):
        labels[labels == label] = label_index
    return labels


def reshape_cube_to_2d_samples(data: np.ndarray,
                               labels: np.ndarray,
                               channels_idx: int = 0,
                               use_unmixing: bool = False) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Reshape the data and labels from [CHANNELS, HEIGHT, WIDTH] to
        [PIXEL,CHANNELS, 1], so it fits the 2D Convolutional modules.
    :param data: Data to reshape.
    :param labels: Corresponding labels.
    :param channels_idx: Index at which the channels are located in the
                         provided data file.
    :param use_unmixing: Boolean indicating whether to perform
            experiments on the unmixing datasets,
            where classes in each pixel are present as fractions.
    :return: Reshape data and labels
    :rtype: tuple with reshaped data and labels
    """
    data = np.rollaxis(data, channels_idx, len(data.shape))
    height, width, channels = data.shape
    data = data.reshape(height * width, channels)
    data = np.expand_dims(np.expand_dims(data, 1),
                          1) if use_unmixing else np.expand_dims(data, -1)
    labels = labels.reshape(height * width,
                            -1) if use_unmixing else labels.reshape(-1)
    data = data.astype(np.float32)
    labels = labels.astype(np.float32) if use_unmixing else labels.astype(
        np.uint8)
    return data, labels


def get_padded_cube(data: np.ndarray, padding_size: int):
    v_padding = np.zeros((padding_size, data.shape[WIDTH], data.shape[DEPTH]))
    data = np.vstack((v_padding, data))
    data = np.vstack((data, v_padding))
    h_padding = np.zeros((data.shape[HEIGHT], padding_size, data.shape[DEPTH]))
    data = np.hstack((h_padding, data))
    data = np.hstack((data, h_padding))
    return data


def spectral_angle_mapper(data: np.ndarray,
                          endmembers: np.ndarray,
                          channel_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the spectral angle mapper between the HSI pixels and endmembers.
    :param data: Data cube.
    :param endmembers: The matrix containing the spectra for each class of
        dimensionality [n_classes, n_bands].
    :param channel_idx: Index of bands in HSI.
    :return: The tuple of both, normalized and unnormalized angle matrix
        for each pixel in HSI.
    """
    if data.min() < 0 or data.max() > 1:
        transform = PerBandMinMaxNormalization(
            **PerBandMinMaxNormalization.get_min_max_vectors(
                data
            ))
        data = next(iter(transform(data, np.zeros((1)))))
    data = data.reshape(-1, data.shape[channel_idx])
    adj_matrix = np.empty((data.shape[0], endmembers.shape[0]))
    for pixel_idx in range(data.shape[0]):
        for spectra_idx in range(endmembers.shape[0]):
            numerator = np.dot(data[pixel_idx], endmembers[spectra_idx])
            denominator = np.sqrt(np.dot(data[pixel_idx], data[pixel_idx]) *
                                  np.dot(endmembers[spectra_idx],
                                         endmembers[spectra_idx]))
            alpha = np.arccos(np.clip(numerator / denominator, -1, 1))
            adj_matrix[pixel_idx, spectra_idx] = alpha
    adj_matrix = adj_matrix.astype(np.float32)
    norm_adj_matrix = adj_matrix / np.expand_dims(adj_matrix.sum(axis=1), -1)
    return adj_matrix, norm_adj_matrix


def reshape_cube_to_3d_samples(data: np.ndarray,
                               labels: np.ndarray,
                               neighborhood_size: int = 5,
                               background_label: int = 0,
                               channels_idx: int = 0,
                               use_unmixing: bool = False) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Reshape data to a array of dimensionality:
        [N_SAMPLES, HEIGHT, WIDTH, CHANNELS] and the labels to
        dimensionality of: [N_SAMPLES, N_CLASSES]
    :param data: Data passed as array.
    :param labels: Labels passed as array.
    :param neighborhood_size: Length of the spatial patch.
    :param background_label: Label to filter out the background.
    :param channels_idx: Index of the channels.
    :param use_unmixing: Boolean indicating whether to perform
        experiments on the unmixing datasets,
        where classes in each pixel are present as abundances fractions.
    :rtype: Tuple of data and labels reshaped to 3D format.
    """
    data = np.rollaxis(data, channels_idx, len(data.shape))
    height, width, _ = data.shape
    padding_size = int(
        neighborhood_size % np.ceil(float(neighborhood_size) / 2.))
    data = get_padded_cube(data, padding_size)
    samples = []
    labels_3d = []
    data = data.astype(np.float32)
    for x, y in product(list(range(height)), list(range(width))):
        if use_unmixing or labels[x, y] != background_label:
            samples.append(data[x:x + padding_size * 2 + 1,
                           y:y + padding_size * 2 + 1])
            labels_3d.append(labels[x, y])
    samples = np.array(samples).astype(np.float32)
    labels3d = np.array(labels_3d).astype(np.float32) if \
        use_unmixing else np.array(labels_3d).astype(np.uint8)
    return samples, labels3d


def align_ground_truth(cube_2d_shape: Tuple[int, int],
                       ground_truth: np.ndarray,
                       cube_to_gt_transform: np.ndarray) -> np.ndarray:
    """
    Align original labels to match the satellite hyperspectral cube using
    transformation matrix
    :param cube_2d_shape: Shape of the hyperspectral data cube
    :param ground_truth: Original labels as 2D array
    :param cube_to_gt_transform: Cube to ground truth transformation matrix
    :return: Transformed ground truth
    """
    gt_to_chan_transform = np.linalg.inv(cube_to_gt_transform)
    gt_transformed = cv2.warpPerspective(ground_truth, gt_to_chan_transform,
                                         cube_2d_shape,
                                         flags=cv2.INTER_NEAREST)
    return gt_transformed


def remove_nan_samples(data: np.ndarray, labels: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Remove samples which contain only nan values
    :param data: Data with dimensions [SAMPLES, ...]
    :param labels: Corresponding labels
    :return: Data and labels with removed samples containing nans
    """
    all_but_samples_axes = tuple(range(1, data.ndim))
    nan_samples_indexes = np.isnan(data).any(axis=all_but_samples_axes).ravel()
    labels = labels[~nan_samples_indexes]
    data = data[~nan_samples_indexes]
    return data, labels


def train_val_test_split(data: np.ndarray, labels: np.ndarray,
                         train_size: Union[List, float, int] = 0.8,
                         val_size: float = 0.1,
                         stratified: bool = True,
                         seed: int = 0,
                         spatial_overlap: bool = True,
                         neighborhood_size: int = None,
                         height: int = None,
                         width: int = None,
                         output_path: str = None) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into train, val and test sets. The size of the training set
    is set by the train_size parameter. All the remaining samples will be
    treated as a test set

    :param data: Data with the [SAMPLES, ...] dimensions
    :param labels: Vector with corresponding labels
    :param train_size: If float, should be between 0.0 and 1.0,
                        if stratified = True, it represents percentage
                        of each class to be extracted.
                 If float and stratified = False, it represents percentage
                    of the whole dataset to be extracted with samples
                    drawn randomly, regardless of their class.
                 If int and stratified = True, it represents number of samples
                    to be drawn from each class.
                 If int and stratified = False, it represents overall number of
                    samples to be drawn regardless of their class, randomly.
                 Defaults to 0.8
    :param val_size: Should be between 0.0 and 1.0. Represents the percentage
                     of each class from the training set to be extracted as a
                     validation set, defaults to 0.1
    :param stratified: Indicated whether the extracted training set should be
                     stratified, defaults to True
    :param seed: Seed used for data shuffling
    :param spatial_overlap: Boolean indicating whether to create non
        overlapping patches of training and test subsets.
    :param neighborhood_size: Size of the spatial patch.
    :param height: Height of the image.
    :param width: Width of the image.
    :return: train_x, train_y, val_x, val_y, test_x, test_y
    :raises AssertionError: When wrong type is passed as train_size
    """
    if spatial_overlap:
        shuffle_arrays_together([data, labels], seed=seed)
        train_indices = _get_set_indices(train_size, labels, stratified)
        val_indices = _get_set_indices(
            val_size, labels[train_indices], stratified)
        val_indices = train_indices[val_indices]
        test_indices = np.setdiff1d(np.arange(len(data)), train_indices)
        train_indices = np.setdiff1d(train_indices, val_indices)
    else:
        np.random.seed(seed)
        start = np.random.randint(0, data.shape[Sample.SAMPLES_DIM])
        end = (start + train_size) % data.shape[Sample.SAMPLES_DIM]
        if end > start:
            train_indices = np.arange(start, end)
        else:
            train_indices = np.concatenate((
                np.arange(start, data.shape[Sample.SAMPLES_DIM]),
                np.arange(0, end)))
        padding_size = int(
            neighborhood_size % np.ceil(float(neighborhood_size) / 2.))
        mask = np.zeros((padding_size + height + padding_size,
                         padding_size + width + padding_size))
        for train_idx in train_indices:
            row_idx, col_idx = train_idx // width, train_idx % width
            row_idx, col_idx = row_idx + padding_size, col_idx + padding_size
            mask[row_idx - padding_size: row_idx + padding_size + 1,
            col_idx - padding_size: col_idx + padding_size + 1] = 1
        # Generate test indices starting from the end up to start,
        # so that the last "sub_test_size" samples will be identical:
        test_indices = []
        omitted = 0
        if end > start:
            for test_idx in range(end, data.shape[Sample.SAMPLES_DIM]):
                row_idx, col_idx = test_idx // width, test_idx % width
                row_idx, col_idx = row_idx + padding_size, col_idx + padding_size
                area = mask[row_idx - padding_size: row_idx + padding_size + 1,
                       col_idx - padding_size: col_idx + padding_size + 1]
                if 1 not in area:
                    test_indices.append(test_idx)
                    mask[row_idx - padding_size: row_idx + padding_size + 1,
                    col_idx - padding_size: col_idx + padding_size + 1] = -1
                else:
                    omitted += 1
            for test_idx in range(start):
                row_idx, col_idx = test_idx // width, test_idx % width
                row_idx, col_idx = row_idx + padding_size, col_idx + padding_size
                area = mask[row_idx - padding_size: row_idx + padding_size + 1,
                       col_idx - padding_size: col_idx + padding_size + 1]
                if 1 not in area:
                    test_indices.append(test_idx)
                    mask[row_idx - padding_size: row_idx + padding_size + 1,
                    col_idx - padding_size: col_idx + padding_size + 1] = -1
                else:
                    omitted += 1
        else:
            for test_idx in range(end, start):
                row_idx, col_idx = test_idx // width, test_idx % width
                row_idx, col_idx = row_idx + padding_size, col_idx + padding_size
                area = mask[row_idx - padding_size: row_idx + padding_size + 1,
                       col_idx - padding_size: col_idx + padding_size + 1]
                if 1 not in area:
                    test_indices.append(test_idx)
                    mask[row_idx - padding_size: row_idx + padding_size + 1,
                    col_idx - padding_size: col_idx + padding_size + 1] = -1
                else:
                    omitted += 1
        diff = width * height - (len(train_indices) + len(test_indices))
        assert diff > 0, 'The number of non-overlapping samples must ' \
                         'be smaller than the total number of pixels.'
        diff -= omitted
        assert diff == 0, 'The difference between the total number of pixels ' \
                          'and the train + test + omitted (due to overlap) ' \
                          'samples must be equal to zero, i.e., the sets must' \
                          'be unique in terms of samples.'
        test_indices = np.asarray(test_indices)
        val_indices = np.arange(int(len(labels[train_indices]) * val_size))
        val_indices = train_indices[val_indices]
        train_indices = np.setdiff1d(train_indices, val_indices)
    return data[train_indices], labels[train_indices], data[val_indices], \
           labels[val_indices], data[test_indices], labels[test_indices]


@functools.singledispatch
def _get_set_indices(size: Union[List, float, int],
                     labels: np.ndarray,
                     stratified: bool = True) -> np.ndarray:
    """
    Extract indices of a subset of specified data according to size and
    stratified parameters.

    :param labels: Vector with corresponding labels
    :param size: If float, should be between 0.0 and 1.0,
                    if stratified = True, it represents percentage
                    of each class to be extracted.
                 If float and stratified = False, it represents
                    percentage of the whole dataset to be extracted
                    with samples drawn randomly, regardless of their class.
                 If int and stratified = True, it represents number of samples
                    to be drawn from each class.
                 If int and stratified = False, it represents overall number of
                    samples to be drawn regardless of their class, randomly.
                 Defaults to 0.8
    :param stratified: Indicated whether the extracted training set should be
                     stratified, defaults to True
    :return: Indexes of the train set
    :raises TypeError: When wrong type is passed as size
    """
    raise NotImplementedError()


@_get_set_indices.register(float)
def _(size: float,
      labels: np.ndarray,
      stratified: bool = True) -> np.ndarray:
    assert 0 < size <= 1
    if stratified:
        label_indices, unique_labels = get_label_indices_per_class(
            labels, return_uniques=True)
        for idx in range(len(unique_labels)):
            samples_per_label = int(len(label_indices[idx]) * size)
            label_indices[idx] = label_indices[idx][:samples_per_label]
        train_indices = np.concatenate(label_indices, axis=0)
    else:
        train_indices = np.arange(int(len(labels) * size))
    return train_indices


@_get_set_indices.register(int)
def _(size: int,
      labels: np.ndarray,
      stratified: bool = True) -> np.ndarray:
    assert size >= 1
    if stratified:
        label_indices, unique_labels = get_label_indices_per_class(
            labels, return_uniques=True)
        for label in range(len(unique_labels)):
            label_indices[label] = label_indices[label][:int(size)]
        train_indices = np.concatenate(label_indices, axis=0)
    else:
        train_indices = np.arange(size, dtype=int)
    return train_indices


@_get_set_indices.register(list)
def _(size: List,
      labels: np.ndarray,
      _) -> np.ndarray:
    label_indices, unique_labels = get_label_indices_per_class(
        labels, return_uniques=True)
    if len(size) == 1:
        size = int(size[0])
    for n_samples, label in zip(size, range(len(unique_labels))):
        label_indices[label] = label_indices[label][:int(n_samples)]
    train_indices = np.concatenate(label_indices, axis=0)
    return train_indices
