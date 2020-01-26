import cv2 as cv2
from PIL import Image

import torch
import numpy as np


def get_image(image_path: str, use_cv2: bool = True):
    if use_cv2:
        return cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH | -1)
    else:
        return Image.open(image_path)


def get_images_and_poses(data_dir: str, label: int):
    with open(data_dir + '/poses.txt', 'r') as f:
        count: int = 0
        samples: list = []
        sample: list = []
        for line in f:
            count += 1
            if count % 2 == 1:
                if not line.strip():
                    print(f'Empty Line: {line} in {data_dir}/poses.txt')
                else:
                    sample: list = [get_image(data_dir + '/' + line.split()[1])]
            else:
                sample.append(label)
                sample.extend([float(i) for i in line.split()])
                samples.append(sample)
        return samples


def get_train_idxs(file: str):
    training_indices: list = []
    with open(file, 'r') as f:
        for line in f:
            training_indices: list = [int(i) for i in line.split(',')]
            return training_indices
    return None  # fixme: remove it


def get_coarse_data(data_dir: str, classes: tuple):
    coarse_data: list = []
    for i in range(5):
        seq_data = get_images_and_poses(data_dir + 'coarse/' + classes[i], i)
        coarse_data.extend(seq_data)
        print(len(seq_data))
    print(len(coarse_data))
    return coarse_data


def get_fine_data(data_dir: str, classes: tuple):
    fine_data: list = []
    for i in range(5):
        seq_data = get_images_and_poses(data_dir + 'fine/' + classes[i], i)
        fine_data.extend(seq_data)
        print(len(seq_data))
    print(len(fine_data))
    return fine_data


def get_real_data(data_dir: str, classes: tuple):
    real_train_data: list = []
    real_test_data: list = []
    train_idxs = get_train_idxs(data_dir + 'real/training_split.txt')

    for i in range(5):
        seq_data = get_images_and_poses(data_dir + 'real/' + classes[i], i)
        for i in range(0, len(seq_data)):
            if i in train_idxs:
                real_train_data.append(seq_data[i])
            else:
                real_test_data.append(seq_data[i])

    print(len(real_train_data))
    print(len(real_test_data))
    return real_train_data, real_test_data


def get_datasets(data_dir: str, classes: tuple):
    coarse_data = get_coarse_data(data_dir, classes)
    fine_data = get_fine_data(data_dir, classes)
    real_train_data, real_test_data = get_real_data(data_dir, classes)
    train_db = fine_data
    train_db.extend(real_train_data)
    return coarse_data, train_db, real_test_data


def get_overall_train_mean(images: list):
    src = images[0]
    n_channels = src.shape[-1]
    mean = np.zeros(n_channels)
    n_images = len(images)

    for image in images:
        image = np.asarray(image, dtype=np.float32)
        mean += image.mean(axis=(0, 1)) / n_images

    return mean


def get_overall_train_std(images: list, mean: float = None):
    if mean is None:
        mean = get_overall_train_mean(images)

    src = images[0]
    n_channels = src.shape[-1]
    std = np.zeros(n_channels)
    n_images = len(images)

    for image in images:
        image = np.asarray(image, dtype=np.float32)
        std_img = np.sum(np.sum(np.square(image - mean), axis=0), axis=0) / (n_images * src.shape[0] * src.shape[1] - 1)
        std += std_img

    return np.sqrt(std)


def get_mean_and_std(data_dir: str, classes: tuple):
    db, train, test = get_datasets(data_dir, classes)
    images = [i[0] for i in train]
    print([len(i) for i in [db, train, test]])
    mean = get_overall_train_mean(images)
    dev = get_overall_train_std(images)
    return mean, dev


def del_theta_quaternion(q1, q2):
    s = 0
    for i in range(4):
        s = s + q1[i] * q2[i]
    s = np.clip(s, -1, 1)
    angle = 2 * np.arccos(abs(s)) * 180 / np.pi
    return angle


if __name__ == '__main__':
    data_dir = "../../dataset/"
    classes = ("ape", "benchvise", "cam", "cat", "duck")
    db, train, test = get_datasets(data_dir, classes)
    print('Loaded!')