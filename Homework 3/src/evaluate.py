import os

import torch

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from utils.data_utils import del_theta_quaternion
from utils.vis_utils import plot_confusion_matrix


class Evaluator(object):
    def __init__(self, dataset_mean, dataset_dev, writer_test=None, k_neighbour_count=5):
        self.k_neighbour_count = k_neighbour_count
        self.dataset_dev = dataset_dev
        self.dataset_mean = dataset_mean
        self.writer_test = writer_test

    def test(self, model, test_loader, nearest_neighbours, knn_dataset_label, knn_dataset_pose, device):
        model.eval()

        test_accuracy = 0
        angular_differences = []
        y_pred = []
        y_true = []

        with torch.no_grad():
            for batch, data in enumerate(test_loader):
                image, label, pose = data[0].permute(dims=[0, 3, 1, 2]).float(), data[1].float(), torch.tensor(data[2:]).float()
                image, label, pose = image.to(device), label.to(device), pose.to(device)
                val_descriptor = model(image)

                prediction_distance, prediction_idx = nearest_neighbours.kneighbors(val_descriptor.numpy(), self.k_neighbour_count)
                prediction_label = int(np.round(knn_dataset_label[prediction_idx[0][0]]))
                y_pred.append(prediction_label)
                y_true.append(label.item())
                if prediction_label == label.item():
                    test_accuracy += 1
                    angular_difference = del_theta_quaternion(knn_dataset_pose[prediction_idx[0][0]],
                                                              pose.numpy())
                    angular_differences.append(angular_difference)

        test_accuracy /= len(test_loader.dataset)

        if self.writer_test is not None:
            self.writer_test.add_scalar('Test Accuracy', test_accuracy)

        print(f'Test Accuracy: {test_accuracy}')
        return test_accuracy, angular_differences, y_true, y_pred

    @staticmethod
    def load_template_descriptor(load_path, device):
        knn_dataset = torch.load(load_path, map_location=device)
        return knn_dataset.reshape(-1, 21)

    def build_template_space(self, load_path, device):
        knn_dataset = self.load_template_descriptor(load_path=load_path, device=device)
        knn_dataset_features = knn_dataset[:, 0:16]
        knn_dataset_label = knn_dataset[:, 16:17]
        knn_dataset_pose = knn_dataset[:, 17:]
        nearest_neighbours = NearestNeighbors(n_neighbors=self.k_neighbour_count, algorithm='ball_tree').fit(knn_dataset_features)
        return nearest_neighbours, knn_dataset_label, knn_dataset_pose

    def build_histogram(self, angular_differences, len_test_dataset, save_path=None):
        bins = list(range(0, 181, 10))
        hist, bin_edges = np.histogram(angular_differences, bins=bins)
        print('Histogram for each degree', hist)
        bins = [0, 10, 20, 40, 180]
        hist, bin_edges = np.histogram(angular_differences, bins=bins)
        hist = hist / len_test_dataset
        print(hist)
        hist = np.cumsum(hist)
        print(hist)
        fig, ax = plt.subplots()
        # Plot the histogram heights against integers on the x axis
        ax.bar(range(len(hist)), hist, width=1)
        # Set the ticks to the middle of the bars
        ax.set_xticks([i for i, j in enumerate(hist)])
        # Set the xticklabels to a string that tells us what the bin edges were
        ax.set_xticklabels(['{} - {}'.format(bins[i], bins[i + 1]) for i, j in enumerate(hist)])
        if save_path is not None:
            plt.savefig(f'{save_path}test_{datetime.now().strftime("%m-%d-%Y_T_%H")}/Test_Histogram_Plot_{datetime.now().strftime("%H-%S")}.png')
        plt.show()

    def evaluate(self, model, test_loader, template_descriptor_path, plot_normalized_confusion_mat, save_path):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        classes = ["ape", "benchvise", "cam", "cat", "duck"]
        torch.backends.cudnn.benchmark = True

        model = model.to(device)
        nearest_neighbours, knn_dataset_label, knn_dataset_pose = self.build_template_space(load_path=template_descriptor_path, device=device)
        test_accuracy, angular_differences, y_true, y_pred = self.test(model=model, test_loader=test_loader, nearest_neighbours=nearest_neighbours,
                                                                       knn_dataset_label=knn_dataset_label, knn_dataset_pose=knn_dataset_pose, device=device)
        confusion_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
        np.set_printoptions(precision=2)
        plt.figure()

        if plot_normalized_confusion_mat:
            plot_confusion_matrix(confusion_matrix=confusion_mat, classes=classes,
                                  normalize=True,
                                  title='Normalized confusion matrix')
        else:
            plot_confusion_matrix(confusion_matrix=confusion_mat, classes=classes,
                                  normalize=False,
                                  title='Without normalization confusion matrix')
        os.makedirs(f'{save_path}test_{datetime.now().strftime("%m-%d-%Y_T_%H")}', exist_ok=True)
        plt.savefig(f'{save_path}test_{datetime.now().strftime("%m-%d-%Y_T_%H")}/Confusion_Matrix_epoch_{datetime.now().strftime("%H-%S")}.png')
        self.build_histogram(angular_differences=angular_differences, len_test_dataset=len(test_loader.dataset), save_path=save_path)
