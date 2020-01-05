//
// Created by theo on 1/14/19.
//

#ifndef TCDV_PROJECT_2_FEATURES_H
#define TCDV_PROJECT_2_FEATURES_H

#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "hog_visualization.hpp"
#include "RandomForest.h"
#include <iostream>

typedef std::vector<std::pair<int, std::vector<float>>> DataSet;
namespace fs = std::filesystem;
using namespace cv;

/**
 * Compute HOG descriptors for image. Images are RESIZED AND PADDED 128x128 pixels
 * to have comparable features and the same amount of features for each image
 * Aspect ratio is preserved.
 *
 * @param img the image for which the descriptors should be calculated
 * @param visualize if true then a visualization of the image with descriptors is shown at the end
 * @return the features
 */
std::vector<float> computeHOG(const Mat &img, bool visualize = false);

/**
 * Compute hog features for all samples under path.
 * Directory structure expected to be:
 * foo/bar/<numeric_class_id>/samples{1,n}
 * eg.  data/train/02/sample1.jpg
 *      data/train/03/sample.jpg
 *
 * @param augment should data be augmented by adding rotation, flip etc to each sample?
 * @param data_set DataSet that should be filled
 * @param path path where samples are located
 */
void compute_data_set(DataSet &data_set, const fs::path &path, bool augment =false);
/**
 * Turn our Dataset to a opencv::ml dataset
 * @param data the dataset to convert
 * @return the cv::ml dataset
 */
Ptr<ml::TrainData> make_opencv_dataset(DataSet data);


#endif //TCDV_PROJECT_2_FEATURES_H
