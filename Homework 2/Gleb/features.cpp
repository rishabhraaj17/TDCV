//
// Created by theo on 1/14/19.
//

#include "features.h"
#include <functional>

std::vector<Mat> create_augmented_samples(const Mat &mat);

std::vector<float> computeHOG(const Mat &img, bool visualize) {
    int nbins = 9;
    Mat working_copy;
    Size blockSize = Size(16, 16);
    Size blockStride = Size(8, 8);
    Size cellSize = Size(8, 8);
    Size winSize = Size(128, 128);
    //pad image so we don't slide the window out of bounds.
    int ypad = max(img.size[1] - img.size[0], 0);
    int xpad = max(img.size[0] - img.size[1], 0);
    copyMakeBorder(img, working_copy, ypad / 2, ypad / 2, xpad / 2, xpad / 2, BORDER_REPLICATE);
    resize(working_copy, working_copy, Size(128, 128));
    HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, nbins);
    std::vector<float> descriptors;
    Size winStride = Size();
    Size padding = Size();
    hog.compute(working_copy, descriptors, winStride, padding);
    if (visualize) {
        visualizeHOG(working_copy, descriptors, hog);
        waitKey(0);
    }
    return descriptors;
}

void compute_data_set(DataSet &data_set, const fs::path &path, bool augment) {
    for (const auto &entry: fs::directory_iterator(path)) {
        int class_id = std::stoi(entry.path().filename());
        for (const auto &sample : fs::directory_iterator(entry)) {
            Mat img = imread(sample.path().string());
            const std::vector<float> descriptors = computeHOG(img);
            if (augment) {
                for (const Mat &augmented_img : create_augmented_samples(img)) {
                    const std::vector<float> aug_descriptors = computeHOG(augmented_img);
                    data_set.emplace_back(class_id, aug_descriptors);
                }
            }
            data_set.emplace_back(class_id, descriptors);
        }
    }
    std::cout << "Loaded data set of size " << data_set.size() <<std::endl;
}

std::vector<Mat> create_augmented_samples(const Mat &img) {
    std::vector<Mat> result;
    std::function<void(const Mat &, Mat &)> transforms[] = {
            [](const Mat &in, Mat &out) {
                flip(in, out, 0);
            },
            [](const Mat &in, Mat &out) {
                flip(in, out, 1);
            },
            [](const Mat &in, Mat &out) {
                rotate(in, out, ROTATE_180);
            },
            [](const Mat &in, Mat &out) {
                rotate(in, out, ROTATE_90_CLOCKWISE);
            },
            [](const Mat &in, Mat &out) {
                rotate(in, out, ROTATE_90_COUNTERCLOCKWISE);
            }
    };
    for (auto &f : transforms) {
        Mat out;
        f(img, out);
        result.push_back(out);
    }
    return result;
};

Ptr<ml::TrainData> make_opencv_dataset(DataSet data) {

    Mat data_mat = Mat(0, static_cast<int>(data[0].second.size()), CV_32F);
    std::vector<int> labels;
    for (const auto &sample : data) {
        auto to_insert = Mat(sample.second, true).t();
        data_mat.push_back(to_insert);
        labels.push_back(sample.first);
    }
    Mat labels_mat = Mat(labels);
    return ml::TrainData::create(data_mat, ml::ROW_SAMPLE, labels_mat);
}
