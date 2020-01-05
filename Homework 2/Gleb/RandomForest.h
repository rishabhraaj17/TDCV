//
// Created by theo on 1/14/19.
//

#ifndef TCDV_PROJECT_2_RANDOMFOREST_H
#define TCDV_PROJECT_2_RANDOMFOREST_H

#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>


class RandomForest {
public:
    RandomForest() {}

    explicit RandomForest(int nTrees, int maxDepth = INT_MAX, int minSampleCount = 10, int maxCategories = 10);

    RandomForest(std::filesystem::path restorePath);

    void train(cv::Ptr<cv::ml::TrainData> &trainData, float subsetRatio = 0.8);
    int predict(const cv::Mat& sample, float* confidence = 0) const;
    void save(const std::filesystem::path &path);
    void restore(const std::filesystem::path &path);

    unsigned long getSize();
    static RandomForest create(int nTrees, int maxDepth = INT_MAX, int minSampleCount = 10, int maxCategories = 10){
        return RandomForest(nTrees,maxDepth,minSampleCount,maxCategories);
    }
private:
    std::vector<cv::Ptr<cv::ml::DTrees>> trees;


};


#endif //TCDV_PROJECT_2_RANDOMFOREST_H
