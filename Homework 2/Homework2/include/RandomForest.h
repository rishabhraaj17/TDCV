//
// Created by rishabh on 29.12.19.
//

#ifndef RF_RANDOMFOREST_H
#define RF_RANDOMFOREST_H


#include <opencv2/opencv.hpp>
#include <vector>
#include <random>
#include "HOGDescriptor.h"

struct Prediction {
    int label;
    float confidence;
    cv::Rect boundingBox;
};

struct GreaterThan {
    inline bool operator()(const Prediction &struct1, const Prediction &struct2) {
        return (struct1.confidence < struct2.confidence);
    }
};

class RandomForest {
public:
    RandomForest();

    // You can create the forest directly in the constructor or create an empty forest and use the below methods to populate it
    RandomForest(int treeCount, int maxDepth, int CVFolds, int minSampleCount, int maxCategories);

    ~RandomForest();

    void setTreeCount(int treeCount);

    void setMaxDepth(int maxDepth);

    void setCVFolds(int cvFols);

    void setMinSampleCount(int minSampleCount);

    void setMaxCategories(int maxCategories);


    void
    train(std::vector<std::pair<int, cv::Mat>> trainingImagesLabelVector, float subsetPercentage, cv::Size winStride,
          cv::Size padding, bool undersampling, bool augment, cv::Size winSize);

    Prediction predict(cv::Mat &testImage, cv::Size winStride, cv::Size padding, cv::Size winSize);

    std::vector<std::pair<int, cv::Mat>> loadTrainDataset();

    std::vector<std::pair<int, cv::Mat>> loadTestDataset();

    static cv::Ptr<RandomForest> createRandomForest(int numberOfClasses, int numberOfDTrees, cv::Size winSize);


    int getCVFolds();

    int getMaxCategories();

    int getMaxDepth();

    int getMinSampleCount();

    int getTreeCount();

    cv::HOGDescriptor createHogDescriptor(cv::Size size);

    cv::Mat resizeToBoundingBox(cv::Mat &inputImage, cv::Size size);

    std::vector<cv::Ptr<cv::ml::DTrees>> getTrees();

private:
    int mTreeCount;
    int mMaxDepth;
    int mCVFolds;

    int mMinSampleCount;
    int mMaxCategories;

    cv::Size mWinSize;
    cv::HOGDescriptor mHogDescriptor;
    std::mt19937 mRandomGenerator;

    // M-Trees for constructing thr forest
    // decision tress
    std::vector<cv::Ptr<cv::ml::DTrees> > mTrees;

    std::vector<int> getRandomUniqueIndices(int start, int end, int numOfSamples);

    cv::Ptr<cv::ml::DTrees> trainDecisionTree(std::vector<std::pair<int, cv::Mat>> &trainingImagesLabelVector,
                                              cv::Size winStride,
                                              cv::Size padding,
                                              cv::Size winSize);

    std::vector<std::pair<int, cv::Mat>>
    generateTrainingImagesLabelSubsetVector(std::vector<std::pair<int, cv::Mat>> &trainingImagesLabelVector,
                                            float subsetPercentage,
                                            bool undersampling);

    std::vector<cv::Mat> augmentImage(cv::Mat &inputImage);

    // copied from external library : https://github.com/takmin/DataAugmentation
    void RandomRotateImage(const cv::Mat &src, cv::Mat &dst, float yaw_range, float pitch_range, float roll_range,
                           const cv::Rect &area = cv::Rect(-1, -1, 0, 0), cv::RNG rng = cv::RNG(),
                           float Z = 1000, int interpolation = cv::INTER_LINEAR, int boarder_mode = cv::BORDER_CONSTANT,
                           const cv::Scalar boarder_color = cv::Scalar(0, 0, 0));

    void ImageRotate(const cv::Mat &src, cv::Mat &dst, float yaw, float pitch, float roll, float Z, int interpolation,
                     int boarder_mode, const cv::Scalar &border_color);

    void composeExternalMatrix(float yaw, float pitch, float roll, float trans_x, float trans_y, float trans_z,
                               cv::Mat &external_matrix);

    cv::Mat Rect2Mat(const cv::Rect &img_rect);

    void CircumTransImgRect(const cv::Size &img_size, const cv::Mat &transM, cv::Rect_<double> &CircumRect);

    void CreateMap(const cv::Size &src_size, const cv::Rect_<double> &dst_rect, const cv::Mat &transMat, cv::Mat &map_x,
                   cv::Mat &map_y);

    cv::Rect ExpandRectForRotate(const cv::Rect &area);
};

#endif //RF_RANDOMFOREST_H
