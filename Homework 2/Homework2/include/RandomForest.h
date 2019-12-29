

#ifndef RF_RANDOMFOREST_H
#define RF_RANDOMFOREST_H


#include <opencv2/opencv.hpp>
#include <vector>
#include <random>
#include "HOGDescriptor.h"

struct Prediction {
    int label;
    float confidence;
    cv::Rect bbox;
};

struct GreaterThan {
    inline bool operator()(const Prediction &struct1, const Prediction &struct2) {
        return (struct1.confidence < struct2.confidence);
    }
};

class RandomForest {
public:
    RandomForest();

    /*RandomForest() : mTreeCount(16) {
        mMaxDepth = 200;
        mCVFolds = 0;
        mMinSampleCount = 2;
        mMaxCategories = 6;
    }*/

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
          cv::Size padding, bool undersampling, bool augment);

    Prediction predict(cv::Mat &testImage, cv::Size winStride, cv::Size padding);

    std::vector<std::pair<int, cv::Mat>> loadTrainDataset();

    std::vector<std::pair<int, cv::Mat>> loadTestDataset();

    static cv::Ptr<RandomForest> createRandomForest(int numberOfClasses, int numberOfDTrees, cv::Size winSize);


private:
    int mTreeCount;
    int mMaxDepth;
    int mCVFolds;
    int mMinSampleCount;
    int mMaxCategories;

    // M-Trees for constructing thr forest
    // decision tress
    std::vector<cv::Ptr<cv::ml::DTrees> > mTrees;
    std::mt19937 mRandomGenerator;

    cv::Size mWinSize;
    cv::HOGDescriptor mHogDescriptor;

    std::vector<int> getRandomUniqueIndices(int start, int end, int numOfSamples);

    cv::HOGDescriptor createHogDescriptor();

    cv::Ptr<cv::ml::DTrees> trainDecisionTree(std::vector<std::pair<int, cv::Mat>> &trainingImagesLabelVector,
                                              cv::Size winStride,
                                              cv::Size padding);

    cv::Mat resizeToBoundingBox(cv::Mat &inputImage);

    std::vector<std::pair<int, cv::Mat>>
    generateTrainingImagesLabelSubsetVector(std::vector<std::pair<int, cv::Mat>> &trainingImagesLabelVector,
                                            float subsetPercentage,
                                            bool undersampling);

    std::vector<cv::Mat> augmentImage(cv::Mat &inputImage);
};

#endif //RF_RANDOMFOREST_H
