#include "RandomForest.h"

RandomForest::RandomForest() {
    mTreeCount = 16;
    mMaxDepth = 200;
    mCVFolds = 0;
    mMinSampleCount = 2;
    mMaxCategories = 6;
}

RandomForest::RandomForest(int treeCount, int maxDepth, int CVFolds, int minSampleCount, int maxCategories)
        : mTreeCount(treeCount), mMaxDepth(maxDepth), mCVFolds(CVFolds), mMinSampleCount(minSampleCount),
          mMaxCategories(maxCategories) {
    /*
      construct a forest with given number of trees and initialize all the trees with the
      given parameters
    */
    for (int i = 0; i < treeCount; i++){
        mTrees.push_back(cv::ml::DTrees::create());
        mTrees[i]->setMaxDepth(maxDepth);
        mTrees[i]->setMinSampleCount(minSampleCount);
        mTrees[i]->setCVFolds(CVFolds);
        // necessary?
        mTrees[i]->setMaxCategories(maxCategories);
    }
}

RandomForest::~RandomForest() {
}

void RandomForest::setTreeCount(int treeCount) {
    // Fill
    mTreeCount = treeCount;
}

void RandomForest::setMaxDepth(int maxDepth) {
    mMaxDepth = maxDepth;
    for (uint treeIdx = 0; treeIdx < mTreeCount; treeIdx++)
        mTrees[treeIdx]->setMaxDepth(mMaxDepth);
}

void RandomForest::setCVFolds(int cvFols) {
    // Fill
    mCVFolds = cvFols;
}

void RandomForest::setMinSampleCount(int minSampleCount) {
    // Fill
    mMinSampleCount = minSampleCount;
}

void RandomForest::setMaxCategories(int maxCategories) {
    // Fill
    mMaxCategories = maxCategories;
}


void RandomForest::train(std::vector<std::pair<int, cv::Mat>> &trainingImagesLabelVector) {
    // Fill
}

float RandomForest::predict(const cv::Mat1f& features, cv::Mat& predicted_labels, cv::Mat& predicted_confidences) {
    // Fill
}

std::vector<std::pair<int, cv::Mat>> RandomForest::loadTrainDataset() {
    std::vector<std::pair<int, cv::Mat>> labelImagesTrain;
    labelImagesTrain.reserve(49 + 67 + 42 + 53 + 67 + 110);
    int numberOfTrainImages[6] = {49, 67, 42, 53, 67, 110};

    for (int i = 0; i < 6; i++)
    {
        for (size_t j = 0; j < numberOfTrainImages[i]; j++)
        {
            std::stringstream imagePath;
            imagePath << std::string(PROJ_DIR) << "/data/task2/train/" << std::setfill('0') << std::setw(2) << i << "/" << std::setfill('0') << std::setw(4) << j << ".jpg";
            std::string imagePathStr = imagePath.str();
            std::cout << imagePathStr << std::endl;
            std::pair<int, cv::Mat> labelImagesTrainPair;
            labelImagesTrainPair.first = i;
            labelImagesTrainPair.second = imread(imagePathStr, cv::IMREAD_UNCHANGED).clone();
            labelImagesTrain.push_back(labelImagesTrainPair);
        }
    }

    return labelImagesTrain;
}

std::vector<std::pair<int, cv::Mat>> RandomForest::loadTestDataset() {
    std::vector<std::pair<int, cv::Mat>> labelImagesTest;
    labelImagesTest.reserve(60);
    int numberOfTestImages[6] = {10, 10, 10, 10, 10, 10};

    for (int i = 0; i < 6; i++)
    {
        for (size_t j = 0; j < numberOfTestImages[i]; j++)
        {
            std::stringstream imagePath;
            imagePath << std::string(PROJ_DIR) << "/data/task2/test/" << std::setfill('0') << std::setw(2) << i << "/" << std::setfill('0') << std::setw(4) << j + numberOfTestImages[i] << ".jpg";
            std::string imagePathStr = imagePath.str();
            std::cout << imagePathStr << std::endl;
            std::pair<int, cv::Mat> labelImagesTestPair;
            labelImagesTestPair.first = i;
            labelImagesTestPair.second = imread(imagePathStr, cv::IMREAD_UNCHANGED).clone();
            labelImagesTest.push_back(labelImagesTestPair);
        }
    }

    return labelImagesTest;
}

