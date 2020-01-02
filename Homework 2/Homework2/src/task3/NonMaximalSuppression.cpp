//
// Created by rishabh on 02.01.20.
//

#include "NonMaximalSuppression.h"

std::vector<std::pair<int, cv::Mat>> NonMaximalSuppression::loadTrainDataset() {
    std::vector<std::pair<int, cv::Mat>> labelImagesTrain;
    labelImagesTrain.reserve(53 + 81 + 51 + 290);
    int numberOfTrainImages[6] = {53, 81, 51, 290};

    for (int i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < numberOfTrainImages[i]; j++)
        {
            std::stringstream imagePath;
            imagePath << std::string(PROJ_DIR) << "/data/task3/train/" << std::setfill('0') << std::setw(2) <<
            i << "/" << std::setfill('0') << std::setw(4) << j << ".jpg";
            std::string imagePathStr = imagePath.str();
            //std::cout << imagePathStr << std::endl;
            std::pair<int, cv::Mat> labelImagesTrainPair;
            labelImagesTrainPair.first = i;
            labelImagesTrainPair.second = imread(imagePathStr, cv::IMREAD_UNCHANGED).clone();
            labelImagesTrain.push_back(labelImagesTrainPair);
        }
    }

    return labelImagesTrain;
}

std::vector<std::pair<int, cv::Mat>> NonMaximalSuppression::loadTestDataset() {
    std::vector<std::pair<int, cv::Mat>> labelImagesTest;
    labelImagesTest.reserve(44);
    int numberOfTestImages[1] = {44};

    for (size_t j = 0; j < numberOfTestImages[0]; j++)
    {
        std::stringstream imagePath;
        imagePath << std::string(PROJ_DIR) << "/data/task3/test/" << std::setfill('0') << std::setw(4) <<
        j << ".jpg";
        std::string imagePathStr = imagePath.str();
        //std::cout << imagePathStr << std::endl;
        std::pair<int, cv::Mat> labelImagesTestPair;
        labelImagesTestPair.first = -1; // These test images have no label
        labelImagesTestPair.second = imread(imagePathStr, cv::IMREAD_UNCHANGED).clone();
        labelImagesTest.push_back(labelImagesTestPair);
    }

    return labelImagesTest;
}

std::vector<std::vector<std::vector<int>>> NonMaximalSuppression::getLabelAndBoundingBoxes() {
    return std::vector<std::vector<std::vector<int>>>();
}

std::vector<float> NonMaximalSuppression::computeTpFpFn(std::vector<Prediction> predictionsNMSVector,
                                                        std::vector<Prediction> groundTruthPredictions) {
    return std::vector<float>();
}

std::vector<float>
NonMaximalSuppression::solver(std::string outputDir, std::vector<std::pair<int, cv::Mat>> &testImagesLabelVector,
                              std::vector<std::vector<std::vector<int>>> &labelAndBoundingBoxes, cv::Scalar *gtColors,
                              float NMS_MIN_IOU_THRESHOLD, float NMS_MAX_IOU_THRESHOLD,
                              float NMS_CONFIDENCE_THRESHOLD) {
    return std::vector<float>();
}

NonMaximalSuppression::NonMaximalSuppression(float max, float min, float confidence) {
    this->NMS_MAX_IOU_THRESHOLD = max;
    this->NMS_MIN_IOU_THRESHOLD = min;
    this->NMS_CONFIDENCE_THRESHOLD = confidence;
}

NonMaximalSuppression::NonMaximalSuppression() = default;

NonMaximalSuppression::~NonMaximalSuppression() = default;
