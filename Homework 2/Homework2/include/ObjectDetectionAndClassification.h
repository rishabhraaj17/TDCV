//
// Created by rishabh on 02.01.20.
//

#ifndef HOMEWORK2_OBJECTDETECTIONANDCLASSIFICATION_H
#define HOMEWORK2_OBJECTDETECTIONANDCLASSIFICATION_H


#include <vector>
#include <opencv2/core/mat.hpp>
#include "RandomForest.h"

class ObjectDetectionAndClassification {
public:
    ObjectDetectionAndClassification();

    ObjectDetectionAndClassification(float max, float min, float confidence,
                                     const cv::Size &winSize,
                                     int numClasses,
                                     float scaleFactor,
                                     int strideX,
                                     int strideY);

    std::vector<std::pair<int, cv::Mat>> loadTrainDataset();

    std::vector<std::pair<int, cv::Mat>> loadTestDataset();

    std::vector<std::pair<int, cv::Mat>> debugTestDataset();

    std::vector<std::vector<std::vector<int>>> getGroundTruth();

    std::vector<float> computeTruePositiveFalsePositive(const std::vector<ModelPrediction> &predictionsAfterNMS,
                                                        const std::vector<ModelPrediction> &groundTruth,
                                                        float thresholdIOU);

    std::vector<float> computeTrueNegativeFalseNegative(const std::vector<ModelPrediction> &predictionsAfterNMS,
                                                        const std::vector<ModelPrediction> &groundTruth,
                                                        float thresholdIOU);

    std::vector<float> precisionRecallNMS(const std::string &savePath,
                                          std::vector<std::pair<int, cv::Mat>> &testDataset,
                                          std::vector<std::vector<std::vector<int>>> &groundTruth,
                                          float nmsMin, float nmsMax,
                                          float nmsConfidence);

    void evaluate_metrics(std::string savePath,
                          std::vector<std::pair<int, cv::Mat>> &testDataset,
                          std::vector<std::vector<std::vector<int>>> &groundTruth);

    void computeBoundingBoxAndConfidence(cv::Ptr<RandomForest> &randomForest,
                                         std::vector<std::pair<int, cv::Mat>> &testDataset,
                                         std::vector<std::vector<std::vector<int>>> &groundTruth,
                                         const cv::Size &winStride,
                                         const cv::Size &padding, cv::Scalar *gtColors,
                                         const std::string &savePath,
                                         const cv::Size &_winSize);

    void solver(std::vector<std::pair<int, cv::Mat>> trainDataset,
                std::vector<std::vector<std::vector<int>>> groundTruth,
                int numTrees,
                const std::string &savePath,
                float subsetPercentage,
                bool underSampling,
                bool augment,
                bool doSaveModel = false,
                bool loadModelFromDisk = false,
                std::string pathToLoadModel = "");

    ~ObjectDetectionAndClassification();

    float NMS_MIN_IOU_THRESHOLD = 0.1f;
    float NMS_MAX_IOU_THRESHOLD = 0.6f;
    float NMS_CONFIDENCE_THRESHOLD = 100.0f;

    cv::Size winSize;
    int numClasses;
    cv::Scalar bBoxColors[4];
    float scaleFactor;
    int strideX;
    int strideY;
};


#endif //HOMEWORK2_OBJECTDETECTIONANDCLASSIFICATION_H
