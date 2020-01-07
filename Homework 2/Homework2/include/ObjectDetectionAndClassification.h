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

    std::vector<float> precisionRecallNMS(std::string outputDir,
                                          std::vector<std::pair<int, cv::Mat>> &testImagesLabelVector,
                                          std::vector<std::vector<std::vector<int>>> &labelAndBoundingBoxes,
                                          cv::Scalar *gtColors,
                                          float NMS_MIN_IOU_THRESHOLD,
                                          float NMS_MAX_IOU_THRESHOLD,
                                          float NMS_CONFIDENCE_THRESHOLD);

    void evaluate_metrics(std::string outputDir,
                          std::vector<std::pair<int, cv::Mat>> &testImagesLabelVector,
                          std::vector<std::vector<std::vector<int>>> &labelAndBoundingBoxes);

    void computeBoundingBoxAndConfidence(cv::Ptr<RandomForest> &randomForest,
                                         std::vector<std::pair<int, cv::Mat>> &testImagesLabelVector,
                                         std::vector<std::vector<std::vector<int>>> &labelAndBoundingBoxes,
                                         int strideX, int strideY,
                                         cv::Size winStride, cv::Size padding,
                                         cv::Scalar *gtColors,
                                         float scaleFactor,
                                         std::string outputDir,
                                         cv::Size winSize);

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
    float NMS_MAX_IOU_THRESHOLD = 0.5f;
    float NMS_CONFIDENCE_THRESHOLD = 100.0f;

    cv::Size winSize;
    int numClasses;
    cv::Scalar bBoxColors[4];
    float scaleFactor;
    int strideX;
    int strideY;
};


#endif //HOMEWORK2_OBJECTDETECTIONANDCLASSIFICATION_H
