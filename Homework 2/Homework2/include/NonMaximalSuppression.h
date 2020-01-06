//
// Created by rishabh on 02.01.20.
//

#ifndef HOMEWORK2_NONMAXIMALSUPPRESSION_H
#define HOMEWORK2_NONMAXIMALSUPPRESSION_H


#include <vector>
#include <opencv2/core/mat.hpp>
#include "RandomForest.h"

class NonMaximalSuppression {
public:
    NonMaximalSuppression();

    NonMaximalSuppression(float max, float min, float confidence);

    std::vector<std::pair<int, cv::Mat>> loadTrainDataset();

    std::vector<std::pair<int, cv::Mat>> loadTestDataset();

    std::vector<std::vector<std::vector<int>>> getLabelAndBoundingBoxes();

    std::vector<float> computeTpFpFn(std::vector<ModelPrediction> predictionsNMSVector,
                                     std::vector<ModelPrediction> groundTruthPredictions);

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

    void solver(float subsetPercentage,
                bool underSampling,
                bool augment);

    ~NonMaximalSuppression();

    float NMS_MIN_IOU_THRESHOLD = 0.1f;
    float NMS_MAX_IOU_THRESHOLD = 0.5f;
    float NMS_CONFIDENCE_THRESHOLD = 100.0f;
};


#endif //HOMEWORK2_NONMAXIMALSUPPRESSION_H
