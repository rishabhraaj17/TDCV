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

    std::vector<float> computeTpFpFn(std::vector<Prediction> predictionsNMSVector,
                                     std::vector<Prediction> groundTruthPredictions);

    std::vector<float> solver(std::string outputDir,
                              std::vector<std::pair<int, cv::Mat>> &testImagesLabelVector,
                              std::vector<std::vector<std::vector<int>>> &labelAndBoundingBoxes,
                              cv::Scalar *gtColors,
                              float NMS_MIN_IOU_THRESHOLD,
                              float NMS_MAX_IOU_THRESHOLD,
                              float NMS_CONFIDENCE_THRESHOLD);

    ~NonMaximalSuppression();

    float NMS_MIN_IOU_THRESHOLD = 0.1f;
    float NMS_MAX_IOU_THRESHOLD = 0.5f;
    float NMS_CONFIDENCE_THRESHOLD = 100.0f;
};


#endif //HOMEWORK2_NONMAXIMALSUPPRESSION_H
