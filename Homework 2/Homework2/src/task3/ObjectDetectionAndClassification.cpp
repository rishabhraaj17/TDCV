//
// Created by rishabh on 02.01.20.
//

#include <sys/ioctl.h>
#include "ObjectDetectionAndClassification.h"
#include <boost/filesystem.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <chrono>
#include <ctime>


std::vector<std::pair<int, cv::Mat>> ObjectDetectionAndClassification::loadTrainDataset() {
    std::vector<std::pair<int, cv::Mat>> trainDataset;
    trainDataset.reserve(53 + 81 + 51 + 290);
    int trainImagesPerClassCount[4] = {53, 81, 51, 290};

    for (int i = 0; i < 4; i++) {
        for (size_t j = 0; j < trainImagesPerClassCount[i]; j++) {
            std::stringstream imagePath;
            imagePath << std::string(PROJ_DIR) << "/data/task3/train/" << std::setfill('0') << std::setw(2) <<
                      i << "/" << std::setfill('0') << std::setw(4) << j << ".jpg";
            std::string imagePathStr = imagePath.str();
            std::pair<int, cv::Mat> labelImagesTrainPair;
            labelImagesTrainPair.first = i;
            labelImagesTrainPair.second = imread(imagePathStr, cv::IMREAD_UNCHANGED).clone();
            trainDataset.push_back(labelImagesTrainPair);
        }
    }
    return trainDataset;
}

std::vector<std::pair<int, cv::Mat>> ObjectDetectionAndClassification::loadTestDataset() {
    std::vector<std::pair<int, cv::Mat>> testDataset;
    testDataset.reserve(44);
    int testImagesPerClassCount = 44;

    for (size_t j = 0; j < testImagesPerClassCount; j++) {
        std::stringstream imagePath;
        imagePath << std::string(PROJ_DIR) << "/data/task3/test/" << std::setfill('0') << std::setw(4) <<
                  j << ".jpg";
        std::string imagePathStr = imagePath.str();
        std::pair<int, cv::Mat> labelImagesTestPair;
        labelImagesTestPair.first = -1;
        labelImagesTestPair.second = imread(imagePathStr, cv::IMREAD_UNCHANGED).clone();
        testDataset.push_back(labelImagesTestPair);
    }
    return testDataset;
}

std::vector<std::pair<int, cv::Mat>> ObjectDetectionAndClassification::debugTestDataset() {
    std::vector<std::pair<int, cv::Mat>> testDataset;
    testDataset.reserve(1);
    int testImagesPerClassCount = 1;

    for (size_t j = 0; j < testImagesPerClassCount; j++) {
        std::stringstream imagePath;
        imagePath << std::string(PROJ_DIR) << "/data/task3/test/" << std::setfill('0') << std::setw(4) <<
                  j << ".jpg";
        std::string imagePathStr = imagePath.str();
        std::pair<int, cv::Mat> labelImagesTestPair;
        labelImagesTestPair.first = -1;
        labelImagesTestPair.second = imread(imagePathStr, cv::IMREAD_UNCHANGED).clone();
        testDataset.push_back(labelImagesTestPair);
    }
    return testDataset;
}

std::vector<std::vector<std::vector<int>>> ObjectDetectionAndClassification::getGroundTruth() {
    /**
     * Parses the ground truth text file and returns vector -- image-objects-labAndBbox
     */
    int testImagesPerClassCount = 44;
    std::vector<std::vector<std::vector<int>>> groundTruthLabelAndBoundingBoxes;
    for (size_t j = 0; j < testImagesPerClassCount; j++) {
        std::stringstream path;
        path << std::string(PROJ_DIR) << "/data/task3/gt/" << std::setfill('0') << std::setw(4) << j << ".gt.txt";
        std::string pathStr = path.str();

        std::fstream groundTruth;
        groundTruth.open(pathStr);
        if (!groundTruth.is_open()) {
            std::cout << "Failed to open file: " << pathStr << std::endl;
            exit(-1);
        }

        std::string line;
        std::vector<std::vector<int>> boundingBoxesPerImage;
        while (std::getline(groundTruth, line)) {
            std::istringstream in(line);
            std::vector<int> labelAndBoundingBox(5);
            int temp;
            for (size_t i = 0; i < 5; i++) {
                in >> temp;
                labelAndBoundingBox.at(i) = temp;
            }
            boundingBoxesPerImage.push_back(labelAndBoundingBox);
        }
        groundTruthLabelAndBoundingBoxes.push_back(boundingBoxesPerImage);
    }
    return groundTruthLabelAndBoundingBoxes;
}

std::vector<float> ObjectDetectionAndClassification::computeTruePositiveFalsePositive(
        const std::vector<ModelPrediction> &predictionsAfterNMS, const std::vector<ModelPrediction> &groundTruth,
        float thresholdIOU = 0.5f) {
    float truePositive = 0, falsePositive = 0;

    for (auto &&pred : predictionsAfterNMS) {
        bool predMatchedGroundTruth = false;
        cv::Rect myRect = pred.boundingBox;
        for (auto &&gt : groundTruth) {
            if (gt.label != pred.label)
                continue;
            cv::Rect gtRect = gt.boundingBox;
            float iouScore = ((myRect & gtRect).area() * 1.0f) / ((myRect | gtRect).area());
            if (iouScore > thresholdIOU) {
                predMatchedGroundTruth = true;
                break;
            }
        }
        if (predMatchedGroundTruth)
            truePositive++;
        else
            falsePositive++;
    }
    std::vector<float> truePositiveFalsePositive;
    truePositiveFalsePositive.push_back(truePositive);
    truePositiveFalsePositive.push_back(falsePositive);
    return truePositiveFalsePositive;
}

std::vector<float> ObjectDetectionAndClassification::computeTrueNegativeFalseNegative(
        const std::vector<ModelPrediction> &predictionsAfterNMS, const std::vector<ModelPrediction> &groundTruth,
        float thresholdIOU = 0.5f) {
    float trueNegative = 0, falseNegative = 0;

    for (auto &&pred : groundTruth) {
        bool gtMissed = true;
        cv::Rect gtRect = pred.boundingBox;
        for (auto &&myPrediction : predictionsAfterNMS) {
            if (pred.label != myPrediction.label)
                continue;
            cv::Rect myRect = myPrediction.boundingBox;
            float iouScore = ((myRect & gtRect).area() * 1.0f) / ((myRect | gtRect).area());
            if (iouScore > thresholdIOU) {
                gtMissed = false;
                break;
            }
        }
        if (gtMissed)
            falseNegative++;
    }
    std::vector<float> falseNegativeTrueNegative;
    falseNegativeTrueNegative.push_back(falseNegative);
    falseNegativeTrueNegative.push_back(trueNegative);
    return falseNegativeTrueNegative;
}


std::vector<float>
ObjectDetectionAndClassification::precisionRecallNMS(const std::string &savePath,
                                                     std::vector<std::pair<int, cv::Mat>> &testDataset,
                                                     std::vector<std::vector<std::vector<int>>> &groundTruth,
                                                     float nmsMin, float nmsMax,
                                                     float nmsConfidence) {
    float fontScale = 0.5;
    int fontFace = cv::FONT_HERSHEY_PLAIN;
    std::ifstream modelPredictions(savePath + "predictions.txt");
    if (!modelPredictions.is_open()) {
        std::cout << "Failed to open" << savePath + "predictions.txt" << std::endl;
        exit(-1);
    }

    float truePositive = 0, falsePositive = 0, falseNegative = 0;
    for (size_t i = 0; i < testDataset.size(); i++) {
        int currentImage;
        modelPredictions >> currentImage;
        assert(currentImage == i);

        int groundTruthCount, skip;
        modelPredictions >> groundTruthCount;
        std::vector<ModelPrediction> groundTruthPredictions;
        for (size_t g = 0; g < groundTruthCount; g++) {
            ModelPrediction groundTruthPrediction;
            groundTruthPrediction.label = groundTruth.at(i).at(g).at(0);
            groundTruthPrediction.boundingBox.x = groundTruth.at(i).at(g).at(1);
            groundTruthPrediction.boundingBox.y = groundTruth.at(i).at(g).at(2);
            groundTruthPrediction.boundingBox.height = groundTruth.at(i).at(g).at(3);
            groundTruthPrediction.boundingBox.height -= groundTruthPrediction.boundingBox.x;
            groundTruthPrediction.boundingBox.width = groundTruth.at(i).at(g).at(4);
            groundTruthPrediction.boundingBox.width -= groundTruthPrediction.boundingBox.y;
            groundTruthPredictions.push_back(groundTruthPrediction);

            modelPredictions >> skip;
            for (size_t k = 0; k < 4; k++) {
                modelPredictions >> skip;
            }
        }

        cv::Mat currentTestImage = testDataset.at(i).second;
        std::vector<ModelPrediction> predictions;
        int totalModelPredictions;
        modelPredictions >> totalModelPredictions;
        predictions.reserve(totalModelPredictions);
        for (size_t i = 0; i < totalModelPredictions; i++) {
            ModelPrediction prediction;
            modelPredictions >> prediction.label;
            modelPredictions >> prediction.boundingBox.x >> prediction.boundingBox.y >> prediction.boundingBox.height
                             >> prediction.boundingBox.width;
            modelPredictions >> prediction.confidence;
            predictions.push_back(prediction);
        }

        // NMS
        cv::Mat clonedFirst = currentTestImage.clone();
        cv::Mat clonedSecond = currentTestImage.clone();
        std::vector<ModelPrediction> NMSPredictions;
        NMSPredictions.reserve(32);

        // Drop Bounding boxes with low threshold.
        std::vector<ModelPrediction>::iterator iter;
        for (iter = predictions.begin(); iter != predictions.end();) {
            if (iter->confidence < nmsConfidence)
                iter = predictions.erase(iter);
            else
                ++iter;
        }

        for (auto &&prediction : predictions) {
            cv::rectangle(clonedFirst, prediction.boundingBox, this->bBoxColors[prediction.label]);
            bool bBoxClusterPresent = false;
            for (auto &&nmsCluster : NMSPredictions) {
                if (nmsCluster.label == prediction.label) {
                    cv::Rect &rect1 = prediction.boundingBox;
                    cv::Rect &rect2 = nmsCluster.boundingBox;
                    float iouScore = ((rect1 & rect2).area() * 1.0f) / ((rect1 | rect2).area());
                    if (iouScore > nmsMax)
                    {
                        nmsCluster.boundingBox = rect1 | rect2;
                        nmsCluster.confidence = std::max(prediction.confidence, nmsCluster.confidence);
                        bBoxClusterPresent = true;
                        break;
                    }
                    else if (iouScore > nmsMin)
                    {
                        if (nmsCluster.confidence < prediction.confidence)
                        {
                            nmsCluster = prediction;
                        }
                        bBoxClusterPresent = true;
                        break;
                    }
                }
            }

            // Adding isolated boxes -- rethink
            if (!bBoxClusterPresent) {
                NMSPredictions.push_back(prediction);
            }
        }

        for (auto &&nmsPrediction : NMSPredictions) {
            cv::rectangle(clonedSecond, nmsPrediction.boundingBox, this->bBoxColors[nmsPrediction.label]);
            cv::putText(
                    clonedSecond,
                    "Class " + std::to_string(nmsPrediction.label) + ":" + std::to_string(nmsPrediction.confidence),
                    cv::Point(nmsPrediction.boundingBox.x + 5, nmsPrediction.boundingBox.y + 5),
                    fontFace,
                    fontScale,
                    this->bBoxColors[nmsPrediction.label]
            );
        }

        // Save Vis
        std::stringstream NMSSavePath;
        NMSSavePath << savePath << std::setfill('0') << std::setw(4) << i << "_NMSOutputWith_" << "Confidence_"
                          << nmsConfidence * 100.0f << "%.png";
        std::string nmsOutputFilePathStr = NMSSavePath.str();
        cv::imwrite(nmsOutputFilePathStr, clonedSecond);

        // Metrics
        std::vector<float> truePositiveFalsePositive = computeTruePositiveFalsePositive(NMSPredictions,
                                                                                        groundTruthPredictions, 0.5f);
        std::vector<float> falseNegativeTrueNegative = computeTrueNegativeFalseNegative(NMSPredictions,
                                                                                        groundTruthPredictions, 0.5f);
        truePositive += truePositiveFalsePositive[0];
        falsePositive += truePositiveFalsePositive[1];
        falseNegative += falseNegativeTrueNegative[0];
    }

    float precision = truePositive / (truePositive + falsePositive);
    float recall = truePositive / (truePositive + falseNegative);
    modelPredictions.close();
    std::vector<float> precisionRecallValue;
    precisionRecallValue.push_back(precision);
    precisionRecallValue.push_back(recall);
    return precisionRecallValue;
}

void ObjectDetectionAndClassification::evaluate_metrics(std::string savePath,
                                                        std::vector<std::pair<int, cv::Mat>> &testDataset,
                                                        std::vector<std::vector<std::vector<int>>> &groundTruth) {
    std::ofstream metricLogCSV;
    metricLogCSV.open(savePath + "metrics.csv");
    if (!metricLogCSV.is_open()) {
        std::cout << "Failed to open" << savePath + "metrics.csv" << std::endl;
        exit(-1);
    }
    metricLogCSV << "Precision,Recall" << std::endl;
    std::cout << "\nNMS_CONFIDENCE_THRESHOLD " << "      Precision           "
                                                  "Recall " << std::endl;
    // 60 to 90 is the reasonable range! 55 to 80 is promising// HyperParam
    for (int confidence = 60; confidence <= 84; confidence += 2) {
        this->NMS_CONFIDENCE_THRESHOLD = confidence / 100.0f;
        std::vector<float> precisionRecallValue = precisionRecallNMS(savePath, testDataset,
                                                                     groundTruth,
                                                                     this->NMS_MIN_IOU_THRESHOLD,
                                                                     this->NMS_MAX_IOU_THRESHOLD,
                                                                     this->NMS_CONFIDENCE_THRESHOLD);
        metricLogCSV << precisionRecallValue[0] << "," << precisionRecallValue[1] << std::endl;

        std::cout << "             " << NMS_CONFIDENCE_THRESHOLD << "             " << precisionRecallValue[0]
                  << "             " << precisionRecallValue[1] << std::endl;
    }
    metricLogCSV.close();
}


void ObjectDetectionAndClassification::computeBoundingBoxAndConfidence(cv::Ptr<RandomForest> &randomForest,
                                                                       std::vector<std::pair<int, cv::Mat>> &testDataset,
                                                                       std::vector<std::vector<std::vector<int>>> &groundTruth,
                                                                       const cv::Size &winStride,
                                                                       const cv::Size &padding, cv::Scalar *gtColors,
                                                                       const std::string &savePath,
                                                                       const cv::Size &_winSize = cv::Size(128, 128)) {
    /**
     * Format :
     * Image Number, e.g 0
     * Count of ground truth bounding boxes, e.g 3
     * Ground Truth Labels and Bounding Box for next count lines
     * Count of predicted bounding boxes
     * Predicted Labels, Bounding Box and confidence for next lines until next image
     */
    std::ofstream modelPredictions(savePath + "predictions.txt");
    if (!modelPredictions.is_open()) {
        std::cout << "Failed to open" << savePath + "predictions.txt" << std::endl;
        exit(-1);
    }

    for (size_t i = 0; i < testDataset.size(); i++) {
        std::cout << "\nPrediction : Image - " << i << ". Total Images : " << testDataset.size() << std::endl;
        modelPredictions << i << std::endl;
        cv::Mat currentTestImage = testDataset.at(i).second;

        // start with high min and low max
        // adaptive search for nice bounding boxes to slide over the image
        int minBoundingBoxSideLength = 1000, maxBoundingBoxSideLength = -1;
        std::vector<std::vector<int>> gt = groundTruth.at(i);
        modelPredictions << gt.size() << std::endl;

        for (auto &g : gt) {
            std::vector<int> boundingBox = g;
            cv::Rect rect(boundingBox[1], boundingBox[2], boundingBox[3] - boundingBox[1],
                          boundingBox[4] - boundingBox[2]);
            modelPredictions << g.at(0) << " " << rect.x << " " << rect.y << " " << rect.height << " " << rect.width
                             << std::endl;
            minBoundingBoxSideLength = std::min(minBoundingBoxSideLength, std::min(rect.width, rect.height));
            maxBoundingBoxSideLength = std::max(maxBoundingBoxSideLength, std::max(rect.width, rect.height));
        }
        minBoundingBoxSideLength -= 10;
        maxBoundingBoxSideLength += 10;

        int currentOptimalBoundingBoxSideLength = minBoundingBoxSideLength;
        std::vector<ModelPrediction> predictions; // Output of Hog Detection
        while (true) {
            std::cout << "Processing at bounding box side length: " << currentOptimalBoundingBoxSideLength << '\n';
            // Sliding window with strideX and StrideY
            for (size_t row = 0; row < currentTestImage.rows - currentOptimalBoundingBoxSideLength; row += this->strideY) {
                for (size_t col = 0; col < currentTestImage.cols - currentOptimalBoundingBoxSideLength; col += this->strideX) {
                    cv::Rect rect(col, row, currentOptimalBoundingBoxSideLength, currentOptimalBoundingBoxSideLength);
                    cv::Mat rectImage = currentTestImage(rect);

                    // Predict each window if object is present inside it
                    ModelPrediction prediction = randomForest->predictPerImage(rectImage, winStride, padding, _winSize);
                    if (prediction.label != 3) {
                        prediction.boundingBox = rect;
                        predictions.push_back(prediction);
                    }
                }
            }

            if (currentOptimalBoundingBoxSideLength == maxBoundingBoxSideLength)
                break;

            currentOptimalBoundingBoxSideLength = lround(currentOptimalBoundingBoxSideLength * this->scaleFactor + 0.5);
            if (currentOptimalBoundingBoxSideLength > maxBoundingBoxSideLength)
                currentOptimalBoundingBoxSideLength = maxBoundingBoxSideLength;
        }

        modelPredictions << predictions.size() << std::endl;
        for (auto &&prediction : predictions) {
            modelPredictions << prediction.label << " " << prediction.boundingBox.x << " " <<
                             prediction.boundingBox.y << " " << prediction.boundingBox.height << " " <<
                             prediction.boundingBox.width << " " << prediction.confidence << std::endl;
        }

        // save for visualization
        cv::Mat predictionImageVis = currentTestImage.clone();
        for (auto &&prediction : predictions)
            cv::rectangle(predictionImageVis, prediction.boundingBox, gtColors[prediction.label]);

        float fontScale = 0.5;
        int fontFace = cv::FONT_HERSHEY_PLAIN;
        gt = groundTruth.at(i);
        cv::Mat groundTruthImageVis = currentTestImage.clone();
        for (auto bBox : gt) {
            cv::Rect rect(bBox[1], bBox[2], bBox[3] - bBox[1], bBox[4] - bBox[2]);
            cv::rectangle(groundTruthImageVis, rect, gtColors[bBox[0]]);
            cv::putText(
                    groundTruthImageVis,
                    "Class " + std::to_string(bBox[0]) + ":" + "1",
                    cv::Point(bBox[1] + 5, bBox[2] + 5),
                    fontFace,
                    fontScale,
                    this->bBoxColors[bBox[0]]
            );
        }

        std::stringstream predictionPath;
        predictionPath << savePath << std::setfill('0') << std::setw(4) << i << "_predictions.png";
        std::string predictionPathStr = predictionPath.str();
        cv::imwrite(predictionPathStr, predictionImageVis);

        std::stringstream groundTruthPath;
        groundTruthPath << savePath << std::setfill('0') << std::setw(4) << i << "_groundTruth.png";
        std::string groundTruthPathStr = groundTruthPath.str();
        cv::imwrite(groundTruthPathStr, groundTruthImageVis);
    }
    modelPredictions.close();
}

void ObjectDetectionAndClassification::solver(std::vector<std::pair<int, cv::Mat>> trainDataset,
                                              std::vector<std::vector<std::vector<int>>> groundTruth,
                                              int numTrees,
                                              const std::string &savePath,
                                              float subsetPercentage = 50.0f,
                                              bool underSampling = false,
                                              bool augment = true,
                                              bool doSaveModel,
                                              bool loadModelFromDisk,
                                              std::string pathToLoadModel) {

    int numClasses = 4;
    cv::Size winSize(128, 128);
    cv::Size winStride(8, 8);
    cv::Size padding(0, 0);
    cv::Ptr<RandomForest> randomForest = RandomForest::createRandomForest(numClasses, numTrees, winSize);

    auto timeNow = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(timeNow);
    std::string loadedModelTime = "";
    std::string oneByOnePath = "../output/models/OneByOne_NMS-treeCount-" + std::to_string(numTrees) + std::to_string(time);

    //load model
    if (loadModelFromDisk) {
        randomForest->setTrees(
                RandomForest::loadModel(pathToLoadModel, numTrees));
    } else {
        randomForest->train(trainDataset, subsetPercentage, winStride, padding, underSampling, augment,
                            winSize, true, true, 300, true,
                            oneByOnePath);
    }
    //save the model for reuse
    if (doSaveModel and not loadModelFromDisk) {
        randomForest->saveModel("../output/models/NMS-treeCount-" + std::to_string(numTrees) + std::to_string(time));
        std::cout<< "Model Saved!"<< std::endl;
    }

    std::vector<std::pair<int, cv::Mat>> testDataset = loadTestDataset();

    cv::utils::fs::createDirectories(savePath);
    computeBoundingBoxAndConfidence(randomForest, testDataset, groundTruth,
                                    winStride, padding, this->bBoxColors, savePath, winSize);

    evaluate_metrics(savePath, testDataset, groundTruth);
}

ObjectDetectionAndClassification::ObjectDetectionAndClassification() = default;

ObjectDetectionAndClassification::~ObjectDetectionAndClassification() = default;

ObjectDetectionAndClassification::ObjectDetectionAndClassification(float max, float min, float confidence,
                                                                   const cv::Size &winSize,
                                                                   int numClasses,
                                                                   float scaleFactor,
                                                                   int strideX,
                                                                   int strideY) {
    this->NMS_MAX_IOU_THRESHOLD = max;
    this->NMS_MIN_IOU_THRESHOLD = min;
    this->NMS_CONFIDENCE_THRESHOLD = confidence;
    this->numClasses = numClasses;
    this->scaleFactor = scaleFactor;
    this->strideX = strideX;
    this->strideY = strideY;
    this->bBoxColors[0] = cv::Scalar(255, 0, 0);
    this->bBoxColors[1] = cv::Scalar(0, 255, 0);
    this->bBoxColors[2] = cv::Scalar(0, 0, 255);
    this->bBoxColors[3] = cv::Scalar(255, 255, 0);
}