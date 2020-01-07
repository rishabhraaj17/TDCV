//
// Created by rishabh on 02.01.20.
//

#include <sys/ioctl.h>
#include "ObjectDetectionAndClassification.h"
#include <boost/filesystem.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <chrono>
#include <ctime>

//#define DISPLAY

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

//todo
std::vector<float>
ObjectDetectionAndClassification::precisionRecallNMS(std::string outputDir,
                                                     std::vector<std::pair<int, cv::Mat>> &testImagesLabelVector,
                                                     std::vector<std::vector<std::vector<int>>> &labelAndBoundingBoxes,
                                                     cv::Scalar *gtColors,
                                                     float NMS_MIN_IOU_THRESHOLD, float NMS_MAX_IOU_THRESHOLD,
                                                     float NMS_CONFIDENCE_THRESHOLD) {
    std::ifstream predictionsFile(outputDir + "predictions.txt");
    if (!predictionsFile.is_open()) {
        std::cout << "Failed to open" << outputDir + "predictions.txt" << std::endl;
        exit(-1);
    }


    float truePositive = 0, falsePositive = 0, falseNegative = 0;
    for (size_t i = 0; i < testImagesLabelVector.size(); i++) {
        int fileNumber;
        predictionsFile >> fileNumber; // Prediction file format: Starts with File number
        assert(fileNumber == i);

        // Ignore the ground truth data in predictions.txt. we already have it.
        int tmp, tmp1;
        predictionsFile >> tmp; // Ignore - number of ground truth
        std::vector<ModelPrediction> groundTruthPredictions;
        for (size_t j = 0; j < tmp; j++) {
            ModelPrediction groundTruthPrediction;
            groundTruthPrediction.label = labelAndBoundingBoxes.at(i).at(j).at(0);
            groundTruthPrediction.boundingBox.x = labelAndBoundingBoxes.at(i).at(j).at(1);
            groundTruthPrediction.boundingBox.y = labelAndBoundingBoxes.at(i).at(j).at(2);
            groundTruthPrediction.boundingBox.height = labelAndBoundingBoxes.at(i).at(j).at(3);
            groundTruthPrediction.boundingBox.height -= groundTruthPrediction.boundingBox.x;
            groundTruthPrediction.boundingBox.width = labelAndBoundingBoxes.at(i).at(j).at(4);
            groundTruthPrediction.boundingBox.width -= groundTruthPrediction.boundingBox.y;
            groundTruthPredictions.push_back(groundTruthPrediction);

            predictionsFile >> tmp1; // Ignore - label;
            for (size_t k = 0; k < 4; k++) {
                predictionsFile >> tmp1; // Ignore - rectangle
            }
        }

        // Read prediction data
        cv::Mat testImage = testImagesLabelVector.at(i).second;
        std::vector<ModelPrediction> predictionsVector; // Output of Hog Detection on ith test image
        int numOfPredictions;
        predictionsFile >> numOfPredictions;
        predictionsVector.reserve(numOfPredictions);
        for (size_t i = 0; i < numOfPredictions; i++) {
            ModelPrediction prediction;
            predictionsFile >> prediction.label;
            predictionsFile >> prediction.boundingBox.x >> prediction.boundingBox.y >> prediction.boundingBox.height
                            >> prediction.boundingBox.width;
            predictionsFile >> prediction.confidence;
            predictionsVector.push_back(prediction);
        }

        // Display all the bounding boxes before NonMaximal Suppression
#ifdef DISPLAY
        cv::Mat testImageClone = testImage.clone(); // For drawing bbox
        for (auto &&prediction : predictionsVector) {
            cv::rectangle(testImageClone, prediction.boundingBox, gtColors[prediction.label]);
        }
        cv::imshow("TestImageOutput", testImageClone);
        cv::waitKey(100);
#endif

        // Apply NonMaximal Suppression
        cv::Mat testImageNms1Clone = testImage.clone(); // For drawing bbox
        cv::Mat testImageNmsClone = testImage.clone();  // For drawing bbox
        std::vector<ModelPrediction> predictionsNMSVector;
        predictionsNMSVector.reserve(20); // 20 should be enough.

        // Ignore boxes with low threshold.
        std::vector<ModelPrediction>::iterator iter;
        for (iter = predictionsVector.begin(); iter != predictionsVector.end();) {
            if (iter->confidence < NMS_CONFIDENCE_THRESHOLD)
                iter = predictionsVector.erase(iter);
            else
                ++iter;
        }

        // std::sort(predictionsVector.begin(), predictionsVector.end(), greater_than_key());

        for (auto &&prediction : predictionsVector) {
            cv::rectangle(testImageNms1Clone, prediction.boundingBox, gtColors[prediction.label]);
            // Check if NMS already has a cluster which shares NMS_IOU_THRESHOLD area with current prediction.bbox and both have same label.
            bool clusterFound = false;
            for (auto &&nmsCluster : predictionsNMSVector) {
                if (nmsCluster.label == prediction.label) { // Only if same label
                    cv::Rect &rect1 = prediction.boundingBox;
                    cv::Rect &rect2 = nmsCluster.boundingBox;
                    float iouScore = ((rect1 & rect2).area() * 1.0f) / ((rect1 | rect2).area());
                    if (iouScore > NMS_MAX_IOU_THRESHOLD) // Merge the two bounding boxes
                    {
                        nmsCluster.boundingBox = rect1 | rect2;
                        nmsCluster.confidence = std::max(prediction.confidence, nmsCluster.confidence);
                        clusterFound = true;
                        break;
                    }
                    // else if (iouScore > NMS_MIN_IOU_THRESHOLD) // ToDo: Improve this.
                    // {
                    //     // Drop the bounding box with lower confidence
                    //     if (nmsCluster.confidence < prediction.confidence)
                    //     {
                    //         nmsCluster = prediction;
                    //     }
                    //     clusterFound = true;
                    //     break;
                    // }
                }
            }

            // If no NMS cluster found, add the prediction as a new cluster
            if (!clusterFound)
                predictionsNMSVector.push_back(prediction);
        }

        // Prediction file format: Next is N Lines of Labels and cv::Rect
        for (auto &&prediction : predictionsNMSVector)
            cv::rectangle(testImageNmsClone, prediction.boundingBox, gtColors[prediction.label]);

#ifdef DISPLAY
        // Display all the bounding boxes before NonMaximal Suppression
        cv::imshow("TestImage NMS BBox Filter", testImageNms1Clone);
        cv::imshow("TestImage NMS Output", testImageNmsClone);
        cv::waitKey(500);
#endif

        // cout << "Boxes count: " << predictionsVector.size();
        // cout << "\nNMS boxes count: " << predictionsNMSVector.size() << '\n';

#ifdef DISPLAY
        // Display all ground truth boxes
        cv::Mat testImageGtClone = testImage.clone(); // For drawing bbox
        for (size_t j = 0; j < groundTruthPredictions.size(); j++)
            cv::rectangle(testImageGtClone, groundTruthPredictions.at(j).boundingBox,
                          gtColors[groundTruthPredictions.at(j).label]);
        cv::imshow("Ground Truth", testImageGtClone);
        cv::waitKey(500);
#endif

        // Write NMS output image
        std::stringstream nmsOutputFilePath;
        nmsOutputFilePath << outputDir << std::setfill('0') << std::setw(4) << i << "-NMSOutput" << "-Confidence-"
                          << NMS_CONFIDENCE_THRESHOLD << ".png";
        std::string nmsOutputFilePathStr = nmsOutputFilePath.str();
        cv::imwrite(nmsOutputFilePathStr, testImageNmsClone);

        // Compute precision and recall using groundTruthPredictions and predictionsNMSVector
#ifdef DISPLAY
        cv::waitKey(0);
#endif
        std::vector<float> truePositiveFalsePositive = computeTruePositiveFalsePositive(predictionsNMSVector,
                groundTruthPredictions, 0.5f);
        std::vector<float> falseNegativeTrueNegative = computeTrueNegativeFalseNegative(predictionsNMSVector,
                groundTruthPredictions, 0.5f);
#ifdef DISPLAY
        cv::waitKey(0);
#endif
        truePositive += truePositiveFalsePositive[0];
        falsePositive += truePositiveFalsePositive[1];
        falseNegative += falseNegativeTrueNegative[0];
    }

    float precision = truePositive / (truePositive + falsePositive);
    float recall = truePositive / (truePositive + falseNegative);
    predictionsFile.close();
    std::vector<float> precisionRecallValue;
    precisionRecallValue.push_back(precision);
    precisionRecallValue.push_back(recall);
    return precisionRecallValue;
}

void ObjectDetectionAndClassification::evaluate_metrics(std::string outputDir,
                                                        std::vector<std::pair<int, cv::Mat>> &testImagesLabelVector,
                                                        std::vector<std::vector<std::vector<int>>> &labelAndBoundingBoxes) {
    #ifdef DISPLAY
    cv::namedWindow("TestImageOutput");
    cv::namedWindow("TestImage NMS Output");
    cv::namedWindow("Ground Truth");
    cv::namedWindow("TestImage NMS BBox Filter");
    cv::waitKey(0);
    #endif
    std::ofstream metricLogCSV;
    metricLogCSV.open(outputDir + "metrics.csv");
    if (!metricLogCSV.is_open()) {
        std::cout << "Failed to open" << outputDir + "metrics.csv" << std::endl;
        exit(-1);
    }
    metricLogCSV << "Precision,Recall" << std::endl;
    std::cout << "\nNMS_CONFIDENCE_THRESHOLD " << "      Precision           "
              "Recall " << std::endl;
    for (int confidence = 0;
         confidence <= 100; confidence += 5)
    {
        this->NMS_CONFIDENCE_THRESHOLD = confidence / 100.0f;
        std::vector<float> precisionRecallValue = precisionRecallNMS(outputDir, testImagesLabelVector,
                                                                     labelAndBoundingBoxes, this->bBoxColors,
                                                                     this->NMS_MIN_IOU_THRESHOLD,
                                                                     this->NMS_MAX_IOU_THRESHOLD,
                                                                     this->NMS_CONFIDENCE_THRESHOLD);
        metricLogCSV << precisionRecallValue[0] << "," << precisionRecallValue[1] << std::endl;

        std::cout << "             " << NMS_CONFIDENCE_THRESHOLD << "             "  << precisionRecallValue[0] << "             "  << precisionRecallValue[1] << std::endl;
    }
    metricLogCSV.close();
}

//todo
void ObjectDetectionAndClassification::computeBoundingBoxAndConfidence(cv::Ptr<RandomForest> &randomForest,
                                                                       std::vector<std::pair<int, cv::Mat>> &testImagesLabelVector,
                                                                       std::vector<std::vector<std::vector<int>>> &labelAndBoundingBoxes,
                                                                       int strideX, int strideY, cv::Size winStride,
                                                                       cv::Size padding, cv::Scalar *gtColors,
                                                                       float scaleFactor,
                                                                       std::string outputDir,
                                                                       cv::Size winSize = cv::Size(128, 128)) {
    std::ofstream predictionsFile(outputDir + "predictions.txt");
    if (!predictionsFile.is_open()) {
        std::cout << "Failed to open" << outputDir + "predictions.txt" << std::endl;
        exit(-1);
    }

    for (size_t i = 0; i < testImagesLabelVector.size(); i++) {
        std::cout << "Running prediction on " << (i + 1) << " of " << testImagesLabelVector.size() << " images.\n";
        predictionsFile << i << std::endl; // Prediction file format: Starts with File number
        cv::Mat testImage = testImagesLabelVector.at(i).second;

        // Run testclean up 2 -- todo :experiment with hyperparamsing on various bounding boxes of different scales
        // int minBoundingBoxSideLength = 70, maxBoundingBoxSideLength = 230;
        int minBoundingBoxSideLength = 1000, maxBoundingBoxSideLength = -1;
        std::vector<std::vector<int>> imageLabelsAndBoundingBoxes = labelAndBoundingBoxes.at(i);
        predictionsFile << imageLabelsAndBoundingBoxes.size()
                        << std::endl; // Prediction file format: Next is Number of Ground Truth Boxes - Say K
        for (size_t j = 0; j < imageLabelsAndBoundingBoxes.size(); j++) {
            std::vector<int> bbox = imageLabelsAndBoundingBoxes.at(j); // fixme - rename
            cv::Rect rect(bbox[1], bbox[2], bbox[3] - bbox[1], bbox[4] - bbox[2]);
            // Prediction file format: Next is K Lines of Labels and cv::Rect
            predictionsFile << imageLabelsAndBoundingBoxes.at(j).at(0) << " " << rect.x <<
                            " " << rect.y << " " << rect.height << " " << rect.width << std::endl;
            minBoundingBoxSideLength = std::min(minBoundingBoxSideLength, std::min(rect.width, rect.height));
            maxBoundingBoxSideLength = std::max(maxBoundingBoxSideLength, std::max(rect.width, rect.height));
        }
        minBoundingBoxSideLength -= 10;
        maxBoundingBoxSideLength += 10;

        int boundingBoxSideLength = minBoundingBoxSideLength;
        std::vector<ModelPrediction> predictionsVector; // Output of Hog Detection
        while (true) {
            std::cout << "Processing at bounding box side length: " << boundingBoxSideLength << '\n';
            // Sliding window with stride
            for (size_t row = 0; row < testImage.rows - boundingBoxSideLength; row += strideY) {
                for (size_t col = 0; col < testImage.cols - boundingBoxSideLength; col += strideX) {
                    cv::Rect rect(col, row, boundingBoxSideLength, boundingBoxSideLength);
                    cv::Mat rectImage = testImage(rect);

                    // Predict on subimage
                    ModelPrediction prediction = randomForest->predictPerImage(rectImage, winStride, padding, winSize);
                    if (prediction.label != 3) // Ignore Background class.
                    {
                        prediction.boundingBox = rect;
                        predictionsVector.push_back(prediction);
                    }
                }
            }

            if (boundingBoxSideLength == maxBoundingBoxSideLength) // Maximum Bounding Box Size from ground truth
                break;
            boundingBoxSideLength = (boundingBoxSideLength * scaleFactor + 0.5); // fixme acc to suggestion
            if (boundingBoxSideLength > maxBoundingBoxSideLength)
                boundingBoxSideLength = maxBoundingBoxSideLength;
        }

        // Prediction file format: Next is N Lines of Labels, cv::Rect and confidence
        predictionsFile << predictionsVector.size() << std::endl;
        for (auto &&prediction : predictionsVector) {
            // Prediction file format: Next is N Lines of Labels and cv::Rect
            predictionsFile << prediction.label << " " << prediction.boundingBox.x << " " <<
                            prediction.boundingBox.y << " " << prediction.boundingBox.height << " " <<
                            prediction.boundingBox.width << " " << prediction.confidence << std::endl;
        }

        cv::Mat testImageClone = testImage.clone(); // For drawing bbox
        for (auto &&prediction : predictionsVector)
            cv::rectangle(testImageClone, prediction.boundingBox, gtColors[prediction.label]);

        // Draw bounding box on the test image using ground truth
        imageLabelsAndBoundingBoxes = labelAndBoundingBoxes.at(i);
        cv::Mat testImageGtClone = testImage.clone(); // For drawing bbox
        for (size_t j = 0; j < imageLabelsAndBoundingBoxes.size(); j++) {
            std::vector<int> bbox = imageLabelsAndBoundingBoxes.at(j);
            cv::Rect rect(bbox[1], bbox[2], bbox[3] - bbox[1], bbox[4] - bbox[2]);
            cv::rectangle(testImageGtClone, rect, gtColors[bbox[0]]);
        }

        std::stringstream modelOutputFilePath;
        modelOutputFilePath << outputDir << std::setfill('0') << std::setw(4) << i << "-ModelOutput.png";
        std::string modelOutputFilePathStr = modelOutputFilePath.str();
        cv::imwrite(modelOutputFilePathStr, testImageClone);

        std::stringstream gtFilePath;
        gtFilePath << outputDir << std::setfill('0') << std::setw(4) << i << "-GroundTruth.png";
        std::string gtFilePathStr = gtFilePath.str();
        cv::imwrite(gtFilePathStr, testImageGtClone);
    }
    predictionsFile.close();
}

void ObjectDetectionAndClassification::solver(std::vector<std::pair<int, cv::Mat>> trainDataset,
                                              std::vector<std::vector<std::vector<int>>> groundTruth,
                                              int numTrees,
                                              const std::string& savePath,
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

    //load model
    if (loadModelFromDisk){
        randomForest->setTrees(RandomForest::loadModel("../output/models/NMS-treeCount-" + std::to_string(numTrees) + loadedModelTime, numTrees));
    } else {
        randomForest->train(trainDataset, subsetPercentage, winStride, padding, underSampling, augment,
                            winSize, false, true);
    }
    //save the model for reuse
    if (doSaveModel) {
        randomForest->saveModel("../output/models/NMS-treeCount-" + std::to_string(numTrees) + std::to_string(time));
    }

    //todo replace real test data later
    //std::vector<std::pair<int, cv::Mat>> testDataset = loadTestDataset();
    std::vector<std::pair<int, cv::Mat>> testDataset = debugTestDataset();

    cv::utils::fs::createDirectories(savePath);
    computeBoundingBoxAndConfidence(randomForest, testDataset, groundTruth, this->strideX, this->strideY,
                                    winStride, padding, this->bBoxColors,
                                    scaleFactor, savePath, winSize);

    evaluate_metrics(savePath, testDataset, groundTruth);
}

ObjectDetectionAndClassification::ObjectDetectionAndClassification() = default;

ObjectDetectionAndClassification::~ObjectDetectionAndClassification() = default;

ObjectDetectionAndClassification::ObjectDetectionAndClassification(float max, float min, float confidence,
                                                                   const cv::Size& winSize,
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