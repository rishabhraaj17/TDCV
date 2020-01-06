//
// Created by rishabh on 02.01.20.
//

#include <sys/ioctl.h>
#include "ObjectDetectionAndClassification.h"
#include <boost/filesystem.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#define DISPLAY

std::vector<std::pair<int, cv::Mat>> ObjectDetectionAndClassification::loadTrainDataset() {
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

std::vector<std::pair<int, cv::Mat>> ObjectDetectionAndClassification::loadTestDataset() {
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

std::vector<std::vector<std::vector<int>>> ObjectDetectionAndClassification::getLabelAndBoundingBoxes() {
    int numberOfTestImages = 44;
    std::vector<std::vector<std::vector<int>>> groundTruthBoundingBoxes;
    for (size_t j = 0; j < numberOfTestImages; j++)
    {
        std::stringstream gtFilePath;
        gtFilePath << std::string(PROJ_DIR) << "/data/task3/gt/" << std::setfill('0') << std::setw(4) << j << ".gt.txt";
        std::string gtFilePathStr = gtFilePath.str();

        std::fstream gtFile;
        gtFile.open(gtFilePathStr);
        if (!gtFile.is_open())
        {
            std::cout << "Failed to open file: " << gtFilePathStr << std::endl;
            exit(-1);
        }

        std::string line;
        std::vector<std::vector<int>> groundTruthBoundingBoxesPerImage;
        while (std::getline(gtFile, line))
        {
            std::istringstream in(line);
            std::vector<int> groundTruthLabelAndBoundingBox(5);
            int temp;
            for (size_t i = 0; i < 5; i++)
            {
                in >> temp;
                groundTruthLabelAndBoundingBox.at(i) = temp;
            }
            groundTruthBoundingBoxesPerImage.push_back(groundTruthLabelAndBoundingBox);
        }
        groundTruthBoundingBoxes.push_back(groundTruthBoundingBoxesPerImage);
    }
    return groundTruthBoundingBoxes;
}

std::vector<float> ObjectDetectionAndClassification::computeTpFpFn(std::vector<ModelPrediction> predictionsNMSVector,
                                                                   std::vector<ModelPrediction> groundTruthPredictions) {
    float tp = 0, fp = 0, fn = 0;
    float matchThresholdIou = 0.5f;

    for (auto &&myPrediction : predictionsNMSVector)
    {
        bool matchesWithAnyGroundTruth = false;
        cv::Rect myRect = myPrediction.boundingBox;

        for (auto &&groundTruth : groundTruthPredictions)
        {
            if (groundTruth.label != myPrediction.label)
                continue;
            cv::Rect gtRect = groundTruth.boundingBox;
            float iouScore = ((myRect & gtRect).area() * 1.0f) / ((myRect | gtRect).area());
            if (iouScore > matchThresholdIou)
            {
                matchesWithAnyGroundTruth = true;
                break;
            }
        }

        if (matchesWithAnyGroundTruth)
            tp++;
        else
            fp++;
    }

    for (auto &&groundTruth : groundTruthPredictions)
    {
        bool isGtBboxMissed = true;
        cv::Rect gtRect = groundTruth.boundingBox;
        for (auto &&myPrediction : predictionsNMSVector)
        {
            if (groundTruth.label != myPrediction.label)
                continue;
            cv::Rect myRect = myPrediction.boundingBox;
            float iouScore = ((myRect & gtRect).area() * 1.0f) / ((myRect | gtRect).area());
            if (iouScore > matchThresholdIou)
            {
                isGtBboxMissed = false;
                break;
            }
        }

        if (isGtBboxMissed)
            fn++;
    }

    std::vector<float> results;
    results.push_back(tp);
    results.push_back(fp);
    results.push_back(fn);
    return results;
}

std::vector<float>
ObjectDetectionAndClassification::precisionRecallNMS(std::string outputDir, std::vector<std::pair<int, cv::Mat>> &testImagesLabelVector,
                                                     std::vector<std::vector<std::vector<int>>> &labelAndBoundingBoxes, cv::Scalar *gtColors,
                                                     float NMS_MIN_IOU_THRESHOLD, float NMS_MAX_IOU_THRESHOLD,
                                                     float NMS_CONFIDENCE_THRESHOLD) {
    std::ifstream predictionsFile(outputDir + "predictions.txt");
    if (!predictionsFile.is_open())
    {
        std::cout << "Failed to open" << outputDir + "predictions.txt" << std::endl;
        exit(-1);
    }


    float tp = 0, fp = 0, fn = 0;
    for (size_t i = 0; i < testImagesLabelVector.size(); i++)
    {
        int fileNumber;
        predictionsFile >> fileNumber; // Prediction file format: Starts with File number
        assert(fileNumber == i);

        // Ignore the ground truth data in predictions.txt. we already have it.
        int tmp, tmp1;
        predictionsFile >> tmp; // Ignore - number of ground truth
        std::vector<ModelPrediction> groundTruthPredictions;
        for (size_t j = 0; j < tmp; j++)
        {
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
            for (size_t k = 0; k < 4; k++)
            {
                predictionsFile >> tmp1; // Ignore - rectangle
            }
        }

        // Read prediction data
        cv::Mat testImage = testImagesLabelVector.at(i).second;
        std::vector<ModelPrediction> predictionsVector; // Output of Hog Detection on ith test image
        int numOfPredictions;
        predictionsFile >> numOfPredictions;
        predictionsVector.reserve(numOfPredictions);
        for (size_t i = 0; i < numOfPredictions; i++)
        {
            ModelPrediction prediction;
            predictionsFile >> prediction.label;
            predictionsFile >> prediction.boundingBox.x >> prediction.boundingBox.y >> prediction.boundingBox.height >> prediction.boundingBox.width;
            predictionsFile >> prediction.confidence;
            predictionsVector.push_back(prediction);
        }

        // Display all the bounding boxes before NonMaximal Suppression
#ifdef DISPLAY
        cv::Mat testImageClone = testImage.clone(); // For drawing bbox
        for (auto &&prediction : predictionsVector)
        {
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
        for (iter = predictionsVector.begin(); iter != predictionsVector.end(); ) {
            if (iter->confidence < NMS_CONFIDENCE_THRESHOLD)
                iter = predictionsVector.erase(iter);
            else
                ++iter;
        }

        // std::sort(predictionsVector.begin(), predictionsVector.end(), greater_than_key());

        for (auto &&prediction : predictionsVector)
        {
            cv::rectangle(testImageNms1Clone, prediction.boundingBox, gtColors[prediction.label]);
            // Check if NMS already has a cluster which shares NMS_IOU_THRESHOLD area with current prediction.bbox and both have same label.
            bool clusterFound = false;
            for (auto &&nmsCluster : predictionsNMSVector)
            {
                if (nmsCluster.label == prediction.label)
                { // Only if same label
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
            cv::rectangle(testImageGtClone, groundTruthPredictions.at(j).boundingBox, gtColors[groundTruthPredictions.at(j).label]);
        cv::imshow("Ground Truth", testImageGtClone);
        cv::waitKey(500);
#endif

        // Write NMS output image
        std::stringstream nmsOutputFilePath;
        nmsOutputFilePath << outputDir << std::setfill('0') << std::setw(4) << i << "-NMSOutput" << "-Confidence-" << NMS_CONFIDENCE_THRESHOLD << ".png";
        std::string nmsOutputFilePathStr = nmsOutputFilePath.str();
        cv::imwrite(nmsOutputFilePathStr, testImageNmsClone);

        // Compute precision and recall using groundTruthPredictions and predictionsNMSVector
#ifdef DISPLAY
        cv::waitKey(0);
#endif
        std::vector<float> tpFpFn = computeTpFpFn(predictionsNMSVector, groundTruthPredictions);
#ifdef DISPLAY
        cv::waitKey(0);
#endif
        tp += tpFpFn[0];
        fp += tpFpFn[1];
        fn += tpFpFn[2];
    }

    float precision = tp / (tp + fp);
    float recall = tp / (tp + fn);
    predictionsFile.close();
    std::vector<float> precisionRecallValue;
    precisionRecallValue.push_back(precision);
    precisionRecallValue.push_back(recall);
    return precisionRecallValue;
    return std::vector<float>();
}

ObjectDetectionAndClassification::ObjectDetectionAndClassification(float max, float min, float confidence) {
    this->NMS_MAX_IOU_THRESHOLD = max;
    this->NMS_MIN_IOU_THRESHOLD = min;
    this->NMS_CONFIDENCE_THRESHOLD = confidence;
}

void ObjectDetectionAndClassification::evaluate_metrics(std::string outputDir,
                                                        std::vector<std::pair<int, cv::Mat>> &testImagesLabelVector,
                                                        std::vector<std::vector<std::vector<int>>> &labelAndBoundingBoxes) {
    cv::Scalar gtColors[4];
    gtColors[0] = cv::Scalar(255, 0, 0);
    gtColors[1] = cv::Scalar(0, 255, 0);
    gtColors[2] = cv::Scalar(0, 0, 255);
    gtColors[3] = cv::Scalar(255, 255, 0);

    float NMS_MAX_IOU_THRESHOLD = 0.5f; // If above this threshold, merge the two bounding boxes.
    float NMS_MIN_IOU_THRESHOLD = 0.1f; // If above this threshold, drop the bounding boxes with lower confidence.
    // float NMS_CONFIDENCE_THRESHOLD = 0.75f;

#ifdef DISPLAY
    cv::namedWindow("TestImageOutput");
    cv::namedWindow("TestImage NMS Output");
    cv::namedWindow("Ground Truth");
    cv::namedWindow("TestImage NMS BBox Filter");
    cv::waitKey(0);
#endif

    std::cout << "\n";
    std::ofstream outputFile;
    outputFile.open(outputDir+"predictionRecallValues.csv");
    if (!outputFile.is_open())
    {
        std::cout << "Failed to open" << outputDir+"predictionRecallValues.csv" << std::endl;
        exit(-1);
    }
    outputFile << "Precision,Recall"<< std::endl;
    for (int confidence = 0; confidence <= 100; confidence += 5) // If float is used, it may overshoot 1.0 - floating point error
    {
        float NMS_CONFIDENCE_THRESHOLD = confidence / 100.0f;
        std::vector<float> precisionRecallValue = precisionRecallNMS(outputDir, testImagesLabelVector,
                                                                     labelAndBoundingBoxes, gtColors,
                                                                     NMS_MIN_IOU_THRESHOLD, NMS_MAX_IOU_THRESHOLD,
                                                                     NMS_CONFIDENCE_THRESHOLD);
        std::cout << "NMS_CONFIDENCE_THRESHOLD: " << NMS_CONFIDENCE_THRESHOLD << ", Precision: " << precisionRecallValue[0] << ", Recall: " << precisionRecallValue[1] << std::endl;
        outputFile << precisionRecallValue[0] << "," << precisionRecallValue[1] << std::endl;
    }
    outputFile.close();

    std::cout << "\n";
}

void ObjectDetectionAndClassification::computeBoundingBoxAndConfidence(cv::Ptr<RandomForest> &randomForest,
                                                                       std::vector<std::pair<int, cv::Mat>> &testImagesLabelVector,
                                                                       std::vector<std::vector<std::vector<int>>> &labelAndBoundingBoxes,
                                                                       int strideX, int strideY, cv::Size winStride,
                                                                       cv::Size padding, cv::Scalar *gtColors, float scaleFactor,
                                                                       std::string outputDir,
                                                                       cv::Size winSize = cv::Size(128, 128)) {
    std::ofstream predictionsFile(outputDir + "predictions.txt");
    if (!predictionsFile.is_open())
    {
        std::cout << "Failed to open" << outputDir + "predictions.txt" << std::endl;
        exit(-1);
    }

    for (size_t i = 0; i < testImagesLabelVector.size(); i++)
    {
        std::cout << "Running prediction on " << (i+1) << " of " << testImagesLabelVector.size() << " images.\n";
        predictionsFile << i << std::endl; // Prediction file format: Starts with File number
        cv::Mat testImage = testImagesLabelVector.at(i).second;

        // Run testclean up 2 -- todo :experiment with hyperparamsing on various bounding boxes of different scales
        // int minBoundingBoxSideLength = 70, maxBoundingBoxSideLength = 230;
        int minBoundingBoxSideLength = 1000, maxBoundingBoxSideLength = -1;
        std::vector<std::vector<int>> imageLabelsAndBoundingBoxes = labelAndBoundingBoxes.at(i);
        predictionsFile << imageLabelsAndBoundingBoxes.size() << std::endl; // Prediction file format: Next is Number of Ground Truth Boxes - Say K
        for (size_t j = 0; j < imageLabelsAndBoundingBoxes.size(); j++)
        {
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
        while (true)
        {
            std::cout << "Processing at bounding box side length: " << boundingBoxSideLength << '\n';
            // Sliding window with stride
            for (size_t row = 0; row < testImage.rows - boundingBoxSideLength; row += strideY)
            {
                for (size_t col = 0; col < testImage.cols - boundingBoxSideLength; col += strideX)
                {
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
        for (auto &&prediction : predictionsVector)
        {
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
        for (size_t j = 0; j < imageLabelsAndBoundingBoxes.size(); j++)
        {
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

void ObjectDetectionAndClassification::solver(float subsetPercentage = 50.0f,
                                              bool underSampling = false,
                                              bool augment = true) {
    std::vector<std::pair<int, cv::Mat>> trainingImagesLabelVector = loadTrainDataset();
    std::vector<std::vector<std::vector<int>>> labelAndBoundingBoxes = getLabelAndBoundingBoxes();

    // Create model
    int numberOfClasses = 4;
    int numberOfDTrees = 62;
    cv::Size winSize(128, 128);
    cv::Ptr<RandomForest> randomForest = RandomForest::createRandomForest(numberOfClasses, numberOfDTrees, winSize);

    // Train the model
    cv::Size winStride(8, 8);
    cv::Size padding(0, 0);
    /*float subsetPercentage = 50.0f;
    bool underSampling = false;
    bool augment = true;*/
    randomForest->train(trainingImagesLabelVector, subsetPercentage, winStride, padding, underSampling, augment, winSize);

    // For each test image
    std::vector<std::pair<int, cv::Mat>> testImagesLabelVector = loadTestDataset();
    cv::Scalar gtColors[4];
    gtColors[0] = cv::Scalar(255, 0, 0);
    gtColors[1] = cv::Scalar(0, 255, 0);
    gtColors[2] = cv::Scalar(0, 0, 255);
    gtColors[3] = cv::Scalar(255, 255, 0);

    float scaleFactor = 1.10f;
    int strideX = 2;
    int strideY = 2;

    // NMS-Not used. Each boundin box is dumped to the text file which contains the confidence value. The thresholding is done in evaluation.cpp
    float NMS_MAX_IOU_THRESHOLD = 0.5f; // If above this threshold, merge the two bounding boxes.
    float NMS_MIN_IOU_THRESHOLD = 0.1f; // If above this threshold, drop the bounding boxes with lower confidence.
    float NMS_CONFIDENCE_THRESHOLD = 0.6f;

    // Loop over multiple values.
    std::ostringstream s;
    s << PROJ_DIR << "/output/Trees-" << numberOfDTrees << "_subsetPercent-" << ((int)subsetPercentage)
      << "-scaleFactor_" << scaleFactor << "-undersampling_" << underSampling << "-augment_" << augment
      << "-strideX_" << strideX << "-strideY_" << strideY << "-NMS_MIN_" << NMS_MIN_IOU_THRESHOLD
      << "-NMS_Max_" << NMS_MAX_IOU_THRESHOLD << "-NMS_CONF_" << NMS_CONFIDENCE_THRESHOLD << "/";
    std::string outputDir = s.str();
    //std::cout<<outputDir<<std::endl;
    bool created_dr = cv::utils::fs::createDirectories(outputDir);
    std::cout<<created_dr<<std::endl;
    computeBoundingBoxAndConfidence(randomForest, testImagesLabelVector, labelAndBoundingBoxes, strideX, strideY, winStride, padding, gtColors,
               scaleFactor, outputDir, winSize);
}

ObjectDetectionAndClassification::ObjectDetectionAndClassification() = default;

ObjectDetectionAndClassification::~ObjectDetectionAndClassification() = default;
