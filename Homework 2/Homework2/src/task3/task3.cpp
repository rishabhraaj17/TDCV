//
// Created by rishabh on 02.01.20.
//
#include <iostream>
#include <ObjectDetectionAndClassification.h>

using namespace std;
using namespace cv;



int main(){
    cout << "OpenCV version : " << CV_VERSION << endl;
    cout << "************************ Task 3 *************************************" << endl;
    cout << "Object Detection" << endl;

    int numClasses = 4;
    int numTrees = 56; // HyperParam
    cv::Size winSize(128, 128);

    float scaleFactor = 1.1f; // HyperParam
    int strideX = 4; // HyperParam
    int strideY = 4; // HyperParam
    float NMS_MAX_IOU_THRESHOLD = 0.55f; // If above this threshold, merge the two bounding boxes. // HyperParam
    float NMS_MIN_IOU_THRESHOLD = 0.3f; // If above this threshold, drop the bounding boxes with lower confidence. // HyperParam
    float NMS_CONFIDENCE_THRESHOLD = 0.6f; // HyperParam

    cout << "Training Trees : " << std::to_string(numTrees) << endl;
    ObjectDetectionAndClassification detector(NMS_MAX_IOU_THRESHOLD, NMS_MIN_IOU_THRESHOLD,
            NMS_CONFIDENCE_THRESHOLD, winSize, numClasses, scaleFactor, strideX, strideY);

    float subsetPercentage = 50.0f;
    bool underSampling = false; // HyperParam
    bool augment = true;

    std::ostringstream path;
    path << PROJ_DIR << "/output/TreesCount_" << numTrees << "_subsetPercent_" << ((int) subsetPercentage)
      << "_scaleFactor_" << scaleFactor << "_underSampling_" << underSampling << "_DataAugment_" << augment
      << "_strideX_" << strideX << "_strideY_" << strideY << "_NMS_MIN_" << NMS_MIN_IOU_THRESHOLD
      << "_NMS_Max_" << NMS_MAX_IOU_THRESHOLD << "_NMS_CONF_" << NMS_CONFIDENCE_THRESHOLD << "/";
    std::string savePath = path.str();

    std::string modelPath = "../pretrained/Gleb/saveFolder";
    std::vector<std::pair<int, cv::Mat>> trainDataset = detector.loadTrainDataset();
    std::vector<std::vector<std::vector<int>>> groundTruth = detector.getGroundTruth();
    detector.solver(trainDataset, groundTruth, numTrees, savePath,
            50.0f, false, true, true, true, modelPath);

    /*std::vector<std::pair<int, cv::Mat>> testDataset = detector.loadTestDataset();
    std::string outputDir = "../output/TreesCount_56_subsetPercent_50_scaleFactor_1.1_underSampling_0_DataAugment_1_strideX_4_strideY_4_NMS_MIN_0.3_NMS_Max_0.55_NMS_CONF_0.6/";
    detector.evaluate_metrics(outputDir, testDataset, groundTruth);*/


    cout << "\n******************* Task 3 Finished! *************************************" << endl;
}