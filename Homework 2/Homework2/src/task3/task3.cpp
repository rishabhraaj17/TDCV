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
    int numTrees = 10; // HyperParam
    cv::Size winSize(128, 128);

    float scaleFactor = 1.10f; // HyperParam
    int strideX = 2; // HyperParam
    int strideY = 2; // HyperParam
    float NMS_MAX_IOU_THRESHOLD = 0.5f; // If above this threshold, merge the two bounding boxes. // HyperParam
    float NMS_MIN_IOU_THRESHOLD = 0.1f; // If above this threshold, drop the bounding boxes with lower confidence. // HyperParam
    float NMS_CONFIDENCE_THRESHOLD = 0.6f; // HyperParam

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

    std::vector<std::pair<int, cv::Mat>> trainDataset = detector.loadTrainDataset();
    std::vector<std::vector<std::vector<int>>> groundTruth = detector.getGroundTruth();
    detector.solver(trainDataset, groundTruth, numTrees, savePath,
            50.0f, false, true, true, false);

    cout << "\n******************* Task 3 Finished! *************************************" << endl;
}