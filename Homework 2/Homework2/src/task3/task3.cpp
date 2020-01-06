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
    /*ObjectDetectionAndClassification detector;
    detector.solver(50.0f, false, false); // Done for tree 80 and tree 59
    std::vector<std::pair<int, cv::Mat>> testImagesLabelVector = detector.loadTestDataset();
    std::vector<std::vector<std::vector<int>>> labelAndBoundingBoxes = detector.getLabelAndBoundingBoxes();
    std::string outputDir = "../output/Trees-60_subsetPercent-50-scaleFactor_1.1-undersampling_0-augment_0-strideX_2-strideY_2-NMS_MIN_0.1-NMS_Max_0.5-NMS_CONF_0.6/";
    detector.evaluate_metrics(outputDir, testImagesLabelVector, labelAndBoundingBoxes);*/

    ObjectDetectionAndClassification detector;
    vector<pair<int, cv::Mat>> trainDataset = detector.loadTestDataset();

    // std::vector<std::pair<int, cv::Mat>> labelImagesTrain = detector.loadTrainDataset();
    // std::vector<std::pair<int, cv::Mat>> labelImagesTest = detector.loadTestDataset();
    // std::vector<std::vector<std::vector<int>>> groundTruthBoundingBoxes = detector.getLabelAndBoundingBoxes();
    cout << "\n******************* Task 3 Finished! *************************************" << endl;
    cout<<"Done"<<endl;
}