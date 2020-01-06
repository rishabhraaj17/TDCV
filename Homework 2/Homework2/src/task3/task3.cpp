//
// Created by rishabh on 02.01.20.
//
#include <iostream>
#include <NonMaximalSuppression.h>

using namespace std;
using namespace cv;



int main(){
    cout << "OpenCV version : " << CV_VERSION << endl;
    cout << "Object Detection" << endl;
    NonMaximalSuppression nonMaximalSuppression;
    //nonMaximalSuppression.solver(50.0f, false, false); // Done for tree 80 and tree 59
    std::vector<std::pair<int, cv::Mat>> testImagesLabelVector = nonMaximalSuppression.loadTestDataset();
    std::vector<std::vector<std::vector<int>>> labelAndBoundingBoxes = nonMaximalSuppression.getLabelAndBoundingBoxes();
    std::string outputDir = "../output/Trees-60_subsetPercent-50-scaleFactor_1.1-undersampling_0-augment_0-strideX_2-strideY_2-NMS_MIN_0.1-NMS_Max_0.5-NMS_CONF_0.6/";
    nonMaximalSuppression.evaluate_metrics(outputDir, testImagesLabelVector, labelAndBoundingBoxes);
    // std::vector<std::pair<int, cv::Mat>> labelImagesTrain = nonMaximalSuppression.loadTrainDataset();
    // std::vector<std::pair<int, cv::Mat>> labelImagesTest = nonMaximalSuppression.loadTestDataset();
    // std::vector<std::vector<std::vector<int>>> groundTruthBoundingBoxes = nonMaximalSuppression.getLabelAndBoundingBoxes();
    cout<<"Done"<<endl;
}