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
    nonMaximalSuppression.solver(50.0f, false, false);
    // std::vector<std::pair<int, cv::Mat>> labelImagesTrain = nonMaximalSuppression.loadTrainDataset();
    // std::vector<std::pair<int, cv::Mat>> labelImagesTest = nonMaximalSuppression.loadTestDataset();
    // std::vector<std::vector<std::vector<int>>> groundTruthBoundingBoxes = nonMaximalSuppression.getLabelAndBoundingBoxes();
    cout<<"Done"<<endl;
}