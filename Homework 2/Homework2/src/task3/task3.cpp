//
// Created by rishabh on 02.01.20.
//
#include <iostream>
#include <NonMaximalSuppression.h>

using namespace std;
using namespace cv;

int main(){
    cout << "Object Detection" << endl;
    NonMaximalSuppression nonMaximalSuppression;
    std::vector<std::pair<int, cv::Mat>> labelImagesTrain = nonMaximalSuppression.loadTrainDataset();
    std::vector<std::pair<int, cv::Mat>> labelImagesTest = nonMaximalSuppression.loadTestDataset();
}