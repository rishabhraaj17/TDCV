

#include "HOGDescriptor.h"
#include <iostream>

void HOGDescriptor::initDetector() {
    // Initialize hog detector
    
    //Fill code here

    is_init = true;

}


void HOGDescriptor::visualizeHOG(cv::Mat img, std::vector<float> &feats, cv::HOGDescriptor hog_detector, int scale_factor) {
    // Fill code here (already provided)
}

void HOGDescriptor::detectHOGDescriptor(cv::Mat &im, std::vector<float> &feat, cv::Size sz, bool show) {
    if (!is_init) {
        initDetector();
    }

   // Fill code here

   /* pad your image
    * resize your image
    * use the built in function "compute" to get the HOG descriptors
    */
}

//returns instance of cv::HOGDescriptor
cv::HOGDescriptor & HOGDescriptor::getHog_detector() {
    // Fill code here
}

