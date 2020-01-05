#ifndef HOG_VISUALIZATION_HPP
#define HOG_VISUALIZATION_HPP

#include <opencv2/opencv.hpp>

void visualizeHOG(cv::Mat& img, std::vector<float> &feats, cv::HOGDescriptor& hog_detector, int scale_factor = 3);


#endif

