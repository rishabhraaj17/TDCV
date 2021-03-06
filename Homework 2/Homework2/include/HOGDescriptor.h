//
// Created by rishabh on 28.12.19.
//

#ifndef RF_HOGDESCRIPTOR_H
#define RF_HOGDESCRIPTOR_H


#include <opencv2/opencv.hpp>
#include <vector>

class HOGDescriptor {

public:

    HOGDescriptor() {
        winSize = cv::Size(128, 128);
        blockSize = cv::Size(16, 16);
        blockStride = cv::Size(8, 8);
        cellSize = cv::Size(8, 8);
        nbins = 9;
        derivAperture = 1;
        winSigma = -1;
        histogramNormType = cv::HOGDescriptor::L2Hys;
        L2HysThreshold = 0.2;
        nlevels= cv::HOGDescriptor::DEFAULT_NLEVELS;
        gammaCorrection = true;
        signedGradient = false;

        
        //if descriptor is already initialized or not
        is_init = false;
    };


    void setWinSize(cv::Size size) {
        winSize = size;
    }

    cv::Size getWinSize(){
        return winSize;
    }

    void setBlockSize(cv::Size size) {
        blockSize = size;
    }

    cv::Size getBlockSize(){
        return blockSize;
    }

    void setBlockStride(cv::Size size) {
       blockStride = size;
    }

    cv::Size getBlockStride(){
        return blockStride;
    }

    void setCellSize(cv::Size size) {
      cellSize = size;
    }

    cv::Size getCellSize(){
        return cellSize;
    }

    void setPadSize(cv::Size sz) {
        pad_size = sz;
    }

    cv::Size getPadSize(){
        return pad_size;
    }

    void setBinCount(int count) {
        nbins = count;
    }

    int getBinCount() {
        return nbins;
    }


    void createHogDescriptor();

    void visualizeHOG(cv::Mat img, std::vector<float> &feats, cv::HOGDescriptor &hog_detector, int scale_factor);

    void detectHOGDescriptor(cv::Mat &im, std::vector<float> &feat, cv::Size sz, bool show);

    ~HOGDescriptor() {};


private:
    cv::Size winSize;
    cv::Size blockSize;
    cv::Size pad_size;
    cv::Size blockStride;
    cv::Size cellSize;
    int nbins;

    int derivAperture;
    double winSigma;
    int histogramNormType;
    double L2HysThreshold;
    bool gammaCorrection;
    int nlevels;
    bool signedGradient;

    cv::Mat img_gray;
    cv::Mat img_resized;

    cv::HOGDescriptor hog_detector;
public:
    cv::HOGDescriptor & getHog_detector();

private:
    bool is_init;
};

#endif //RF_HOGDESCRIPTOR_H
