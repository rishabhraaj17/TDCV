//
// Created by rishabh on 28.12.19.
//

#include "../../include/HOGDescriptor.h"
#include <iostream>

void HOGDescriptor::initDetector() {
    // Initialize hog detector
    
    //Fill code here

    // if change of params init
    cv::Size size(128, 128);
    cv::Size stride(8, 8);
    cv::Size cell(8, 8);
    int bins(9);
    int aperture(1);
    double sigma(-1);
    int normType(cv::HOGDescriptor::L2Hys);
    double l2HysThreshold(0.2);
    bool gcorrection(true);
    //! Maximum number of detection window increases. Default value is 64
    int n_levels(cv::HOGDescriptor::DEFAULT_NLEVELS);
    bool gradient(false);
    cv::HOGDescriptor hogDescriptor(size, blockSize, stride, cell, bins, aperture, sigma, cv::HOGDescriptor::L2Hys,
                                    l2HysThreshold, gcorrection, n_levels, gradient);

    hog_detector = hogDescriptor;
    is_init = true;
}


void HOGDescriptor::visualizeHOG(cv::Mat img, std::vector<float> &feats, cv::HOGDescriptor hog_detector, int scale_factor) {
    // Fill code here (already provided)

    cv::Mat visual_image;
    resize(img, visual_image, cv::Size(img.cols * scale_factor, img.rows * scale_factor));

    int n_bins = hog_detector.nbins;
    float rad_per_bin = 3.14 / (float) n_bins;
    cv::Size win_size = hog_detector.winSize;
    cv::Size cell_size = hog_detector.cellSize;
    cv::Size block_size = hog_detector.blockSize;
    cv::Size block_stride = hog_detector.blockStride;

    // prepare data structure: 9 orientation / gradient strenghts for each cell
    int cells_in_x_dir = win_size.width / cell_size.width;
    int cells_in_y_dir = win_size.height / cell_size.height;
    int n_cells = cells_in_x_dir * cells_in_y_dir;
    int cells_per_block = (block_size.width / cell_size.width) * (block_size.height / cell_size.height);

    int blocks_in_x_dir = (win_size.width - block_size.width) / block_stride.width + 1;
    int blocks_in_y_dir = (win_size.height - block_size.height) / block_stride.height + 1;
    int n_blocks = blocks_in_x_dir * blocks_in_y_dir;

    float ***gradientStrengths = new float **[cells_in_y_dir];
    int **cellUpdateCounter = new int *[cells_in_y_dir];
    for (int y = 0; y < cells_in_y_dir; y++) {
        gradientStrengths[y] = new float *[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x = 0; x < cells_in_x_dir; x++) {
            gradientStrengths[y][x] = new float[n_bins];
            cellUpdateCounter[y][x] = 0;

            for (int bin = 0; bin < n_bins; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }


    // compute gradient strengths per cell
    int descriptorDataIdx = 0;


    for (int block_x = 0; block_x < blocks_in_x_dir; block_x++) {
        for (int block_y = 0; block_y < blocks_in_y_dir; block_y++) {
            int cell_start_x = block_x * block_stride.width / cell_size.width;
            int cell_start_y = block_y * block_stride.height / cell_size.height;

            for (int cell_id_x = cell_start_x;
                 cell_id_x < cell_start_x + block_size.width / cell_size.width; cell_id_x++)
                for (int cell_id_y = cell_start_y;
                     cell_id_y < cell_start_y + block_size.height / cell_size.height; cell_id_y++) {

                    for (int bin = 0; bin < n_bins; bin++) {
                        float val = feats.at(descriptorDataIdx++);
                        gradientStrengths[cell_id_y][cell_id_x][bin] += val;
                    }
                    cellUpdateCounter[cell_id_y][cell_id_x]++;
                }
        }
    }


    // compute average gradient strengths
    for (int celly = 0; celly < cells_in_y_dir; celly++) {
        for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {

            float NrUpdatesForThisCell = (float) cellUpdateCounter[celly][cellx];

            // compute average gradient strenghts for each gradient bin direction
            for (int bin = 0; bin < n_bins; bin++) {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }


    for (int celly = 0; celly < cells_in_y_dir; celly++) {
        for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {
            int drawX = cellx * cell_size.width;
            int drawY = celly * cell_size.height;

            int mx = drawX + cell_size.width / 2;
            int my = drawY + cell_size.height / 2;

            rectangle(visual_image,
                      cv::Point(drawX * scale_factor, drawY * scale_factor),
                      cv::Point((drawX + cell_size.width) * scale_factor,
                                (drawY + cell_size.height) * scale_factor),
                      CV_RGB(100, 100, 100),
                      1);

            for (int bin = 0; bin < n_bins; bin++) {
                float currentGradStrength = gradientStrengths[celly][cellx][bin];

                if (currentGradStrength == 0)
                    continue;

                float currRad = bin * rad_per_bin + rad_per_bin / 2;

                float dirVecX = cos(currRad);
                float dirVecY = sin(currRad);
                float maxVecLen = cell_size.width / 2;
                float scale = scale_factor / 5.0; // just a visual_imagealization scale,

                // compute line coordinates
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

                // draw gradient visual_imagealization
                line(visual_image,
                     cv::Point(x1 * scale_factor, y1 * scale_factor),
                     cv::Point(x2 * scale_factor, y2 * scale_factor),
                     CV_RGB(0, 0, 255),
                     1);

            }

        }
    }


    for (int y = 0; y < cells_in_y_dir; y++) {
        for (int x = 0; x < cells_in_x_dir; x++) {
            delete[] gradientStrengths[y][x];
        }
        delete[] gradientStrengths[y];
        delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;
    cv::imshow("HOG vis", visual_image);
    cv::waitKey(-1);
    cv::imwrite("hog_vis.jpg", visual_image);
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
   // vector of found locations returned by compute
   std::vector<cv::Point> foundLocations;
   // resize the image
   // Way 1 : looks like pdf
   //cv::resize(im, img_resized, sz, 0, 0, cv::INTER_AREA);
   // Way 2 : Looks diiferent than pdf sample. TODO: decide
   cv::Size w_size(128, 128);
   img_resized = resizeToBoundingBox(im, w_size);
   // convert to grayscale
   cv::cvtColor(img_resized, img_gray, cv::COLOR_BGR2GRAY);
   // compute the descriptors
   hog_detector.compute(img_resized, feat, cv::Size(8, 8), cv::Size(0, 0), foundLocations);
   if (show) {
       visualizeHOG(img_resized, feat, hog_detector, 6);
   }
}

//returns instance of cv::HOGDescriptor
cv::HOGDescriptor & HOGDescriptor::getHog_detector() {
    // Fill code here
    return hog_detector;
}

cv::Mat HOGDescriptor::resizeToBoundingBox(cv::Mat &inputImage, cv::Size &winSize)
{
    cv::Mat resizedInputImage;
    if (inputImage.rows < winSize.height || inputImage.cols < winSize.width)
    {
        float scaleFactor = fmax((winSize.height * 1.0f) / inputImage.rows, (winSize.width * 1.0f) / inputImage.cols);
        cv::resize(inputImage, resizedInputImage, cv::Size(0, 0), scaleFactor, scaleFactor, cv::INTER_LINEAR);
    }
    else
    {
        resizedInputImage = inputImage;
    }

    cv::Rect r = cv::Rect((resizedInputImage.cols - winSize.width) / 2, (resizedInputImage.rows - winSize.height) / 2,
                  winSize.width, winSize.height);
    return resizedInputImage(r);
}