
#include <opencv2/opencv.hpp>

#include "../../include/HOGDescriptor.h"




int main(){
    cv::Mat im = cv::imread("../data/task1/obj1000.jpg");
    /*cv::imshow("image", im);
    cv::waitKey(0);*/
	//Fill Code here

    /*
    	* Create instance of HOGDescriptor and initialize
    	* Compute HOG descriptors
    	* visualize
    */

    std::vector<float> descriptors;

    HOGDescriptor hogDescriptor;
    hogDescriptor.detectHOGDescriptor(im, descriptors, cv::Size(128, 128), true);

    return 0;
}