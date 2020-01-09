
#include <opencv2/opencv.hpp>

#include "../../include/HOGDescriptor.h"




int main(){
    std::cout << "************************ Task 1 *************************************" << std::endl;
    cv::Mat im = cv::imread("../data/task1/obj1000.jpg");
    std::vector<float> descriptors;

    HOGDescriptor hogDescriptor;
    hogDescriptor.detectHOGDescriptor(im, descriptors, cv::Size(128, 128), true);
    std::cout << "\n******************* Task 1 Finished! *************************************" << std::endl;
    return 0;
}