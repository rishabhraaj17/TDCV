#include <opencv2/opencv.hpp>
#include "hog_visualization.hpp"
#include <vector>
#include <functional>

using namespace cv;

void computeHOG(Mat& img) {
	Size winSize = img.size();
	Size blockSize = Size(16, 16);
	Size blockStride = Size(8, 8);
	Size cellSize = Size(8, 8);
	int nbins = 9;

	HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, nbins);
	std::vector<float> descriptors;
	Size winStride = Size();
	Size padding = Size();

	hog.compute(img, descriptors, winStride, padding);
	visualizeHOG(img, descriptors, hog);
}

int main(int argc, char *argv[]) {

	// read input image
	Mat resimg;
	Mat img = imread("data/task1/obj1000.jpg");
	resize(img, resimg, Size(128, 128));
	imshow("orig", img);
	waitKey(0);

	// transformations
	std::function<void (Mat&, Mat&)> transforms[] = {
		[](Mat &in, Mat &out) {
			resize(in, out, Size(256, 64));
		},
		[](Mat &in, Mat &out) {
			Mat white = Mat::zeros(in.rows, in.cols, in.type());
			white.setTo(Scalar(255,255,255));
			addWeighted(in, 0.5, white, 0.5, 0.0, out);
		},
		[](Mat &in, Mat &out) {
			flip(in, out, -1);
		},
		[](Mat &in, Mat &out) {
			cvtColor(in, out, COLOR_BGR2GRAY);
		},
		[](Mat &in, Mat &out) {
			Sobel(in, out, -1, 1, 0, 0);
		},
		[](Mat &in, Mat &out) {
			Point center = Point(in.cols/2, in.rows/2);
			Mat rot_mat = getRotationMatrix2D(center, 30, 1);
			warpAffine(in, out, rot_mat, in.size());
		},
		[](Mat &in, Mat &out) {
			copyMakeBorder(in, out, 8, 8, 8, 8, BORDER_CONSTANT);
		}
	};

	// apply and compute and visualize
	for(auto& f : transforms) {
		Mat modmat;
		f(resimg, modmat);
		computeHOG(modmat);
	}

	return EXIT_SUCCESS;
}

