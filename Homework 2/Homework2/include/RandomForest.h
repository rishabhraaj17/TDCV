

#ifndef RF_RANDOMFOREST_H
#define RF_RANDOMFOREST_H


#include <opencv2/opencv.hpp>
#include <vector>
#include <random>
#include "HOGDescriptor.h"

struct Prediction
{
    int label;
    float confidence;
    cv::Rect bbox;
};

struct GreaterThan
{
    inline bool operator() (const Prediction& struct1, const Prediction& struct2)
    {
        return (struct1.confidence < struct2.confidence);
    }
};

class RandomForest
{
public:
	RandomForest();

	/*RandomForest() : mTreeCount(16) {
	    mMaxDepth = 200;
	    mCVFolds = 0;
	    mMinSampleCount = 2;
	    mMaxCategories = 6;
	}*/

    // You can create the forest directly in the constructor or create an empty forest and use the below methods to populate it
	RandomForest(int treeCount, int maxDepth, int CVFolds, int minSampleCount, int maxCategories);
    
    ~RandomForest();

    void setTreeCount(int treeCount);
    void setMaxDepth(int maxDepth);
    void setCVFolds(int cvFols);
    void setMinSampleCount(int minSampleCount);
    void setMaxCategories(int maxCategories);
	

    void train(std::vector<std::pair<int, cv::Mat>> &trainingImagesLabelVector);

    float predict(const cv::Mat1f& features, cv::Mat& predicted_labels, cv::Mat& predicted_confidences);

    std::vector<std::pair<int, cv::Mat>> loadTrainDataset();

    std::vector<std::pair<int, cv::Mat>> loadTestDataset();


private:
	int mTreeCount;
	int mMaxDepth;
	int mCVFolds;
	int mMinSampleCount;
	int mMaxCategories;

    // M-Trees for constructing thr forest
    // decision tress
    std::vector<cv::Ptr<cv::ml::DTrees> > mTrees;
    std::mt19937 m_randomGenerator;

    cv::Size m_winSize;
    HOGDescriptor hogDescriptor;
};

#endif //RF_RANDOMFOREST_H
