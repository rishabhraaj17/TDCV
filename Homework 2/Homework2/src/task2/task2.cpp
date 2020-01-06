//
// Created by rishabh on 29.12.19.
//

#include <opencv2/opencv.hpp>
#include <iostream>


#include "HOGDescriptor.h"
#include "RandomForest.h"

using namespace std;

void
singleDecisionTreeClassifier(int num_classes = 6,
                             int max_depth = 6,
                             const cv::Size &winSize = cv::Size(128, 128)) {
    auto *randomForest = new RandomForest(1, max_depth, 0, 2, num_classes);

    //Load Train Dataset and Train the tree
    vector<pair<int, cv::Mat>> trainDataset = randomForest->loadTrainDataset();
    cout << "Training!" << endl;
    randomForest->trainSingleTree(randomForest, trainDataset);

    // Load from Model file
    //randomForest->setTrees(RandomForest::loadModel("../output/models/test", 1));

    //Load Test Dataset
    vector<pair<int, cv::Mat>> testDataset = randomForest->loadTestDataset();
    cout << "\nEvaluating!" << endl;

    //Prepare Test Dataset
    float accuracy = 0;
    //cv::Size winSize(128, 128);
    cv::HOGDescriptor hogDescriptor = randomForest->createHogDescriptor(winSize);
    cv::Size winStride(8, 8);
    cv::Size padding(0, 0); // padding decreases accuracy

    // int j = 0;
    for (auto &i : testDataset) {
        cv::Mat inputImage = i.second;
        cv::Mat resizedInputImage = randomForest->resizeToBoundingBox(inputImage, winSize);

        // Compute Hog Descriptors
        vector<float> descriptors;
        vector<cv::Point> foundLocations;
        hogDescriptor.compute(resizedInputImage, descriptors, winStride, padding, foundLocations);

        // cout << j++ << ": Expected: " << i.first << ", Found: " << randomForest->getTrees()[0]->predict(cv::Mat(descriptors)) << endl ;
        if (i.first == randomForest->getTrees()[0]->predict(cv::Mat(descriptors)))
            accuracy += 1;
    }
    cout << "Single Decision Tree Classification Accuracy : " << (accuracy / testDataset.size()) * 100.0f
         << "%" << endl;

    // Save Model
    //randomForest->saveModel("../output/models/test");
}


void randomForestClassifier(int numberOfClasses = 6,
                            int numberOfDTrees = 70,
                            const cv::Size& winSize = cv::Size(128, 128),
                            float subsetPercentage = 50.0f,
                            bool underSampling = true,
                            bool augment = false) {
    //TODO Experiment with the MaxDepth & TreeCount parameters, to see how it affects the performance

    cv::Ptr<RandomForest> randomForest = RandomForest::createRandomForest(numberOfClasses, numberOfDTrees, winSize);

    vector<pair<int, cv::Mat>> trainDataset = randomForest->loadTrainDataset();
    vector<pair<int, cv::Mat>> testDataset = randomForest->loadTestDataset();

    // Train the model
    cv::Size winStride(8, 8);
    cv::Size padding(0, 0); // padding decreases accuracy

    randomForest->train(trainDataset, subsetPercentage, winStride, padding, underSampling, augment, winSize);

    // Predict on test dataset
    float accuracy = 0;
    float accuracyPerClass[6] = {0};
    for (size_t i = 0; i < testDataset.size(); i++) {
        cv::Mat testImage = testDataset.at(i).second;
        Prediction prediction = randomForest->predict(testImage, winStride, padding, winSize);
        if (testDataset.at(i).first == prediction.label) {
            accuracy += 1;
            accuracyPerClass[prediction.label] += 1;
        }
    }

    cout << "Random Forest Classification Accuracy : " << (accuracy / testDataset.size()) * 100.0f << "%" << endl;

    int numberOfTestImages[6] = {10, 10, 10, 10, 10, 10};
    cout << "Per-Class Classification Accuracy ---> " << endl;
    for (size_t i = 0; i < numberOfClasses; i++) {
        cout << "Class " << i << " : " << (accuracyPerClass[i] / numberOfTestImages[i]) * 100.0f << "%" << endl;
    }
}


int main() {
    cout << "************************ Task 2 *************************************" << endl;
    cout << "*********** Single Decision Tree Classification *************************************" << endl;
    int numClasses = 6;
    int maxDepth = 6; // increase maxDepth -- same acc after the value of 6. 6 max accuracy
    cv::Size winSize(128, 128);
    singleDecisionTreeClassifier(numClasses, maxDepth, winSize);
    cout << "\n*********** Random Forest Classification *************************************" << endl;
    int numberOfDTrees = 2; // TODO: 70 trees gives 90% accuracy
    float subsetPercentage = 50.0f;
    bool underSampling = true; //under sample the dataset or not (Class Imbalance)
    bool augment = false;
    randomForestClassifier(numClasses, numberOfDTrees, winSize, subsetPercentage, underSampling, augment);
    cout << "\n******************* Task 2 Finished! *************************************" << endl;
    return 0;
}
