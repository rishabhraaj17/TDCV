//
// Created by rishabh on 29.12.19.
//

#include <opencv2/opencv.hpp>
#include <iostream>


#include "HOGDescriptor.h"
#include "RandomForest.h"

RandomForest * trainDecisionTree(RandomForest *randomForest, std::vector<std::pair<int, cv::Mat>> &trainingImagesLabelVector);

using namespace std;

template<class ClassifierType>
void performanceEval(cv::Ptr<ClassifierType> classifier, cv::Ptr<cv::ml::TrainData> data) {

	/* 

		Fill Code 	

	*/

};





void testDTrees() {

    int num_classes = 6;

    /* 
      * Create your data (i.e Use HOG from task 1 to compute the descriptor for your images)
      * Train a single Decision Tree and evaluate the performance 
      * Experiment with the MaxDepth parameter, to see how it affects the performance

    */
    RandomForest *randomForest = new RandomForest(1, 6, 0, 2, num_classes);

    //Load Train Dataset and Train the tree
    vector<pair<int, cv::Mat>> trainingImagesLabelVector = randomForest->loadTrainDataset();
    trainDecisionTree(randomForest, trainingImagesLabelVector);

    //Load Test Dataset
    vector<pair<int, cv::Mat>> testImagesLabelVector = randomForest->loadTestDataset();
    cout << "Test Dataset Loaded!" << endl;
    //Prepare Test Dataset
    float accuracy = 0;
    cv::Size winSize(128, 128);
    cv::HOGDescriptor testHog = randomForest->createHogDescriptor(winSize);
    cv::Size winStride(8, 8);
    cv::Size padding(0, 0);

    for (size_t i = 0; i < testImagesLabelVector.size(); i++)
    {
        cv::Mat inputImage = testImagesLabelVector.at(i).second;
        //cv::imshow("input", inputImage);
        //cv::waitKey(0);
        cv::Mat resizedInputImage = randomForest->resizeToBoundingBox(inputImage, winSize);
        //cv::imshow("resized", resizedInputImage);

        // Compute Hog only of center crop of grayscale image
        vector<float> descriptors;
        vector<cv::Point> foundLocations;
        vector<double> weights;
        testHog.compute(resizedInputImage, descriptors, winStride, padding, foundLocations);

        // Store the features and labels for model training.
        // cout << i << ": Expected: " << testImagesLabelVector.at(i).first << ", Found: " << model->predict(cv::Mat(descriptors)) << endl ;
        if (testImagesLabelVector.at(i).first == randomForest->getTrees()[0]->predict(cv::Mat(descriptors)))
            accuracy += 1;
    }

    cout << "==================================================" << endl;
    cout << "TASK 2 - Single Decision Tree Accuracy is: [" << accuracy / testImagesLabelVector.size() << "]." << endl;
    cout << "==================================================" << endl;

    //performanceEval<cv::ml::DTrees>(tree, train_data);
    //performanceEval<cv::ml::DTrees>(tree, test_data);

}

RandomForest * trainDecisionTree(RandomForest *randomForest, vector<pair<int, cv::Mat>> &trainingImagesLabelVector) {
    //RandomForest *randomForest = new RandomForest(1, 6, 0, 2, 6);
    cout << "Number of cross validation folds are: " << randomForest->getCVFolds() << endl;
    cout << "Max Categories are: " << randomForest->getMaxCategories() << endl;
    cout << "Max depth is: " << randomForest->getMaxDepth() << endl;
    cout << "Minimum Sample Count: " << randomForest->getMinSampleCount() << endl;
    cout << "Total Number of Decision Trees: " << randomForest->getTreeCount() << endl;

    // Compute Hog Features for all the training images
    cv::Size winSize(128, 128);
    cv::HOGDescriptor hog = randomForest->createHogDescriptor(winSize);
    cv::Size winStride(8, 8);
    cv::Size padding(0, 0);
    cout << "Hog Descriptor window size:" << hog.winSize << endl;

    cv::Mat feats, labels;
    for (size_t i = 0; i < trainingImagesLabelVector.size(); i++)
    {
        cv::Mat inputImage = trainingImagesLabelVector.at(i).second;
        cv::Mat resizedInputImage = randomForest->resizeToBoundingBox(inputImage, winSize);

        // Compute Hog only of center crop of grayscale image
        vector<float> descriptors;
        vector<cv::Point> foundLocations;
        vector<double> weights;
        hog.compute(resizedInputImage, descriptors, winStride, padding, foundLocations);

        // Store the features and labels for model training.
        // cout << "=====================================" << endl;
        // cout << "Number of descriptors are: " << descriptors.size() << endl;
        feats.push_back(cv::Mat(descriptors).clone().reshape(1, 1));
        // cout << "New size of training features" << feats.size() << endl;
        labels.push_back(trainingImagesLabelVector.at(i).first);
        // cout << "New size of training labels" << labels.size() << endl;
    }

    cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(feats, cv::ml::ROW_SAMPLE, labels);
    randomForest->getTrees()[0]->train(trainData);
    cout<< "Decision Tree Trained!" << endl;
    return randomForest;  // do not need to return it
}


void testForest(){

    int num_classes = 6;

    /* 
      * 
      * Create your data (i.e Use HOG from task 1 to compute the descriptor for your images)
      * Train a Forest and evaluate the performance 
      * Experiment with the MaxDepth & TreeCount parameters, to see how it affects the performance

    */

    //TODO

    // Create model
    int numberOfClasses = 6;
    int numberOfDTrees = 70;
    cv::Size winSize(128, 128);
    cv::Ptr<RandomForest> randomForest = RandomForest::createRandomForest(numberOfClasses, numberOfDTrees, winSize);

    vector<pair<int, cv::Mat>> trainingImagesLabelVector = randomForest->loadTrainDataset();
    vector<pair<int, cv::Mat>> testImagesLabelVector = randomForest->loadTestDataset();
    cout << "Datasets Loaded!" << endl;

    // Train the model
    cv::Size winStride(8, 8);
    cv::Size padding(0, 0);
    float subsetPercentage = 50.0f;
    bool undersampling = true;
    bool augment = false;
    randomForest->train(trainingImagesLabelVector, subsetPercentage, winStride, padding, undersampling, augment, winSize);

    // Predict on test dataset
    float accuracy = 0;
    float accuracyPerClass[6] = {0};
    for (size_t i = 0; i < testImagesLabelVector.size(); i++)
    {
        cv::Mat testImage = testImagesLabelVector.at(i).second;
        Prediction prediction = randomForest->predict(testImage, winStride, padding, winSize);
        if (testImagesLabelVector.at(i).first == prediction.label)
        {
            accuracy += 1;
            accuracyPerClass[prediction.label] += 1;
        }
    }

    cout << "==================================================" << endl;
    cout << "TASK 2 - Random Forest Accuracy is: [" << accuracy / testImagesLabelVector.size() << "]." << endl;

    int numberOfTrainImages[6] = {49, 67, 42, 53, 67, 110};
    int numberOfTestImages[6] = {10, 10, 10, 10, 10, 10};
    for (size_t i = 0; i < numberOfClasses; i++)
    {
        cout << "Class " << i << " accuracy: [" << accuracyPerClass[i] / numberOfTestImages[i] << "]." << endl;
    }
    cout << "==================================================" << endl;

    //performanceEval<RandomForest>(forest, train_data);
    //performanceEval<RandomForest>(forest, test_data);
}


int main(){
    //cout << "OpenCV version : " << CV_VERSION << endl;
    testDTrees();
    testForest();
    return 0;
}
