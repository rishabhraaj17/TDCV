//
//
//


#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "hog_visualization.hpp"
#include "RandomForest.h"
#include <iostream>
#include "features.h"

typedef std::vector<std::pair<int, std::vector<float>>> DataSet;
namespace fs = std::filesystem;
using namespace cv;

struct evalResult {
    int errors;
    int total;
    float avgConfidence;
};


inline std::string resultString(const std::string &modelName, evalResult result) {
    return modelName +": "+ std::to_string(result.total - result.errors) + "/"
           + std::to_string(result.total) + " correctly classified. Avg. confidence: " +
           std::to_string(result.avgConfidence);
}

evalResult evaluateModel(const RandomForest &model, DataSet data) {
    int errors = 0;
    float avg_confidence = 0;
    for (const auto &sample : data) {
        float confidence;
        float prediction = model.predict(Mat(sample.second), &confidence);
        errors += ((int) prediction != sample.first);
        avg_confidence += confidence;
    }
    return evalResult{errors, static_cast<int>(data.size()), avg_confidence / data.size()};
}

int main(int argc, char *argv[]) {

    int forestSize = 50;
    auto train_path = fs::path("data/task2/train");
    auto test_path = fs::path("data/task2/test");

    std::cout << "Read images and extract HOG Descriptors" <<std::endl;
    DataSet training_set;
    DataSet test_set;
    compute_data_set(training_set, train_path);
    compute_data_set(test_set, test_path);
    auto train_data = make_opencv_dataset(training_set);
    std::cout << "Done!\n" << std::endl;

    std::cout << "Training single tree" << std::endl;
    RandomForest singleTree = RandomForest(1);
    singleTree.train(train_data);
    std::cout << "Done!\n" << std::endl;

    std::cout << "Training forest of size " << forestSize << std::endl;
    RandomForest forest = RandomForest(forestSize);
    forest.train(train_data);
    std::cout << "Done!\n" << std::endl;

    std::cout << "Evaluating Models:" <<std::endl;
    evalResult singleTreeResult = evaluateModel(singleTree, test_set);
    evalResult forestResult = evaluateModel(forest, test_set);
    std::cout << resultString("Single Binary Tree", singleTreeResult) << std::endl;
    std::cout << resultString("Random Forest", forestResult) << std::endl;

    return EXIT_SUCCESS;
}
