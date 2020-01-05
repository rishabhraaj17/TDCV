//
// Created by theo on 1/14/19.
//

#include <filesystem>
#include <random>
#include "RandomForest.h"

RandomForest::RandomForest(std::filesystem::path restorePath) {
    for (const auto &file : std::filesystem::directory_iterator(restorePath)) {
        if (file.path().extension().compare("tree")) {
            auto newTree = cv::ml::DTrees::create();
            this->trees.push_back(newTree->load(file.path().string()));
        }
    }
}

RandomForest::RandomForest(int nTrees, int maxDepth, int minSampleCount, int maxCategories) {
    for (int i = 0; i < nTrees; i++) {
        auto newTree = cv::ml::DTrees::create();
        newTree->setMaxDepth(maxDepth);
        newTree->setMinSampleCount(minSampleCount);
        newTree->setMaxCategories(maxCategories);
        newTree->setUseSurrogates(false);
        newTree->setCVFolds(1); // the number of cross-validation folds
        newTree->setUse1SERule(true);
        newTree->setTruncatePrunedTree(false);
        this->trees.push_back(newTree);
    }
}

void RandomForest::save(const std::filesystem::path &path) {
    std::filesystem::create_directory(path);
    int treeindex = 0;
    for (const auto &tree : this->trees) {
        tree->save(path.string() + '/' + std::to_string(treeindex++) + ".tree");
    }
}

void RandomForest::train(cv::Ptr<cv::ml::TrainData> &trainData, float subsetRatio) {
    int ti = 1;
    for (const auto &tree : this->trees) {
        std::cout << "Training tree " << ti << "/" << this->getSize() << "\r" << std::flush;
        if (this->trees.size() > 1) trainData->setTrainTestSplitRatio(subsetRatio, true);
        tree->train(trainData);
        ti++;
    }
    std::cout << std::endl;
}

int RandomForest::predict(const cv::Mat &sample, float *confidence) const {
    std::map<float, int> results;
    for (const auto &tree : this->trees) {
        float prediction = tree->predict(sample);
        if (!results.count(prediction)) {
            results.emplace(prediction, 1);
        } else {
            results[prediction] += 1;
        }
    }
    int maxVotes = -1;
    float maxResult = -1;
    for (std::pair<const float, int> &res : results) {
        if (res.second > maxVotes) {
            maxVotes = res.second;
            maxResult = res.first;
        }
    }
    if (confidence != nullptr)
        *confidence = maxVotes / (float) this->trees.size();
    return (int) maxResult;
}

unsigned long RandomForest::getSize() {
    return this->trees.size();
}
