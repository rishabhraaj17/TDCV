//
// Created by rishabh on 29.12.19.
//

#include <Util.h>
#include <opencv2/core/utils/filesystem.hpp>
#include "RandomForest.h"


RandomForest::RandomForest() {
    mTreeCount = 30;
    mMaxDepth = 300;
    mCVFolds = 0;
    mMinSampleCount = 2;
    mMaxCategories = 6;
}

RandomForest::RandomForest(int treeCount, int maxDepth, int CVFolds, int minSampleCount, int maxCategories)
        : mTreeCount(treeCount), mMaxDepth(maxDepth), mCVFolds(CVFolds), mMinSampleCount(minSampleCount),
          mMaxCategories(maxCategories) {

    for (int i = 0; i < treeCount; i++) {
        mTrees.push_back(cv::ml::DTrees::create());
        mTrees[i]->setMaxDepth(maxDepth);
        mTrees[i]->setMinSampleCount(minSampleCount);
        mTrees[i]->setCVFolds(CVFolds);
        mTrees[i]->setMaxCategories(maxCategories);
    }
}

RandomForest::~RandomForest() {
}

void RandomForest::setTreeCount(int treeCount) {
    mTreeCount = treeCount;
}

int RandomForest::getTreeCount() {
    return mTreeCount;
}

void RandomForest::setMaxDepth(int maxDepth) {
    mMaxDepth = maxDepth;
    for (uint treeIdx = 0; treeIdx < mTreeCount; treeIdx++)
        mTrees[treeIdx]->setMaxDepth(mMaxDepth);
}

int RandomForest::getMaxDepth() {
    return mMaxDepth;
}

void RandomForest::setCVFolds(int cvFols) {
    mCVFolds = cvFols;
}

int RandomForest::getCVFolds() {
    return mCVFolds;
}

void RandomForest::setMinSampleCount(int minSampleCount) {
    mMinSampleCount = minSampleCount;
}

int RandomForest::getMinSampleCount() {
    return mMinSampleCount;
}

void RandomForest::setMaxCategories(int maxCategories) {
    mMaxCategories = maxCategories;
}

int RandomForest::getMaxCategories() {
    return mMaxCategories;
}

std::vector<cv::Ptr<cv::ml::DTrees> > RandomForest::getTrees() {
    return mTrees;
}


void
RandomForest::train(std::vector<std::pair<int, cv::Mat>> trainDataset, float perTreeTrainDatasetSubsetPercentage,
                    const cv::Size &winStride,
                    const cv::Size &padding, bool underSampling, bool dataAugmentation,
                    const cv::Size &winSize = cv::Size(128, 128)) {
    // Augment the dataset
    int counter = 0;
    std::vector<std::pair<int, cv::Mat>> augmentedTrainDataset;
    augmentedTrainDataset.reserve(trainDataset.size() * 60);
    if (dataAugmentation) {
        for (auto &&sample : trainDataset) {
            std::vector<cv::Mat> augmentedImages = augmentImage(sample.second);
            std::cout << "Augmented Images Iteration : " << counter++ << std::endl;
            for (auto &&augmentedImage : augmentedImages) {
                augmentedTrainDataset.emplace_back(sample.first, augmentedImage);
            }
        }
    } else {
        augmentedTrainDataset = trainDataset;
    }

    std::cout << "Train set size before augmentation : " << trainDataset.size() << std::endl;
    std::cout << "Train set size after augmentation : " << augmentedTrainDataset.size() << std::endl;

    // Train each decision tree
    for (size_t i = 0; i < mTreeCount; i++) {
        std::cout << "Training Decision Tree: " << i << " - Total Trees : " << mTreeCount << "\n";
        std::vector<std::pair<int, cv::Mat>> subsetOfTrainDataset =
                trainDatasetSubsetSampler(augmentedTrainDataset,
                                          perTreeTrainDatasetSubsetPercentage,
                                          underSampling);

        cv::Ptr<cv::ml::DTrees> model = trainSingleDecisionTree(subsetOfTrainDataset,
                                                                winStride,
                                                                padding,
                                                                winSize, 100, 6);
        mTrees.push_back(model);
    }
}

ModelPrediction
RandomForest::predictPerImage(cv::Mat &testImage, cv::Size winStride, cv::Size padding, cv::Size winSize = cv::Size(128,
                                                                                                                    128)) {
    cv::Mat resizedImage = imageResize(testImage, winSize);

    std::vector<float> descriptors;
    std::vector<cv::Point> foundLocations;
    mHogDescriptor.compute(resizedImage, descriptors, winStride, padding, foundLocations);

    // Predictions from all the models
    std::map<int, int> predictedLabels;
    int finalPrediction = -1;
    for (auto &&model : mTrees) {
        int label = model->predict(cv::Mat(descriptors));
        if (predictedLabels.count(label) > 0)
            predictedLabels[label]++;
        else
            predictedLabels[label] = 1;

        if (finalPrediction == -1)
            finalPrediction = label;
        else if (predictedLabels[label] > predictedLabels[finalPrediction])
            finalPrediction = label;
    }

    return ModelPrediction{.label = finalPrediction,
            .confidence = (predictedLabels[finalPrediction] * 1.0f) / mTreeCount};
}

std::vector<std::pair<int, cv::Mat>> RandomForest::loadTrainDataset() {
    // < label, image > trainDataset
    std::vector<std::pair<int, cv::Mat>> trainDataset;
    trainDataset.reserve(49 + 67 + 42 + 53 + 67 + 110);
    int trainImagesPerClassCount[6] = {49, 67, 42, 53, 67, 110};

    for (int i = 0; i < 6; i++) {
        for (size_t j = 0; j < trainImagesPerClassCount[i]; j++) {
            std::stringstream path;
            path << std::string(PROJ_DIR) << "/data/task2/train/" << std::setfill('0') << std::setw(2) << i <<
                 "/" << std::setfill('0') << std::setw(4) << j << ".jpg";
            std::string imagePathStr = path.str();
            std::pair<int, cv::Mat> pair;
            pair.first = i;
            pair.second = imread(imagePathStr, cv::IMREAD_UNCHANGED).clone();
            trainDataset.push_back(pair);
        }
    }

    return trainDataset;
}

std::vector<std::pair<int, cv::Mat>> RandomForest::loadTestDataset() {
    std::vector<std::pair<int, cv::Mat>> testDataset;
    testDataset.reserve(60);
    int testImagesPerClassCount[6] = {10, 10, 10, 10, 10, 10};
    int trainImagesPerClassCount[6] = {49, 67, 42, 53, 67, 110};

    for (int i = 0; i < 6; i++) {
        for (size_t j = 0; j < testImagesPerClassCount[i]; j++) {
            std::stringstream path;
            path << std::string(PROJ_DIR) << "/data/task2/test/" << std::setfill('0') << std::setw(2) << i <<
                 "/" << std::setfill('0') << std::setw(4) << j + trainImagesPerClassCount[i] << ".jpg";
            std::string imagePathStr = path.str();
            std::pair<int, cv::Mat> pair;
            pair.first = i;
            pair.second = imread(imagePathStr, cv::IMREAD_UNCHANGED).clone();
            testDataset.push_back(pair);
        }
    }

    return testDataset;
}

cv::HOGDescriptor RandomForest::createHogDescriptor(cv::Size size = cv::Size(128, 128)) {
    cv::Size wSize = size;
    cv::Size blockSize(16, 16);
    cv::Size stride(8, 8);
    cv::Size cell(8, 8);
    int bins(18);
    int aperture(1);
    double sigma(-1);
    double l2HysThreshold(0.2);
    bool gCorrection(true);
    int n_levels(cv::HOGDescriptor::DEFAULT_NLEVELS);
    bool gradient(true);
    cv::HOGDescriptor hog_descriptor(wSize, blockSize, stride, cell, bins, aperture, sigma, cv::HOGDescriptor::L2Hys,
                                     l2HysThreshold, gCorrection, n_levels, gradient);
    mHogDescriptor = hog_descriptor;
    return hog_descriptor;
}

cv::Ptr<cv::ml::DTrees>
RandomForest::trainSingleDecisionTree(std::vector<std::pair<int, cv::Mat>> &trainingImagesLabelVector,
                                      const cv::Size &winStride, const cv::Size &padding,
                                      const cv::Size &winSize = cv::Size(128, 128), int maxDepth = 100,
                                      int maxClasses = 6) {
    // Model for single tree
    cv::Ptr<cv::ml::DTrees> model = cv::ml::DTrees::create();
    model->setCVFolds(0);
    model->setMaxCategories(maxClasses);
    model->setMaxDepth(maxDepth);
    model->setMinSampleCount(2);

    cv::Mat features, labels;
    std::cout << "Size of training set : " << trainingImagesLabelVector.size() << std::endl;
    for (auto & i : trainingImagesLabelVector) {
        cv::Mat inputImage = i.second;
        cv::Mat resizedInputImage = imageResize(inputImage, winSize);

        std::vector<float> descriptors;
        std::vector<cv::Point> foundLocations;
        try {
            mHogDescriptor.compute(resizedInputImage, descriptors, winStride, padding, foundLocations);
        } catch (cv::Exception &exception) {
            std::cout << "Unable to compute Hog descriptors!!!" << std::endl;
        }
        features.push_back(cv::Mat(descriptors).clone().reshape(1, 1));
        labels.push_back(i.first);
    }
    cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(features, cv::ml::ROW_SAMPLE, labels);
    model->train(trainData);
    return model;
}

cv::Mat RandomForest::imageResize(cv::Mat &inputImage, cv::Size size) {
    mWinSize = size;
    cv::Mat resizedImage;
    if (inputImage.rows < mWinSize.height || inputImage.cols < mWinSize.width) {
        float scaleFactor = fmax((mWinSize.height * 1.0f) / inputImage.rows, (mWinSize.width * 1.0f) / inputImage.cols);
        cv::resize(inputImage, resizedImage, cv::Size(0, 0), scaleFactor, scaleFactor, cv::INTER_LINEAR);
    } else {
        resizedImage = inputImage;
    }

    cv::Rect r = cv::Rect((resizedImage.cols - mWinSize.width) / 2, (resizedImage.rows - mWinSize.height) / 2,
                          mWinSize.width, mWinSize.height);
    return resizedImage(r);
}

std::vector<std::pair<int, cv::Mat>>
RandomForest::trainDatasetSubsetSampler(std::vector<std::pair<int, cv::Mat>> &trainDataset,
                                        float perTreeTrainDatasetSubsetPercentage, bool underSampling) {
    std::vector<std::pair<int, cv::Mat>> trainDatasetSubset;
    // Samples randomly subset of images

    // A high value to start with
    int minSample = 999999;

    if (underSampling) {
        int minSamplesPerClass[mMaxCategories];
        // Initialize each class to 0
        for (size_t i = 0; i < mMaxCategories; i++)
            minSamplesPerClass[i] = 0;
        // Finding number of sample for each class
        for (auto &&trainingSample : trainDataset)
            minSamplesPerClass[trainingSample.first]++;
        // Get the minimum samples one class has
        for (size_t i = 1; i < mMaxCategories; i++)
            if (minSamplesPerClass[i] < minSample)
                minSample = minSamplesPerClass[i];
    }

    for (size_t label = 0; label < mMaxCategories; label++) {
        std::vector<std::pair<int, cv::Mat>> workingSubset;
        workingSubset.reserve(100);
        for (auto &&sample : trainDataset)
            if (sample.first == label)
                workingSubset.push_back(sample);

        // Number of samples to choose for each label for current subset.
        // Under Sampling maintains class balance
        int numOfElements = 0;
        if (underSampling) {
            numOfElements = (perTreeTrainDatasetSubsetPercentage * minSample) / 100;
        } else {
            numOfElements = (workingSubset.size() * perTreeTrainDatasetSubsetPercentage) / 100;
        }

        // Generating random indexes to pick samples from for subset train Dataset
        std::vector<int> idxs;
        idxs.reserve(workingSubset.size() - 0);
        for (size_t i = 0; i < workingSubset.size(); i++)
            idxs.push_back(i);

        std::shuffle(idxs.begin(), idxs.end(), mRandomGenerator);
        std::vector<int> finalIdxs = std::vector<int>(idxs.begin(), idxs.begin() + numOfElements);

        for (int finalIdx : finalIdxs) {
            std::pair<int, cv::Mat> subsetSample = workingSubset.at(finalIdx);
            trainDatasetSubset.push_back(subsetSample);
        }
    }

    return trainDatasetSubset;
}

//Todo
std::vector<cv::Mat> RandomForest::augmentImage(cv::Mat &inputImage) {
    std::vector<cv::Mat> augmentations;
    cv::Mat currentImage = inputImage;
    cv::Mat rotatedImage, flippedImage;
    for (size_t j = 0; j < 4; j++) {
        std::vector<cv::Mat> currentImageAugmentations;
        if (j == 0)
            rotatedImage = currentImage;
        else
            cv::rotate(currentImage, rotatedImage, cv::ROTATE_90_CLOCKWISE);
        currentImageAugmentations.push_back(rotatedImage);

        int numOfRandomRotations = 4;
        for (size_t i = 0; i < numOfRandomRotations; i++) {
            cv::Mat randomlyRotatedImage(rotatedImage.size(), rotatedImage.type());
            cv::RNG rng(time(0));
            RandomRotateImage(rotatedImage, randomlyRotatedImage, 90, 30, 30, cv::Rect(0, 0, 0, 0), rng);
            currentImageAugmentations.push_back(randomlyRotatedImage);
            // cv::imshow("input image", rotatedImage);
            // cv::imshow("RandomRotationImage", randomlyRotatedImage);
            // cv::waitKey(1000);
        }

        int imagesToFlip = currentImageAugmentations.size();
        for (int i = 0; i < imagesToFlip; i++) {
            cv::flip(currentImageAugmentations[i], flippedImage, 0);
            currentImageAugmentations.push_back(flippedImage);
            // cv::Mat dest;
            // cv::vconcat(currentImageAugmentations[i], flippedImage, dest);
            // cv::imshow("AugmentedImage", dest);
            // cv::waitKey(1000);
            cv::flip(currentImageAugmentations[i], flippedImage, 1);
            currentImageAugmentations.push_back(flippedImage);
            // cv::vconcat(currentImageAugmentations[i], flippedImage, dest);
            // cv::imshow("AugmentedImage", dest);
            // cv::waitKey(1000);
        }

        augmentations.insert(augmentations.end(), currentImageAugmentations.begin(), currentImageAugmentations.end());
        currentImage = rotatedImage;
    }

    return augmentations;
}

cv::Ptr<RandomForest> RandomForest::createRandomForest(int numberOfClasses, int numberOfDTrees, cv::Size winSize) {
    cv::Ptr<RandomForest> randomForest = new RandomForest();
    randomForest->mMaxCategories = numberOfClasses;
    randomForest->mTreeCount = numberOfDTrees;
    randomForest->mWinSize = winSize;
    randomForest->mTrees.reserve(numberOfDTrees);
    randomForest->mHogDescriptor = randomForest->createHogDescriptor(winSize);
    auto timestamp = static_cast<long unsigned int>(time(0));
    randomForest->mRandomGenerator = std::mt19937(timestamp);
    return randomForest;
}

void RandomForest::setTrees(std::vector<cv::Ptr<cv::ml::DTrees>> trees) {
    mTrees = trees;
}

void RandomForest::saveModel(const std::string &path) {
    cv::utils::fs::createDirectories(path);
    int idx = 0;
    for (const auto &tree : this->mTrees) {
        tree->save(path + '/' + std::to_string(idx++) + ".tree");
    }
}

std::vector<cv::Ptr<cv::ml::DTrees>> RandomForest::loadModel(const std::string &path, int treeCount) {
    std::vector<cv::Ptr<cv::ml::DTrees> > totalTrees;
    for (size_t i = 0; i < treeCount; i++) {
        totalTrees.push_back(cv::ml::DTrees::load(path + '/' + std::to_string(i) + ".tree"));
    }
    return totalTrees;
}

void RandomForest::trainSingleTree(RandomForest *randomForest,
                                   std::vector<std::pair<int, cv::Mat>> &trainingImagesLabelVector) {
    cv::Size winSize(128, 128);
    cv::HOGDescriptor hog = randomForest->createHogDescriptor(winSize);
    cv::Size winStride(8, 8);
    cv::Size padding(0, 0);

    cv::Mat features, labels;
    for (auto &i : trainingImagesLabelVector) {
        cv::Mat inputImage = i.second;
        cv::Mat resizedInputImage = randomForest->imageResize(inputImage, winSize);

        std::vector<float> descriptors;
        std::vector<cv::Point> foundLocations;
        std::vector<double> weights;
        hog.compute(resizedInputImage, descriptors, winStride, padding, foundLocations);

        features.push_back(cv::Mat(descriptors).clone().reshape(1, 1));
        labels.push_back(i.first);
    }

    cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(features, cv::ml::ROW_SAMPLE, labels);
    randomForest->getTrees()[0]->train(trainData);
    std::cout << "Single Decision Tree Trained!" << std::endl;
}


// taken from git : link in header file
void
RandomForest::RandomRotateImage(const cv::Mat &src, cv::Mat &dst, float yaw_range, float pitch_range, float roll_range,
                                const cv::Rect &area, cv::RNG rng, float Z, int interpolation, int boarder_mode,
                                cv::Scalar boarder_color) {
    double yaw = rng.uniform(-yaw_range / 2, yaw_range / 2);
    double pitch = rng.uniform(-pitch_range / 2, pitch_range / 2);
    double roll = rng.uniform(-roll_range / 2, roll_range / 2);

    cv::Rect rect = (area.width <= 0 || area.height <= 0) ? cv::Rect(0, 0, src.cols, src.rows) :
                    ExpandRectForRotate(area);
    rect = util::TruncateRectKeepCenter(rect, src.size());

    cv::Mat rot_img;
    ImageRotate(src(rect).clone(), rot_img, yaw, pitch, roll, Z, interpolation, boarder_mode, boarder_color);

    cv::Rect dst_area((rot_img.cols - rect.width) / 2, (rot_img.rows - rect.height) / 2, rect.width, rect.height);
    dst_area = util::TruncateRectKeepCenter(dst_area, rot_img.size());
    dst = rot_img(dst_area).clone();
}

void RandomForest::ImageRotate(const cv::Mat &src, cv::Mat &dst, float yaw, float pitch, float roll,
                               float Z = 1000, int interpolation = cv::INTER_LINEAR,
                               int boarder_mode = cv::BORDER_CONSTANT,
                               const cv::Scalar &border_color = cv::Scalar(0, 0, 0)) {
    // rotation matrix
    cv::Mat rotMat_3x4;
    composeExternalMatrix(yaw, pitch, roll, 0, 0, Z, rotMat_3x4);

    cv::Mat rotMat = cv::Mat::eye(4, 4, rotMat_3x4.type());
    rotMat_3x4.copyTo(rotMat(cv::Rect(0, 0, 4, 3)));

    // From 2D coordinates to 3D coordinates
    // The center of image is (0,0,0)
    cv::Mat invPerspMat = cv::Mat::zeros(4, 3, CV_64FC1);
    invPerspMat.at<double>(0, 0) = 1;
    invPerspMat.at<double>(1, 1) = 1;
    invPerspMat.at<double>(3, 2) = 1;
    invPerspMat.at<double>(0, 2) = -(double) src.cols / 2;
    invPerspMat.at<double>(1, 2) = -(double) src.rows / 2;

    // �R�������W����Q�������W�֓����ϊ�
    cv::Mat perspMat = cv::Mat::zeros(3, 4, CV_64FC1);
    perspMat.at<double>(0, 0) = Z;
    perspMat.at<double>(1, 1) = Z;
    perspMat.at<double>(2, 2) = 1;

    // ���W�ϊ����A�o�͉摜�̍��W�͈͂�擾
    cv::Mat transMat = perspMat * rotMat * invPerspMat;
    cv::Rect_<double> CircumRect;
    CircumTransImgRect(src.size(), transMat, CircumRect);

    // �o�͉摜�Ɠ��͉摜�̑Ή��}�b�v��쐬
    cv::Mat map_x, map_y;
    CreateMap(src.size(), CircumRect, rotMat, map_x, map_y);
    cv::remap(src, dst, map_x, map_y, interpolation, boarder_mode, border_color);
}

void
RandomForest::composeExternalMatrix(float yaw, float pitch, float roll, float trans_x, float trans_y, float trans_z,
                                    cv::Mat &external_matrix) {
    external_matrix.release();
    external_matrix.create(3, 4, CV_64FC1);

    double sin_yaw = sin((double) yaw * CV_PI / 180);
    double cos_yaw = cos((double) yaw * CV_PI / 180);
    double sin_pitch = sin((double) pitch * CV_PI / 180);
    double cos_pitch = cos((double) pitch * CV_PI / 180);
    double sin_roll = sin((double) roll * CV_PI / 180);
    double cos_roll = cos((double) roll * CV_PI / 180);

    external_matrix.at<double>(0, 0) = cos_pitch * cos_yaw;
    external_matrix.at<double>(0, 1) = -cos_pitch * sin_yaw;
    external_matrix.at<double>(0, 2) = sin_pitch;
    external_matrix.at<double>(1, 0) = cos_roll * sin_yaw + sin_roll * sin_pitch * cos_yaw;
    external_matrix.at<double>(1, 1) = cos_roll * cos_yaw - sin_roll * sin_pitch * sin_yaw;
    external_matrix.at<double>(1, 2) = -sin_roll * cos_pitch;
    external_matrix.at<double>(2, 0) = sin_roll * sin_yaw - cos_roll * sin_pitch * cos_yaw;
    external_matrix.at<double>(2, 1) = sin_roll * cos_yaw + cos_roll * sin_pitch * sin_yaw;
    external_matrix.at<double>(2, 2) = cos_roll * cos_pitch;

    external_matrix.at<double>(0, 3) = trans_x;
    external_matrix.at<double>(1, 3) = trans_y;
    external_matrix.at<double>(2, 3) = trans_z;
}


cv::Mat RandomForest::Rect2Mat(const cv::Rect &img_rect) {
    cv::Mat srcCoord(3, 4, CV_64FC1);
    srcCoord.at<double>(0, 0) = img_rect.x;
    srcCoord.at<double>(1, 0) = img_rect.y;
    srcCoord.at<double>(2, 0) = 1;
    srcCoord.at<double>(0, 1) = img_rect.x + img_rect.width;
    srcCoord.at<double>(1, 1) = img_rect.y;
    srcCoord.at<double>(2, 1) = 1;
    srcCoord.at<double>(0, 2) = img_rect.x + img_rect.width;
    srcCoord.at<double>(1, 2) = img_rect.y + img_rect.height;
    srcCoord.at<double>(2, 2) = 1;
    srcCoord.at<double>(0, 3) = img_rect.x;
    srcCoord.at<double>(1, 3) = img_rect.y + img_rect.height;
    srcCoord.at<double>(2, 3) = 1;

    return srcCoord;
}

void RandomForest::CircumTransImgRect(const cv::Size &img_size, const cv::Mat &transM, cv::Rect_<double> &CircumRect) {
    cv::Mat cornersMat = Rect2Mat(cv::Rect(0, 0, img_size.width, img_size.height));

    cv::Mat dstCoord = transM * cornersMat;
    double min_x = std::min(dstCoord.at<double>(0, 0) / dstCoord.at<double>(2, 0),
                            dstCoord.at<double>(0, 3) / dstCoord.at<double>(2, 3));
    double max_x = std::max(dstCoord.at<double>(0, 1) / dstCoord.at<double>(2, 1),
                            dstCoord.at<double>(0, 2) / dstCoord.at<double>(2, 2));
    double min_y = std::min(dstCoord.at<double>(1, 0) / dstCoord.at<double>(2, 0),
                            dstCoord.at<double>(1, 1) / dstCoord.at<double>(2, 1));
    double max_y = std::max(dstCoord.at<double>(1, 2) / dstCoord.at<double>(2, 2),
                            dstCoord.at<double>(1, 3) / dstCoord.at<double>(2, 3));

    CircumRect.x = min_x;
    CircumRect.y = min_y;
    CircumRect.width = max_x - min_x;
    CircumRect.height = max_y - min_y;
}

void RandomForest::CreateMap(const cv::Size &src_size, const cv::Rect_<double> &dst_rect, const cv::Mat &transMat,
                             cv::Mat &map_x, cv::Mat &map_y) {
    map_x.create(dst_rect.size(), CV_32FC1);
    map_y.create(dst_rect.size(), CV_32FC1);

    double Z = transMat.at<double>(2, 3);

    cv::Mat invTransMat = transMat.inv();    // �t�s��
    cv::Mat dst_pos(3, 1, CV_64FC1);    // �o�͉摜��̍��W
    dst_pos.at<double>(2, 0) = Z;
    for (int dy = 0; dy < map_x.rows; dy++) {
        dst_pos.at<double>(1, 0) = dst_rect.y + dy;
        for (int dx = 0; dx < map_x.cols; dx++) {
            dst_pos.at<double>(0, 0) = dst_rect.x + dx;
            cv::Mat rMat = -invTransMat(cv::Rect(3, 2, 1, 1)) / (invTransMat(cv::Rect(0, 2, 3, 1)) * dst_pos);
            cv::Mat src_pos = invTransMat(cv::Rect(0, 0, 3, 2)) * dst_pos * rMat + invTransMat(cv::Rect(3, 0, 1, 2));
            map_x.at<float>(dy, dx) = src_pos.at<double>(0, 0) + (float) src_size.width / 2;
            map_y.at<float>(dy, dx) = src_pos.at<double>(1, 0) + (float) src_size.height / 2;
        }
    }
}

// Keep center and expand rectangle for rotation
cv::Rect RandomForest::ExpandRectForRotate(const cv::Rect &area) {
    cv::Rect exp_rect;

    int w = cvRound(std::sqrt((double) (area.width * area.width + area.height * area.height)));

    exp_rect.width = w;
    exp_rect.height = w;
    exp_rect.x = area.x - (exp_rect.width - area.width) / 2;
    exp_rect.y = area.y - (exp_rect.height - area.height) / 2;

    return exp_rect;
}
