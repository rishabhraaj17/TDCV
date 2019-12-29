#include "RandomForest.h"

RandomForest::RandomForest() {
    mTreeCount = 16;
    mMaxDepth = 200;
    mCVFolds = 0;
    mMinSampleCount = 2;
    mMaxCategories = 6;
}

RandomForest::RandomForest(int treeCount, int maxDepth, int CVFolds, int minSampleCount, int maxCategories)
        : mTreeCount(treeCount), mMaxDepth(maxDepth), mCVFolds(CVFolds), mMinSampleCount(minSampleCount),
          mMaxCategories(maxCategories) {
    /*
      construct a forest with given number of trees and initialize all the trees with the
      given parameters
    */
    for (int i = 0; i < treeCount; i++){
        mTrees.push_back(cv::ml::DTrees::create());
        mTrees[i]->setMaxDepth(maxDepth);
        mTrees[i]->setMinSampleCount(minSampleCount);
        mTrees[i]->setCVFolds(CVFolds);
        // necessary?
        mTrees[i]->setMaxCategories(maxCategories);
    }
}

RandomForest::~RandomForest() {
}

void RandomForest::setTreeCount(int treeCount) {
    // Fill
    mTreeCount = treeCount;
}

void RandomForest::setMaxDepth(int maxDepth) {
    mMaxDepth = maxDepth;
    for (uint treeIdx = 0; treeIdx < mTreeCount; treeIdx++)
        mTrees[treeIdx]->setMaxDepth(mMaxDepth);
}

void RandomForest::setCVFolds(int cvFols) {
    // Fill
    mCVFolds = cvFols;
}

void RandomForest::setMinSampleCount(int minSampleCount) {
    // Fill
    mMinSampleCount = minSampleCount;
}

void RandomForest::setMaxCategories(int maxCategories) {
    // Fill
    mMaxCategories = maxCategories;
}


void
RandomForest::train(std::vector<std::pair<int, cv::Mat>> trainingImagesLabelVector, float subsetPercentage, cv::Size winStride,
                    cv::Size padding, bool undersampling, bool augment) {
    // Fill
    // Augment the dataset
    std::vector<std::pair<int, cv::Mat>> augmentedTrainingImagesLabelVector;
    augmentedTrainingImagesLabelVector.reserve(trainingImagesLabelVector.size() * 60);
    if (augment)
    {
        for(auto&& trainingImagesLabelSample : trainingImagesLabelVector)
        {
            std::vector<cv::Mat> augmentedImages = augmentImage(trainingImagesLabelSample.second);
            for (auto &&augmentedImage : augmentedImages)
            {
                augmentedTrainingImagesLabelVector.push_back(std::pair<int, cv::Mat>(trainingImagesLabelSample.first, augmentedImage));
            }
        }
    } else {
        augmentedTrainingImagesLabelVector = trainingImagesLabelVector;
    }

    std::cout << trainingImagesLabelVector.size() << std::endl;
    std::cout << augmentedTrainingImagesLabelVector.size() << std::endl;

    // Train each decision tree
    for (size_t i = 0; i < mTreeCount; i++)
    {
        std::cout << "Training decision tree: " << i + 1 << " of " << mTreeCount << ".\n";
        std::vector<std::pair<int, cv::Mat>> trainingImagesLabelSubsetVector =
                generateTrainingImagesLabelSubsetVector(augmentedTrainingImagesLabelVector,
                                                        subsetPercentage,
                                                        undersampling);

        cv::Ptr<cv::ml::DTrees> model = trainDecisionTree(trainingImagesLabelSubsetVector,
                                                          winStride,
                                                          padding);
        mTrees.push_back(model);
    }
}

Prediction RandomForest::predict(cv::Mat &testImage, cv::Size winStride, cv::Size padding) {
    // Fill
    cv::Mat resizedInputImage = resizeToBoundingBox(testImage);

    // Compute Hog only of center crop of grayscale image
    std::vector<float> descriptors;
    std::vector<cv::Point> foundLocations;
    std::vector<double> weights;
    mHogDescriptor.compute(resizedInputImage, descriptors, winStride, padding, foundLocations);

    // Store the features and labels for model training.
    // cout << i << ": Expected: " << testImagesLabelVector.at(i).first << ", Found: " << model->predict(cv::Mat(descriptors)) << endl ;
    // if(testImagesLabelVector.at(i).first == randomForest.at(0)->predict(cv::Mat(descriptors)))
    //     accuracy += 1;
    std::map<int, int> labelCounts;
    int maxCountLabel = -1;
    for (auto &&model : mTrees)
    {
        int label = model->predict(cv::Mat(descriptors));
        if (labelCounts.count(label) > 0)
            labelCounts[label]++;
        else
            labelCounts[label] = 1;

        if (maxCountLabel == -1)
            maxCountLabel = label;
        else if (labelCounts[label] > labelCounts[maxCountLabel])
            maxCountLabel = label;
    }

    return Prediction{.label = maxCountLabel,
            .confidence = (labelCounts[maxCountLabel] * 1.0f) / mTreeCount};

}

std::vector<std::pair<int, cv::Mat>> RandomForest::loadTrainDataset() {
    std::vector<std::pair<int, cv::Mat>> labelImagesTrain;
    labelImagesTrain.reserve(49 + 67 + 42 + 53 + 67 + 110);
    int numberOfTrainImages[6] = {49, 67, 42, 53, 67, 110};

    for (int i = 0; i < 6; i++)
    {
        for (size_t j = 0; j < numberOfTrainImages[i]; j++)
        {
            std::stringstream imagePath;
            imagePath << std::string(PROJ_DIR) << "/data/task2/train/" << std::setfill('0') << std::setw(2) << i << "/" << std::setfill('0') << std::setw(4) << j << ".jpg";
            std::string imagePathStr = imagePath.str();
            std::cout << imagePathStr << std::endl;
            std::pair<int, cv::Mat> labelImagesTrainPair;
            labelImagesTrainPair.first = i;
            labelImagesTrainPair.second = imread(imagePathStr, cv::IMREAD_UNCHANGED).clone();
            labelImagesTrain.push_back(labelImagesTrainPair);
        }
    }

    return labelImagesTrain;
}

std::vector<std::pair<int, cv::Mat>> RandomForest::loadTestDataset() {
    std::vector<std::pair<int, cv::Mat>> labelImagesTest;
    labelImagesTest.reserve(60);
    int numberOfTestImages[6] = {10, 10, 10, 10, 10, 10};

    for (int i = 0; i < 6; i++)
    {
        for (size_t j = 0; j < numberOfTestImages[i]; j++)
        {
            std::stringstream imagePath;
            imagePath << std::string(PROJ_DIR) << "/data/task2/test/" << std::setfill('0') << std::setw(2) << i << "/" << std::setfill('0') << std::setw(4) << j + numberOfTestImages[i] << ".jpg";
            std::string imagePathStr = imagePath.str();
            std::cout << imagePathStr << std::endl;
            std::pair<int, cv::Mat> labelImagesTestPair;
            labelImagesTestPair.first = i;
            labelImagesTestPair.second = imread(imagePathStr, cv::IMREAD_UNCHANGED).clone();
            labelImagesTest.push_back(labelImagesTestPair);
        }
    }

    return labelImagesTest;
}

std::vector<int> RandomForest::getRandomUniqueIndices(int start, int end, int numOfSamples) {
    std::vector<int> indices;
    indices.reserve(end - start);
    for (size_t i = start; i < end; i++)
        indices.push_back(i);

    std::shuffle(indices.begin(), indices.end(), mRandomGenerator);
    // copy(indices.begin(), indices.begin() + numOfSamples, std::ostream_iterator<int>(std::cout, ", "));
    // cout << endl;
    return std::vector<int>(indices.begin(), indices.begin() + numOfSamples);
}

cv::HOGDescriptor RandomForest::createHogDescriptor() {
    cv::Size wsize(128, 128);
    cv::Size blockSize(16, 16);
    cv::Size stride(8, 8);
    cv::Size cell(8, 8);
    int bins(18); // 9 to 18
    int aperture(1);
    double sigma(-1);
    int normType(cv::HOGDescriptor::L2Hys);
    double l2HysThreshold(0.2);
    bool gcorrection(true);
    //! Maximum number of detection window increases. Default value is 64
    int n_levels(cv::HOGDescriptor::DEFAULT_NLEVELS);
    //TODO: Observe true
    bool gradient(true);
    cv::HOGDescriptor hog_descriptor(wsize, blockSize, stride, cell, bins, aperture, sigma, normType,
                                    l2HysThreshold, gcorrection, n_levels, gradient);
    //TODO: observe copying
    mHogDescriptor = hog_descriptor;
    return mHogDescriptor;
}

cv::Ptr<cv::ml::DTrees>
RandomForest::trainDecisionTree(std::vector<std::pair<int, cv::Mat>> &trainingImagesLabelVector, cv::Size winStride,
                                cv::Size padding) {
    // Create the model
    cv::Ptr<cv::ml::DTrees> model = cv::ml::DTrees::create();
    // See https://docs.opencv.org/3.0-beta/modules/ml/doc/decision_trees.html#dtrees-params
    model->setCVFolds(0);        // set num cross validation folds - Not implemented in OpenCV
    model->setMaxCategories(10); // set max number of categories
    model->setMaxDepth(20);      // set max tree depth
    model->setMinSampleCount(2); // set min sample count
    // ToDo - Tweak this
    // cout << "Number of cross validation folds are: " << model->getCVFolds() << endl;
    // cout << "Max Categories are: " << model->getMaxCategories() << endl;
    // cout << "Max depth is: " << model->getMaxDepth() << endl;
    // cout << "Minimum Sample Count: " << model->getMinSampleCount() << endl;

    // Compute Hog Features for all the training images
    cv::Mat feats, labels;
    for (size_t i = 0; i < trainingImagesLabelVector.size(); i++)
    {
        cv::Mat inputImage = trainingImagesLabelVector.at(i).second;
        cv::Mat resizedInputImage = resizeToBoundingBox(inputImage);

        // Compute Hog only of center crop of grayscale image
        std::vector<float> descriptors;
        std::vector<cv::Point> foundLocations;
        std::vector<double> weights;
        mHogDescriptor.compute(resizedInputImage, descriptors, winStride, padding, foundLocations);

        // Store the features and labels for model training.
        // cout << "=====================================" << endl;
        // cout << "Number of descriptors are: " << descriptors.size() << endl;
        feats.push_back(cv::Mat(descriptors).clone().reshape(1, 1));
        // cout << "New size of training features" << feats.size() << endl;
        labels.push_back(trainingImagesLabelVector.at(i).first);
        // cout << "New size of training labels" << labels.size() << endl;
    }

    cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(feats, cv::ml::ROW_SAMPLE, labels);
    model->train(trainData);
    return model;
}

cv::Mat RandomForest::resizeToBoundingBox(cv::Mat &inputImage) {
    cv::Mat resizedInputImage;
    if (inputImage.rows < mWinSize.height || inputImage.cols < mWinSize.width)
    {
        float scaleFactor = fmax((mWinSize.height * 1.0f) / inputImage.rows, (mWinSize.width * 1.0f) / inputImage.cols);
        cv::resize(inputImage, resizedInputImage, cv::Size(0, 0), scaleFactor, scaleFactor, cv::INTER_LINEAR);
    }
    else
    {
        resizedInputImage = inputImage;
    }

    cv::Rect r = cv::Rect((resizedInputImage.cols - mWinSize.width) / 2, (resizedInputImage.rows - mWinSize.height) / 2,
                          mWinSize.width, mWinSize.height);
    // cv::imshow("Resized", resizedInputImage(r));
    // cv::imshow("Original", inputImage);
    // cv::waitKey(0);
    return resizedInputImage(r);
}

std::vector<std::pair<int, cv::Mat>>
RandomForest::generateTrainingImagesLabelSubsetVector(std::vector<std::pair<int, cv::Mat>> &trainingImagesLabelVector,
                                                      float subsetPercentage, bool undersampling) {
    std::vector<std::pair<int, cv::Mat>> trainingImagesLabelSubsetVector;

    // Compute minimum number of samples a class label has.
    int minimumSample = trainingImagesLabelVector.size(); // A high enough value

    if (undersampling)
    {
        int minimumClassSamples[mMaxCategories];
        for (size_t i = 0; i < mMaxCategories; i++)
            minimumClassSamples[i] = 0;
        for (auto &&trainingSample : trainingImagesLabelVector)
            minimumClassSamples[trainingSample.first]++;
        for (size_t i = 1; i < mMaxCategories; i++)
            if (minimumClassSamples[i] < minimumSample)
                minimumSample = minimumClassSamples[i];
    }

    for (size_t label = 0; label < mMaxCategories; label++)
    {
        // Create a subset vector for all the samples with class label.
        std::vector<std::pair<int, cv::Mat>> temp;
        temp.reserve(100);
        for (auto &&sample : trainingImagesLabelVector)
            if (sample.first == label)
                temp.push_back(sample);

        // Compute how many samples to choose for each label for random subset.
        int numOfElements;
        if (undersampling)
        {
            numOfElements = (subsetPercentage * minimumSample) / 100;
        }
        else
        {
            numOfElements = (temp.size() * subsetPercentage) / 100;
        }

        // Filter numOfElements elements from temp and append to trainingImagesLabelSubsetVector
        std::vector<int> randomUniqueIndices = getRandomUniqueIndices(0, temp.size(), numOfElements);
        for (size_t j = 0; j < randomUniqueIndices.size(); j++)
        {
            std::pair<int, cv::Mat> subsetSample = temp.at(randomUniqueIndices.at(j));
            trainingImagesLabelSubsetVector.push_back(subsetSample);
        }
    }

    return trainingImagesLabelSubsetVector;
}

std::vector<cv::Mat> RandomForest::augmentImage(cv::Mat &inputImage) {
    std::vector<cv::Mat> augmentations;
    cv::Mat currentImage = inputImage;
    cv::Mat rotatedImage, flippedImage;
    for (size_t j = 0; j < 4; j++)
    {
        std::vector<cv::Mat> currentImageAugmentations;
        if (j == 0)
            rotatedImage = currentImage;
        else
            cv::rotate(currentImage, rotatedImage, cv::ROTATE_90_CLOCKWISE);
        currentImageAugmentations.push_back(rotatedImage);

        int numOfRandomRotations = 4;
        for(size_t i = 0; i < numOfRandomRotations; i++)
        {
            cv::Mat randomlyRotatedImage(rotatedImage.size(), rotatedImage.type());
            cv::RNG rng(time(0));
            //FIXME RandomRotateImage(rotatedImage, randomlyRotatedImage, 90, 30, 30, cv::Rect(0, 0, 0, 0), rng);
            currentImageAugmentations.push_back(randomlyRotatedImage);
            // cv::imshow("input image", rotatedImage);
            // cv::imshow("RandomRotationImage", randomlyRotatedImage);
            // cv::waitKey(1000);
        }

        int imagesToFlip = currentImageAugmentations.size();
        for (int i = 0; i < imagesToFlip; i++)
        {
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
    randomForest->mHogDescriptor = randomForest->createHogDescriptor();
    long unsigned int timestamp = static_cast<long unsigned int>(time(0));
    std::cout << timestamp << std::endl;
    randomForest->mRandomGenerator = std::mt19937(timestamp);
    return randomForest;
}

