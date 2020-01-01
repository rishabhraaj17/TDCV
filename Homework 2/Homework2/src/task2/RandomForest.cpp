#include <Util.h>
#include "RandomForest.h"
#include "RandomRotation.h"

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

int RandomForest::getTreeCount(){
    return mTreeCount;
}

void RandomForest::setMaxDepth(int maxDepth) {
    mMaxDepth = maxDepth;
    for (uint treeIdx = 0; treeIdx < mTreeCount; treeIdx++)
        mTrees[treeIdx]->setMaxDepth(mMaxDepth);
}

int RandomForest::getMaxDepth(){
    return mMaxDepth;
}

void RandomForest::setCVFolds(int cvFols) {
    // Fill
    mCVFolds = cvFols;
}

int RandomForest::getCVFolds(){
    return mCVFolds;
}

void RandomForest::setMinSampleCount(int minSampleCount) {
    // Fill
    mMinSampleCount = minSampleCount;
}

int RandomForest::getMinSampleCount(){
    return mMinSampleCount;
}

void RandomForest::setMaxCategories(int maxCategories) {
    // Fill
    mMaxCategories = maxCategories;
}

int RandomForest::getMaxCategories(){
    return mMaxCategories;
}

std::vector<cv::Ptr<cv::ml::DTrees> > RandomForest::getTrees(){
    return mTrees;
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
    cv::Mat resizedInputImage = resizeToBoundingBox(testImage, cv::Size());

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
            //std::cout << imagePathStr << std::endl;
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
    int numberOfTrainImages[6] = {49, 67, 42, 53, 67, 110};

    for (int i = 0; i < 6; i++)
    {
        for (size_t j = 0; j < numberOfTestImages[i]; j++)
        {
            std::stringstream imagePath;
            imagePath << std::string(PROJ_DIR) << "/data/task2/test/" << std::setfill('0') << std::setw(2) << i << "/" << std::setfill('0') << std::setw(4) << j + numberOfTrainImages[i] << ".jpg";
            std::string imagePathStr = imagePath.str();
            //std::cout << imagePathStr << std::endl;
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

cv::HOGDescriptor RandomForest::createHogDescriptor(cv::Size size = cv::Size(128, 128)) {
    //cv::Size wsize(128, 128);
    cv::Size wsize = size;
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
        cv::Mat resizedInputImage = resizeToBoundingBox(inputImage, cv::Size());

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

cv::Mat RandomForest::resizeToBoundingBox(cv::Mat &inputImage, cv::Size size) {
    mWinSize = size;
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
            RandomRotateImage(rotatedImage, randomlyRotatedImage, 90, 30, 30, cv::Rect(0, 0, 0, 0), rng);
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
    randomForest->mHogDescriptor = randomForest->createHogDescriptor(cv::Size());
    long unsigned int timestamp = static_cast<long unsigned int>(time(0));
    std::cout << timestamp << std::endl;
    randomForest->mRandomGenerator = std::mt19937(timestamp);
    return randomForest;
}

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

void RandomForest::ImageRotate(const cv::Mat& src, cv::Mat& dst, float yaw, float pitch, float roll,
                 float Z = 1000, int interpolation = cv::INTER_LINEAR, int boarder_mode = cv::BORDER_CONSTANT, const cv::Scalar& border_color = cv::Scalar(0, 0, 0))
{
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
    invPerspMat.at<double>(0, 2) = -(double)src.cols / 2;
    invPerspMat.at<double>(1, 2) = -(double)src.rows / 2;

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

void RandomForest::composeExternalMatrix(float yaw, float pitch, float roll, float trans_x, float trans_y, float trans_z, cv::Mat& external_matrix)
{
    external_matrix.release();
    external_matrix.create(3, 4, CV_64FC1);

    double sin_yaw = sin((double)yaw * CV_PI / 180);
    double cos_yaw = cos((double)yaw * CV_PI / 180);
    double sin_pitch = sin((double)pitch * CV_PI / 180);
    double cos_pitch = cos((double)pitch * CV_PI / 180);
    double sin_roll = sin((double)roll * CV_PI / 180);
    double cos_roll = cos((double)roll * CV_PI / 180);

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


cv::Mat RandomForest::Rect2Mat(const cv::Rect& img_rect)
{
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

void RandomForest::CircumTransImgRect(const cv::Size& img_size, const cv::Mat& transM, cv::Rect_<double>& CircumRect)
{
    cv::Mat cornersMat = Rect2Mat(cv::Rect(0, 0, img_size.width, img_size.height));

    cv::Mat dstCoord = transM * cornersMat;
    double min_x = std::min(dstCoord.at<double>(0, 0) / dstCoord.at<double>(2, 0), dstCoord.at<double>(0, 3) / dstCoord.at<double>(2, 3));
    double max_x = std::max(dstCoord.at<double>(0, 1) / dstCoord.at<double>(2, 1), dstCoord.at<double>(0, 2) / dstCoord.at<double>(2, 2));
    double min_y = std::min(dstCoord.at<double>(1, 0) / dstCoord.at<double>(2, 0), dstCoord.at<double>(1, 1) / dstCoord.at<double>(2, 1));
    double max_y = std::max(dstCoord.at<double>(1, 2) / dstCoord.at<double>(2, 2), dstCoord.at<double>(1, 3) / dstCoord.at<double>(2, 3));

    CircumRect.x = min_x;
    CircumRect.y = min_y;
    CircumRect.width = max_x - min_x;
    CircumRect.height = max_y - min_y;
}

void RandomForest::CreateMap(const cv::Size& src_size, const cv::Rect_<double>& dst_rect, const cv::Mat& transMat, cv::Mat& map_x, cv::Mat& map_y)
{
    map_x.create(dst_rect.size(), CV_32FC1);
    map_y.create(dst_rect.size(), CV_32FC1);

    double Z = transMat.at<double>(2, 3);

    cv::Mat invTransMat = transMat.inv();	// �t�s��
    cv::Mat dst_pos(3, 1, CV_64FC1);	// �o�͉摜��̍��W
    dst_pos.at<double>(2, 0) = Z;
    for (int dy = 0; dy<map_x.rows; dy++){
        dst_pos.at<double>(1, 0) = dst_rect.y + dy;
        for (int dx = 0; dx<map_x.cols; dx++){
            dst_pos.at<double>(0, 0) = dst_rect.x + dx;
            cv::Mat rMat = -invTransMat(cv::Rect(3, 2, 1, 1)) / (invTransMat(cv::Rect(0, 2, 3, 1)) * dst_pos);
            cv::Mat src_pos = invTransMat(cv::Rect(0, 0, 3, 2)) * dst_pos * rMat + invTransMat(cv::Rect(3, 0, 1, 2));
            map_x.at<float>(dy, dx) = src_pos.at<double>(0, 0) + (float)src_size.width / 2;
            map_y.at<float>(dy, dx) = src_pos.at<double>(1, 0) + (float)src_size.height / 2;
        }
    }
}

// Keep center and expand rectangle for rotation
cv::Rect RandomForest::ExpandRectForRotate(const cv::Rect& area)
{
    cv::Rect exp_rect;

    int w = cvRound(std::sqrt((double)(area.width * area.width + area.height * area.height)));

    exp_rect.width = w;
    exp_rect.height = w;
    exp_rect.x = area.x - (exp_rect.width - area.width) / 2;
    exp_rect.y = area.y - (exp_rect.height - area.height) / 2;

    return exp_rect;
}

cv::Ptr<RandomForest> RandomForest::create(int numberOfClasses, int numberOfDTrees, cv::Size winSize) {
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
