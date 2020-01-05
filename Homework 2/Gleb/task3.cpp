
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "hog_visualization.hpp"
#include "RandomForest.h"
#include <iostream>
#include "features.h"
#include <list>
#include <vector>
#include <fstream>
#include <assert.h>

typedef std::vector<std::pair<int, std::vector<float>>> DataSet;
namespace fs = std::filesystem;
using namespace cv;

/**
 * Square bounding box.
 * bool overlaps(boundingBox) checks whether another boundingbox overlaps
 * int intersection_area(boundingBox) area of intersection (number of pixels in both boxes)
 * int union_area(boundingBox) union area (number of pixels in either box)
 *
 * (x,y)   : pixel coordinates of upper left hand corner
 * w       : width & height of box
 * class_id: detected object class in box
 * confidence : confidence of forest during detection
 *
 */
struct boundingBox {
    int x, y, w;
    int class_id;
    float confidence;

    bool overlaps(const boundingBox &o) const {
        return overlaps1d(x, x + w, o.x, o.x + o.w) && overlaps1d(y, y + w, o.y, o.y + o.w);
    }

    int intersection_area(const boundingBox &o) const {
        return intersection1d(x, x + w, o.x, o.x + o.w) * intersection1d(y, y + w, o.y, o.y + o.w);
    }

    int union_area(const boundingBox &o) const {
        return w * w + o.w * o.w - intersection_area(o);
    }

    bool operator==(const boundingBox &o) const {
        return x == o.x && y == o.y && w == o.w && class_id == o.class_id && confidence == o.confidence;
    }

private:
    static bool overlaps1d(int L1, int R1, int L2, int R2) {
        if (L1 > L2)
            return overlaps1d(L2, R2, L1, R1);
        return L2 < R1;
    }

    static int intersection1d(int L1, int R1, int L2, int R2) {
        if (L1 > L2)
            return intersection1d(L2, R2, L1, R1);
        int diff = R1 - L2;
        return diff > 0 ? diff : 0;
    }
};

std::list<boundingBox> filter_proposals(std::list<boundingBox> proposals, float thresh);

void draw_result(std::list<boundingBox> boxes, Mat &img, bool saveVis, std::string filename);

/**
 * Take frame and produce first guesses for bounding boxes of objects in frame
 * Use square sliding window of different sizes
 * @param img frame to be detected on
 * @param forest trained randomForest classifier
 * @param confidence_thresh discard all detected objectes with lower confidence
 * @param stepsize stepsize of sliding window
 * @param background_class class id of background
 * @param sizes sizes of the sliding window to be tried
 * @return list of proposed bounding boxes
 */
std::list<boundingBox>
find_boundingbox_proposals(const Mat &img, const RandomForest &forest,
                           int stepsize = 8, int background_class = 3,
                           const std::vector<int> sizes = {172,148, 136, 124, 115, 112, 105, 100, 95, 90, 85}) {
    std::list<boundingBox> proposals;
    for (const int &size : sizes) {
        for (int y = 0; y < img.rows - size; y += stepsize) {
            for (int x = 0; x < img.cols - size; x += stepsize) {
                Mat cutout = img.colRange(x, x + size).rowRange(y, y + size);
                std::vector<float> feats = computeHOG(cutout);

                float confidence = 0;
                int result = forest.predict(Mat(feats), &confidence);
                if (result != background_class) {
                    proposals.push_back(boundingBox{x, y, size, result, confidence});
                }
            }
        }
    }
    return proposals;
}

void validate_result(const fs::path &sample_path, const std::list<boundingBox> &found_boxes, float cur_thresh,
                     float correctly_classified_threshold, const std::string &folder) {
    //do validation here
    // Literally do what it says on the exercise sheet. Remember multiple minimum overlaps oder so was wollten die ja
    // wegen dieser PR kurve
    // Any good plotting options for c++ ? If not -> dump it to stdout and use matplot lib in python or whatever
    fs::path gt_path = fs::path("data/task3/gt/" + sample_path.stem().string() + ".gt.txt");

    std::list<boundingBox> true_boxes;
    std::ifstream infile(gt_path.string());
    int i, x1, y1, x2, y2;
    while (infile >> i >> x1 >> y1 >> x2 >> y2) {
        assert(x2 - x1 == y2 - y1);
        assert(x2 - x1 > 0);
        true_boxes.push_back(boundingBox{x1, y1, x2 - x1, i, 1.0f});
    }
    infile.close();
    fs::create_directory(fs::path("PR_curves"));
    fs::create_directory("PR_curves/" + folder);
    std::ostringstream str;
    str << "PR_curves/" << folder << "/PR_curve_" << sample_path.stem().string() << ".csv";
    std::ofstream csv(str.str(), std::ios_base::app);

    // this will not give correct results if a found bbox can be associated with multiple gt bboxes
    std::vector<float> ious;
    for (const boundingBox &gt : true_boxes) {
        float max_iou = 0;
        for (const boundingBox &found : found_boxes) {
            if (gt.class_id != found.class_id)
                continue;
            float iou = found.intersection_area(gt);
            iou /= found.union_area(gt);
            if (max_iou < iou)
                max_iou = iou;
        }
        if (max_iou > 0)
            ious.push_back(max_iou);
    }

    int correctly_identified = 0;
    for (float iou : ious)
        if (iou > correctly_classified_threshold)
            correctly_identified++;
    float precision = correctly_identified, recall = correctly_identified;
    precision /= found_boxes.size();
    recall /= true_boxes.size();
    csv << precision << ',' << recall << '\n';
    csv.close();
}


std::list<boundingBox>
apply_nonmaximum_suppression(std::list<boundingBox> proposal, int overlap_threshold, bool binary_overlap) {
    //Do NMS here
    // Take proposals, find overlaps. In each overlapping group only keep the one with the highest confidence.
    // This can probably be done in a naive way as runtime isnt really an issue here
    proposal.sort([](const boundingBox &lhs, const boundingBox &rhs) { return lhs.confidence >= rhs.confidence; });
    std::list<boundingBox> result;
    while (!proposal.empty()) {
        boundingBox current = proposal.front();
        proposal.pop_front();
        result.push_back(current);
        std::list<boundingBox> keep_boxes;
        for (const auto &box : proposal) {
            if (box.intersection_area(current) <= (binary_overlap ? 0 : current.w * current.w / overlap_threshold)) {
                keep_boxes.push_back(box);
            }
        }
        proposal = keep_boxes;
    }

    return result;

}

// ./task3 --size forestSize --load [PATH_TO_RESTORE_FROM] --save [PATH_TO_SAVE_TO] --augment false
int main(int argc, char *argv[]) {
    int forestSize = 50;
    bool shouldSave = false;
    bool shouldLoad = false;
    bool shouldAugment = false;
    bool shouldVisualize = false;
    bool shouldbinarynms = false;
    bool saveVis = false;
    int nmsthresh = 12;
    int stepsize = 6;
    std::vector<int> window_sizes = { 116, 108, 95, 84};

    fs::path savePath;
    fs::path loadPath;
    for (int i = 1; i < argc; i += 2) {
        if (std::string(argv[i]) == "--size") {
            forestSize = std::stoi(argv[i + 1]);
        } else if (std::string(argv[i]) == "--save") {
            shouldSave = true;
            savePath = fs::path(argv[i + 1]);
        } else if (std::string(argv[i]) == "--load") {
            shouldLoad = true;
            loadPath = fs::path(argv[i + 1]);
        } else if (std::string(argv[i]) == "--augment") {
            shouldAugment = std::string(argv[i + 1]) == "true";
        } else if (std::string(argv[i]) == "--visualize") {
            shouldVisualize = std::string(argv[i + 1]) == "true";
        } else if (std::string(argv[i]) == "--save_vis") {
            saveVis = std::string(argv[i + 1]) == "true";
        } else if (std::string(argv[i]) == "--nms_thresh") {
            nmsthresh = std::stoi(argv[i + 1]);
            shouldbinarynms = nmsthresh == 0;
        } else if (std::string(argv[i]) == "--slide_step") {
            stepsize = std::stoi(argv[i + 1]);
        } else if (std::string(argv[i]) == "--win_sizes") {
            int no_wins = std::stoi(argv[i + 1]);
            window_sizes.clear();
            for (int j = 0; j < no_wins; j++) {
                window_sizes.push_back(std::stoi(argv[i + j + 1]));
            }
            i += no_wins;
        }
    }


    //Step 1: Augment test data
    auto train_path = fs::path("data/task3/train");
    std::cout << "Read images and extract HOG Descriptors" << std::endl;
    DataSet training_set;
    compute_data_set(training_set, train_path, shouldAugment);
    auto train_data = make_opencv_dataset(training_set);
    RandomForest forest;
    if (!shouldLoad) {
        std::cout << "Training forest of size " << forestSize << std::endl;
        forest = RandomForest(forestSize);
        forest.train(train_data);
        std::cout << "Done!\n" << std::endl;
        if (shouldSave) {
            std::cout << "Saving forest at " << savePath << std::endl;
            forest.save(savePath);
        }
    } else {
        forest = RandomForest(loadPath);
        forestSize = static_cast<int>(forest.getSize());
        std::cout << "Restored forest of size " << forest.getSize() << " from " << loadPath << std::endl;
    }
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::string foldername = "prcurves_" + std::to_string(forestSize) + "_" + std::ctime(&now);
    for (const auto &sample : fs::directory_iterator(fs::path("data/task3/test"))) {
        std::cout << "Running detection on frame " << sample.path().filename() << "\r" << std::flush;
        Mat image = imread(sample.path().string());
        auto proposals = find_boundingbox_proposals(image, forest, stepsize, 3, window_sizes);
        for (int thresh = 40; thresh < 100; thresh += 2) {
            auto final_boxes = apply_nonmaximum_suppression(filter_proposals(proposals, thresh / 100.0f), nmsthresh,
                                                            shouldbinarynms);
            if (shouldVisualize && thresh == 60)
                draw_result(final_boxes, image, saveVis, foldername.substr(24,8)+sample.path().filename().string());
            validate_result(sample, final_boxes, thresh, 0.5, foldername);
        }
    }
    std::cout << std::endl;


    return EXIT_SUCCESS;
}

std::list<boundingBox> filter_proposals(std::list<boundingBox> proposals, float thresh) {
    std::list<boundingBox> result;
    for (const auto &box : proposals) {
        if (box.confidence >= thresh) {
            result.push_back(box);
        }
    }
    return result;
}

void draw_result(std::list<boundingBox> boxes, Mat &img, bool saveVis, std::string filename = "") {
    for (const auto &box : boxes) {
        Scalar color;
        switch (box.class_id) {
            case 0:
                color = Scalar(155 + box.confidence * 100, 0, 0);
                break;
            case 1:
                color = Scalar(0, 70 + box.confidence * 100, 0);
                break;
            case 2:
                color = Scalar(0, 0, 70 + box.confidence * 100);
                break;
            default:
                break;
        }
        rectangle(img, Point(box.x, box.y), Point(box.x + box.w, box.y + box.w), color);
        putText(img, std::to_string(box.class_id) + "  : c 0." + std::to_string(int(box.confidence * 100)),
                Point(std::max(box.x - 2, 0), std::max(box.y - 2, 0)), FONT_HERSHEY_SIMPLEX, 0.3, color);
    }
    fs::create_directory(fs::path("out"));
    if (saveVis) { imwrite("out/" + filename, img); }
    else {
        imshow("result", img);
        waitKey(0);
    }
}



