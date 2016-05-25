/*
 * 
 * 
 */
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main(int argc, char* argv[]) {

  if (argc < 3) {
    cout << "Usage: file_1 file_2" << endl;
    exit(0);
  }

    Mat img_1, img_2;
//    Mat img_1c = imread("/scratch/CVDP/foo.tif");
//    Mat img_2c = imread("/scratch/CVDP/foo.tif");
    Mat img_1c = imread(argv[1]);
    Mat img_2c = imread(argv[2]);
    cvtColor(img_1c, img_1, CV_BGR2GRAY);
    cvtColor(img_2c, img_2, CV_BGR2GRAY);

    vector<KeyPoint> keypoints_1;
    vector<KeyPoint> keypoints_2;
    //SurfFeatureDetector surf(2.50e3);
    Ptr<SURF> surf = SURF::create(2.50e2);
    surf->detect(img_1, keypoints_1);
    surf->detect(img_2, keypoints_2);

    cv::Mat descriptors_1, descriptors_2;
    //compute descriptors
    surf->compute(img_1, keypoints_1, descriptors_1);
    surf->compute(img_2, keypoints_2, descriptors_2);

    //use brute force method to match vectors
    BFMatcher* matcher = new BFMatcher(NORM_L1);
    vector<DMatch>matches;
    matcher->match(descriptors_1, descriptors_2, matches);

    //draw results
    Mat img_matches;
    drawMatches(img_1c, keypoints_1, img_2c, keypoints_2, matches, img_matches);
    imshow("surf_Matches", img_matches);

    int keyCode;
    do {
        keyCode = waitKey(0);
        cout << keyCode << endl;
    } while (keyCode != 113); // "q"

}
