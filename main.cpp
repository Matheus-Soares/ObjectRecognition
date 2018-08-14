//1a
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"

#include <iostream>
#include <stdio.h>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

char *camWindow = (char*) "Camera Image";
char *testWindow = (char*) "Test Image";

int num = 0;

vector<KeyPoint> keypointsRef, keypointsCam;

Mat imgRef, img, imgDraw, imgRefGray, imgGray, descriptorsRef, descriptorsCam;

VideoCapture cap(0);

void sel_method();
void draw_object(Mat &img, vector <DMatch> &Matches);
void flann_matcher(Mat &frame);
void cam();
void draw(Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors);
void sift_method(Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors);
void surf_method(Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors);
void orb_method(Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors);
void kaze_method(Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors);
void akaze_method(Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors);


int main(int argc, char** argv){

    if(!cap.isOpened()){
        printf("Couldn't open camera\n");
        return -1;
    }

    printf("Usage: ./Feature_detection ../../../Images/rl/22.jpg ../../../Images/rl/0.3.jpg");

    namedWindow(camWindow, WINDOW_NORMAL);
    namedWindow(testWindow, WINDOW_NORMAL);

    imgRef = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    img = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    cvtColor(imgRef, imgRefGray, COLOR_RGB2GRAY);
    cvtColor(img, imgGray, COLOR_RGB2GRAY);

    sel_method();

}

void sel_method(){

    while (true){
        int key = waitKey();

        switch (key) {
            case 49:                //Key = 1;
                num = 1;
                cout << "Sift" << endl;

                sift_method(imgRefGray, keypointsRef, descriptorsRef);
                sift_method(imgGray, keypointsCam, descriptorsCam);
                flann_matcher(imgGray);

                //draw(imgRef, keypointsRef, descriptorsRef);
                //cam();
                break;

            case 50:                //Key = 2;
                num = 2;
                cout << "Surf" << endl;

                surf_method(imgRefGray, keypointsRef, descriptorsRef);
                surf_method(imgGray, keypointsCam, descriptorsCam);
                flann_matcher(imgGray);

                //draw(imgRef, keypointsRef, descriptorsRef);
                //cam();
                break;

            case 51:                //Key = 3;
                num = 3;
                cout << "Orb" << endl;

                orb_method(imgRefGray, keypointsRef, descriptorsRef);
                orb_method(imgGray, keypointsCam, descriptorsCam);
                flann_matcher(imgGray);

                //draw(imgRef, keypointsRef, descriptorsRef);
                //cam();
                break;

            case 52:                 //key = 4;
                num = 4;
                cout << "Kaze" << endl;

                kaze_method(imgRefGray, keypointsRef, descriptorsRef);
                kaze_method(imgGray, keypointsCam, descriptorsCam);
                flann_matcher(imgGray);

                //draw(imgRef, keypointsRef, descriptorsRef);
                //cam();
                break;

            case 53:                 //key = 5;
                num = 5;
                cout << "Akaze" << endl;

                akaze_method(imgRefGray, keypointsRef, descriptorsRef);
                akaze_method(imgGray, keypointsCam, descriptorsCam);
                flann_matcher(imgGray);

                //draw(imgRef, keypointsRef, descriptorsRef);
                //cam();
                break;

            case 27:                //key = Esc;
                printf("exit\n");
                destroyAllWindows();
                return;

            default:
                break;
        }
    }
}

void draw_object(Mat &img, vector <DMatch> &Matches){
    vector<Point2f> obj;
    vector<Point2f> scene;
    float x1, x2, y1, y2, delta;
    x1 = img.size().width;
    y1 = img.size().height;
    x2 = y2 = 0;
    delta = 0;

    img.copyTo(imgDraw);

    if (!Matches.empty()) {
        for (size_t i = 0; i < Matches.size(); i++) {
            //-- get the keypoints from the good matches
            scene.push_back(keypointsCam[Matches[i].trainIdx].pt);
        }

        for (int i = 0; i < scene.size(); i++) {
            if (scene[i].x < x1) x1 = scene[i].x;
            if (scene[i].x > x2) x2 = scene[i].x;
            if (scene[i].y < y1) y1 = scene[i].y;
            if (scene[i].y > y2) y2 = scene[i].y;
        }

        vector<Point2f> objCorners(4);
        Point2f d(delta, delta);

        objCorners[0] = Point2f(x1, y1) + d;
        objCorners[1] = Point2f(x1, y2) + d;
        objCorners[2] = Point2f(x2, y2) + d;
        objCorners[3] = Point2f(x2, y1) + d;


        /*Rect Rec(x1, y1, (x2 - x1), (y2 - y1));

        Mat roi = img(Rec);
*/
        //Draw the lines
        line(imgDraw, objCorners[0], objCorners[1], Scalar(0,255,0), 3);
        line(imgDraw, objCorners[1], objCorners[2], Scalar(0,255,0), 3);
        line(imgDraw, objCorners[2], objCorners[3], Scalar(0,255,0), 3);
        line(imgDraw, objCorners[3], objCorners[0], Scalar(0,255,0), 3);


        imshow(testWindow, imgDraw);
    }
    else imshow(testWindow, imgDraw);
}

void flann_matcher(Mat &frame){

    FlannBasedMatcher matcher;
    vector <DMatch> matches;

    if(descriptorsRef.type() != CV_32F) descriptorsRef.convertTo(descriptorsRef, CV_32F);
    if(descriptorsCam.type() != CV_32F) descriptorsCam.convertTo(descriptorsCam, CV_32F);

    matcher.match(descriptorsRef, descriptorsCam, matches);

    double maxDist = 0, minDist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for(int i = 0; i < descriptorsRef.rows; i++){
        double dist = matches[i].distance;
        if(dist < minDist) minDist = dist;
        if(dist > maxDist) maxDist = dist;
    }

    vector <DMatch> goodMatches;

    //-- Select only "good" matches (d < 2*min_dist || d < 0.02)
    for(int i = 0; i < descriptorsRef.rows; i++){
        if(matches[i].distance <= max(2*minDist, 0.02)){
            goodMatches.push_back(matches[i]);
        }
    }

    //-- Draw only "good" matches
    Mat imgMatches;
    drawMatches(imgRefGray, keypointsRef, frame, keypointsCam,
                goodMatches, imgMatches, Scalar(0, 0, 255), Scalar(255, 0, 0),
                vector<char>());

    //-- Show detected matches
    imshow(camWindow, imgMatches);
    draw_object(img, goodMatches);

    cout << "Good matches: " << goodMatches.size() << endl;

    waitKey(0);
}

void cam(){
    Mat frame, imgKeypoints;
    namedWindow(camWindow, WINDOW_NORMAL);

    while(true){

        cap >> frame;

        cvtColor(frame, frame, COLOR_BGR2GRAY);

        switch (num){
            case 1:
                sift_method(frame, keypointsCam, descriptorsCam);
                break;
            case 2:
                surf_method(frame, keypointsCam, descriptorsCam);
                break;
            case 3:
                orb_method(frame, keypointsCam, descriptorsCam);
                break;
            case 4:
                kaze_method(frame, keypointsCam, descriptorsCam);
                break;
            case 5:
                akaze_method(frame, keypointsCam, descriptorsCam);
                break;
            default:
                break;
        }

        flann_matcher(frame);

        if(waitKey(30) >= 0)
            break;

    }
}

void draw(Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors){
    Mat imgKeypoints;

    drawKeypoints(img, keypoints, imgKeypoints, Scalar(0, 255, 0), DrawMatchesFlags::DEFAULT );

    namedWindow(testWindow, WINDOW_NORMAL);

    while(waitKey(30) <= 0)
        imshow(testWindow, imgKeypoints);

    destroyWindow(testWindow);

}

void sift_method(Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors){
    Ptr<SIFT> detector = SIFT::create();

    detector->detectAndCompute(img, Mat(), keypoints, descriptors);

}

void surf_method(Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors){
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create(minHessian);

    detector->detectAndCompute(img, Mat(), keypoints, descriptors);

}

void orb_method(Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors){
    Ptr<ORB> detector = ORB::create();

    detector->detectAndCompute(img, noArray(), keypoints, descriptors);

}

void kaze_method(Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors){
    Ptr<KAZE> detector = KAZE::create();

    detector->detectAndCompute(img, Mat(), keypoints, descriptors);

}

void akaze_method(Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors){
    Ptr<AKAZE> detector = AKAZE::create();

    detector->detectAndCompute(img, noArray(), keypoints, descriptors);

}

