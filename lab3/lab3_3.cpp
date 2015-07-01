#include <iostream>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <cmath>
#include <boost/algorithm/string.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

#include <pthread.h>
#include <ctime>

#define DEBUG 0

// Working well

using namespace cv;
using namespace std;

Mat stereoL, stereoR, undistL, undistR;

vector<Point2f> leftPoints, rightPoints;

Mat readIntrinsicFile(string fileName){
	ifstream File;
	File.open(fileName.c_str());
	string line;
	getline (File,line);
	istringstream iss(line);
	vector<string> fields;
	split(fields, line, boost::is_any_of(" ") );

	Mat intrinsics = Mat::ones(3,3, CV_64F);

	for(int i = 0; i < 9; i++){
		intrinsics.at<double>(i / 3, i %3) = (double)atof(fields[i].c_str());
	}
	return intrinsics;
}

Mat readDistortionFile(string fileName){
	ifstream File;
	File.open(fileName.c_str());
	string line;
	getline (File,line);
	istringstream iss(line);
	vector<string> fields;
	split(fields, line, boost::is_any_of(" ") );

	Mat distort = Mat::ones(5,1, CV_64F);

	for(int i = 0; i < 5; i++){
		distort.at<double>(i) = (double)atof(fields[i].c_str());
	}
	return distort;
}

void readStereoParams(string fileName, Mat R, Mat T, Mat E, Mat F){
	vector<Mat> values; 
	ifstream File;
	File.open(fileName.c_str());
	string line;

	getline (File,line);
	istringstream iss(line);
	vector<string> fields;
	split(fields, line, boost::is_any_of(" ") );
	for(int i = 0; i < 9; i++){
		R.at<double>(i / 3, i %3) = (double)atof(fields[i].c_str());
	}

	getline (File,line);
	iss.clear();
	iss.str(line);
	fields.clear();
	split(fields, line, boost::is_any_of(" ") );
	for(int i = 0; i < 3; i++){
		T.at<double>(i) = (double)atof(fields[i].c_str());
	}

	getline (File,line);
	iss.clear();
	iss.str(line);
	fields.clear();
	split(fields, line, boost::is_any_of(" ") );
	for(int i = 0; i < fields.size(); i++){
		E.at<double>(i / 3, i %3) = (double)atof(fields[i].c_str());
	}

	getline (File,line);
	iss.clear();
	iss.str(line);
	fields.clear();
	split(fields, line, boost::is_any_of(" ") );
	for(int i = 0; i < fields.size(); i++){
		F.at<double>(i / 3, i %3) = (double)atof(fields[i].c_str());
	}
}
/*
void mouseHandlerL(int event, int x, int y, int flags, void* param)
{
    // user press left button 
    if (event == CV_EVENT_LBUTTONDOWN )
    {
        Point2f point = Point2f(x, y);
		cout << x << "  " << y << endl;
		circle(undistL, point, 3, Scalar(35, 36, 200), 1, 1);
		imshow("left", undistL);
		leftPoints.push_back(point);
    }
}

void mouseHandlerR(int event, int x, int y, int flags, void* param)
{
    // user press left button
    if (event == CV_EVENT_LBUTTONDOWN )
    {
        Point2f point = Point2f(x, y);
		cout << x << "  " << y << endl;
		circle(undistR, point, 3, Scalar(35, 36, 200), 1, 1);
		imshow("right", undistR);
		rightPoints.push_back(point);
    }
}
*/

int main(int argc, char *argv[])
{
	Mat T = Mat(3,1, CV_64FC1);
	Mat R = Mat(3,3, CV_64FC1);
	Mat E = Mat(3,3, CV_64FC1);
	Mat F = Mat(3,3, CV_64FC1);
	if (DEBUG)
		readStereoParams("stereoParamsNew.txt", R, T, E, F);
	else
		readStereoParams("stereoParams.txt", R, T, E, F);
	// Read in the R, T, E, and F matrices from the stereoCalibrate function.
	cout << "R = " << R << endl << "T = " <<  T << endl << "E = " << E << endl << "F = " << F << endl;

	namedWindow("left", WINDOW_NORMAL);
	namedWindow("right", WINDOW_NORMAL);

	Mat camMat1 = readIntrinsicFile("lIntrinsic.txt");
	Mat camMat2 = readIntrinsicFile("rIntrinsic.txt");
	Mat distMat1 = readDistortionFile("lDistortion.txt");
	Mat distMat2 = readDistortionFile("rDistortion.txt");

	cout << camMat1 << endl << camMat2 << endl << distMat1 << endl << distMat2 << endl;

	stereoL = imread("new_images/stereo1L0.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	stereoR = imread("new_images/stereo1R0.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	Size boardSize(10, 7);
	vector<Point2f> cornersL, cornersR;
	TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1);
	findChessboardCorners(stereoL, boardSize, cornersL, CV_CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
	cornerSubPix(stereoL, cornersL, Size(11,11), Size(-1, -1), criteria);
	findChessboardCorners(stereoR, boardSize, cornersR, CV_CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
	cornerSubPix(stereoR, cornersR, Size(11,11), Size(-1, -1), criteria);

	cvtColor(stereoL, stereoL, CV_GRAY2BGR);
	cvtColor(stereoR, stereoR, CV_GRAY2BGR);

	undistort(stereoL, undistL, camMat1, distMat1);
	undistort(stereoR, undistR, camMat2, distMat2);

	Mat diffL, diffR;
	absdiff(undistL, stereoL, diffL);
	absdiff(undistR, stereoR, diffR);

	imshow("right", diffR);
	imshow("left", diffL);

	waitKey(0);
// Get points from the image in order to draw epipolar lines through them.
//	cvSetMouseCallback("left", mouseHandlerL, NULL);
//	cvSetMouseCallback("right", mouseHandlerR, NULL);

	for (int i = 0; i < 7; i++){
		rightPoints.push_back(cornersR[i * 9 + 7]);
		circle(undistR, cornersR[i * 9 + 7], 7, Scalar(35, 36, 200), 1, 1);
		leftPoints.push_back(cornersL[i * 9 + 7]);
		circle(undistL, cornersL[i * 9 + 7], 7, Scalar(35, 36, 200), 1, 1);
	}

	cout << "Hello..." << endl;

	Mat leftLines = Mat(3,1, CV_32FC1);
	Mat rightLines = Mat(3,1, CV_32FC1);
	computeCorrespondEpilines(leftPoints, 1, F, rightLines);
	computeCorrespondEpilines(rightPoints, 2, F, leftLines);

	//cout << "Left " << leftLines << endl << "Right " << rightLines << endl;

// Compute lines going across the whole image corresponding to the epipolar lines
	for (int i = 0; i < leftPoints.size(); i++){
		float a = leftLines.at<float>(i,0);
		float b = leftLines.at<float>(i,1);
		float c = leftLines.at<float>(i,2);
		int y1 = (int)(-1 * c / b);
		int y2 = (int)(-1 * a / b * 640 - c / b); // At 640
		//cout << a << " " << b << " " << c << " " << y1 << " " << y2 << endl;
		Point pt1(0, y1);
		Point pt2(640, y2);
		line(undistL, pt1, pt2, Scalar(100, 245, 60), 2);
	}

	for (int i = 0; i < rightPoints.size(); i++){
		float a = rightLines.at<float>(i,0);
		float b = rightLines.at<float>(i,1);
		float c = rightLines.at<float>(i,2);
		int y1 = (int)(-1 * c / b);
		int y2 = (int)(-1 * a / b * 640 - c / b); // At 640
		//cout << a << " " << b << " " << c << " " << y1 << " " << y2 << endl;
		Point pt1(0, y1);
		Point pt2(640, y2);
		line(undistR, pt1, pt2, Scalar(100, 245, 60), 2);
	}

	cout << "Drawing lines...\n";

	imshow("right", undistR);
	imshow("left", undistL);

	imwrite("leftEpi.jpg", undistL);
	imwrite("rightEpi.jpg", undistR);
	waitKey(0);

    return 0;
}
