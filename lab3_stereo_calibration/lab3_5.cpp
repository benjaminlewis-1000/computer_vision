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
#include <math.h>

using namespace cv;
using namespace std;

vector<Point2f> leftPoints, rightPoints;
Mat imageL, imageR;

void mouseHandlerL(int event, int x, int y, int flags, void* param)
{
    /* user press left button */
    if (event == CV_EVENT_LBUTTONDOWN )
    {
        Point2f point = Point2f(x, y);
		cout << x << "  " << y << endl;
		circle(imageL, point, 3, Scalar(35, 36, 200), 1, 1);
		imshow("left", imageL);
		leftPoints.push_back(point);
    }
}

void mouseHandlerR(int event, int x, int y, int flags, void* param)
{
    /* user press left button */
    if (event == CV_EVENT_LBUTTONDOWN )
    {
        Point2f point = Point2f(x, y);
		cout << x << "  " << y << endl;
		circle(imageR, point, 3, Scalar(35, 36, 200), 1, 1);
		imshow("right", imageR);
		rightPoints.push_back(point);
    }
}


void readRectifyParams(string fileName, Mat R1, Mat R2, Mat P1, Mat P2, Mat Q){
	ifstream File;
	File.open(fileName.c_str());
	string line;

	getline (File,line);
	istringstream iss(line);
	vector<string> fields;
	split(fields, line, boost::is_any_of(" ") );
	for(int i = 0; i < 9; i++){
		R1.at<double>(i / 3, i % 3) = (double)atof(fields[i].c_str());
	}
//cout << R1 << endl;
	getline (File,line);
	iss.clear();
	iss.str(line);
	fields.clear();
	split(fields, line, boost::is_any_of(" ") );
	for(int i = 0; i < 9; i++){
		R2.at<double>(i / 3, i % 3) = (double)atof(fields[i].c_str());
	}

	getline (File,line);
	iss.clear();
	iss.str(line);
	fields.clear();
	split(fields, line, boost::is_any_of(" ") );
	for(int i = 0; i < 12; i++){
		P1.at<double>(i / 4, i % 4) = (double)atof(fields[i].c_str());
	}

	getline (File,line);
	iss.clear();
	iss.str(line);
	fields.clear();
	split(fields, line, boost::is_any_of(" ") );
	for(int i = 0; i < 12; i++){
		P2.at<double>(i / 4, i % 4) = (double)atof(fields[i].c_str());
	}
		
	getline (File,line);
	iss.clear();
	iss.str(line);
	fields.clear();
	split(fields, line, boost::is_any_of(" ") );
	for(int i = 0; i < 16; i++){
		Q.at<double>(i / 4, i % 4) = (double)atof(fields[i].c_str());
	}

}

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


int main(int argc, char** argv){

	Mat R1 = Mat(3,3, CV_64FC1);
	Mat R2 = Mat(3,3, CV_64FC1);
	Mat P1 = Mat(3,4, CV_64FC1);
	Mat P2 = Mat(3,4, CV_64FC1);
	Mat Q  = Mat(4,4, CV_64FC1);
	
	readRectifyParams("stereoRectifyParams.txt", R1, R2, P1, P2, Q);

	cout << R1 << endl << R2 << endl << P1 << endl << P2 << endl << Q << endl;

	//Mat imageL, imageR;
	imageL = imread("new_images/stereo1L0.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	imageR = imread("new_images/stereo1R0.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	
/* Finding chessboard corners */
// Find chessboard inner corners, 10 corners per row/7 per column
	int numCornersHorizontal = 10;
	int numCornersVertical = 7;
	int numSquares = numCornersHorizontal * numCornersVertical;
	Size boardSize = Size(numCornersHorizontal, numCornersVertical);
	
	vector<Point2f> lPointBuf, rPointBuf;

	findChessboardCorners(imageL, boardSize, lPointBuf, CV_CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
	TermCriteria criteria = TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1 );
	cornerSubPix(imageL, lPointBuf, Size(11,11), Size(-1, -1), criteria);
	
	findChessboardCorners(imageR, boardSize, rPointBuf, CV_CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
	cornerSubPix(imageR, rPointBuf, Size(11,11), Size(-1, -1), criteria);

/* Finished finding corners  */ 

	namedWindow("left", CV_WINDOW_NORMAL);
	namedWindow("right", CV_WINDOW_NORMAL);
	cvtColor(imageL, imageL, CV_GRAY2BGR);
	cvtColor(imageR, imageR, CV_GRAY2BGR);

	Mat camMat1 = readIntrinsicFile("lIntrinsic.txt");
	Mat camMat2 = readIntrinsicFile("rIntrinsic.txt");
	Mat distMat1 = readDistortionFile("lDistortion.txt");
	Mat distMat2 = readDistortionFile("rDistortion.txt");

	imshow("right", imageR);
	imshow("left", imageL);
	
//	cvSetMouseCallback("left", mouseHandlerL, NULL);
//	cvSetMouseCallback("right", mouseHandlerR, NULL);

	for(int i = 0; i < 6; i++){
		int num;
		if (i < 2) num = i;
		else if (i < 4) num = i + 10;
		else num = i%2 + 24 + 9 * (i % 2);

		num += 4;

		leftPoints.push_back(lPointBuf[num]);
		circle(imageL, lPointBuf[num], 7, Scalar(255, 0, 0), 1, 1);

		rightPoints.push_back(rPointBuf[num]);
		circle(imageR, rPointBuf[num], 7, Scalar(0, 0, 255), 1, 1);
	}
	imshow("right", imageR);
	imshow("left", imageL);

	imwrite("measurementL.jpg", imageL);
	imwrite("measurementR.jpg", imageR);

	waitKey(0);

	vector<Point2f> undistLeftPoints, undistRightPoints;	
	
	undistortPoints(leftPoints, undistLeftPoints, camMat1, distMat1, R1, P1);
	// Use R1 and P1 from stereoRectify
	undistortPoints(rightPoints, undistRightPoints, camMat2, distMat2, R2, P2);

	for(int i; i < undistLeftPoints.size(); i++){
		cout << rightPoints[i] << "  " << undistRightPoints[i] << endl;
	}

	cout << undistLeftPoints.size() << "  " <<  undistRightPoints.size() << endl;
	
	vector<Point3f> left3dPoints, right3dPoints;
	
	for (int i = 0; i < min(undistLeftPoints.size(), undistRightPoints.size()); i++){
		double disparity = undistLeftPoints[i].x - undistRightPoints[i].x;
		Point3f lpt = Point3f(undistLeftPoints[i].x, undistLeftPoints[i].y, disparity);
		Point3f rpt = Point3f(undistRightPoints[i].x, undistRightPoints[i].y, disparity);
		left3dPoints.push_back(lpt);
		right3dPoints.push_back(rpt);
	}

	//for (int i = 0; i < 4; i++){
	//	cout << leftPoints[i] << "   " << undistLeftPoints[i] << endl;
	//}
	
	vector<Point3f> lPerspectiveFromR, rPerspectiveFromL;
	perspectiveTransform(left3dPoints, rPerspectiveFromL, Q.clone());
	perspectiveTransform(right3dPoints, lPerspectiveFromR, Q.clone());

	for (int i = 0; i < 4; i++){
		cout << undistLeftPoints[i].x << "   " << lPerspectiveFromR[i] << rPerspectiveFromL[i] << endl;
	}
	
	for (int i = 0; i < min(undistLeftPoints.size(), undistRightPoints.size()) - 1; i++){
		double xdistance = lPerspectiveFromR[i].x - lPerspectiveFromR[i+1].x;
		double ydistance = lPerspectiveFromR[i].y - lPerspectiveFromR[i+1].y;
		double zdistance = lPerspectiveFromR[i].z - lPerspectiveFromR[i+1].z;
		double distance = sqrt(pow(xdistance, 2) + pow(ydistance, 2) + pow(zdistance, 2));
		cout << "Distance is " << distance << " inches." << endl;
	}

	return 0;

}
