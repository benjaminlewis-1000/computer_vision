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

using namespace cv;
using namespace std;

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
	for(int i = 0; i < 9; i++){
		E.at<double>(i / 3, i %3) = (double)atof(fields[i].c_str());
	}

	getline (File,line);
	iss.clear();
	iss.str(line);
	fields.clear();
	split(fields, line, boost::is_any_of(" ") );
	for(int i = 0; i < 9; i++){
		F.at<double>(i / 3, i %3) = (double)atof(fields[i].c_str());
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

	Mat imageL, imageR;
	imageL = imread("new_images/stereo1L13.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	imageR = imread("new_images/stereo1R13.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	Mat T = Mat(3,1, CV_64FC1);
	Mat R = Mat(3,3, CV_64FC1);
	Mat E = Mat(3,3, CV_64FC1);
	Mat F = Mat(3,3, CV_64FC1);
	if (DEBUG)
		readStereoParams("stereoParamsNew.txt", R, T, E, F);
	else
		readStereoParams("stereoParams.txt", R, T, E, F);
	cout << R << endl << T << endl << E << endl << F << endl;

	imwrite("origRemapL.jpg", imageL);
	imwrite("origRemapR.jpg", imageR);

	Mat outR1, outR2, outP1, outP2, outQ;

	Mat camMat1 = readIntrinsicFile("lIntrinsic.txt");
	Mat camMat2 = readIntrinsicFile("rIntrinsic.txt");
	Mat distMat1 = readDistortionFile("lDistortion.txt");
	Mat distMat2 = readDistortionFile("rDistortion.txt");

	stereoRectify(camMat1, distMat1, camMat2, distMat2, imageL.size(), R, T, outR1, outR2, outP1, outP2, outQ);
	cout << "R1 " << outR1 << endl << "R2 " << outR2 << endl << "P1 " << outP1 << endl << "P2 " << outP2 << endl << "Q " << outQ << endl;
	
	ofstream rectifyParams;
	rectifyParams.open("stereoRectifyParams.txt");
	
	for(int i = 0; i < 3; i++){
		for(int j =0; j < 3; j++){
			rectifyParams << outR1.at<double>(i, j) << " ";
		}
	}
	
	rectifyParams << endl;

	for(int i = 0; i < 3; i++){
		for(int j =0; j < 3; j++){
			rectifyParams << outR2.at<double>(i, j) << " ";
		}
	}
	
	rectifyParams << endl;
	
	for(int i = 0; i < 3; i++){
		for(int j =0; j < 4; j++){
			rectifyParams << outP1.at<double>(i, j) << " ";
		}
	}
	
	rectifyParams << endl;
	
	for(int i = 0; i < 3; i++){
		for(int j =0; j < 4; j++){
			rectifyParams << outP2.at<double>(i, j) << " ";
		}
	}
	
	rectifyParams << endl;
	
	for(int i = 0; i < 4; i++){
		for(int j =0; j < 4; j++){
			rectifyParams << outQ.at<double>(i, j) << " ";
		}
	}

	rectifyParams << endl;

	Mat lOutMap1, lOutMap2, rOutMap1, rOutMap2;

//Left
	initUndistortRectifyMap(camMat1, distMat1, outR1, outP1, imageL.size(), CV_32FC1, lOutMap1, lOutMap2);
//Right
	initUndistortRectifyMap(camMat2, distMat2, outR2, outP2, imageR.size(), CV_32FC1, rOutMap1, rOutMap2);

	Mat remapL, remapR;
	remap(imageL, remapL, lOutMap1, lOutMap2, INTER_CUBIC);
	remap(imageR, remapR, rOutMap1, rOutMap2, INTER_CUBIC);

	namedWindow("Left", CV_WINDOW_NORMAL);
	namedWindow("Right", CV_WINDOW_NORMAL);

	cvtColor(remapL, remapL, CV_GRAY2BGR);
	cvtColor(remapR, remapR, CV_GRAY2BGR);

	imwrite("RemapL.jpg", remapL);
	imwrite("RemapR.jpg", remapR);

	Mat diffL, diffR;
	cvtColor(imageL, imageL, CV_GRAY2BGR);
	absdiff(remapL, imageL, diffL);
	cvtColor(imageR, imageR, CV_GRAY2BGR);
	absdiff(remapR, imageR, diffR);
	imshow("Right", diffR);
	imshow("Left", diffL);
	imwrite("remapDiffR.jpg", diffR);
	imwrite("remapDiffL.jpg", diffL);
	waitKey(0);

	for (int i = 10; i < 640; i+= 30){
		line(remapL, Point(0, i), Point(640, i), Scalar(200, 245, 60), 2);
		line(remapR, Point(0, i), Point(640, i), Scalar(200, 245, 60), 2);
	}

	imshow("Left", remapL);
	imshow("Right", remapR);

	imwrite("linesRemapL.jpg", remapL);
	imwrite("linesRemapR.jpg", remapR);

	waitKey(0);
	
}
