#include <iostream>
#include <boost/algorithm/string.hpp>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace boost;
using namespace cv;

int main(){
	// Read in intrinsic file
	ifstream intrinsicFile;
	intrinsicFile.open("intrinsic.txt");
	string line;
	getline (intrinsicFile,line);
	istringstream iss(line);
	vector<string> fields;
	split(fields, line, boost::is_any_of(" ") );

	Mat intrinsics = Mat::ones(3,3, CV_64F);

	for(int i = 0; i < fields.size(); i++){
		intrinsics.at<double>(i / 3, i %3) = (double)atof(fields[i].c_str());
	}

	cout << intrinsics << endl;
	//Read in distortion coefficient file
	ifstream distortionFile;
	distortionFile.open("distortion.txt");
	getline (distortionFile,line);
	istringstream issDist(line);
	split(fields, line, boost::is_any_of(" ") );

	Mat distCoeffs = Mat::ones(5,1,CV_64F);

	for(int i = 0; i < fields.size(); i++){
		distCoeffs.at<double>(i) = (double)atof(fields[i].c_str());
	}

	// Read in the three images
	Mat far, close, turned;
	far    = imread("Far.jpg",    CV_LOAD_IMAGE_GRAYSCALE);
	close  = imread("Close.jpg",  CV_LOAD_IMAGE_GRAYSCALE);
	turned = imread("Turned.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	Mat far_un, close_un, turned_un;

	undistort(far, far_un, intrinsics.clone(), distCoeffs.clone());
	undistort(close, close_un, intrinsics.clone(), distCoeffs.clone());
	undistort(turned, turned_un, intrinsics.clone(), distCoeffs.clone());

	namedWindow("Far", WINDOW_NORMAL);
	namedWindow("Near", WINDOW_NORMAL);
	namedWindow("Turn", WINDOW_NORMAL);


	Mat far_diff, close_diff, turned_diff;

	absdiff(far_un, far, far_diff);
	absdiff(close_un, close, close_diff);
	absdiff(turned_un, turned, turned_diff);

	imshow("Far", far_diff);
	imshow("Near", close_diff);
	imshow("Turn", turned_diff);

	imwrite("far_diff.jpg", far_diff);
	imwrite("close_diff.jpg", close_diff);
	imwrite("turned_diff.jpg", turned_diff);

	waitKey(0);

}
