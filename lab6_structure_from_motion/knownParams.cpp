#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <string>
#include <sstream>
#include <vector>
#include <iostream> 
#include <fstream>

using namespace std;
using namespace cv;

#define CLOSEST_DISTANCE 20

int main(int argc, char** argv){
	// Read in image 10 of the cube
	Mat paraCube10 = imread("parallel_cube/ParallelCube10.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat turnCube10 = imread("turned_cube/TurnCube10.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	
	Mat corners; // 23 * 11
	bool pattern = findChessboardCorners(paraCube10, Size(11, 11), corners);
	TermCriteria criteria = TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 );
	cornerSubPix(paraCube10, corners, Size(10,10), Size(-1, -1), criteria);
	cvtColor(paraCube10, paraCube10, CV_GRAY2BGR);
	drawChessboardCorners(paraCube10, cvSize(23,11), corners, pattern);
	/*
	pattern = findChessboardCorners(turnCube10, Size(11, 11), corners);
	cornerSubPix(turnCube10, corners, Size(10,10), Size(-1, -1), criteria);
	cvtColor(turnCube10, turnCube10, CV_GRAY2BGR);
	drawChessboardCorners(turnCube10, cvSize(23,11), corners, pattern);
	*/
	imshow("lol", paraCube10);
	waitKey(0);
	imshow("lol2", turnCube10);
	waitKey(0);
}
