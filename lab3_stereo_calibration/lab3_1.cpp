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

#define NUM_THREADS 2
volatile int running_threads = 0;

using namespace cv;
using namespace std;
using namespace boost;

struct calibration{
	Mat Intrinsic;
	Mat DistCoeffs;
	Size imageSize;
	vector<Mat> rvecs;
	vector<Mat> tvecs;
	vector<vector<Point3f> > object_points;
	vector<vector<Point2f> > image_points;
	string intrinsicFilename;
	string distFilename;
};

Mat readIntrinsicFile(string fileName){
	ifstream File;
	File.open(fileName.c_str());
	string line;
	getline (File,line);
	istringstream iss(line);
	vector<string> fields;
	split(fields, line, boost::is_any_of(" ") );

	Mat intrinsics = Mat(3,3, CV_64F);

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

void *calibrate(void *p){
	cout << "Starting..." << endl;
	running_threads++;
	cout << "Number of threads left is " << running_threads << endl;
	struct timespec start, finish;
	clock_gettime(CLOCK_MONOTONIC, &start);
	calibration* cal = (calibration*)p;
	double rms = calibrateCamera(cal->object_points, cal->image_points, cal->imageSize, cal->Intrinsic, cal->DistCoeffs, cal->rvecs, cal->tvecs);
	std::cout << "Matrix is " << cal->Intrinsic << std::endl << cal->DistCoeffs << std::endl;
	cout << "RMS is " << rms << endl;

	ofstream IntrinsicFile;
	IntrinsicFile.open(cal->intrinsicFilename.c_str());
	ofstream DistortionParams;
	DistortionParams.open(cal->distFilename.c_str());

	for(int i = 0; i < 3; i++){
		for(int j =0; j < 3; j++){
			IntrinsicFile << cal->Intrinsic.at<double>(i, j) << " ";
		}
	}

	for(int k = 0; k < 5; k++){
		DistortionParams << cal->DistCoeffs.at<double>(k) << " ";
	}

	IntrinsicFile.close();
	DistortionParams.close();

	clock_gettime(CLOCK_MONOTONIC, &finish);
	double elapsed_secs = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec)  / CLOCKS_PER_SEC;
	cout << "Time passed was " << elapsed_secs << endl;
	running_threads--;
	cout << "Number of threads left is " << running_threads << endl;
}

///////////////////////////////////////////////////////////////////////////////////


int main(int argc, char** argv){
	pthread_t threads[NUM_THREADS];	
	
	int num_images = 32;
// Set up the size of the chessboard in number of interior corners
	int numCornersHorizontal = 10;
	int numCornersVertical = 7;
	int numSquares = numCornersHorizontal * numCornersVertical;
	Size boardSize(10,7);

	vector<vector<Point3f> > lObject_points;
	vector<vector<Point2f> > lImage_points;

	vector<vector<Point3f> > rObject_points;
	vector<vector<Point2f> > rImage_points;

	Mat lImage, lGray;
	Mat rImage, rGray;
// Push an object with equidistant points, representing the grid of the chessboard. Equivalent of looping two nested
// for loops, the interior loop for the horizontal (second argument) and the vertical in the outer loop (first argument)
// with 0 distance from the camera. 
	vector < Point3f > obj;
	for (int j = 0; j < numCornersHorizontal * numCornersVertical; j++)
		obj.push_back(Point3f(j / numCornersHorizontal * 3.88f, 
				      j % numCornersHorizontal * 3.88f, 0.0f));

	namedWindow("Left", CV_WINDOW_NORMAL);
	int countL = 0;
	int countR = 0; 
	int both = 0;
	vector<Point2f> cornersL, cornersR;

// Read each image and find the subpixel coordinates of the chessboard corners, then push them into a vector. 
	for (int i = 0; i < num_images; i++){
		cout << "On iteration " << i << endl;
		ostringstream leftName, rightName;
		leftName  << "new_images/leftL"  << i << ".bmp";
		rightName << "new_images/rightR" << i << ".bmp";
		string lFilename = leftName.str();
		string rFilename = rightName.str();

		lImage = imread(lFilename, CV_LOAD_IMAGE_COLOR);
		rImage = imread(rFilename, CV_LOAD_IMAGE_COLOR);

		cvtColor(lImage, lGray, CV_BGR2GRAY);
		cvtColor(rImage, rGray, CV_BGR2GRAY);

	// Find chessboard inner corners, 10 corners per row/7 per column
		bool lFound = findChessboardCorners(lGray, boardSize, cornersL, CV_CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
		bool rFound = findChessboardCorners(rGray, boardSize, cornersR, CV_CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);

	//	if(lFound) countL++;
	//	if(rFound) countR++;
		
		TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1);
		if(lFound){
		// Find subpixels with the criteria if both boards have all the corners found. 
			cornerSubPix(lGray, cornersL, Size(11,11), Size(-1, -1), criteria);
		//	cvtColor(lGray, lImage, CV_GRAY2BGR);
			drawChessboardCorners(lImage, cvSize(10,7), cornersL, lFound);
			lImage_points.push_back(cornersL);
			lObject_points.push_back(obj);
		}
		if(rFound){
			cornerSubPix(rGray, cornersR, Size(11,11), Size(-1, -1), criteria);
		//	cvtColor(rGray,rImage, CV_GRAY2BGR);
			drawChessboardCorners(rImage, cvSize(10,7), cornersR, rFound);
			rImage_points.push_back(cornersR);
			rObject_points.push_back(obj);
		}
		//	imshow("Left", rImage);
			//waitKey(1000);
	}

	vector<vector<Point3f> > bothObject_points;
	vector<vector<Point2f> > lBothImage_points;
	vector<vector<Point2f> > rBothImage_points;

	for (int i = 0; i < num_images; i++){
		cout << "On iteration " << i << endl;
		ostringstream leftName, rightName;
		leftName  << "new_images/stereo1L"  << i << ".bmp";
		rightName << "new_images/stereo1R" << i << ".bmp";
		string lFilename = leftName.str();
		string rFilename = rightName.str();

		lImage = imread(lFilename, CV_LOAD_IMAGE_COLOR);
		rImage = imread(rFilename, CV_LOAD_IMAGE_COLOR);

		cvtColor(lImage, lGray, CV_BGR2GRAY);
		cvtColor(rImage, rGray, CV_BGR2GRAY);

	// Find chessboard inner corners, 10 corners per row/7 per column
		bool lFound = findChessboardCorners(lGray, boardSize, cornersL, CV_CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
		bool rFound = findChessboardCorners(rGray, boardSize, cornersR, CV_CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);

		if(lFound) countL++;
		if(rFound) countR++;
		
		TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1);
		if(lFound && rFound){
		// Find subpixels with the criteria if both boards have all the corners found. 
			cornerSubPix(lGray, cornersL, Size(11,11), Size(-1, -1), criteria);
		//	cvtColor(lGray, lImage, CV_GRAY2BGR);
			drawChessboardCorners(lImage, cvSize(10,7), cornersL, lFound);

			cornerSubPix(rGray, cornersR, Size(11,11), Size(-1, -1), criteria);
		//	cvtColor(rGray,rImage, CV_GRAY2BGR);
			drawChessboardCorners(rImage, cvSize(10,7), cornersR, rFound);

			rBothImage_points.push_back(cornersR);
			lBothImage_points.push_back(cornersL);
			bothObject_points.push_back(obj);
		}
		//	imshow("Left", rImage);
			//waitKey(1000);
	}

cout << "Left: " << countL << endl << "Right: " << countR << endl << "Both: " << both << endl;

// Init two structs to pass to the threads. 
	calibration* left  = new calibration();
	calibration* right = new calibration();

	left->Intrinsic = Mat(3,3, CV_32FC1);
	left->imageSize = lImage.size();
	left->object_points = lObject_points;
	left->image_points = lImage_points;
	left->intrinsicFilename = "lIntrinsic.txt";
	left->distFilename = "lDistortion.txt";
	left->Intrinsic.ptr<float>(0)[0] = 1;
	left->Intrinsic.ptr<float>(1)[1] = 1;

	right->Intrinsic = Mat(3,3, CV_32FC1);
	right->imageSize = rImage.size();
	right->object_points = rObject_points;
	right->image_points = rImage_points;
	right->intrinsicFilename = "rIntrinsic.txt";
	right->distFilename = "rDistortion.txt";
	right->Intrinsic.ptr<float>(0)[0] = 1;
	right->Intrinsic.ptr<float>(1)[1] = 1;

	int rc_l = pthread_create(&threads[0], NULL, calibrate, (void*)(left));
	int rc_r = pthread_create(&threads[1], NULL, calibrate, (void*)(right));
// Join the threads so they synchronize or something like that. 
	pthread_join(threads[0], NULL);
	pthread_join(threads[1], NULL);

//	pthread_exit(NULL);

	while(running_threads > 0){
		sleep(3);
	}

	cout << "Threads are finished!" << endl;

	Mat camMat1 = readIntrinsicFile("lIntrinsic.txt");
	Mat camMat2 = readIntrinsicFile("rIntrinsic.txt");
	Mat distMat1 = readDistortionFile("lDistortion.txt");
	Mat distMat2 = readDistortionFile("rDistortion.txt");

	cout << camMat1 << endl << camMat2 << endl << distMat1 << endl << distMat2 << endl;

	Mat R;
	Mat T;
	Mat E;
	Mat F;

	cout << "Calibrating stereo..." << endl;

	// Pass the object points (physical coordinates) plus the image points, camera and distortion matrices from each calibrated camera. Pass in R, T, E, and F to use as outputs. 
	double rms = stereoCalibrate(bothObject_points, lBothImage_points, rBothImage_points, camMat1, distMat1, camMat2, distMat2, lImage.size(), R, T, E, F);

	cout << R << endl << T << endl << E << endl << F << endl << "Stereo RMS is " << rms << endl;

	cout << "Finished " << endl;

// Write the R, T, E, and F matrices into a file. 
	ofstream stereoParams;
	stereoParams.open("stereoParams.txt");

	for(int i = 0; i < 3; i++){
		for(int j =0; j < 3; j++){
			stereoParams << R.at<double>(i, j) << " ";
		}
	}

	stereoParams << endl;

	for(int i = 0; i < 3; i++){
		stereoParams << T.at<double>(i) << " ";
	}

	stereoParams << endl;

	for(int i = 0; i < 3; i++){
		for(int j =0; j < 3; j++){
			stereoParams << E.at<double>(i, j) << " ";
		}
	}

	stereoParams << endl;

	for(int i = 0; i < 3; i++){
		for(int j =0; j < 3; j++){
			stereoParams << F.at<double>(i, j) << " ";
		}
	}

	stereoParams.close();

	cout << "Finished writing params" << endl;

	return 0;
}

