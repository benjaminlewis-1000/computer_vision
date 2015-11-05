#include <iostream>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv){

ofstream intrinsicFile;
intrinsicFile.open("intrinsic.txt");
ofstream distortionParams;
distortionParams.open("distortion.txt");

// Task 2
	int num_images = 40;
	int numCornersHorizontal = 10;
	int numCornersVertical = 7;
	int numSquares = numCornersHorizontal * numCornersVertical;
	Size boardSize = Size(numCornersHorizontal, numCornersVertical);

	vector<vector<Point3f> > object_points;
	vector<vector<Point2f> > image_points;
	Mat image;

	vector<Point3f> obj;
	for(int j=0;j<numSquares;j++){
		obj.push_back(Point3f( j/numCornersHorizontal , j%numCornersHorizontal , 0.0f));
	}

	for (int i = 0; i < num_images; i++){
		vector<Point2f> pointBuf;
		ostringstream name;
		name << "jpg_calibration/AR" << i + 1 << ".jpg";
		string filename = name.str();
		image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

	//	namedWindow("Screen",WINDOW_NORMAL);

	// Find chessboard inner corners, 10 corners per row/7 per column
		bool found = findChessboardCorners(image, boardSize, pointBuf, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
		
		if(found){
		// Find subpixels with the criteria
			TermCriteria criteria = TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.01 );
			cornerSubPix(image, pointBuf, Size(10,10), Size(-1, -1), criteria);
			drawChessboardCorners(image, cvSize(10,7), pointBuf, found);

			cvtColor(image, image, CV_GRAY2BGR);
			image_points.push_back(pointBuf);
			object_points.push_back(obj);
		// Draw the corners on a colored image
	/*	// Write the image
			//imwrite("Task_1.jpg", image[i]);

			//imshow("Screen", image[i]);
			//waitKey(100);*/
			std::cout << i + 1 << " processed " << pointBuf[0] << "\n";
		}
	}

	Mat intrinsic = Mat(3,3, CV_32FC1);
	Mat distCoeffs;
	Size imageSize = image.size();
	vector<Mat> rvecs, tvecs;
	intrinsic.ptr<double>(0)[0] = 1;
	intrinsic.ptr<double>(1)[1] = 1;

	calibrateCamera(object_points, image_points, imageSize, intrinsic, distCoeffs, rvecs, tvecs);
	std::cout << "Matrix is " << intrinsic << std::endl << distCoeffs << std::endl;

	for(int i = 0; i < 3; i++){
		for(int j =0; j < 3; j++){
			intrinsicFile << intrinsic.at<double>(i, j) << " ";
		}
	}

	for(int k = 0; k < 5; k++)
		distortionParams << distCoeffs.at<double>(k) << " ";
	
//	intrinsicFile << intrinsic;
//	distortionParams << distCoeffs;

	intrinsicFile.close();
	distortionParams.close();
	return 0;
}



