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

/* Keep the webcam from locking up when you interrupt a frame capture */
volatile int quit_signal=0;
#ifdef __unix__
#include <signal.h>
extern "C" void quit_signal_handler(int signum) {
 if (quit_signal!=0) exit(0); // just exit already
 quit_signal=1;
 printf("Will quit at next camera frame (repeat to kill now)\n");
}
#endif


using namespace std;
using namespace boost;
using namespace cv;

int main(){
	// Read in intrinsic file
	ifstream intrinsicFile;
	intrinsicFile.open("intrinsic_webcam.txt");
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
	distortionFile.open("distortion_webcam.txt");
	getline (distortionFile,line);
	istringstream issDist(line);
	split(fields, line, boost::is_any_of(" ") );

	Mat distCoeffs = Mat::ones(5,1,CV_64F);

	for(int i = 0; i < fields.size(); i++){
		distCoeffs.at<double>(i) = (double)atof(fields[i].c_str());
	}

	// Read in the three images
	Mat far, close, turned;
	far    = imread("calibrate_webcam/img13.jpg",    CV_LOAD_IMAGE_GRAYSCALE);
	close  = imread("calibrate_webcam/img6.jpg",  CV_LOAD_IMAGE_GRAYSCALE);
	turned = imread("calibrate_webcam/img26.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	Mat far_un, close_un, turned_un;

	undistort(far, far_un, intrinsics.clone(), distCoeffs.clone());
	undistort(close, close_un, intrinsics.clone(), distCoeffs.clone());
	undistort(turned, turned_un, intrinsics.clone(), distCoeffs.clone());

	cout << "Window size is " << far.size() << endl;

	namedWindow("One", WINDOW_NORMAL);
	namedWindow("Two", WINDOW_NORMAL);
	namedWindow("Three", WINDOW_NORMAL);

	Mat far_diff, close_diff, turned_diff;

	absdiff(far_un, far, far_diff);
	absdiff(close_un, close, close_diff);
	absdiff(turned_un, turned, turned_diff);

	imshow("One", far_diff);
	imshow("Two", close_diff);
	imshow("Three", turned_diff);

	imwrite("webcam1.jpg", far_diff);
	imwrite("webcam2.jpg", close_diff);
	imwrite("webcam3.jpg", turned_diff);

	waitKey(0);

	/*VideoCapture cap(0); // open the default camera
	if(!cap.isOpened()){ // check if we succeeded
		return -1;
	}
	#ifdef __unix__
	   signal(SIGINT,quit_signal_handler); // listen for ctrl-C
	#endif

	namedWindow("Screen", CV_WINDOW_NORMAL);
	//system("rm calibrate_webcam/*"); // Very much playing with fire here, yes. 

	for(;;){  // Loop through, getting new images from the camera.
		Mat frame;
		cap >> frame; // get a new frame from camera
		//cvtColor(frame, frame, CV_RGB2GRAY);
		if (quit_signal) exit(0); // exit cleanly on interrupt
		Mat frameOut;
		undistort(frame, frameOut, intrinsics.clone(), distCoeffs.clone());

		imshow("Screen", frameOut);  // Show the image on the screen

		waitKey(30);
	}*/
}
