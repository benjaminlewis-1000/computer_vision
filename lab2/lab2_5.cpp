#include <iostream>
#include <boost/algorithm/string.hpp>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <cmath>

//#include <system.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

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

int capture_video_spacebar(VideoCapture cap){
	unsigned int count = 0;
	namedWindow("Screen", CV_WINDOW_NORMAL);
	char setting = 'a';
	//system("rm calibrate_webcam/*"); // Very much playing with fire here, yes. 

	for(;;){  // Loop through, getting new images from the camera.
		//cout << "Setting " << setting + 'a' << endl;
		Mat frame;
		cap >> frame; // get a new frame from camera
		if (quit_signal) exit(0); // exit cleanly on interrupt

		imshow("Screen", frame);  // Show the image on the screen

		char key = waitKey(30);
		if (key == ' '){
			//cout << "setting is " << key + 0<< endl;
			cout << "space bar\n";
			ostringstream name;
			name << "calibrate_webcam/img" << count++ << ".jpg";
			string filename = name.str();
			imwrite(filename, frame);
		}else if(key == 'x'){
			return count;
		}
	}
}

void calibrate(int numImages, int numCornersHorizontal, int numCornersVertical, const char* intrinsic_name, const char* distortion_name){
	ofstream intrinsicFile;
	intrinsicFile.open(intrinsic_name);
	ofstream distortionParams;
	distortionParams.open(distortion_name);

// Task 2
	int numSquares = numCornersHorizontal * numCornersVertical;
	Size boardSize = Size(numCornersHorizontal, numCornersVertical);

	vector<vector<Point3f> > object_points;
	vector<vector<Point2f> > image_points;
	Mat image;

	vector<Point3f> obj;
	for(int j=0;j<numSquares;j++){
		obj.push_back(Point3f( j/numCornersHorizontal , j%numCornersHorizontal , 0.0f));
	}

	for (int i = 0; i < numImages; i++){
		vector<Point2f> pointBuf;
		ostringstream name;
		name << "calibrate_webcam/img" << i << ".jpg";
		string filename = name.str();
		image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

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
	return;
}

int main(int argc, char** argv){
	VideoCapture cap(0); // open the default camera
	if(!cap.isOpened()){ // check if we succeeded
		return -1;
	}
	#ifdef __unix__
	   signal(SIGINT,quit_signal_handler); // listen for ctrl-C
	#endif

	int count = capture_video_spacebar(cap);

	int cornersHoriz = 9; 
	int cornersVert = 7;
	calibrate(count, cornersHoriz, cornersVert, "intrinsic_webcam.txt", "distortion_webcam.txt");
	

// the camera will be deinitialized automatically in VideoCapture destructor

return 0;
}
