// Author: Alex Wilson
// Compiles with:
//   g++ `pkg-config --cflags opencv` tennis_ball.cpp -o tennis_ball `pkg-config --libs opencv`

#include "opencv2/highgui/highgui_c.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <stdio.h>
#include <iostream>
#include <sys/time.h>

double When(){
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1e-6);
}

using namespace std;
using namespace cv;

// Debug function to identify frame types
string type2str(int type)
{
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch ( depth ) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans+'0');

	return r;
}

int main(int argc, char** argv )
{
	VideoCapture sequenceL("/home/lewis/Dropbox/631_robot_vision/tennis_ball/ball_launch/ball1L%02d.bmp");
	VideoCapture sequenceR("/home/lewis/Dropbox/631_robot_vision/tennis_ball/ball_launch/ball1R%02d.bmp");

	if (!sequenceL.isOpened()) {
		fprintf(stderr,"Failed to open Image Sequence!\n");
		return 1;
	}
	if (!sequenceR.isOpened()) {
		fprintf(stderr,"Failed to open Image Sequence!\n");
		return 1;
	}

	Mat frameL, frameR;
	int key;
	namedWindow("tennis ball right", CV_WINDOW_NORMAL);
	namedWindow("tennis ball left", CV_WINDOW_NORMAL);

	/*VideoWriter Vout;
	Vout.open("tennis_ball.avi", CV_FOURCC('M', 'P', 'E', 'G'), 30, Size(640,480), 1);*/

	// Set up blob detector
	SimpleBlobDetector::Params params;
	params.minDistBetweenBlobs = 50.0f;
	params.filterByInertia = true;
	params.filterByConvexity = true;
	params.filterByColor = false;
	params.filterByCircularity = true;
	params.filterByArea = true;
	params.minArea = 20.0f;
	params.maxArea = 40000.0f;

	SimpleBlobDetector blob_detectorL(params);
	SimpleBlobDetector blob_detectorR(params);
	vector<cv::KeyPoint> keypointsL, keypointsR;

	Mat frameROIR;
	float blob_x_r = 320;
	float blob_y_r = 100;
	float blob_r_r = 5;
	float blob_x_range_r = 4*blob_r_r;
	float blob_y_range_r = 4*blob_r_r;

	Mat frameROIL;
	float blob_x_l = 290;
	float blob_y_l = 100;
	float blob_r_l = 5;
	float blob_x_range_l = 4*blob_r_l;
	float blob_y_range_l = 4*blob_r_l;

	bool running = true;

	while (running) {
		double start = When();
		sequenceL >> frameL;
		sequenceR >> frameR;
		if (frameR.empty() || frameL.empty())
			break;

		// Check ROI boundaries L
		if (blob_x_l-blob_x_range_l < 0)
			blob_x_range_l = blob_x_l;
		if (blob_x_l + blob_x_range_l > 640)
			blob_x_range_l = 640 - blob_x_l;

		if (blob_y_l-blob_y_range_l < 0)
			blob_y_range_l = blob_y_l;
		if (blob_y_l + blob_y_range_l > 480)
			blob_y_range_l = 480 - blob_y_l;


		// Check ROI boundaries R
		if (blob_x_r-blob_x_range_r < 0)
			blob_x_range_r = blob_x_r;
		if (blob_x_r + blob_x_range_r > 640)
			blob_x_range_r = 640 - blob_x_r;

		if (blob_y_r-blob_y_range_r < 0)
			blob_y_range_r = blob_y_r;
		if (blob_y_r + blob_y_range_r > 480)
			blob_y_range_r = 480 - blob_y_r;


		// Create ROI
		frameROIL = frameL(Rect(blob_x_l-blob_x_range_l, blob_y_l-blob_y_range_l,
			blob_x_range_l*2, blob_y_range_l*2));
		frameROIR = frameR(Rect(blob_x_r-blob_x_range_r, blob_y_r-blob_y_range_r,
			blob_x_range_r*2, blob_y_range_r*2));

		blob_detectorR.detect(frameROIR, keypointsR);
		blob_detectorL.detect(frameROIL, keypointsL);

		// Draw ROI
		line(frameL, Point(blob_x_l-blob_x_range_l, blob_y_l-blob_y_range_l), 
			Point(blob_x_l+blob_x_range_l, blob_y_l-blob_y_range_l), 
			Scalar(255, 0, 0), 1);
		line(frameL, Point(blob_x_l-blob_x_range_l, blob_y_l-blob_y_range_l), 
			Point(blob_x_l-blob_x_range_l, blob_y_l+blob_y_range_l), 
			Scalar(255, 0, 0), 1);
		line(frameL, Point(blob_x_l+blob_x_range_l, blob_y_l+blob_y_range_l), 
			Point(blob_x_l+blob_x_range_l, blob_y_l-blob_y_range_l), 
			Scalar(255, 0, 0), 1);
		line(frameL, Point(blob_x_l-blob_x_range_l, blob_y_l+blob_y_range_l), 
			Point(blob_x_l+blob_x_range_l, blob_y_l+blob_y_range_l), 
			Scalar(255, 0, 0), 1);

		// Draw ROI R
		line(frameR, Point(blob_x_r-blob_x_range_r, blob_y_r-blob_y_range_r), 
			Point(blob_x_r+blob_x_range_r, blob_y_r-blob_y_range_r), 
			Scalar(255, 0, 0), 1);
		line(frameR, Point(blob_x_r-blob_x_range_r, blob_y_r-blob_y_range_r), 
			Point(blob_x_r-blob_x_range_r, blob_y_r+blob_y_range_r), 
			Scalar(255, 0, 0), 1);
		line(frameR, Point(blob_x_r+blob_x_range_r, blob_y_r+blob_y_range_r), 
			Point(blob_x_r+blob_x_range_r, blob_y_r-blob_y_range_r), 
			Scalar(255, 0, 0), 1);
		line(frameR, Point(blob_x_r-blob_x_range_r, blob_y_r+blob_y_range_r), 
			Point(blob_x_r+blob_x_range_r, blob_y_r+blob_y_range_r), 
			Scalar(255, 0, 0), 1);

		// Draw and update next position
		for (int i=0; i<keypointsL.size(); i++){
			float XL = keypointsL[i].pt.x + blob_x_l-blob_x_range_l; 
			float YL = keypointsL[i].pt.y + blob_y_l-blob_y_range_l;
			float RL = keypointsL[i].size;

			float XR = keypointsR[i].pt.x + blob_x_r-blob_x_range_r; 
			float YR = keypointsR[i].pt.y + blob_y_r-blob_y_range_r;
			float RR = keypointsR[i].size;

			if (i == 0) {
				line(frameL, Point(XL-RL/2.0, YL), Point(XL+RL/2.0, YL), Scalar(0, 255, 0), 1);
				line(frameL, Point(XL, YL-RL/2.0), Point(XL, YL+RL/2.0), Scalar(0, 255, 0), 1);
				//blob_x += (X - blob_x) * 2;
				//blob_y += (Y - blob_y) * 2;
				blob_x_r = XR;
				blob_y_r = YR;
				blob_r_r = RR;
				blob_x_range_r = 4*blob_r_r;
				blob_y_range_r = 4*blob_r_r;

				line(frameR, Point(XR-RR/2.0, YR), Point(XR+RR/2.0, YR), Scalar(0, 255, 0), 1);
				line(frameR, Point(XR, YR-RR/2.0), Point(XR, YR+RR/2.0), Scalar(0, 255, 0), 1);
				//blob_x += (X - blob_x) * 2;
				//blob_y += (Y - blob_y) * 2;
				blob_x_l = XL;
				blob_y_l = YL;
				blob_r_l = RL;
				blob_x_range_l = 4*blob_r_l;
				blob_y_range_l = 4*blob_r_l;
				blob_x_r = XR;
				blob_y_r = YR;
				blob_r_r = RR;
				blob_x_range_r = 4*blob_r_r;
				blob_y_range_r = 4*blob_r_r;
				
			} else {
				//circle(frame, Point(X, Y), R/0.9, Scalar(0, 255, 0), -1, 8, 0);
				line(frameR, Point(XR-RR/2.0, YR), Point(XR+RR/2.0, YR), Scalar(0, 0, 255), 1);
				line(frameR, Point(XR, YR-RR/2.0), Point(XR, YR+RR/2.0), Scalar(0, 0, 255), 1);
				line(frameL, Point(XL-RL/2.0, YL), Point(XL+RL/2.0, YL), Scalar(0, 0, 255), 1);
				line(frameL, Point(XL, YL-RL/2.0), Point(XL, YL+RL/2.0), Scalar(0, 0, 255), 1);
			}
		/*	if (count++ == 0){
				blob_x = 290;
				blob_y = 100;
				blob_r = 5;
				blob_x_range = 4*blob_r;
				blob_y_range = 4*blob_r;
				cout << "reset" << endl;
			}
			if (count == 32){
				count = 0;
			}*/
		}

	/*	Vout.write(frame);
		Vout.write(frame);
		Vout.write(frame);
		Vout.write(frame);
		Vout.write(frame);*/

		imshow("tennis ball left", frameL);
		imshow("tennis ball right", frameR);
		double end = When();
		double timePassed = end - start;
		std::cout << timePassed << std::endl;

		key = waitKey(0);
		switch (key) {
		case 27:
			running = false;
			break;
		default:
			break;
		}
	}

	//Vout.release();
	sequenceL.release();
	sequenceR.release();

	return 0;
}

