#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <cmath>


using namespace cv;
using namespace std;

#define WINWIDTH 1920
#define WINHEIGHT 1080

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

vector<Point> ROI_Points;
int ROI_done = 0;

Mat drawCroppedImage(Mat& sourceImg, Mat& dstImg, vector<Point> ROI);
Mat translateImg(Mat &img, int offsetx, int offsety);

Mat translateImg(Mat &img, int offsetx, int offsety){
    Mat trans_mat = (Mat_<double>(2,3) << 1, 0, offsetx, 0, 1, offsety);
    warpAffine(img,img,trans_mat,img.size());
    return trans_mat;
}

Mat drawCroppedImage(Mat& sourceImg, Mat& dstImg, vector<Point> ROI){
	Mat mask = Mat::zeros(sourceImg.rows, sourceImg.cols, CV_8UC1);
	
	vector<Point> ROI_Poly;
	approxPolyDP(ROI, ROI_Poly, 1.0, true);

	fillConvexPoly(mask, &ROI_Poly[0], ROI_Poly.size(), 255, 8, 0);
	
	sourceImg.copyTo(dstImg , mask);
	return mask;
}


void mouseHandlerHSV(int event, int x, int y, int flags, void* param)
{
    /* user press left button */
    // Get the HSV value of the point.
    if (event == CV_EVENT_LBUTTONDOWN )
    {
    	//Mat* img = (Mat*) param;
    	//imshow("Show", *img);
    	
			x = (x > WINWIDTH) ? WINWIDTH : x;  // Makes sure the click is within the image, puts it on the edge of the screen. 
			y = (y > WINHEIGHT) ? WINHEIGHT : y;
			x = (x < 0 ) ? 0 : x;
			y = (y < 0 ) ? 0 : y;
			
			
    	//cout << img->at<Vec3b>(x, y) << endl;
    	Point point(x,y);
		//circle(*img, point, 3, Scalar(255, 255, 255), 3, 5);
		//imshow("Screen", *img);
    	ROI_Points.push_back( point );
    	
    	///Mat tmp = *img;
    }
    if (event == CV_EVENT_RBUTTONDOWN ){
    	ROI_done = 1;
    }
}

#define INFILL 2
#define CIRCLE 3

void mouseHandlerRunning(int event, int x, int y, int flags, void* param)
{
    /* user press left button */
    // Get the HSV value of the point.
    if (event == CV_EVENT_LBUTTONDOWN )
    {
    	ROI_done = INFILL;
    }
    if (event == CV_EVENT_RBUTTONDOWN ){
    	ROI_done = CIRCLE;
    }
}

int main(int argc, char** argv){
	
	VideoCapture cap(2);
	
	VideoWriter Vout;
	Vout.open("output.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, Size(640,480), 1);

	if(!cap.isOpened()){ // check if we succeeded
		return -1;
	}
	#ifdef __unix__
	   signal(SIGINT,quit_signal_handler); // listen for ctrl-C
	#endif
	
	namedWindow("Screen", WINDOW_NORMAL);
	resizeWindow("Screen", WINWIDTH, WINHEIGHT);
	Mat lastFrame;
	
again:
	cvSetMouseCallback("Screen", mouseHandlerHSV, NULL);
	cap >> lastFrame;

	while(!ROI_done){  // Loop through, getting new images from the camera.
		//cout << "Setting " << setting + 'a' << endl;
		Mat frame;
		cap >> frame; // get a new frame from camera
		
		if (quit_signal) exit(0); // exit cleanly on interrupt

		for (int i = 0; i < ROI_Points.size(); i++){
			circle(frame, ROI_Points[i], 2, Scalar(255, 255, 255), 3, 5);
		}

		imshow("Screen", frame);  // Show the image on the screen
		Vout << frame;

		char key = waitKey(1);
		if (key == ' '){
			//cout << "setting is " << key + 0<< endl;
			cout << "space bar\n";
		}
	}

	int maxCorners = 500;
	double qualityLevel = 0.01f;
	int minDistance = 1;
	int blockSize = 5;
	int useHarrisDetector = 0;
	double k = 0.04f;
	TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,30,0.01);
	Size winSize(20,20);

	Mat last, cropped, last_bw;
	cap >> last;
	Mat mask = drawCroppedImage(last, cropped, ROI_Points);
	vector<Point2f> lastCorners;
	cvtColor(last, last_bw, CV_BGR2GRAY);

	goodFeaturesToTrack(last_bw, lastCorners, maxCorners, qualityLevel, minDistance, 
		Mat(), blockSize, useHarrisDetector, k);
		
	for (int i = lastCorners.size() - 1; i >= 0; i--){
		int x = lastCorners[i].x;
		int y = lastCorners[i].y;
		if (mask.at<uchar>(y,x) == 0){
			// Not within ROI, remove point
			lastCorners.erase(lastCorners.begin() + i);
		}
	}

	cout << lastCorners.size() << endl; 
	int dilation_size = 12;
	Mat element = getStructuringElement( MORPH_ELLIPSE,
                                   Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                   Point( dilation_size, dilation_size ) );
	cvSetMouseCallback("Screen", mouseHandlerRunning, NULL);
	ROI_done = CIRCLE;

	while (1) {
		Mat frame, frame_bw;
		cap >> frame;
		vector<Point2f> corners;
		
		if (lastCorners.size() <= 2){
			corners.clear();
			lastCorners.clear();
			ROI_done = 0;
			ROI_Points.clear();
			goto again;
		}
  		
		cvtColor(frame, frame_bw, CV_BGR2GRAY);
		
		vector<Point2f> movedLastCorners;
		vector<uchar> status;
		vector<float> err;
  		calcOpticalFlowPyrLK(last_bw, frame_bw, lastCorners, movedLastCorners, status, err, winSize, 2, termcrit);
		goodFeaturesToTrack(frame_bw, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
		
  		Point2f center;
  		float radius;
  		
  		minEnclosingCircle(movedLastCorners, center, radius);
  		
  		if (radius > 200){
			corners.clear();
			lastCorners.clear();
			ROI_done = 0;
			ROI_Points.clear();
			goto again;
  		}
  		
  		radius += 7;
		
		Mat mask = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
		circle(mask, center, radius, 255, -1);
		
		for (int i = corners.size() - 1; i >= 0; i--){
			int x = corners[i].x;
			int y = corners[i].y;
			if (mask.at<uchar>(y,x) == 0){
				// Not within ROI, remove point
				corners.erase(corners.begin() + i);
			}
		}
		
		if (corners.size() <= 2){
			corners.clear();
			lastCorners.clear();
			ROI_done = 0;
			ROI_Points.clear();
			goto again;
		}

  		minEnclosingCircle(corners, center, radius);
  		mask = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
		circle(mask, center, radius + 10, 255, -1);
		
		if (ROI_done == INFILL){
			inpaint(frame, mask, frame, 2, INPAINT_NS);
		}else if (ROI_done == CIRCLE){
  			circle(frame, center, radius, Scalar(255,255,255), 3, 5);
		}		

	//	vector<float> err;
		
		if (quit_signal) exit(0); // exit cleanly on interrupt

		//imshow("Screen", inpaintMask);  // Show the image on the screen
		imshow("Screen", frame);
		Vout << frame;

		char key = waitKey(1);
		if (key == ' '){
			//cout << "setting is " << key + 0<< endl;
			cout << "space bar\n";
		}
		last_bw = frame_bw;
		lastCorners = corners;
	}
	
	return 0;
}




		//for (int i = 0; i < corners.size(); i++){
			//circle(inpaintMask, corners[i], 3, 255, 3, 5);
			//circle(frame, corners[i], 2, Scalar(255, 255, 255), 3, 5);
		//}
		
		//dilate(inpaintMask, inpaintMask, element);
		//dilate(inpaintMask, inpaintMask, element);
		
		//inpaint(frame, inpaintMask, frame, 1, INPAINT_NS);

//calcOpticalFlowPyrLK(last_bw, frame_bw, lastCorners, corners, status, err, winSize, 2, termcrit);

	/*	for (int i = status.size()-1; i >= 0; i--) {
			if (status.at(i) == 0) {
				corners.erase(corners.begin()+i);
				lastCorners.erase(lastCorners.begin()+i);
			}
		}*/

  	/*	vector<Point> intCorners;
  		for (int i = 0; i < corners.size(); i++){
  			int x = (int) corners[i].x;
  			int y = (int) corners[i].y;
  			intCorners.push_back(Point(x,y) );
  		}
  		
  		Mat t1 = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
  		Mat t2 = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
  		Mat inpaintMask = drawCroppedImage(t1, t2, intCorners) ;*/
  		
  	//	Mat inpaintMask = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
  	//	vector<Point2f> hull;
  	//	convexHull( corners, hull, false, true );
//	approxPolyDP(hull, hull, 1.0, true);
  	//	vector<Point> intHull;
  	/*	for (int i = 0; i < corners.size(); i++){
  			int x = (int) hull[i].x;
  			int y = (int) hull[i].y;
  			intHull.push_back(Point(x,y) );
  		}
  		fillConvexPoly(inpaintMask, &intHull[0], intHull.size(), 255, 8, 0);
  		*/
  		
  	/*	RotatedRect ellipseRect = fitEllipse(corners);
  		if (corners.size() > 5){
	  		ellipse(frame, ellipseRect, Scalar(255,255,255), 1) ;
	  		
	  		rectangle(inpaintMask, boundingRect(corners), 255, CV_FILLED, 8);
	  	}*/
  		
//  		fillConvexPoly(mask, &ROI_Poly[0], ROI_Poly.size(), 255, 8, 0);

