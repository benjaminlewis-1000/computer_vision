#include <iostream>
#include <string>
#include <sstream>
#include <stdlib.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]){
	Mat last_frame;
	namedWindow("diff", 1);
	namedWindow("Camera", 1);
	Mat image;
	Mat in;

	VideoWriter VOut;
	VOut.open("tennis_tracking.avi", CV_FOURCC('M', 'P', 'E', 'G'), 30, Size(640, 480), 1);

	for(;;){
		for (int i = 34; i <= 72; i += 2){ // i = 34

			ostringstream name;
			name << "image0" << i << ".jpg";
			string filename = name.str();
			in = imread(filename, CV_LOAD_IMAGE_COLOR);
			//cout << filename << endl;
			if (last_frame.empty()){
				cvtColor(in, last_frame, CV_BGR2GRAY);
			}else{
				last_frame = image.clone();  // Which was the last gray frame, before I update it. 
			}
			cvtColor(in, image, CV_BGR2GRAY);
			Mat out;
			GaussianBlur(image, out, Size(7,7), 3.8, 3.8);
			Mat t_out;

		 	vector<Vec3f> circles;

		  /// Apply the Hough Transform to find the circles
			threshold(out, t_out, 130, 255, THRESH_BINARY);
		    HoughCircles( t_out, circles, CV_HOUGH_GRADIENT, 1, out.rows/30, 28, 13, 4, 30 ); // Good for large ball without too much blur (2/3 of frames)
		if (circles.size() == 0){  // Works for the very small ball at the beginning. Small ball detector, low threshold. 
			threshold(out, t_out, 30, 255, THRESH_BINARY);
		    HoughCircles( t_out, circles, CV_HOUGH_GRADIENT, 1, out.rows/30, 32, 12, 4, 10 );
		}
		if (circles.size() == 0){ // Last frame with a lot of blur... just had to test it out!
			threshold(out, t_out, 140, 255, THRESH_BINARY);
		    HoughCircles( t_out, circles, CV_HOUGH_GRADIENT, 1, out.rows/30, 17, 11, 4, 35 );  // Very arbitrary, yet it works. 140/17/11. 
		}
		 for( size_t i = 0; i < circles.size(); i++ )
		  {
		//	if (i == 0){
		//		cout << "found " << circles.size() << endl;
		//	}
			  Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			  int radius = cvRound(circles[i][2]);
			  // circle center
			  circle( in, center, 3, Scalar(0,255,0), -1, 3, 0 );
			  // circle outline
			  circle( in, center, radius, Scalar(0,0,255), 3, 3, 0 );
		   }

			imshow("diff", t_out);
			imshow("Camera", in);
			VOut << in;
			waitKey(100);
		}
	}
	return 0;
}
