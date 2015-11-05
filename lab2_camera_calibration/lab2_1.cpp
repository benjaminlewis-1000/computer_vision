#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char **argv){

// Task 1
	Mat image;
	image = imread("jpg_calibration/AR40.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	namedWindow("Screen",WINDOW_NORMAL);

	Mat corners; 
// Find chessboard inner corners, 10 corners per row/7 per column
	bool pattern = findChessboardCorners(image, cvSize(10, 7), corners);
// Find subpixels with the criteria
	TermCriteria criteria = TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 );
	cornerSubPix(image, corners, Size(10,10), Size(-1, -1), criteria);

	cvtColor(image, image, CV_GRAY2BGR);
// Draw the corners on a colored image
	drawChessboardCorners(image, cvSize(10,7), corners, pattern);
// Write the image
	imwrite("Task_1.jpg", image);

	imshow("Screen", image);
	waitKey(0);

// Task 2

	return 0;
}
