#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	VideoCapture cam(0);
	if (!cam.isOpened()) {
		fprintf(stderr, "Failed to open camera\n");
		exit(EXIT_FAILURE);
	}
	Mat frame;	

	namedWindow("cam", CV_WINDOW_NORMAL);
	moveWindow("cam", 500, 500);

	int i;
	for (i = 0; i < 1000; i++) {
		cam >> frame;
		imshow("cam", frame);
		waitKey(1);
	}

	cam.release();

	exit(EXIT_SUCCESS);
}

