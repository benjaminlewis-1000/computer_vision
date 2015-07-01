// Author: Alex Wilson
// Compiles with:
//   g++ `pkg-config --cflags opencv` intrinsic.cpp -o intrinsic `pkg-config --libs opencv`

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

using namespace std;
using namespace cv;

int main(int argc, char **argv )
{
	if (argc < 3) {
		fprintf(stderr, "usage: %s <image(s)> <output intrinsic\n", argv[0]);
		exit(EXIT_FAILURE);	
	}

	VideoCapture sequence(argv[1]);

	if (!sequence.isOpened()) {
		fprintf(stderr,"Failed to open Image Sequence!\n");
		return 1;
	}

	Mat frame, gray;
	namedWindow("intrinsic", CV_WINDOW_NORMAL);

	Size patternsize(10,7);
	vector < Point2f > corners;
	vector < vector < Point3f > > object_points;
	vector < vector < Point2f > > image_points;
	
	vector < Point3f > obj;
	for (int j = 0; j < patternsize.width * patternsize.height; j++)
		obj.push_back(Point3f(j / patternsize.width * 3.88f, 
				      j % patternsize.width * 3.88f, 0.0f));

	int key;
	bool running = true;

	int count = 0;
	while (running) {
		sequence >> frame;
		if (frame.empty())
			break;

		cvtColor(frame, gray, CV_BGR2GRAY);

		bool patternfound = findChessboardCorners(gray, patternsize, 
			corners, CALIB_CB_ADAPTIVE_THRESH + 
			CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);

		if(patternfound) {
			cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
				     TermCriteria(CV_TERMCRIT_EPS + 
				                  CV_TERMCRIT_ITER, 30, 0.1));
			drawChessboardCorners(frame, patternsize, Mat(corners), 
				patternfound);
			image_points.push_back(corners);
			object_points.push_back(obj);
			count ++;
		}

		imshow("intrinsic", frame);

		key = waitKey(10);
		switch (key) {
		case 27:
			running = false;
			break;
		default:
			break;
		}
	}

cout << count << endl;

	sequence.release();
	sequence.open(argv[1]);
	sequence >> frame;

	Size image_size = frame.size();
	Mat intrinsic = Mat(3, 3, CV_32FC1);
	intrinsic.ptr < float >(0)[0] = 1;
	intrinsic.ptr < float >(1)[1] = 1;
	Mat dist_coeffs;
	vector < Mat > rvecs;
	vector < Mat > tvecs;

	double rms = calibrateCamera(object_points, image_points, image_size,
		intrinsic, dist_coeffs, rvecs, tvecs);
		
	cout << "intrinsic:\n" << intrinsic << "\n\n";
	cout << "dist_coeffs:\n" << dist_coeffs << "\n\n";

	sequence.release();
	
	// Serialize the intrinsic and dist_coeffs Matrices into a file
	int intr_fd = open(argv[2], O_WRONLY | O_CREAT, 0755);
	
	int cols = intrinsic.cols;
	int rows = intrinsic.rows;
	int chan = intrinsic.channels();
	int eSiz = (intrinsic.dataend - intrinsic.datastart)/(cols*rows*chan);
	write(intr_fd, (char*)&cols, sizeof(int));
	write(intr_fd, (char*)&rows, sizeof(int));
	write(intr_fd, (char*)&chan, sizeof(int));
	write(intr_fd, (char*)&eSiz, sizeof(int));
	write(intr_fd, (char*)intrinsic.data, cols*rows*chan*eSiz);
	
	cols = dist_coeffs.cols;
	rows = dist_coeffs.rows;
	chan = dist_coeffs.channels();
	eSiz = (dist_coeffs.dataend - dist_coeffs.datastart)/(cols*rows*chan);
	write(intr_fd, (char*)&cols, sizeof(int));
	write(intr_fd, (char*)&rows, sizeof(int));
	write(intr_fd, (char*)&chan, sizeof(int));
	write(intr_fd, (char*)&eSiz, sizeof(int));
	write(intr_fd, (char*)dist_coeffs.data, cols*rows*chan*eSiz);
	
	close(intr_fd);

	exit(EXIT_SUCCESS);
}
