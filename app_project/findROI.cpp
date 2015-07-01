#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <string>
#include <sstream>
#include <vector>

using namespace cv;
using namespace std;

vector<Point> ROI_Points;

void drawCroppedImage(Mat& sourceImg, Mat& dstImg, int centerX, int centerY, double scaleX, double scaleY, vector<Point> ROI);
Mat translateImg(Mat &img, int offsetx, int offsety);

Mat translateImg(Mat &img, int offsetx, int offsety){
    Mat trans_mat = (Mat_<double>(2,3) << 1, 0, offsetx, 0, 1, offsety);
    warpAffine(img,img,trans_mat,img.size());
    return trans_mat;
}

void drawCroppedImage(Mat& sourceImg, Mat& dstImg, int centerX, int centerY, double scaleX, double scaleY, vector<Point> ROI){
	int max_x = 0;
	int max_y = 0;
	int min_x = 99999;
	int min_y = 99999;
	
	Mat mask = cvCreateMat(sourceImg.rows, sourceImg.cols, CV_8UC1);
	for(int i=0; i<mask.cols; i++)
	   for(int j=0; j<mask.rows; j++)
		   mask.at<uchar>(Point(i,j)) = 0;

	vector<Point> ROI_Poly;
	approxPolyDP(ROI, ROI_Poly, 1.0, true);

	for (int i = 0; i < ROI.size(); i++){
		if (ROI[i].x > max_x){
			max_x = ROI[i].x;}
		if (ROI[i].x < min_x){
			min_x = ROI[i].x;}
		if (ROI[i].y > max_y){
			max_y = ROI[i].y;}
		if (ROI[i].y < min_y){
			min_y = ROI[i].y;}
	}
	
	// Sanity check for outside the edges of the source image
	max_x = (max_x > sourceImg.cols) ? sourceImg.cols : max_x;
	max_y = (max_y > sourceImg.rows) ? sourceImg.rows : max_y;
	min_x = (min_x < 0 ) ? 0 : min_x;
	min_y = (min_y < 0 ) ? 0 : min_y;
	
	int width = max_x - min_x;
	int height = max_y - min_y;
	
	int topLeftX = centerX - width  * scaleX / 2;
	int topLeftY = centerY - height * scaleY / 2;
	
	cout << topLeftX << " " << topLeftY << endl;
	
	topLeftX = max(topLeftX, 0);
	topLeftY = max(topLeftY, 0);
	
	fillConvexPoly(mask, &ROI_Poly[0], ROI_Poly.size(), 255, 8, 0); 
	
	// Move mask to the origin
	translateImg(mask, (-1 * min_x) , (-1 * min_y)  );//, 50);
	
	cout << mask.size() << endl;
			
	Mat tmp = sourceImg(Rect(min_x, min_y, width, height) );
	
	resize(tmp, tmp, Size(width * scaleX, height * scaleY), 0, 0, INTER_CUBIC);
	resize(mask, mask, Size(mask.cols * scaleX, mask.rows * scaleY), 0, 0, INTER_CUBIC);
	
	// By playing around, it seems that the thing to do is move the mask to the origin, where it is 
	// applied to the dstImg(rectangle). Then that masked image is applied to the rectangle in the 
	// destination image. We don't need to move the mask to line up with the destination image. 
	tmp.copyTo(dstImg(Rect(topLeftX, topLeftY, width * scaleX, height * scaleY) )  , mask);
}


void mouseHandlerHSV(int event, int x, int y, int flags, void* param)
{
    /* user press left button */
    // Get the HSV value of the point.
    if (event == CV_EVENT_LBUTTONDOWN )
    {
    	Mat* img = (Mat*) param;
    	//imshow("Show", *img);
    	
			x = (x > img->cols) ? img->cols : x;
			y = (y > img->rows) ? img->rows : y;
			x = (x < 0 ) ? 0 : x;
			y = (y < 0 ) ? 0 : y;
			
			
    	cout << img->at<Vec3b>(x, y) << endl;
    	Point point(x,y);
		circle(*img, point, 3, Scalar(255, 255, 255), 3, 5);
		imshow("Show", *img);
    	ROI_Points.push_back( point );
    	
    	///Mat tmp = *img;
    }
}

int main(int argc, char** argv){

	Mat img = imread("test.jpg");
	namedWindow("Show", WINDOW_NORMAL);
	resizeWindow("Show", 1920, 1200);
	
	Mat replace = imread("t2.jpg");
	
	//cvtColor(img, img, CV_RGB2HSV);
	
	cvSetMouseCallback("Show", mouseHandlerHSV, &img);
	
	imshow("Show", img);
	while(1){
		char cmd = waitKey(30);
		if(cmd == 'f' && ROI_Points.size() > 2){
			
			
		//	drawCroppedImage(Mat& sourceImg, Mat& dstImg, int centerX, int centerY, double scaleX, double scaleY, vector<Point> ROI)
			drawCroppedImage(img, replace, 500, 500, 1, 1, ROI_Points);
			
			imshow("Show", replace);  
			ROI_Points.clear();
		}else if(cmd == 'x'){
			break;
		}else if(cmd == 'c'){
			imshow("Show", img);
		}else if(cmd == 'q'){
			img = imread("test.jpg");
			imshow("Show", img);
		}
	}
}

/*

int numImages = 17;
	namedWindow("Screen",WINDOW_NORMAL);
	resizeWindow("Screen", 1800, 1000);
	
	// Max 500 points to track, quality of 0.01, min. distance 15 pixels, etc. 
	const int MAX_COUNT = 500;
	const float QUALITY = 0.01;
	const int MIN_DIST = 10;
	Size winSize(10,10);
	vector<Point2f> points[2];  // Index 0 = last image, index 1 = this image. 
	TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03);
	Mat image, prevImage;
	
	VideoWriter Vout;
	//optical_flow.avi
	Vout.open("junk.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, Size(640,480), 1);
	
	for (int j = 1; j < 4; j++){
	// Do for subsequent frames, 2 frame difference, and 3 frame difference. 
		for (int i = 0; i < numImages - j; i++){
		
		// Open a previous and a next image to compute the optical flow between them. 
			ostringstream prev;
			prev << "optical_flow/O" << i + 1 << ".jpg";
			ostringstream next;
			next << "optical_flow/O" << i + 1 + j << ".jpg";
			
			
			/*ostringstream prev;
			prev << "parallel_cube/ParallelCube" << i + 1 << ".jpg";
			ostringstream next;
			next << "parallel_cube/ParallelCube" << i + 1 + j << ".jpg";
		
			Mat lastImage, nextImage;
		
			string lastFilename = prev.str();
			lastImage = imread(lastFilename, CV_LOAD_IMAGE_GRAYSCALE);
			string nextFilename = next.str();
			nextImage = imread(nextFilename, CV_LOAD_IMAGE_GRAYSCALE);
			
			// Get good features to track, which will be fed into the optical flow algorithm.
			goodFeaturesToTrack(lastImage, points[0], MAX_COUNT, 0.01, MIN_DIST, Mat(), 3, 0, 0.04);
			cornerSubPix(lastImage, points[0], winSize, Size(-1,-1), termcrit);
			
			// Set up some output vectors, then compute optical flow. 
			vector<uchar> status;
			vector<float> err;
			calcOpticalFlowPyrLK(lastImage, nextImage, points[0], points[1], status, err, winSize, 3, termcrit);

			cvtColor(nextImage, nextImage, CV_GRAY2BGR);
			
			// Graph previous point (circle) and then a line between the two points, showing where the point has moved. 
			for (int k = 0; k < points[0].size(); k++){
				circle(nextImage,points[0][k],3,cv::Scalar(0,255,0),1);
				line(nextImage, points[0][k], points[1][k], Scalar(0,0,255), 1);
			}
			
			// Write out the image. 
			imshow("Screen", nextImage);
			for (int vid = 0; vid < 9; vid++){
				Vout << nextImage;  // SUPER HACK!!! YAY! Video writer is super fast at 30fps, so write more frames. 
			}
			waitKey(0);
			
		}
	
	}
*/



/*Mat mask = cvCreateMat(img.rows, img.cols, CV_8UC1);
			for(int i=0; i<mask.cols; i++)
			   for(int j=0; j<mask.rows; j++)
				   mask.at<uchar>(Point(i,j)) = 0;
			cout << "Hey guys!\n";
			vector<Point> ROI_Poly;
			approxPolyDP(ROI_Points, ROI_Poly, 1.0, true);
			
			int max_x = 0;
			int max_y = 0;
			int min_x = 99999;
			int min_y = 99999;
			
			// TODO: Order points around centroid for best shape. I have code in MATLAB for this already.
			
			for (int i = 0; i < ROI_Points.size(); i++){
				if (ROI_Points[i].x > max_x){
					max_x = ROI_Points[i].x;}
				if (ROI_Points[i].x < min_x){
					min_x = ROI_Points[i].x;}
				if (ROI_Points[i].y > max_y){
					max_y = ROI_Points[i].y;}
				if (ROI_Points[i].y < min_y){
					min_y = ROI_Points[i].y;}
			}
			
			// Sanity check for outside the bounds.
			max_x = (max_x > img.cols) ? img.cols : max_x;
			max_y = (max_y > img.rows) ? img.rows : max_y;
			min_x = (min_x < 0 ) ? 0 : min_x;
			min_y = (min_y < 0 ) ? 0 : min_y;
			
			int width = max_x - min_x;
			int height = max_y - min_y;
			
			rectangle(replace, Point(min_x, min_y), Point(max_x, max_y), Scalar(255, 0, 100) );
			
			cout << max_x << " " << max_y << " " << min_x << " " << min_y  << endl;
			
			fillConvexPoly(mask, &ROI_Poly[0], ROI_Poly.size(), 255, 8, 0); 
			//bitwise_not(mask, mask);
			/*Mat blank = Mat::zeros(img.rows, img.cols, CV_8UC1);
			img.copyTo(blank, mask);
			imshow("Show", blank);  
			waitKey(0);*/
			
			// Have to translate the mask with an affine transformation so that it still lies on top of the 
			// translated image region.
			/*translateImg(mask, -1 * min_x, -1 *min_y);//, 50);
			
			Mat tmp = img(Rect(min_x, min_y, width, height) );
			
			resize(tmp, tmp, Size(width/2, height/2), 0, 0, INTER_CUBIC);
			resize(mask, mask, Size(mask.cols/2, mask.rows/2), 0, 0, INTER_CUBIC);
			
			tmp.copyTo(replace(Rect(0, 0, width / 2, height / 2) ) , mask);
			
			// Applies mask -- but it's getting shrunk to scale. 
			// Copies masked source image to a rectangle at 0,0 in the copiedTo matrix
			// Then displays the area of the original rectangle...
			
			//seamlessClone(img(Rect(min_x, min_y, width, height) ), replace, mask, Point(200, 200));
			// OpenCV 3 stuff, looks really really nice though.
			
			cout << "Mask " << mask.at<Vec3b>(img.cols/2, img.rows/2) << endl;*/
