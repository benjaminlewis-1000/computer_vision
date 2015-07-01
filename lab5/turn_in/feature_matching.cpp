#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <string>
#include <sstream>
#include <vector>

#define WIDTH 640
#define HEIGHT 480

/*
*  Feature Matching Algorithm, by Ben Lewis, 3/18/2015
*	Matches features in one image. Finds good features to track, draws a template around them in the first image,
*	then searches for that template in a larger field in the second image.
*	Then draw a line between the feature and its location in the second image.
*/

using namespace cv;
using namespace std;

int main(int argc, char** argv){
	
	int numImages = 17;
	namedWindow("Screen",WINDOW_NORMAL);
	resizeWindow("Screen", 1800, 1000);
	
	// Max 500 points to track, quality of 0.01, min. distance 15 pixels, etc. 
	const int MAX_COUNT = 300;
	const float QUALITY = 0.01;
	const int MIN_DIST = 10;
	Size winSize(10,10);
	vector<Point2f> points[2];  // Index 0 = last image, index 1 = this image. 
	TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03);
	Mat image, prevImage;
	
	int templateXSize = 59;
	int templateYSize = 59;

	int searchXSize = 89;
	int searchYSize = 89;
	
	VideoWriter Vout;
	// feature_tracking.avi
	Vout.open("junk.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, Size(640,480), 1);
	
	for (int j = 1; j < 4; j++){
	//for (int j = 1; j < 2; j++){
	// Do for subsequent frames, 2 frame difference, and 3 frame difference. 
		for (int i = 0; i < numImages - j; i++){
		
		// Open a previous and a next image to compute the optical flow between them.
		// Load in the previous image and the current image. We take the template from source and try to match it to compareImg.  
			ostringstream prev;
			prev << "optical_flow/O" << i + 1 << ".jpg";
			ostringstream next;
			next << "optical_flow/O" << i + 1 + j << ".jpg";
			
			/*
			ostringstream prev;
			prev << "parallel_cube/ParallelCube" << i + 1 << ".jpg";
			ostringstream next;
			next << "parallel_cube/ParallelCube" << i + 1 + j << ".jpg";*/
		
			Mat source, compareImg;
		
			string lastFilename = prev.str();
			source = imread(lastFilename, CV_LOAD_IMAGE_GRAYSCALE);
			string nextFilename = next.str();
			compareImg = imread(nextFilename, CV_LOAD_IMAGE_GRAYSCALE);
			
			// Get good features to track, which will be fed into the optical flow algorithm.
			goodFeaturesToTrack(source, points[0], MAX_COUNT, 0.01, MIN_DIST, Mat(), 3, 0, 0.04); // Get a field of 500 features to track
			cornerSubPix(source, points[0], winSize, Size(-1,-1), termcrit);

			Mat outImg;
			cvtColor(source, outImg, CV_GRAY2BGR);
			for (int k = 0; k < points[0].size(); k++){
			// Get the corners for the template and the search area, and fix for edge cases. 
				int xTempCorner = std::max( (int)points[0][k].x - templateXSize/2, 0);
				int yTempCorner = std::max( (int)points[0][k].y - templateYSize/2, 0);
				int xSearchCorner = std::max( (int)points[0][k].x - searchXSize/2, 0);
				int ySearchCorner = std::max( (int)points[0][k].y - searchYSize/2, 0);

				// Same for width of the ROIs. 
				int templateWidth = std::min(templateXSize, WIDTH - xTempCorner);
				int templateHeight = std::min(templateYSize, HEIGHT - yTempCorner);
				int searchWidth = std::min(searchXSize, WIDTH - xSearchCorner);
				int searchHeight = std::min(searchYSize, HEIGHT - ySearchCorner);

				if (!templateWidth || !templateHeight || !searchWidth || !searchHeight){
					continue;
				}

				Mat templateImg(source, Rect(xTempCorner, yTempCorner, templateWidth, templateHeight) );  // Smaller
				Mat comparison(compareImg, Rect(xSearchCorner, ySearchCorner , searchWidth, searchHeight) ); 

				// Formulate the result matrix, which is the size of the difference of the 2 mats + 1. This holds
				// the result of the convolutions.
				int rCols = comparison.cols - templateImg.cols + 1; // This is key... 9x9 total size of result.
				int rRows = comparison.rows - templateImg.rows + 1;
				Mat result = Mat(rRows, rCols, CV_8UC3);

				matchTemplate(comparison, templateImg, result, CV_TM_SQDIFF);//CV_TM_CCORR);
				normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );

				Point minLoc, maxLoc;
				double minVal, maxVal;
				// Finds the index of the max and min values. For CV_TM_SQDIFF, best match is at minLoc. 
				minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
				Point matchLoc = minLoc;  // This will be a location within the bigger Mat,
											// i.e. in the comparison rectangle. Has to be. 
				// After doing some working it out on paper, I came up with this as the correct algorithm for figuring out
				// where the corresponding point is in the larger image. The result vector's top left point is (searchCols - tempCols)/2 from the 
				// left of the search field, similar for y, and then add in the search field's x and y. 
				matchLoc = minLoc;
				matchLoc.x = xSearchCorner + matchLoc.x + (searchXSize - rCols) / 2;  // Add value to x and y to get a translation vector. 
				matchLoc.y = ySearchCorner + matchLoc.y + (searchYSize - rRows) / 2; 
				// Draw circle around previous point and a line pointing to next point. 
				circle(outImg,points[0][k],3,cv::Scalar(0,255,0),1);
				line(outImg, points[0][k], matchLoc, Scalar(0,0,255), 1);
			}
			
			imshow("Screen", outImg);
			for (int vid = 0; vid < 9; vid++){
				Vout << outImg;  // SUPER HACK!!! YAY! Video writer is super fast at 30fps, so write more frames. 
			}
			waitKey(0);
			
		}
	
	}
	
}
