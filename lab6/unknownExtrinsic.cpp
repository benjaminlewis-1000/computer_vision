#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <string>
#include <sstream>
#include <vector>
#include <iostream> 
#include <fstream>

using namespace std;
using namespace cv;

Mat cameraCalib;
float fx;
float fy;
float cx;
float cy;
Mat distortionParams;

struct correctMotion{ // Return struct for returning values from the findRandT function.
	Mat whichT;
	Mat whichR;
	Mat normE;
	double Zr;
};

// Helper function to split a string on a given delimiter and return it in the vector<string> elems.
vector<string>& split(const string &s, char delim, vector<string> &elems) {
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

// Helper function for reading a list of points that are in a file as xval,yval\n for each pair. 
void readPointList(string filename, vector<Point2f>& points){
	ifstream myfile;
	myfile.open(filename.c_str());
	if (!myfile.is_open()){
		cout << "File " << filename << " does not exist!" << endl;
		return;
	}
	
	string line;
	while (getline (myfile, line) ){
		vector<string> elems;
		split(line, ',', elems);
		float x = atof(elems[0].c_str());
		float y = atof(elems[1].c_str());
		points.push_back(Point2f(x, y) );
	}
	
	myfile.close();
	//cout << filename << " complete!\n";
}

// Read in the points, undistort them with the camera calibration and distortion matrices.
void readUndistort(string filename, vector<Point2f>& points){
	readPointList(filename, points);
	undistortPoints(points, points, cameraCalib, distortionParams);
	for (int i = 0; i < points.size(); i++){
		points[i] = Point2f(points[i].x * fx + cx, points[i].y * fy + cy);
	}
}

// Matrix transpose operation that doesn't have to have a Mat defined for the transpose. 
Mat trans(Mat M){
	Mat Mprime;
	transpose(M, Mprime);
	return Mprime;
}

// Nifty function found online for determining the type of a variable. 
string type2str(int type) {
// Useful debugging tool for finding the type of an array. 
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

// Helper function, convert a double to a string
string double2string(double number){
    std::ostringstream buff;
    buff<<number;
    return buff.str();   
}

// Convert any Mat into TEX format for LaTEX documents.
string texFormatMat(Mat M){
	string val = "\\begin{equation*}\n\\begin{bmatrix}\n";
	for (int i = 0; i < M.rows; i++){
		for (int j = 0; j < M.cols - 1; j++){
			val += double2string(M.at<double>(i, j)); 
			val += " & ";
		}
		val += double2string(M.at<double>(i, M.cols - 1) ); 
		if (i < M.rows - 1){
			val += " \\\\ ";
		}else{
			val += "\n\\end{bmatrix}\n\\end{equation*}\n";
		}
	}
	
	return val;
}

// The main function for this program.
correctMotion findCorrectRandT(Mat F, vector<Point2f> start, vector<Point2f> end, ofstream& output, string sequence, int point ) {

	correctMotion retVal; 
	Mat K = cameraCalib;
	
	Mat E = trans(K) * F * K;  // Compute the essential matrix from the fundamental matrices.
	
	// Step 1: Obtain normalized E
	Mat W, U, Vt;
	SVD::compute(E, W, U, Vt);
	Mat sigma = (Mat_<double>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 0);
	
	retVal.normE =  U * sigma * Vt;
	
	Mat RzT_plus  = (Mat_<double>(3,3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
	Mat RzT_minus = (Mat_<double>(3,3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
	
	// Step 2: Calculate all possible T and R
	Mat R1, R2, That1, That2;
	R1 = U * RzT_plus * Vt;
	R2 = U * RzT_minus * Vt;
	That1 = U * trans(RzT_plus) * sigma * trans(U);
	That2 = U * trans(RzT_minus) * sigma * trans(U);
	double Tx1, Ty1, Tz1, Tx2, Ty2, Tz2;
	Tx1 = That1.at<double>(2,1);
	Ty1 = That1.at<double>(0,2);
	Tz1 = That1.at<double>(1,0);
	Tx2 = That2.at<double>(2,1);
	Ty2 = That2.at<double>(0,2);
	Tz2 = That2.at<double>(1,0);
	
	Mat T1 = (Mat_<double>(3,1) << Tx1, Ty1, Tz1);
	Mat T2 = (Mat_<double>(3,1) << Tx2, Ty2, Tz2);
	
	// Step 3: Reconstruct Zl and Zr using triangulation.
	
	// I can get P_phys in the left using the equation on slide 27 of lecture "3D Reconstruction", triangulation method. 
	// Pl = (fr * R1 - xr * R3) * T / (fr*R1 - xr *R3), where I assume R1 and R3 are rows of the rotation matrix.
	// Iterate through all four combinations, and I get two physically viable points, i.e. that have a positive Z. 
	Mat Zl[4];
	//double point = 1;
	Zl[0] = fx * (fx * R1.row(0) - start[point].x * R1.row(2) ) * T1 / ( (fx * R1.row(0) - start[point].x * R1.row(2)) * ( Mat_<double>(3,1) << start[point].x, start[point].y, 1) );
	Zl[1] = fx * (fx * R1.row(0) - start[point].x * R1.row(2) ) * T2 / ( (fx * R1.row(0) - start[point].x * R1.row(2)) * ( Mat_<double>(3,1) << start[point].x, start[point].y, 1) );
	Zl[2] = fx * (fx * R2.row(0) - start[point].x * R2.row(2) ) * T1 / ( (fx * R2.row(0) - start[point].x * R2.row(2)) * ( Mat_<double>(3,1) << start[point].x, start[point].y, 1) );
	Zl[3] = fx * (fx * R2.row(0) - start[point].x * R2.row(2) ) * T2 / ( (fx * R2.row(0) - start[point].x * R2.row(2)) * ( Mat_<double>(3,1) << start[point].x, start[point].y, 1) );
	
	Mat Zr[4];
	Zr[0] = fx * (fx * R1.row(0) - end[point].x * R1.row(2) ) * T1 / ( (fx * R1.row(0) - end[point].x * R1.row(2)) * ( Mat_<double>(3,1) << end[point].x, end[point].y, 1) );
	Zr[1] = fx * (fx * R1.row(0) - end[point].x * R1.row(2) ) * T2 / ( (fx * R1.row(0) - end[point].x * R1.row(2)) * ( Mat_<double>(3,1) << end[point].x, end[point].y, 1) );
	Zr[2] = fx * (fx * R2.row(0) - end[point].x * R2.row(2) ) * T1 / ( (fx * R2.row(0) - end[point].x * R2.row(2)) * ( Mat_<double>(3,1) << end[point].x, end[point].y, 1) );
	Zr[3] = fx * (fx * R2.row(0) - end[point].x * R2.row(2) ) * T2 / ( (fx * R2.row(0) - end[point].x * R2.row(2)) * ( Mat_<double>(3,1) << end[point].x, end[point].y, 1) );
	
	int which;
	int count = 0;
	for (int i = 0; i < 4; i++){
		if ( (Zr[i].at<double>(0) > 0) && (Zl[i].at<double>(0) > 0) ){
		//	cout << i << " is good\n";
		//	cout << Zr[i];
			count++;
			which = i;
			retVal.Zr = Zr[i].at<double>(0);
		}
	}
	
	if (count != 1){
		//cout << "Error! We have the wrong number of solutions: " << count << endl;
	}
	
	// Z in frame "end", i.e. the 10th frame, is going to be Zr. (We're moving left, therefore points 
	// move to the right as the sequence progresses. 
		
/*	Mat Pr[4];
	Pr[0] = R1 * (trans(Zl[0] ) - T1);
	Pr[1] = R1 * (trans(Zl[1] ) - T2);
	Pr[2] = R2 * (trans(Zl[2] ) - T1);
	Pr[3] = R2 * (trans(Zl[3] ) - T2);*/
	// OK, cool, T2 looks right. Pr = R * (Pl - T). L -(-x) = Something to the right, so good. X value should be negative.
	// Now, my guess is that viable pairs are 1-3 and 2-4, but we'll play it safe. 
	
/*	for (int i = 0; i < 4; i++){
		if (Pl[i].at<double>(2) > 0){
			cout << i << "   " ;
		}
	}
	cout << endl;*/
	
	int ans_index = -1;
	
	/*for (int i = 0; i < 4; i++){
		if ( (Pr[i].at<double>(2) > 0) && (Pl[i].at<double>(2) > 0) ){
			ans_index = i;
			cout << "possible answer is " << i << endl;
		}
	}
	cout << "The answer is " << ans_index << endl;
	cout << T1 << endl << T2 << endl;
	*/
	switch(which){
		case 0:
			retVal.whichR = R1;
			retVal.whichT = T1;
			break;
		case 1:
			retVal.whichR = R1;
			retVal.whichT = T2;
			break;
		case 2:
			retVal.whichR = R2;
			retVal.whichT = T1;
			break;
		case 3:
			retVal.whichR = R2;
			retVal.whichT = T2;
			break;
		default:
			retVal.whichR = (Mat_<double>(3,3) << 99999, 99999, 99999, 99999, 99999, 99999, 99999, 99999, 99999);
			retVal.whichT = (Mat_<double>(3,1) << 99999, 99999, 99999);
	}
			
	output << "F -- " << sequence << ": \n" << texFormatMat(F) << endl << "E -- " << sequence << ": \n" <<
		texFormatMat(retVal.normE) << endl	<< "R -- " << sequence << ": \n"  << texFormatMat(retVal.whichR) <<
		endl << "T -- " << sequence << ": \n" << texFormatMat( trans(retVal.whichT ) ) << endl;
	return retVal;
}

int clicks = 0;
vector<Point2f> ROI_corners;
RNG rng;

void mouseHandler(int event, int x, int y, int flags, void* param)
{
    /* user press left button */
    Mat* img = (Mat*) param;
    if (event == CV_EVENT_LBUTTONDOWN )
    {
    	if (clicks == 2){
    		clicks = 1;
    		ROI_corners.clear();
    	}else{
	    	clicks++;
	    }
        Point2f point = Point2f(x, y);
        ROI_corners.push_back(point);
		//cout << x << "  " << y << endl;
		
		if (clicks == 2){
			Point2f corner1 = ROI_corners[0];
			Point2f corner2 = ROI_corners[1];
			rectangle(*img, corner1, corner2, Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255)), 1, 8);
			imshow("Screen", *img);
		}
    }
}


		
int isIn(Point2f test, int top, int bottom, int left, int right){
	if (test.x < right && test.x > left && test.y > top && test.y < bottom){
		return 1;
	}
	return 0;
}

int main(int argc, char** argv){

	// Camera calibration and distortion matrices are now given for this assignment. fx, fy, cx, and cy can be extracted. 
	cameraCalib = (Mat_<double>(3,3) << 825.0900600547, 0.0000, 331.6538103208, 0.00, 824.2672147458, 252.9284287373, 0.0, 0.0, 1.0);
	fx = cameraCalib.at<double>(0, 0);
	fy = cameraCalib.at<double>(1, 1);
	cx = cameraCalib.at<double>(0, 2);
	cy = cameraCalib.at<double>(1, 2);
	distortionParams = (Mat_<double>(1,5) << -0.2380769337, 0.0931325835, 0.0003242537, -0.0021901930, 0.4641735616);

	Mat F_paraReal, F_paraCube, F_turnCube, F_turnReal;
	
	vector<Point2f> paraCubeStart;
	readUndistort("paraCubeStart.txt", paraCubeStart);
	
	vector<Point2f> paraCubeEnd;
	readUndistort("paraCubeEnd.txt", paraCubeEnd);
	
	vector<Point2f> paraRealStart;
	readUndistort("paraRealStart.txt", paraRealStart);
	
	vector<Point2f> paraRealEnd;
	readUndistort("paraRealEnd.txt", paraRealEnd);
	
	vector<Point2f> turnCubeStart;
	readUndistort("turnCubeStart.txt", turnCubeStart);
	
	vector<Point2f> turnCubeEnd;
	readUndistort("turnCubeEnd.txt", turnCubeEnd);
	
	vector<Point2f> turnRealStart;
	readUndistort("turnRealStart.txt", turnRealStart);
	
	vector<Point2f> turnRealEnd;
	readUndistort("turnRealEnd.txt", turnRealEnd);
	
	F_paraReal = findFundamentalMat(paraRealStart, paraRealEnd, FM_RANSAC, 1, 0.99);
	F_paraCube = findFundamentalMat(paraCubeStart, paraCubeEnd, FM_RANSAC, 1, 0.99);
	F_turnReal = findFundamentalMat(turnRealStart, turnRealEnd, FM_RANSAC, 1, 0.99);
	F_turnCube = findFundamentalMat(turnCubeStart, turnCubeEnd, FM_RANSAC, 1, 0.99);

	ofstream myfile;
	myfile.open("task2_params.tex");
	
	correctMotion ansParaReal = findCorrectRandT(F_paraReal, paraRealStart, paraRealEnd, myfile, "parallel real", 1);
	correctMotion ansParaCube = findCorrectRandT(F_paraCube, paraCubeStart, paraCubeEnd, myfile, "parallel cube", 1);
	correctMotion ansTurnReal = findCorrectRandT(F_turnReal, turnRealStart, turnRealEnd, myfile, "turned real", 1);
	correctMotion ansTurnCube = findCorrectRandT(F_turnCube, turnCubeStart, turnCubeEnd, myfile, "turned cube", 1);

	myfile.close();
	
	myfile.open("junk~");

// We have all of these on frame 10 for end points.

	namedWindow("Screen", WINDOW_NORMAL);
	Mat image = imread("turned_cube/TurnCube10.jpg");
	cvSetMouseCallback("Screen", mouseHandler, &image);
	imshow("Screen", image);
	// With turnCubeEnd, find points in ROI.
	waitKey(0);
	/******************************************************/
	/*    Turned Cube									  */
	/******************************************************/

	vector<int> indices;
	if (clicks == 2){
		// I've got a ROI. 
		cout << "ROI found" << turnCubeEnd.size()  << endl;
		Point2f corners1 = ROI_corners[0];
		Point2f corners2 = ROI_corners[1];
		int left, right, top, bottom;
		if (corners1.x < corners2.x){
			left = corners1.x;
			right = corners2.x;
		}else{
			left = corners2.x;
			right = corners1.x;
		}
		if (corners1.y < corners2.y){
			top = corners1.y;
			bottom = corners2.y;
		}else{
			top = corners2.y;
			bottom = corners1.y;
		}
		for (int i = 0; i < turnCubeEnd.size(); i++){
			if ( isIn(turnCubeEnd[i], top, bottom, left, right) ){
				indices.push_back(i);
			//	cout << i << "   ";
				circle(image, turnCubeEnd[i], 3, Scalar(0,255,36) );
				circle(image, turnCubeStart[i], 3, Scalar(255,255,36) );
			}
		}// Indices holds the points that are within the ROI.
	}
	imshow("Screen", image);
	
	// Now I need to find what the closest point is, scale it in Z to 20 inches. 
	double closestZr = 100;
	int closestIndex = 0;
	for (int i = 0; i < indices.size(); i++){
		correctMotion tmp = findCorrectRandT(F_turnCube, turnCubeStart, turnCubeEnd, myfile, "turned cube", indices[i]);
		if (tmp.Zr < closestZr){
			closestZr = tmp.Zr;
			closestIndex = i;
			cout << "A closer one is at: " << closestZr << "  " << i << endl;
		}
	}
	cout << closestIndex << endl;
	correctMotion tmp = findCorrectRandT(F_turnCube, turnCubeStart, turnCubeEnd, myfile, "turned cube", closestIndex);
	double turnScaleFactor = 20/tmp.Zr;
	cout << "Turned Cube T: " << endl << tmp.whichT * turnScaleFactor << endl;
	
	//waitKey(0);
	
	/******************************************************/
	/*    Parallel Cube									  */
	/******************************************************/
	
	indices.clear();
	clicks = 0;
	ROI_corners.clear();
	
	image = imread("parallel_cube/ParallelCube10.jpg");
	cvSetMouseCallback("Screen", mouseHandler, &image);
	imshow("Screen", image);
	waitKey(0);

	if (clicks == 2){
		// I've got a ROI. 
		cout << "ROI found" << paraCubeEnd.size()  << endl;
		vector<int> indices;
		Point2f corners1 = ROI_corners[0];
		Point2f corners2 = ROI_corners[1];
		int left, right, top, bottom;
		if (corners1.x < corners2.x){
			left = corners1.x;
			right = corners2.x;
		}else{
			left = corners2.x;
			right = corners1.x;
		}
		if (corners1.y < corners2.y){
			top = corners1.y;
			bottom = corners2.y;
		}else{
			top = corners2.y;
			bottom = corners1.y;
		}
		for (int i = 0; i < paraCubeEnd.size(); i++){
			if ( isIn(paraCubeEnd[i], top, bottom, left, right) ){
				indices.push_back(i);
			//	cout << i << "   ";
				circle(image, paraCubeEnd[i], 3, Scalar(0,255,36) );
				circle(image, paraCubeStart[i], 3, Scalar(255,255,36) );
			}
		}// Indices holds the points that are within the ROI.
	}
	imshow("Screen", image);
	//waitKey(0);
	
	closestZr = 100.0;
	closestIndex = 0;
	for (int i = 0; i < indices.size(); i++){
		correctMotion tmp = findCorrectRandT(F_paraCube, paraCubeStart, paraCubeEnd, myfile, "parallel cube", indices[i]);
		if (tmp.Zr < closestZr){
			closestZr = tmp.Zr;
			closestIndex = i;
			cout << "A closer one is at: " << closestZr << "  " << i << endl;
		}
	}
	cout << closestIndex << endl;
	tmp = findCorrectRandT(F_paraCube, paraCubeStart, paraCubeEnd, myfile, "parallel cube", closestIndex);
	double paraScaleFactor = 20/tmp.Zr;
	cout << "Parallel Cube T: " << endl << tmp.whichT * paraScaleFactor << endl;
	
	
	/********************************************************/
	/* Apply these same scale factors to the Real sequences */
	/********************************************************/
	Mat T_parallelReal = paraScaleFactor * ansParaReal.whichT;
	Mat T_turnReal = turnScaleFactor * ansTurnReal.whichT;
	
	cout << "Turned Real T: " << endl << T_turnReal << endl;
	cout << "Parallel Real T: " << endl << T_parallelReal << endl;
	
	cout << endl << "Parallel Cube" << endl << endl;
	
	int randPoints[4] = {3, 26, 49, 78};
	string nums[4] = {"3", "26", "49", "78"};
	Mat paraCubeIm = imread("parallel_cube/ParallelCube5.jpg");
	for (int i = 0; i < 4; i++){
		int index = randPoints[i]; 
		tmp = findCorrectRandT(F_paraCube, paraCubeStart, paraCubeEnd, myfile, "parallel cube", index);
		double realX = paraCubeStart[index].x * tmp.Zr / 825 * paraScaleFactor; // 825 is about the focal length. 
		double realY = paraCubeStart[index].y * tmp.Zr / 825 * paraScaleFactor; // 825 is about the focal length. 
		circle(paraCubeIm, paraCubeStart[index], 3, Scalar(255, 255, 0)) ;
		cout << "Real-world xyz coordinates for point " << index << " are " << realX << ", " << realY << ", " << tmp.Zr  * paraScaleFactor << endl;
		putText(paraCubeIm, nums[i], paraCubeStart[index], FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255) );
	}
	imwrite("parallelCube_3dinfo.jpg", paraCubeIm);
	
	cout << endl << "Turned Cube" << endl << endl;
	
	Mat turnCubeIm = imread("turned_cube/TurnCube5.jpg");
	for (int i = 0; i < 4; i++){
		int index = randPoints[i]; 
		tmp = findCorrectRandT(F_turnCube, turnCubeStart, turnCubeEnd, myfile, "turned cube", index);
		double realX = turnCubeStart[index].x * tmp.Zr / 825 * turnScaleFactor; // 825 is about the focal length. 
		double realY = turnCubeStart[index].y * tmp.Zr / 825 * turnScaleFactor; // 825 is about the focal length. 
		circle(turnCubeIm, turnCubeStart[index], 3, Scalar(255, 255, 0)) ;
		cout << "Real-world xyz coordinates for point " << index << " are " << realX << ", " << realY << ", " << tmp.Zr  * turnScaleFactor << endl;
		putText(turnCubeIm, nums[i], turnCubeStart[index], FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255) );
	}
	imwrite("turnCube_3dinfo.jpg", turnCubeIm);
	
	cout << endl << "Parallel Real" << endl << endl;
	
	Mat paraRealIm = imread("parallel_real/ParallelReal5.jpg");
	for (int i = 0; i < 4; i++){
		int index = randPoints[i]; 
		tmp = findCorrectRandT(F_paraReal, paraRealStart, paraRealEnd, myfile, "parallel real", index);
		double realX = paraRealStart[index].x * tmp.Zr / 825 * paraScaleFactor; // 825 is about the focal length. 
		double realY = paraRealStart[index].y * tmp.Zr / 825 * paraScaleFactor; // 825 is about the focal length. 
		circle(paraRealIm, paraRealStart[index], 3, Scalar(255, 255, 0)) ;
		cout << "Real-world xyz coordinates for point " << index << " are " << realX << ", " << realY << ", " << tmp.Zr  * paraScaleFactor << endl;
		putText(paraRealIm, nums[i], paraRealStart[index], FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255) );
	}
	imwrite("parallelreal_3dinfo.jpg", paraRealIm);
	
	cout << endl << "Turned Real" << endl << endl;
	
	Mat turnRealIm = imread("turned_real/TurnReal5.jpg");
	for (int i = 0; i < 4; i++){
		int index = randPoints[i]; 
		tmp = findCorrectRandT(F_turnReal, turnRealStart, turnRealEnd, myfile, "turned real", index);
		double realX = turnRealStart[index].x * tmp.Zr / 825 * turnScaleFactor; // 825 is about the focal length. 
		double realY = turnRealStart[index].y * tmp.Zr / 825 * turnScaleFactor; // 825 is about the focal length. 
		circle(turnRealIm, turnRealStart[index], 3, Scalar(255, 255, 0)) ;
		cout << "Real-world xyz coordinates for point " << index << " are " << realX << ", " << realY << ", " << tmp.Zr  * turnScaleFactor << endl;
		putText(turnRealIm, nums[i], turnRealStart[index], FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255) );
	}
	imwrite("turnReal_3dinfo.jpg", turnRealIm);
	
	// x_real = x_image/focal_length * Z_real
	// Similar for y_real
	
	//waitKey(0);
	
	myfile.close();
	
	
}
