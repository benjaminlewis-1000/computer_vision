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

std::vector<std::string>& split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

void readPointList(string filename, vector<Point2f>& points){
	ifstream myfile;
	myfile.open(filename.c_str());
	if (!myfile.is_open()){
		cout << "File " << filename << " does not exist!" << endl;
		return;
	}
	
	string line;
	//vector<Point2f> points;
	while (getline (myfile, line) ){
		//cout << line << endl;
		vector<string> elems;
		split(line, ',', elems);
		double x = atof(elems[0].c_str());
		double y = atof(elems[1].c_str());
		points.push_back(Point2f(x, y) );
	}
	
	myfile.close();
	cout << filename << " complete!\n";
}

void writeF(string filename, Mat F){
	ofstream myfile;
	myfile.open(filename.c_str());
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			myfile << F.at<double>(i, j) << ",";
		}
	}
	myfile.close();
}

int main(int argc, char** argv){
	// Read in the pairs of start and end corresponding points

	vector<Point2f> paraCubeStart;
	readPointList("paraCubeStart.txt", paraCubeStart);
	
	vector<Point2f> paraCubeEnd;
	readPointList("paraCubeEnd.txt", paraCubeEnd);
	
	vector<Point2f> paraRealStart;
	readPointList("paraRealStart.txt", paraRealStart);
	
	vector<Point2f> paraRealEnd;
	readPointList("paraRealEnd.txt", paraRealEnd);
	
	vector<Point2f> turnCubeStart;
	readPointList("turnCubeStart.txt", turnCubeStart);
	
	vector<Point2f> turnCubeEnd;
	readPointList("turnCubeEnd.txt", turnCubeEnd);
	
	vector<Point2f> turnRealStart;
	readPointList("turnRealStart.txt", turnRealStart);
	
	vector<Point2f> turnRealEnd;
	readPointList("turnRealEnd.txt", turnRealEnd);
	
	// Compute fundamental matrices for each pair
	Mat F_paraCube, F_paraReal, F_turnCube, F_turnReal;
	
	vector<uchar> status;
	F_paraCube = findFundamentalMat(paraCubeStart, paraCubeEnd, FM_RANSAC, 2, 0.99, status);
	// paraCube is having issues; calculate RANSAC of fundamental matrix and throw out points that don't work.
	//cout << paraCubeStart.size() << endl;
	
	/*for (int i = paraCubeStart.size() - 1; i >= 0; i--){
		if (status[i] == 0){
			paraCubeStart.erase(paraCubeStart.begin() + i);
			paraCubeEnd.erase(paraCubeEnd.begin() + i);
		}
	}*/
	
	//cout << paraCubeStart.size() << endl;
	//writeF("F_paraCube.txt", F_paraCube);
	F_paraReal = findFundamentalMat(paraRealStart, paraRealEnd, FM_RANSAC, 2, 0.99);
	cout << F_paraReal << endl << endl;
	//writeF("F_paraReal.txt", F_paraReal);
	F_turnCube = findFundamentalMat(turnCubeStart, turnCubeEnd, FM_RANSAC, 2, 0.99);
	//writeF("F_turnCube.txt", F_turnCube);
	F_turnReal = findFundamentalMat(turnRealStart, turnRealEnd, FM_RANSAC, 2, 0.99);
	//writeF("F_turnReal.txt", F_turnReal);
	
	Mat H1_paraCube, H2_paraCube, H1_paraReal, H2_paraReal, H1_turnCube, H2_turnCube, H1_turnReal, H2_turnReal;
	
	stereoRectifyUncalibrated(paraCubeStart, paraCubeEnd, F_paraCube, Size(640, 480), H1_paraCube, H2_paraCube);
	
	stereoRectifyUncalibrated(paraRealStart, paraRealEnd, F_paraReal, Size(640, 480), H1_paraReal, H2_paraReal);
	stereoRectifyUncalibrated(turnCubeStart, turnCubeEnd, F_turnCube, Size(640, 480), H1_turnCube, H2_turnCube);
	stereoRectifyUncalibrated(turnRealStart, turnRealEnd, F_turnReal, Size(640, 480), H1_turnReal, H2_turnReal);
	
	Mat M = (Mat_<double>(3,3) << 1500, 0, 320, 0, 1500, 240, 0, 0, 1); // Going to be a rough guess at the camera matrix
	Mat M2 = (Mat_<double>(3,3) << 1145.242651309859, 0, 328.1354097078708, 0, 1143.630735895785, 222.3441185708712, 0, 0, 1); // Going to be a rough guess at the camera matrix
	Mat R1_paraCube = M2.inv() * H1_paraCube * M2;
	Mat R2_paraCube = M2.inv() * H2_paraCube * M2;
	Mat R1_paraReal = M.inv() * H1_paraReal * M;
	Mat R2_paraReal = M.inv() * H2_paraReal * M;
	Mat R1_turnCube = M.inv() * H1_turnCube * M;
	Mat R2_turnCube = M.inv() * H2_turnCube * M;
	Mat R1_turnReal = M.inv() * H1_turnReal * M;
	Mat R2_turnReal = M.inv() * H2_turnReal * M;
	
	cout << F_paraCube << endl;
	
	Mat distCoeffs_paraReal = (Mat_<double>(5,1) << 0, 0, 0, 0, 0);//0.05, .2, 0.01, -0.004, -.002); // Just a guess all around. 
	Mat distCoeffs_paraCube = distCoeffs_paraReal;
	Mat distCoeffs_turnCube = distCoeffs_paraReal;
	Mat distCoeffs_turnReal = distCoeffs_paraReal;
	
	const int start = 9;
	const int end = 14;
	
	//cout << "R1 values are:: " << endl << R1_turnCube << endl << R1_paraCube << endl << R1_paraReal << endl << R1_turnReal <<  endl;
	
	Mat paraCubeStartPic = imread("parallel_cube/ParallelCube9.jpg");
	Mat paraCubeEndPic = imread("parallel_cube/ParallelCube14.jpg");
	Mat paraRealStartPic = imread("parallel_real/ParallelReal9.jpg");
	Mat paraRealEndPic = imread("parallel_real/ParallelReal14.jpg");
	Mat turnCubeStartPic = imread("turned_cube/TurnCube9.jpg");
	Mat turnCubeEndPic = imread("turned_cube/TurnCube14.jpg");
	Mat turnRealStartPic = imread("turned_real/TurnReal9.jpg");
	Mat turnRealEndPic = imread("turned_real/TurnReal14.jpg");
	
	Mat map1, map2;
	
	initUndistortRectifyMap(M, distCoeffs_turnCube, R1_turnCube, M, Size(640, 480), CV_32FC1, map1, map2);
	remap(turnCubeStartPic, turnCubeStartPic, map1, map2, INTER_CUBIC);
	for (int i = 0; i < 10; i++)
		line(turnCubeStartPic, Point(0, i * 43 + 20), Point(640, i * 43 + 20), Scalar(20, 100, 70) );
	imshow("lol", turnCubeStartPic);
	
	initUndistortRectifyMap(M, distCoeffs_turnCube, R2_turnCube, M, Size(640, 480), CV_32FC1, map1, map2);
	remap(turnCubeEndPic, turnCubeEndPic, map1, map2, INTER_CUBIC);
	for (int i = 0; i < 10; i++)
		line(turnCubeEndPic, Point(0, i * 43 + 20), Point(640, i * 43 + 20), Scalar(20, 100, 70) );
	imshow("lol2", turnCubeEndPic);
	waitKey(0);
	
	initUndistortRectifyMap(M2, distCoeffs_paraCube, R2_paraCube, M2, Size(640, 480), CV_32FC1, map1, map2);
	remap(paraCubeStartPic, paraCubeStartPic, map1, map2, INTER_CUBIC);
	for (int i = 0; i < 10; i++)
		line(paraCubeStartPic, Point(0, i * 43 + 20), Point(640, i * 43 + 20), Scalar(200, 100, 10) );
	imshow("lol", paraCubeStartPic);
	
	initUndistortRectifyMap(M2, distCoeffs_paraCube, R2_paraCube, M2, Size(640, 480), CV_32FC1, map1, map2);
	remap(paraCubeEndPic, paraCubeEndPic, map1, map2, INTER_CUBIC);
	for (int i = 0; i < 10; i++)
		line(paraCubeEndPic, Point(0, i * 43 + 20), Point(640, i * 43 + 20), Scalar(200, 100, 10) );
	imshow("lol2", paraCubeEndPic);
	waitKey(0);
	
	initUndistortRectifyMap(M, distCoeffs_paraReal, R1_paraReal, M, Size(640, 480), CV_32FC1, map1, map2);
	remap(paraRealStartPic, paraRealStartPic, map1, map2, INTER_CUBIC);
	for (int i = 0; i < 10; i++)
		line(paraRealStartPic, Point(0, i * 43 + 20), Point(640, i * 43 + 20), Scalar(200, 100, 10) );
	imshow("lol", paraRealStartPic);
	
	initUndistortRectifyMap(M, distCoeffs_paraReal, R2_paraReal, M, Size(640, 480), CV_32FC1, map1, map2);
	remap(paraRealEndPic, paraRealEndPic, map1, map2, INTER_CUBIC);
	for (int i = 0; i < 10; i++)
		line(paraRealEndPic, Point(0, i * 43 + 20), Point(640, i * 43 + 20), Scalar(200, 100, 10) );
	imshow("lol2", paraRealEndPic);
	waitKey(0);
	
	initUndistortRectifyMap(M, distCoeffs_turnReal, R1_turnReal, M, Size(640, 480), CV_32FC1, map1, map2);
	remap(turnRealStartPic, turnRealStartPic, map1, map2, INTER_CUBIC);
	for (int i = 0; i < 10; i++)
		line(turnRealStartPic, Point(0, i * 43 + 20), Point(640, i * 43 + 20), Scalar(200, 100, 10) );
	imshow("lol", turnRealStartPic);
	
	initUndistortRectifyMap(M, distCoeffs_turnReal, R2_turnReal, M, Size(640, 480), CV_32FC1, map1, map2);
	remap(turnRealEndPic, turnRealEndPic, map1, map2, INTER_CUBIC);
	for (int i = 0; i < 10; i++)
		line(turnRealEndPic, Point(0, i * 43 + 20), Point(640, i * 43 + 20), Scalar(200, 100, 10) );
	imshow("lol2", turnRealEndPic);
	waitKey(0);
	
//	imwrite("paraCubeS.jpg", paraCubeStartPic);
//	imwrite("paraCubeE.jpg", paraCubeEndPic);
	imwrite("turnCubeS.jpg", turnCubeStartPic);
	imwrite("turnCubeE.jpg", turnCubeEndPic);
	imwrite("paraRealS.jpg", paraRealStartPic);
	imwrite("paraRealE.jpg", paraRealEndPic);
	imwrite("turnRealS.jpg", turnRealStartPic);
	imwrite("turnRealE.jpg", turnRealEndPic);

	cout << "\nProgram complete!\n";
	
}
