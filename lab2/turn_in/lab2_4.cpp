#include <iostream>
#include <boost/algorithm/string.hpp>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace boost;
using namespace std;

int main(int argc, char** argv){
	int numPoints = 20;
	ifstream dataPoints;
	dataPoints.open("Data_points.txt");
	string line;
	vector<Point3f>  objectPoints;
	vector<Point2f>  imagePoints;

	for (int i = 0; i < numPoints; i++){
		getline(dataPoints,line);
		istringstream iss(line);
		vector<string> fields;
		split(fields, line, boost::is_any_of(" ") );
		vector<Point2f> obj;
		imagePoints.push_back(Point2f( (double)atof(fields[0].c_str()), (double)atof(fields[1].c_str()) ) );
		//imagePoints.push_back(obj);
	}
	for (int i = 0; i < numPoints; i++){
		getline (dataPoints,line);
		istringstream iss(line);
		vector<string> fields;
		split(fields, line, boost::is_any_of(" ") );
	//	vector<Point3f> obj;
		objectPoints.push_back(Point3f( (double)atof(fields[0].c_str()), (double)atof(fields[1].c_str()),  (double)atof(fields[2].c_str()) ) );
		//objectPoints.push_back(obj);
	}

	//cout << objectPoints.size() << " " << imagePoints.size() << endl;

	// Read in calibration parameters from task 2
	// Read in intrinsic matrix
	ifstream intrinsicFile;
	intrinsicFile.open("intrinsic.txt");
	getline (intrinsicFile,line);
	istringstream iss(line);
	vector<string> fields;
	split(fields, line, boost::is_any_of(" ") );

	Mat intrinsics = Mat::ones(3,3, CV_64F);

	for(int i = 0; i < fields.size(); i++){
		intrinsics.at<double>(i / 3, i %3) = (double)atof(fields[i].c_str());
	}

	//cout << intrinsics << endl;

	//Read in distortion coefficient file
	ifstream distortionFile;
	distortionFile.open("distortion.txt");
	getline (distortionFile,line);
	istringstream issDist(line);
	split(fields, line, boost::is_any_of(" ") );

	Mat distCoeffs = Mat::ones(5,1,CV_64F);

	for(int i = 0; i < 5; i++){
		distCoeffs.at<double>(i) = (double)atof(fields[i].c_str() );
	}
	
	//cout << distCoeffs.size() << endl;
	Mat rvecs, tvecs; 

	solvePnP(objectPoints, imagePoints, intrinsics, distCoeffs, rvecs, tvecs, false, CV_ITERATIVE);

	Rodrigues(rvecs, rvecs); // Convert from the phi, theta, psi of rotation vector to the rotation matrix.

	cout << rvecs << endl << tvecs << endl;
}
