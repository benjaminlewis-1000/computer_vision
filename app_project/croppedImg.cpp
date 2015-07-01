void drawCroppedImage(Mat& sourceImg, vector<Point> ROI){
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
	
	fillConvexPoly(mask, &ROI_Poly[0], ROI_Poly.size(), 255, 8, 0); 
	// Sanity check for outside the edges of the source image
	/*max_x = (max_x > sourceImg.cols) ? sourceImg.cols : max_x;
	max_y = (max_y > sourceImg.rows) ? sourceImg.rows : max_y;
	min_x = (min_x < 0 ) ? 0 : min_x;
	min_y = (min_y < 0 ) ? 0 : min_y;
	
	int width = max_x - min_x;
	int height = max_y - min_y;*/
	
//	int topLeftX = centerX - width  * scaleX / 2;
//	int topLeftY = centerY - height * scaleY / 2;
	
//	cout << topLeftX << " " << topLeftY << endl;
	
//	topLeftX = max(topLeftX, 0);
//	topLeftY = max(topLeftY, 0);
	
	
	// Move mask to the origin
//	translateImg(mask, (-1 * min_x) , (-1 * min_y)  );//, 50);
	
//	cout << mask.size() << endl;
			
//	Mat tmp = sourceImg(Rect(min_x, min_y, width, height) );
	
//	resize(tmp, tmp, Size(width * scaleX, height * scaleY), 0, 0, INTER_CUBIC);
//	resize(mask, mask, Size(mask.cols * scaleX, mask.rows * scaleY), 0, 0, INTER_CUBIC);
	
	// By playing around, it seems that the thing to do is move the mask to the origin, where it is 
	// applied to the dstImg(rectangle). Then that masked image is applied to the rectangle in the 
	// destination image. We don't need to move the mask to line up with the destination image. 
//	tmp.copyTo(dstImg(Rect(topLeftX, topLeftY, width * scaleX, height * scaleY) )  , mask);
}

