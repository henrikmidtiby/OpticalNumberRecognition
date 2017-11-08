#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;

int main( int argc, char** argv )
{
	Mat image;
	Mat imageGray;
	image = imread("../shapes.png", 1);

	cvtColor( image, imageGray, CV_BGR2GRAY );


	Mat canny_output;
	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;

	/// Detect edges using canny
	int thresh = 100;
	Canny( imageGray, canny_output, thresh, thresh*2, 3 );

	/// Find contours
	findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	// Show input image
	namedWindow( "Display Image", WINDOW_AUTOSIZE );
	imshow( "Display Image", imageGray );

	// Start random number generator with a known seed
	RNG rng(12345);

	/// Draw contours
	Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
	for( int i = 0; i< contours.size(); i++ )
	{
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
	}

	/// Show in a window
	namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
	imshow( "Contours", drawing );

	waitKey(0);

	return 0;
}
