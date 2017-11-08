#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;

int main( int argc, char** argv )
{
	Mat image;
	Mat imageGray;
	image = imread("../shapes.png", 1);

	cvtColor( image, imageGray, CV_BGR2GRAY );

	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;

	/// Find contours
	findContours( imageGray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	// Start random number generator with a known seed
	RNG rng(12345);

	/// Draw contours
	Mat drawing = Mat::zeros( image.size(), CV_8UC3 );
	for( int i = 0; i< contours.size(); i++ )
	{
		// Calculate perimeter length
		double perimeter = arcLength(contours[i], 1);
		double area = contourArea(contours[i], true);
		double compactness = perimeter * perimeter / (4 * 3.141592 * area);
		printf("perimeter: %8.3f  area: %8.3f   compactness: %8.3f\n", perimeter, area, compactness);

		if(compactness < 1.4)
		{
			// Draw contour
			Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
			drawContours( drawing, contours, i, color, 4, 8, hierarchy, 0, Point() );
		}
		else
		{
			// Draw contour
			Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
			drawContours( drawing, contours, i, color, 1, 8, hierarchy, 0, Point() );
		}
	
	}

	/// Show in a window
	namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
	imshow( "Contours", drawing );

	waitKey(0);

	return 0;
}

