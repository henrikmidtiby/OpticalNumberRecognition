#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;

vector< vector< Point > > findContoursInFile(std::string filename)
{
	Mat image;
	Mat imageGray;
	image = imread(filename, 1);

	cvtColor( image, imageGray, CV_BGR2GRAY );

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Find contours
	findContours( imageGray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	return contours;
}

void analyzeContours(vector< vector< Point > > contours, int objectClass)
{
	for( int i = 0; i< contours.size(); i++ )
	{
		// Calculate perimeter length
		double perimeter = arcLength(contours[i], 1);
		double area = contourArea(contours[i], true);
		if(area <= 0)
		{
			continue;
		}

		double compactness = perimeter * perimeter / (4 * 3.141592 * area);
		printf("%d\t", objectClass);
		printf("%.3f\t%.0f\t%.3f", perimeter, area, compactness);

		/// Get the moments
		Moments mu;
		mu = moments(contours[i], false); 
		double hu[7];
		cv::HuMoments(mu, hu); 

		printf("\t%.0f", mu.mu20);
		printf("\t%.0f", mu.mu11);
		printf("\t%.0f", mu.mu02);
		printf("\t%.0f", mu.mu30);
		printf("\t%.0f", mu.mu21);
		printf("\t%.0f", mu.mu12);
		printf("\t%.0f", mu.mu03);


		printf("\t%.6f", hu[0]);
		printf("\t%.6f", hu[1]);
		printf("\t%.6f", hu[2]);
		printf("\t%.10f", hu[3]);
		printf("\t%.10f", hu[4]);
		printf("\t%.12f", hu[5]);
		printf("\t%.12f", hu[6]);
		printf("\n");
	}
	return;
}

int main( int argc, char** argv )
{
	printf("class\tperimeter\tarea\tcompactness\tmu20\tmu11\tmu02\tmu30\tmu21\tmu12\tmu03\thu1\thu2\thu3\thu4\thu5\thu6\thu7\n");
	analyzeContours(findContoursInFile("../numbers/1.png"), 1);
	analyzeContours(findContoursInFile("../numbers/2.png"), 2);
	analyzeContours(findContoursInFile("../numbers/3.png"), 3);
	analyzeContours(findContoursInFile("../numbers/4.png"), 4);
	analyzeContours(findContoursInFile("../numbers/5.png"), 5);
	analyzeContours(findContoursInFile("../numbers/6.png"), 6);
	analyzeContours(findContoursInFile("../numbers/7.png"), 7);
	analyzeContours(findContoursInFile("../numbers/8.png"), 8);
	analyzeContours(findContoursInFile("../numbers/9.png"), 9);

	return 0;
}

