#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;

std::vector< std::vector< Point > > findContoursInFile(char* filename)
{
	Mat image;
	Mat imageGray;
	image = imread(filename, 1);

	cvtColor( image, imageGray, CV_BGR2GRAY );

	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;

	/// Find contours
	findContours( imageGray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	return contours;
}

int main( int argc, char** argv )
{
	Mat image;
	image = imread("../as.png", 1);

	std::vector< std::vector< Point > >contours = findContoursInFile("../as.png");
	
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

		if(area > 0)
		{
			printf("perimeter: %8.3f  area: %8.3f   compactness: %8.3f\n", perimeter, area, compactness);

			/// Get the moments
			Moments mu;
			mu = moments(contours[i], false); 
			double hu[7];
			cv::HuMoments(mu, hu); 

			// Draw contour
			Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
			drawContours( drawing, contours, i, color, 1, 8);
		}
	}
	/// Show in a window
	namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
	imshow( "Contours", drawing );
	
	waitKey(0);


	// Set up training data
	float labels[4] = {1.0, -1.0, -1.0, -1.0};
	Mat labelsMat(4, 1, CV_32FC1, labels);
	
	float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
	Mat trainingDataMat(4, 2, CV_32FC1, trainingData);
	
	// Train the SVM
	cv::ml::SVM* svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm->train(trainingDataMat, cv::ml::ROW_SAMPLE, labelsMat);

	// Use the SVM
	float testData[1][2] = { {601, 10} };
	Mat testDataMat(1, 2, CV_32FC1, testData);
        float response = svm->predict(testDataMat);

	printf("Response: %5.3f\n", response);

	return 0;
}

