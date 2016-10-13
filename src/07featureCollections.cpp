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

std::vector< std::vector <double > > analyzeContours(vector< vector< Point > > contours, int objectClass, std::vector< std::vector <double > > output)
{
	for( int i = 0; i< contours.size(); i++ )
	{
		// Calculate perimeter length
		double perimeter = arcLength(contours[i], 1);
		double area = contourArea(contours[i], true);
		if(area <= 10)
		{
			continue;
		}

		/// Get the moments
		Moments mu;
		mu = moments(contours[i], false); 
		double hu[7];
		cv::HuMoments(mu, hu); 

		std::vector< double > temp;
		int coordinatex = mu.m10 / mu.m00;
		int coordinatey = mu.m01 / mu.m00;
		temp.push_back(objectClass);
		temp.push_back(area);
		temp.push_back(perimeter);
		temp.push_back(mu.mu12);

		output.push_back(temp);
	}
	return output;
}

std::vector< std::vector <double > > analyzeContours(vector< vector< Point > > contours, int objectClass)
{
	std::vector< std::vector <double > > output;

	return analyzeContours(contours, objectClass, output);
}

int main( int argc, char** argv )
{
	std::vector< std::vector <double > > output;

	output = analyzeContours(findContoursInFile("../numbers/1.png"), 1);
	output = analyzeContours(findContoursInFile("../numbers/2.png"), 2, output);
	output = analyzeContours(findContoursInFile("../numbers/3.png"), 3, output);
	output = analyzeContours(findContoursInFile("../numbers/4.png"), 4, output);
	output = analyzeContours(findContoursInFile("../numbers/5.png"), 5, output);
	output = analyzeContours(findContoursInFile("../numbers/6.png"), 6, output);
	output = analyzeContours(findContoursInFile("../numbers/7.png"), 7, output);
	output = analyzeContours(findContoursInFile("../numbers/8.png"), 8, output);
	output = analyzeContours(findContoursInFile("../numbers/9.png"), 9, output);

	cv::Mat trainingDataLabels(output.size(), 1, CV_64F);
	cv::Mat trainingData(output.size(), output.at(0).size() - 1, CV_64F);

	for (size_t i = 0; i < output.size(); i++)
	{   
		trainingDataLabels.at<double>(i, 1) = output[i][0];
		printf("%8.3f\t ", output[i][0]);
	    	for (size_t j = 1; j < output.at(0).size(); j++)
	    	{   
			trainingData.at<double>(i, j - 1) = output[i][j];
			printf("%8.3f\t ", output[i][j]);
	    	}   
		printf("\n");
	}   

	std::cout << trainingData.rows << std::endl;
	std::cout << trainingData.cols << std::endl;

	return 0;
}

