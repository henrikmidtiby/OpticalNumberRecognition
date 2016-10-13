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

std::vector< std::vector <double > > analyzeContours(vector< vector< Point > > contours, int objectClass)
{
	std::vector< std::vector <double > > output;

	for( int i = 0; i< contours.size(); i++ )
	{
		// Calculate perimeter length
		double perimeter = arcLength(contours[i], 1);
		double area = contourArea(contours[i], true);
		if(area <= 0)
		{
			continue;
		}

		/// Get the moments
		Moments mu;
		mu = moments(contours[i], false); 
		double hu[7];
		cv::HuMoments(mu, hu); 

		std::vector< double > temp;
		temp.push_back(perimeter);
		temp.push_back(mu.mu12);

		output.push_back(temp);

		std::cout << "perimeter: " << perimeter << std::endl;
	}
	return output;
}

int main( int argc, char** argv )
{
	std::vector< std::vector <double > > output;

	output = analyzeContours(findContoursInFile("../numbers/1.png"), 1);

	cv::Mat trainingData(output.size(), output.at(0).size(), CV_64F);

	for (size_t i = 0; i < output.size(); i++)
	{   
	    for (size_t j = 0; j < output.at(0).size(); j++)
	    {   
		trainingData.at<double>(i,j) = output[i][j];
	    }   
	}   

	std::cout << trainingData.rows << std::endl;
	std::cout << trainingData.cols << std::endl;

	return 1;

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

