#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;

vector< vector< Point > > findContoursInFile(std::string filename)
{
	Mat image;
	Mat imageGray;

	image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	if(!image.data)
	{
		printf("Could not open or find the image\n");
		assert(false);
	}

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Find contours
	findContours( image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

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
		temp.push_back(coordinatex);
		temp.push_back(coordinatey);
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

	cv::Mat trainingDataLabels(output.size(), 1, CV_32SC1);
	cv::Mat trainingData(output.size(), output.at(0).size() - 3, CV_32FC1);

	for (size_t i = 0; i < output.size(); i++)
	{   
	//	trainingDataLabels.at<float>(i, 0) = output[i][0];
		//printf("%8.3f\t ", output[i][0]);
	    	for (size_t j = 3; j < output.at(0).size(); j++)
	    	{   
	//		trainingData.at<float>(i, j - 3) = output[i][j];
			//printf("%8.3f\t ", output[i][j]);
	    	}   
		//printf("\n");
	}   

	// Set up SVM's parameters
	CvSVMParams params;
	params.svm_type= CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	
	// Train the SVM
	CvSVM SVM;
	printf("trainingData size: %d x %d\n", trainingData.rows, trainingData.cols);
	printf("trainingDataLabels size: %d x %d\n", trainingDataLabels.rows, trainingDataLabels.cols);
	//SVM.train(trainingData, trainingDataLabels, Mat(), Mat(), params);

	Mat image;
	image = imread("../sudokuer/01bw.png", CV_LOAD_IMAGE_COLOR);

	printf("Locating contours in test image.\n");
	vector< vector< Point > > contours = findContoursInFile("../sudokuer/01bw.png");
	
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
			//printf("perimeter: %8.3f  area: %8.3f   compactness: %8.3f\n", perimeter, area, compactness);

			/// Get the moments
			Moments mu;
			mu = moments(contours[i], false); 
			double hu[7];
			cv::HuMoments(mu, hu); 

			// Draw contour
			Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
			drawContours( image, contours, i, color, 3, 8);
		}
	}
	printf("Training completed.\n");
	/// Show in a window
	namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
	imshow( "Contours", image);
	
	waitKey(0);


	// Use the SVM
//	float testData[1][2] = { {601, 10} };
//	Mat testDataMat(1, 2, CV_32FC1, testData);
//        float response = SVM.predict(testDataMat);

//	printf("Response: %5.3f\n", response);


}

