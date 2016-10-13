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
		temp.push_back(mu.mu12 / 1000);
		temp.push_back(mu.mu11 / 1000);

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

	cv::Mat trainingDataLabels(output.size(), 1, CV_32FC1);
	cv::Mat trainingData(output.size(), output.at(0).size() - 3, CV_32FC1);

	for (size_t i = 0; i < output.size(); i++)
	{   
		trainingDataLabels.at<float>(i, 0) = output[i][0];
	    	for (size_t j = 3; j < output.at(0).size(); j++)
	    	{   
			trainingData.at<float>(i, j - 3) = output[i][j];
	    	}   
	}   

	std::cout << "trainingData = "<< std::endl << " "  << trainingData << std::endl << std::endl;

	// Set up SVM's parameters
	CvSVMParams params;
	params.svm_type= CvSVM::C_SVC;
	params.kernel_type = CvSVM::POLY;
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	params.degree      = 2;
	params.coef0       = 2;
	
	// Train the SVM
	CvSVM SVM;
	printf("trainingData size: %d x %d\n", trainingData.rows, trainingData.cols);
	printf("trainingDataLabels size: %d x %d\n", trainingDataLabels.rows, trainingDataLabels.cols);
	//SVM.train_auto(trainingData, trainingDataLabels, Mat(), Mat(), params);
	SVM.train(trainingData, trainingDataLabels, Mat(), Mat(), params);

	Mat image;
	image = imread("../sudokuer/01bw.png", CV_LOAD_IMAGE_COLOR);

	printf("Locating contours in test image.\n");
	vector< vector< Point > > contours = findContoursInFile("../sudokuer/01bw.png");

	output = analyzeContours(contours, 1);

	// Get feature descriptors.
	cv::Mat objectData(output.size(), output.at(0).size() - 3, CV_32FC1);
	for (size_t i = 0; i < output.size(); i++)
	{   
	    	for (size_t j = 3; j < output.at(0).size(); j++)
	    	{   
			objectData.at<float>(i, j - 3) = output[i][j];
	    	}   
	}   

	// Start random number generator with a known seed
	RNG rng(12345);

	/// Draw contours
	Mat drawing = Mat::zeros( image.size(), CV_8UC3 );
	for( int i = 0; i < output.size(); i++ )
	{
		float area = output[i][3];
		if(area > 400)
		{
			int coordx = output[i][1];
			int coordy = output[i][2];
			Mat sampleMat = (Mat_<float>(1, 4) << output[i][3], output[i][4], output[i][5], output[i][6]);
			float recognizedClass = SVM.predict(sampleMat);
			printf("%5d, %5d, %2f\n", coordx, coordy, recognizedClass);
			// Draw recognized class
			char str[20];
			sprintf(str,"%.0f", recognizedClass);
			putText(image, str, Point2f(coordx, coordy), FONT_HERSHEY_TRIPLEX, 1,  Scalar(0, 0, 255));
		}
	}
	printf("Training completed.\n");
	/// Show in a window
	namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
	imshow( "Contours", image);
	
	waitKey(0);
}

