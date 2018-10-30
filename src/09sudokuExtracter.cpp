#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;
const int N = 3;            // The size of 1 square in the sudoku.
const int EMPTY = 0;        // A sign for an empty cell.

//====== Function Declaration ======//
void input_sud(int sud[][N*N]);
bool fill_sud(int sud[][N*N], int row, int col);
void print_sud(const int sud[][N*N]);
bool is_legal(const int sud[][N*N], int row, int col, int val);
bool is_row_ok(const int row[], int col, int val);
bool is_col_ok(const int sud[][N*N], int row, int col, int val);
bool is_sqr_ok(const int sud[][N*N], int row, int col, int val);

std::vector< std::vector< Point > > findContoursInFile(std::string filename)
{
	Mat image;
	Mat imageGray;

	image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	if(!image.data)
	{
		printf("Could not open or find the image\n");
		assert(false);
	}

	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;

	/// Find contours
	findContours( image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	return contours;
}

std::vector< std::vector <double > > analyzeContours(std::vector< std::vector< Point > > contours, int objectClass, std::vector< std::vector <double > > output)
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

std::vector< std::vector <double > > analyzeContours(std::vector< std::vector< Point > > contours, int objectClass)
{
	std::vector< std::vector <double > > output;

	return analyzeContours(contours, objectClass, output);
}









//======== Fill Sudoku =========//
// Tries to fill-in the given sudoku board
// according to the sudoku rules.
// Returns whether it was possible to solve it or not.
bool fill_sud(int sud[][N*N], int row, int col)
{
    // Points to the row number of the next cell.
    int next_row = (col == N*N - 1) ? row + 1 : row;

    // Points to the column number of the next cell.
    int next_col = (col + 1) % (N*N);

    // If we get here, it means we succeed to solve the sudoku.
    if(row == N*N)  
        return true;

    // Checks if we are allowed to change the value of the current cell.
    // If we're not, then we're moving to the next one.
    if(sud[row][col] != EMPTY)
        return fill_sud(sud, next_row, next_col);

    // We're about to try and find the legal and appropriate value
    // to put in the current cell.
    for(int value = 1; value <= N*N; value++)
    {
        sud[row][col] = value;

        // Checks if 'value' can stay in the current cell,
        // and returns true if it does.
        if(is_legal(sud, row, col, value) && fill_sud(sud, next_row, next_col))
            return true;

        // Trial failed!
        sud[row][col] = EMPTY;
    }

    // None of the values solved the sudoku.
    return false;
}

//======== Print Sudoku ========//
// Prints the sudoku Graphically.
void print_sud(const int sud[][N*N])
{
    for(int i = 0; i < N*N; i++)
    {
        for(int j = 0; j < N*N; j++)
            std::cout << sud[i][j] << ' ';
	std::cout << std::endl;
    }
}

//========== Is Legal ==========//
// Checks and returns whether it's legal
// to put 'val' in A specific cell.
bool is_legal(const int sud[][N*N], int row, int col, int val)
{
    return (is_row_ok(sud[row], col, val) &&
            is_col_ok(sud, row, col, val) &&
            is_sqr_ok(sud, row, col, val));
}

//========= Is Row OK =========//
// Checks and returns whether it's legal
// to put 'val' in A specific row.
bool is_row_ok(const int row[], int col, int val)
{
    for(int i = 0; i < N*N; i++)
        if(i != col && row[i] == val)
            return false;       // Found the same value again!

    return true;
}

//========= Is Column OK =========//
// Checks and returns whether it's legal
// to put 'val' in A specific column.
bool is_col_ok(const int sud[][N*N], int row, int col, int val)
{
    for(int i = 0; i < N*N; i++)
        if(i != row && sud[i][col] == val)
            return false;       // Found the same value again!

    return true;
}

//========= Is Square OK =========//
// Checks and returns whether it's legal
// to put 'val' in A specific square.
bool is_sqr_ok(const int sud[][N*N], int row, int col, int val)
{
    int row_corner = (row / N) * N;
    // Holds the row number of the current square corner cell.

    int col_corner = (col / N) * N;
    // Holds the column number of the current square corner cell.

    for(int i = row_corner; i < (row_corner + N); i++)
        for(int j = col_corner; j < (col_corner + N); j++)
            if((i != row || j != col) && sud[i][j] == val)
                return false;       // Found the same value again!

    return true;
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

	cv::Mat trainingDataLabels(output.size(), 1, CV_32S);
	cv::Mat trainingData(output.size(), output.at(0).size() - 3, CV_32F);

	for (size_t i = 0; i < output.size(); i++)
	{   
		trainingDataLabels.at<int>(i, 0) = output[i][0];
	    	for (size_t j = 3; j < output.at(0).size(); j++)
	    	{   
			trainingData.at<float>(i, j - 3) = output[i][j];
	    	}   
	}   

	std::cout << "trainingData = "<< std::endl << " "  << trainingData << std::endl << std::endl;

	// Train the SVM
	printf("trainingData size: %d x %d\n", trainingData.rows, trainingData.cols);
	printf("trainingDataLabels size: %d x %d\n", trainingDataLabels.rows, trainingDataLabels.cols);

	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::POLY);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm->setDegree(2);
	svm->setCoef0(2);
	svm->train(trainingData, cv::ml::ROW_SAMPLE, trainingDataLabels);

	Mat image;
	image = imread("../sudokuer/01bw.png", CV_LOAD_IMAGE_COLOR);

	printf("Locating contours in test image.\n");
	std::vector< std::vector< Point > > contours = findContoursInFile("../sudokuer/01bw.png");

	output = analyzeContours(contours, 1);

	// Get feature descriptors.
	cv::Mat objectData(output.size(), output.at(0).size() - 3, CV_32F);
	for (size_t i = 0; i < output.size(); i++)
	{   
	    	for (size_t j = 3; j < output.at(0).size(); j++)
	    	{   
			objectData.at<float>(i, j - 3) = output[i][j];
	    	}   
	}   

	// Start random number generator with a known seed
	RNG rng(12345);

	// Fade image
	Mat detected_numbers;
	addWeighted(image, 1.0, image, 0.0, 200, detected_numbers);


	// Draw lines
	int spacing = 89;
	for( int i = 0; i < 10; i++)
	{
		int coordx = i * spacing;
		line(detected_numbers, Point2f(coordx, 0), Point2f(coordx, 800), Scalar(0, 0, 0));
		line(detected_numbers, Point2f(0, coordx), Point2f(800, coordx), Scalar(0, 0, 0));
	}
	

	/// Draw contours
	for( int i = 0; i < output.size(); i++ )
	{
		float area = output[i][3];
		if(area > 400)
		{
			int coordx = output[i][1];
			int coordy = output[i][2];
			Mat sampleMat = (Mat_<float>(1, 4) << output[i][3], output[i][4], output[i][5], output[i][6]);
			float recognizedClass = svm->predict(sampleMat);
			printf("%5d, %5d, %2f\n", coordx, coordy, recognizedClass);
			// Draw recognized class
			char str[20];
			sprintf(str,"%.0f", recognizedClass);
			putText(detected_numbers, str, Point2f(coordx - 20, coordy + 20), FONT_HERSHEY_TRIPLEX, 2,  Scalar(0, 0, 0));

			printf("%d %d %d\n", coordx, coordy, recognizedClass);
		}
	}

	int dilation_type = cv::MORPH_CROSS;
	int dilation_size = 1;
	Mat element = getStructuringElement( dilation_type,
                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                       Point( dilation_size, dilation_size ) );
	erode( detected_numbers, detected_numbers, element );

	printf("Training completed.\n");
	/// Show in a window
	namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
	imshow( "Contours", image);
	namedWindow( "Detected numbers", CV_WINDOW_AUTOSIZE );
	imshow( "Detected numbers", detected_numbers);
	
	waitKey(0);
}

