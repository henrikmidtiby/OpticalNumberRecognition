CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`
# Change 'opencv' to 'opencv-3.2.0-dev' for making it work with opencv from ROS.

# General rule for building opencv programs
% : %.cpp
	g++ $(CFLAGS) -o $@ $< $(LIBS)


all: 01locateContours 02contourFeatures 03circleFinder 04huMoments 05simpleOCR 06digitFeatures 07featureCollections 08simpleOCR 09sudokuExtracter

clean:
	rm -f 01locateContours 02contourFeatures 03circleFinder 04huMoments 05simpleOCR 06digitFeatures 07featureCollections 08simpleOCR
