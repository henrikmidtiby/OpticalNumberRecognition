
FLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

# General rule for building opencv programs
% : %.cpp
	g++ $(CFLAGS) -o $@ $< $(LIBS)


all: 01locateContours 02contourFeatures 03circleFinder 04huMoments 05simpleOCR 06digitsFeatures 07featureCollection 08simpleOCR

clean:
	rm -f 01locateContours 02contourFeatures 03circleFinder 04huMoments 05simpleOCR 06digitFeatures 07featureCollections 08simpleOCR
