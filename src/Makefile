
FLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

# General rule for building opencv programs
% : %.cpp
	g++ $(CFLAGS) -o $@ $< $(LIBS)

