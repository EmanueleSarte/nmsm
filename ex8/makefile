# Makefile for compiling offlattice_lj.cpp

CXX = g++
CXXFLAGS = -Ofast
TARGET = offlattice_lj
SRC = offlattice_lj.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: all clean

all: $(TARGET)
