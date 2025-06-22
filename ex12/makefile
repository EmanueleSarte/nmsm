# Makefile for compiling ex_12_MHM_ising.cpp

CXX = g++
CXXFLAGS = -Ofast
TARGET = ex_12_MHM_ising
SRC = ex_12_MHM_ising.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: all clean

all: $(TARGET)
