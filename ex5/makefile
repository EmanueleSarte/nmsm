# Makefile for compiling ex_5_ising.cpp

CXX = g++
CXXFLAGS = -Ofast
TARGET = ex_5_ising
SRC = ex_5_ising.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: all clean

all: $(TARGET)
