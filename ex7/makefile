# Makefile for compiling ex_7.cpp

CXX = g++
CXXFLAGS = -Ofast
TARGET = ex_7
SRC = ex_7.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: all clean

all: $(TARGET)
