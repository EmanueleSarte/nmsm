# Makefile for compiling thermostats.cpp

CXX = g++
CXXFLAGS = -Ofast
TARGET = thermostats
SRC = thermostats.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: all clean

all: $(TARGET)
