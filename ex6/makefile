# Makefile for compiling ex_6_mcc.cpp

CXX = g++
CXXFLAGS = -Ofast
TARGET = ex_6_mcc
SRC = ex_6_mcc.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: all clean

all: $(TARGET)
