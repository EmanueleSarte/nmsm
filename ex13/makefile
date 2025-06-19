# Makefile for compiling cell_list_alg.cpp

CXX = g++
CXXFLAGS = -Ofast
TARGET = cell_list_alg
SRC = cell_list_alg.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: all clean

all: $(TARGET)
