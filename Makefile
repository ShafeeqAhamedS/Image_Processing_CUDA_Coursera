# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv)

all: clean build run

build: 
	$(CXX) ./src/sharpening.cu --std c++17 `pkg-config opencv --cflags --libs` -o ./bin/sharpening.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda

run:
	./bin/sharpening.exe $(ARGS)

clean:
	rm -f ./bin/sharpening.exe ./output/*