.PHONY: all clean

CUDA_LIB_DIR := /usr/local/cuda/lib64

LIBS += -lcudart
LIBS += -lcuda
CU_SRCS :=  ./resnet_extern.cu
CU_OBJS := $(CU_SRCS:%.cu=%.o)

C_SRCS := ./*.cc
C_OBJS := $(C_SRCS:%.c=%.o)

CPP_SRCS := $(./lib_json/*.cpp)
CPP_OBJS := $(CPP_SRCS:%.cpp=%.o)

EXECUTABLE := hello

all : $(EXECUTABLE)

$(CU_OBJS):$(CU_SRCS)
	nvcc -c $^ ./lib_json/*.cpp -std=c++11

$(CPP_OBJS):$(CPP_SRCS)
	g++ -c $^ -std=c++11

$(C_OBJS):$(C_SRCS)
	g++ -c $^ -std=c++11

$(EXECUTABLE):$(CU_OBJS) $(C_OBJS) *.o
	g++ -o $@ $^ -L$(CUDA_LIB_DIR) $(LIBS) -std=c++11

clean:
	rm $(EXECUTABLE) $(CU_OBJS)