all: 
	hipcc -D__HIP_PLATFORM_AMD__ Host.cpp -o host
	
	g++ matrixFill matrixFill.cpp