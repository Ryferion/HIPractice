all: 
	#hipcc -D__HIP_PLATFORM_AMD__ Host.cpp -o host
	hipcc -D__HIP_PLATFORM_AMD__ Host_pthreads.cpp -o host_pthread