cmake_minimum_required(VERSION 3.5...3.22)
set(CMAKE_C_FLAGS "-D__HIP_PLATFORM_AMD__")
set(CMAKE_CXX_FLAGS "-D__HIP_PLATFORM_AMD__")
set(CMAKE_BUILD_TYPE Release)
set(GPU_TARGETS "gfx906" CACHE STRING "GPU targets to compile for")

project (hostTest) 

if(DEFINED ENV{ROCM_PATH})
	message(STATUS "CMaking Now")
	list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH})
	list(APPEND CMAKE_PREFIX_PATH "/home/rquac004/rocm.bkup/comgr")
	list(APPEND CMAKE_PREFIX_PATH "/home/rquac004/rocm.bkup/rocdl/lib/cmake/AMDDeviceLibs")
	list(APPEND CMAKE_PREFIX_PATH "/home/rquac004/.opt/rocm/hip")

	find_package(hip REQUIRED)
	if (hip_FOUND)
		message(STATUS "hip found")
		add_library(hipLib hip)
	endif(hip_FOUND)

	find_package(HIP REQUIRED)
	if (HIP_FOUND)
		message(STATUS "HIP found")
		add_library(HIPLib HIP)
	endif(HIP_FOUND)

	find_package(rocm_smi REQUIRED)
	set(RSMI_INC_DIR ${ROCM_DIR}/include)
	set(RSMI_LIB_DIR ${ROCM_DIR}/lib)
	message(STATUS "rocm_smi found")
	if (RSMI_INC_DIR)
		message(STATUS "RSMI_INC_DIR found")
	endif(RSMI_INC_DIR)
	if (RSMI_LIB_DIR)
		message(STATUS "RSMI_LIB_DIR found")
	endif(RSMI_LIB_DIR)


	add_executable(hostBuilt "Host.cpp")
	set_target_properties(hostBuilt PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
	set_property(TARGET hostBuilt PROPERTY CXX_STANDARD 17)
	set_source_files_properties("Host.cpp" PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

	target_link_libraries(hostBuilt hipLib)
	target_link_libraries(hostBuilt HIPLib)
	target_link_libraries(hostBuilt RSMI_LIB_DIR)

endif(DEFINED ENV{ROCM_PATH})
