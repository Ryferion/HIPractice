cmake_minimum_required(VERSION 3.5...3.22)
set(CMAKE_BUILD_TYPE Release)
set(GPU_TARGETS "gfx906" CACHE STRING "GPU targets to compile for")

if(DEFINED ENV{ROCM_PATH})
	list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH})

	find_package(rocm_smi REQUIRED)
	set(RSMI_INC_DIR ${ROCM_DIR}/include)
	set(RSMI_LIB_DIR ${ROCM_DIR}/lib)

	message("--------RSMI Lib Dir: " ${RSMI_LIB_DIR})

	if (RSMI_INC_DIR)
		message(STATUS "RSMI_INC_DIR found")
	endif(RSMI_INC_DIR)
	if (RSMI_LIB_DIR)
		message(STATUS "RSMI_LIB_DIR found")
	endif(RSMI_LIB_DIR)

	# target_link_libraries()

	add_executable(hostBuilt Host.cpp)
	set_target_properties(hostBuilt PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
	
endif(DEFINED ENV{ROCM_PATH})
