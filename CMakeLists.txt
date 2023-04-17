cmake_minimum_required(VERSION 3.5...3.22)
set(CMAKE_C_FLAGS "-D__HIP_PLATFORM_AMD__")
set(CMAKE_CXX_FLAGS "-D__HIP_PLATFORM_AMD__")
set(CMAKE_BUILD_TYPE Release)
set(GPU_TARGETS "gfx906" CACHE STRING "GPU targets to compile for")

project (hostTest)

if(DEFINED ENV{ROCM_PATH})

        list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH})
        list(APPEND CMAKE_PREFIX_PATH "/home/rquac004/rocm.bkup/comgr")
        list(APPEND CMAKE_PREFIX_PATH "/home/rquac004/rocm.bkup/rocdl/lib/cmake/AMDDeviceLibs")
        list(APPEND CMAKE_PREFIX_PATH "/home/rquac004/.opt/rocm/hip")
        list(APPEND CMAKE_PREFIX_PATH "/home/rquac004/.opt/rocm/rocm_smi/lib")

        find_package(hip REQUIRED)
        message(STATUS ${hip_DIR})
        add_library(hipLib ${hip_DIR})
        target_link_libraries(hipLib hip::device)
        set_target_properties(hipLib PROPERTIES LINKER_LANGUAGE CXX)

        #find_package(rocm_smi REQUIRED)
        #message(STATUS ${rocm_smi_DIR})
        #add_library(rsmiLib ${rocm_smi_DIR})
        #target_link_libraries(rsmiLib hip::device)
        #set_target_properties(rsmiLib PROPERTIES LINKER_LANGUAGE CXX)


        set(RSMI_INC_DIR "/home/rquac004/.opt/rocm/rocm_smi/include")
        set(RSMI_LIB_DIR "/home/rquac004/.opt/rocm/rocm_smi/lib")

        #add_library(RSMI_LIB ${RSMI_LIB_DIR})
        # Determine RSMI Header files are present
        #  (no external source dependencies)
        # Determine RSMI Library file is present
        if(NOT EXISTS ${RSMI_LIB_DIR}/librocm_smi64.so)
                        message("ERROR: Define RSMI_LIB_DIR pointing to RSMI library is not set")
                        message(" missing: ${RSMI_LIB_DIR}/librocm_smi64.so")
                        return()
        endif()

        LINK_DIRECTORIES(${RSMI_LIB_DIR} ${RSMI_INC_DIR})
        include_directories(${RSMI_INC_DIR})
        include_directories(${RSMI_LIB_DIR})

        set(RSMI_LIBS rocm_smi64)
        add_executable(hostBuilt "Host.cpp")
        set_target_properties(hostBuilt PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
        set_property(TARGET hostBuilt PROPERTY CXX_STANDARD 17)
        set_source_files_properties("Host.cpp" PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

        #target_link_libraries(hostBuilt ${rocm_smi})
        target_link_libraries(hostBuilt ${RSMI_LIB_DIR}/librocm_smi64.so)



endif(DEFINED ENV{ROCM_PATH})