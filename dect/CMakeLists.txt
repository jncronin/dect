find_package(OpenCL)
find_package(TIFF REQUIRED)
find_package(OpenMP)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(DECT_SOURCES "main.cpp" "XGetOpt.cpp")
set(EXTRA_LIBS ${EXTRA_LIBS} dectlib)

include_directories(../libdect)
include_directories(${TIFF_INCLUDE_DIRS})
include_directories(${OCL_INCLUDE_DIRS})

add_executable(dect ${DECT_SOURCES})
target_link_libraries(dect ${EXTRA_LIBS})
target_link_libraries(dect ${TIFF_LIBRARIES})
target_link_libraries(dect ${OCL_LIBRARIES})

if(OpenMP_CXX_FOUND)
	target_link_libraries(dect OpenMP::OpenMP_CXX)
endif()

if(OpenCL_FOUND)
	target_link_libraries(dect OpenCL::OpenCL)
endif()

target_compile_options(dect PUBLIC ${OCL_CFLAGS_OTHER})
target_compile_options(dect PUBLIC ${TIFF_CFLAGS_OTHER})

install (TARGETS dect RUNTIME DESTINATION bin)

