cmake_minimum_required(VERSION 2.8.9)
project(dect)

include(FindPkgConfig)
pkg_search_module(OCL OpenCL)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(LIBDECT_SOURCES "libdect/libdect.cpp" "libdect/cpu.cpp" "libdect/simul.cpp" )
if(OCL_FOUND)
	set(LIBDECT_SOURCES ${LIBDECT_SOURCES} "libdect/opencl.cpp")
endif(OCL_FOUND)

set(DECT_SOURCES "dect/main.cpp")
set(EXTRA_LIBS ${EXTRA_LIBS} dect)

include_directories(libdect)

configure_file (
	"${PROJECT_SOURCE_DIR}/libdect/config.h.in" 
	"${PROJECT_BINARY_DIR}/libdect/config.h"
	)

add_library(dect STATIC ${LIBDECT_SOURCES})

add_executable(dect/dect ${DECT_SOURCES})
target_link_libraries(dect/dect ${EXTRA_LIBS})
target_link_libraries(dect/dect tiff)
target_link_libraries(dect/dect ${OCL_LIBRARIES})

install (TARGETS dect/dect RUNTIME DESTINATION bin)
install (TARGETS dect ARCHIVE DESTINATION lib)
install(FILES libdect/libdect.h DESTINATION include)

