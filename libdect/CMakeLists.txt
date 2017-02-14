include(FindPkgConfig)
pkg_search_module(OCL OpenCL)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(LIBDECT_SOURCES "libdect.cpp" "cpu.cpp" "simul.cpp" )
if(OCL_FOUND)
	set(LIBDECT_SOURCES ${LIBDECT_SOURCES} "opencl.cpp")
endif(OCL_FOUND)

configure_file (
	"${CMAKE_CURRENT_SOURCE_DIR}/config.h.in" 
	"${CMAKE_CURRENT_BINARY_DIR}/config.h"
	)

include_directories("${CMAKE_CURRENT_BINARY_DIR}")

add_library(dectlib STATIC ${LIBDECT_SOURCES})
set_target_properties(dectlib PROPERTIES OUTPUT_NAME dect)

install(TARGETS dectlib ARCHIVE DESTINATION lib)
install(FILES libdect.h DESTINATION include)

