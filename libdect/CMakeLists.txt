include(FindPkgConfig)
pkg_search_module(OCL OpenCL)

execute_process(
	COMMAND git log -1 --format=%h
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	OUTPUT_VARIABLE GIT_VER
	OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(LIBDECT_SOURCES "libdect.cpp" "simul.cpp" "cpud16.cpp" "cpud8.cpp" "cpudf32.cpp" "cpudf64.cpp" "cpuf16.cpp" "cpuf8.cpp" "cpuff32.cpp" "cpuff64.cpp" )
if(OCL_FOUND)
	set(LIBDECT_SOURCES ${LIBDECT_SOURCES} "opencl.cpp")
endif(OCL_FOUND)

configure_file (
	"${CMAKE_CURRENT_SOURCE_DIR}/config.h.in" 
	"${CMAKE_CURRENT_BINARY_DIR}/config.h"
	)

configure_file (
	"${CMAKE_CURRENT_SOURCE_DIR}/git.version.h.in"
	"${CMAKE_CURRENT_BINARY_DIR}/git.version.h"
	)

include_directories("${CMAKE_CURRENT_BINARY_DIR}")

add_library(dectlib STATIC ${LIBDECT_SOURCES})
set_target_properties(dectlib PROPERTIES OUTPUT_NAME dect)

add_library(dectlibshared SHARED ${LIBDECT_SOURCES})
set_target_properties(dectlibshared PROPERTIES OUTPUT_NAME dect)
set_target_properties(dectlibshared PROPERTIES CXX_VISIBILITY_PRESET hidden)

install(TARGETS dectlib dectlibshared
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib)
install(FILES libdect.h DESTINATION include)

