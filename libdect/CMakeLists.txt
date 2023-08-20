find_package(OpenCL)
find_package(OpenMP)

message(OpenCL="${OpenCL_FOUND}")
message(OpenMP="${OpenMP_CXX_FOUND}")

set (CMAKE_CXX_STANDARD 17)

execute_process(
	COMMAND git log -1 --format=%h
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	OUTPUT_VARIABLE GIT_VER
	OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(LIBDECT_SOURCES "libdect.cpp" "simul.cpp" "cpud16.cpp" "cpud8.cpp" "cpudf32.cpp" "cpudf64.cpp" "cpuf16.cpp" "cpuf8.cpp" "cpuff32.cpp" "cpuff64.cpp" )
if(OpenCL_FOUND)
	set(LIBDECT_SOURCES ${LIBDECT_SOURCES} "opencl.cpp")
endif(OpenCL_FOUND)

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

if(OpenMP_CXX_FOUND)
	target_link_libraries(dectlib OpenMP::OpenMP_CXX)
	target_link_libraries(dectlibshared OpenMP::OpenMP_CXX)
endif()

if(OpenCL_FOUND)
	target_link_libraries(dectlib OpenCL::OpenCL)
	target_link_libraries(dectlibshared OpenCL::OpenCL)
endif()

install(TARGETS dectlib dectlibshared
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib)
install(FILES libdect.h DESTINATION include)

