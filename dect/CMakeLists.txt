include(FindPkgConfig)
pkg_search_module(OCL OpenCL)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(DECT_SOURCES "main.cpp")
set(EXTRA_LIBS ${EXTRA_LIBS} dectlib)

include_directories(libdect)

add_executable(dect ${DECT_SOURCES})
target_link_libraries(dect ${EXTRA_LIBS})
target_link_libraries(dect tiff)
target_link_libraries(dect ${OCL_LIBRARIES})

install (TARGETS dect RUNTIME DESTINATION bin)

