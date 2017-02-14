cmake_minimum_required(VERSION 2.8.9)

project(dect)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/jncronin/dect")
set(EXTENSION_CATEGORY "Examples")
set(EXTENSION_CONTRIBUTORS "John Cronin (KCL)")
set(EXTENSION_DESCRIPTION "Slicer interface to external dect (Dual-Energy CT) application available from https://github.com/jncronin/dect")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/dect.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/dect/Screenshots/1.png")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(slicerdect)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_CPACK})
