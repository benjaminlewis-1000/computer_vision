# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lewis/Dropbox/631_robot_vision/lab3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lewis/Dropbox/631_robot_vision/lab3

# Include any dependencies generated for this target.
include CMakeFiles/pointGet.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pointGet.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pointGet.dir/flags.make

CMakeFiles/pointGet.dir/lab3_3.cpp.o: CMakeFiles/pointGet.dir/flags.make
CMakeFiles/pointGet.dir/lab3_3.cpp.o: lab3_3.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/lewis/Dropbox/631_robot_vision/lab3/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/pointGet.dir/lab3_3.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/pointGet.dir/lab3_3.cpp.o -c /home/lewis/Dropbox/631_robot_vision/lab3/lab3_3.cpp

CMakeFiles/pointGet.dir/lab3_3.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pointGet.dir/lab3_3.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/lewis/Dropbox/631_robot_vision/lab3/lab3_3.cpp > CMakeFiles/pointGet.dir/lab3_3.cpp.i

CMakeFiles/pointGet.dir/lab3_3.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pointGet.dir/lab3_3.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/lewis/Dropbox/631_robot_vision/lab3/lab3_3.cpp -o CMakeFiles/pointGet.dir/lab3_3.cpp.s

CMakeFiles/pointGet.dir/lab3_3.cpp.o.requires:
.PHONY : CMakeFiles/pointGet.dir/lab3_3.cpp.o.requires

CMakeFiles/pointGet.dir/lab3_3.cpp.o.provides: CMakeFiles/pointGet.dir/lab3_3.cpp.o.requires
	$(MAKE) -f CMakeFiles/pointGet.dir/build.make CMakeFiles/pointGet.dir/lab3_3.cpp.o.provides.build
.PHONY : CMakeFiles/pointGet.dir/lab3_3.cpp.o.provides

CMakeFiles/pointGet.dir/lab3_3.cpp.o.provides.build: CMakeFiles/pointGet.dir/lab3_3.cpp.o

# Object files for target pointGet
pointGet_OBJECTS = \
"CMakeFiles/pointGet.dir/lab3_3.cpp.o"

# External object files for target pointGet
pointGet_EXTERNAL_OBJECTS =

pointGet: CMakeFiles/pointGet.dir/lab3_3.cpp.o
pointGet: CMakeFiles/pointGet.dir/build.make
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_ts.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
pointGet: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
pointGet: CMakeFiles/pointGet.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable pointGet"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pointGet.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pointGet.dir/build: pointGet
.PHONY : CMakeFiles/pointGet.dir/build

CMakeFiles/pointGet.dir/requires: CMakeFiles/pointGet.dir/lab3_3.cpp.o.requires
.PHONY : CMakeFiles/pointGet.dir/requires

CMakeFiles/pointGet.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pointGet.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pointGet.dir/clean

CMakeFiles/pointGet.dir/depend:
	cd /home/lewis/Dropbox/631_robot_vision/lab3 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lewis/Dropbox/631_robot_vision/lab3 /home/lewis/Dropbox/631_robot_vision/lab3 /home/lewis/Dropbox/631_robot_vision/lab3 /home/lewis/Dropbox/631_robot_vision/lab3 /home/lewis/Dropbox/631_robot_vision/lab3/CMakeFiles/pointGet.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pointGet.dir/depend
