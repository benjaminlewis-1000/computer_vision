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
CMAKE_SOURCE_DIR = /home/lewis/Dropbox/631_robot_vision/lab2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lewis/Dropbox/631_robot_vision/lab2

# Include any dependencies generated for this target.
include CMakeFiles/pose.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pose.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pose.dir/flags.make

CMakeFiles/pose.dir/lab2_4.cpp.o: CMakeFiles/pose.dir/flags.make
CMakeFiles/pose.dir/lab2_4.cpp.o: lab2_4.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/lewis/Dropbox/631_robot_vision/lab2/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/pose.dir/lab2_4.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/pose.dir/lab2_4.cpp.o -c /home/lewis/Dropbox/631_robot_vision/lab2/lab2_4.cpp

CMakeFiles/pose.dir/lab2_4.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pose.dir/lab2_4.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/lewis/Dropbox/631_robot_vision/lab2/lab2_4.cpp > CMakeFiles/pose.dir/lab2_4.cpp.i

CMakeFiles/pose.dir/lab2_4.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pose.dir/lab2_4.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/lewis/Dropbox/631_robot_vision/lab2/lab2_4.cpp -o CMakeFiles/pose.dir/lab2_4.cpp.s

CMakeFiles/pose.dir/lab2_4.cpp.o.requires:
.PHONY : CMakeFiles/pose.dir/lab2_4.cpp.o.requires

CMakeFiles/pose.dir/lab2_4.cpp.o.provides: CMakeFiles/pose.dir/lab2_4.cpp.o.requires
	$(MAKE) -f CMakeFiles/pose.dir/build.make CMakeFiles/pose.dir/lab2_4.cpp.o.provides.build
.PHONY : CMakeFiles/pose.dir/lab2_4.cpp.o.provides

CMakeFiles/pose.dir/lab2_4.cpp.o.provides.build: CMakeFiles/pose.dir/lab2_4.cpp.o

# Object files for target pose
pose_OBJECTS = \
"CMakeFiles/pose.dir/lab2_4.cpp.o"

# External object files for target pose
pose_EXTERNAL_OBJECTS =

pose: CMakeFiles/pose.dir/lab2_4.cpp.o
pose: CMakeFiles/pose.dir/build.make
pose: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_ts.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
pose: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
pose: CMakeFiles/pose.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable pose"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pose.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pose.dir/build: pose
.PHONY : CMakeFiles/pose.dir/build

CMakeFiles/pose.dir/requires: CMakeFiles/pose.dir/lab2_4.cpp.o.requires
.PHONY : CMakeFiles/pose.dir/requires

CMakeFiles/pose.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pose.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pose.dir/clean

CMakeFiles/pose.dir/depend:
	cd /home/lewis/Dropbox/631_robot_vision/lab2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lewis/Dropbox/631_robot_vision/lab2 /home/lewis/Dropbox/631_robot_vision/lab2 /home/lewis/Dropbox/631_robot_vision/lab2 /home/lewis/Dropbox/631_robot_vision/lab2 /home/lewis/Dropbox/631_robot_vision/lab2/CMakeFiles/pose.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pose.dir/depend

