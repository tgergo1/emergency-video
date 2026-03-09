cmake_minimum_required(VERSION 3.16)
project(emergency_video_codec LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat
    libavcodec
    libavutil
    libswresample
)

include(GNUInstallDirs)

option(EV_BUILD_TESTS "Build and run codec/unit tests with CTest" ON)
option(EV_PORTABLE_BUNDLE_SYSTEM_LIBS "Bundle system shared libraries for maximum artifact portability" OFF)

option(EV_FETCH_OPENCV "Fetch OpenCV with CMake FetchContent if it is not found locally" ON)
set(EV_OPENCV_TAG "4.10.0" CACHE STRING "OpenCV git tag used by FetchContent fallback")

find_package(OpenCV QUIET COMPONENTS core imgproc highgui videoio objdetect)

if(NOT OpenCV_FOUND)
    if(NOT EV_FETCH_OPENCV)
        message(FATAL_ERROR
            "OpenCV not found. Install OpenCV and set OpenCV_DIR, or configure with -DEV_FETCH_OPENCV=ON.")
    endif()

    include(FetchContent)

    # Keep fetched OpenCV focused on the modules used by this demo.
    set(BUILD_LIST "core,imgproc,highgui,videoio,objdetect" CACHE STRING "" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_apps OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_js OFF CACHE BOOL "" FORCE)
    set(BUILD_JAVA OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_python2 OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_python3 OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_world OFF CACHE BOOL "" FORCE)
    set(WITH_GSTREAMER OFF CACHE BOOL "" FORCE)
    set(WITH_FFMPEG OFF CACHE BOOL "" FORCE)
    set(WITH_OBSENSOR OFF CACHE BOOL "" FORCE)
    set(OBSENSOR_USE_ORBBEC_SDK OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        opencv
        GIT_REPOSITORY https://github.com/opencv/opencv.git
        GIT_TAG ${EV_OPENCV_TAG}
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(opencv)

    set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_videoio opencv_objdetect)
    set(OpenCV_INCLUDE_DIRS
        "${opencv_SOURCE_DIR}/include"
        "${opencv_SOURCE_DIR}/modules/core/include"
        "${opencv_SOURCE_DIR}/modules/imgproc/include"
        "${opencv_SOURCE_DIR}/modules/highgui/include"
        "${opencv_SOURCE_DIR}/modules/videoio/include"
        "${opencv_SOURCE_DIR}/modules/objdetect/include"
        "${opencv_SOURCE_DIR}/modules/imgcodecs/include"
        "${CMAKE_BINARY_DIR}"
        "${opencv_BINARY_DIR}"
    )
    set(OpenCV_FOUND TRUE)
    message(STATUS "Using fetched OpenCV from git tag ${EV_OPENCV_TAG}")
else()
    message(STATUS "Using local OpenCV from ${OpenCV_DIR}")
endif()

add_executable(emergency_video
    web_main.cpp
    bitstream.cpp
    codec.cpp
    encoder.cpp
    decoder.cpp
    metrics.cpp
    util.cpp
    communicator_protocol.cpp
    queue_manager.cpp
    router.cpp
    persistent_store.cpp
    transport_manager.cpp
    acoustic_link.cpp
    acoustic_modem.cpp
    fallback_controller.cpp
    crypto.cpp
    audio_io.cpp
    media_ffmpeg.cpp
    third_party/miniaudio.c
)
if(NOT MSVC)
    set_source_files_properties(third_party/miniaudio.c PROPERTIES COMPILE_OPTIONS "-w")
endif()
set_target_properties(emergency_video PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
if(APPLE)
    set_target_properties(emergency_video PROPERTIES
        INSTALL_RPATH "@executable_path"
        BUILD_WITH_INSTALL_RPATH OFF
    )
elseif(UNIX)
    set_target_properties(emergency_video PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH OFF
    )
endif()

target_link_libraries(emergency_video PRIVATE ${OpenCV_LIBS} PkgConfig::FFMPEG)
if(WIN32)
    target_link_libraries(emergency_video PRIVATE ws2_32)
endif()
target_include_directories(emergency_video PRIVATE ${OpenCV_INCLUDE_DIRS} ${FFMPEG_INCLUDE_DIRS})

if(MSVC)
    target_compile_options(emergency_video PRIVATE /W4)
else()
    target_compile_options(emergency_video PRIVATE -Wall -Wextra -Wpedantic)
endif()

set(EV_RUNTIME_SEARCH_DIRS "")
if(DEFINED FFMPEG_LIBRARY_DIRS)
    list(APPEND EV_RUNTIME_SEARCH_DIRS ${FFMPEG_LIBRARY_DIRS})
endif()

function(ev_collect_runtime_dirs out_var)
    set(_dirs "${${out_var}}")
    foreach(_item IN LISTS ARGN)
        if(TARGET "${_item}")
            get_target_property(_imported_loc "${_item}" IMPORTED_LOCATION)
            if(NOT _imported_loc)
                get_target_property(_imported_loc "${_item}" IMPORTED_LOCATION_RELEASE)
            endif()
            if(_imported_loc AND EXISTS "${_imported_loc}")
                get_filename_component(_dir "${_imported_loc}" DIRECTORY)
                list(APPEND _dirs "${_dir}")
            endif()

            get_target_property(_iface_libs "${_item}" INTERFACE_LINK_LIBRARIES)
            foreach(_iface_lib IN LISTS _iface_libs)
                if(IS_ABSOLUTE "${_iface_lib}" AND EXISTS "${_iface_lib}")
                    get_filename_component(_iface_dir "${_iface_lib}" DIRECTORY)
                    list(APPEND _dirs "${_iface_dir}")
                endif()
            endforeach()
        elseif(IS_ABSOLUTE "${_item}" AND EXISTS "${_item}")
            get_filename_component(_dir "${_item}" DIRECTORY)
            list(APPEND _dirs "${_dir}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _dirs)
    set(${out_var} "${_dirs}" PARENT_SCOPE)
endfunction()

ev_collect_runtime_dirs(EV_RUNTIME_SEARCH_DIRS ${OpenCV_LIBS} PkgConfig::FFMPEG)

if(DEFINED OpenCV_DIR AND EXISTS "${OpenCV_DIR}")
    get_filename_component(_ev_opencv_root "${OpenCV_DIR}" DIRECTORY)
    if(EXISTS "${_ev_opencv_root}/lib")
        list(APPEND EV_RUNTIME_SEARCH_DIRS "${_ev_opencv_root}/lib")
    endif()
endif()
if(EXISTS "${CMAKE_BINARY_DIR}/_deps/opencv-build/lib")
    list(APPEND EV_RUNTIME_SEARCH_DIRS "${CMAKE_BINARY_DIR}/_deps/opencv-build/lib")
endif()
if(APPLE)
    list(APPEND EV_RUNTIME_SEARCH_DIRS "/opt/homebrew/lib" "/usr/local/lib")
    file(GLOB _ev_brew_opt_lib_dirs LIST_DIRECTORIES TRUE
        "/opt/homebrew/opt/*/lib"
        "/usr/local/opt/*/lib"
    )
    list(APPEND EV_RUNTIME_SEARCH_DIRS ${_ev_brew_opt_lib_dirs})
elseif(UNIX)
    list(APPEND EV_RUNTIME_SEARCH_DIRS "/usr/local/lib" "/usr/lib" "/lib")
endif()
list(REMOVE_DUPLICATES EV_RUNTIME_SEARCH_DIRS)

set(EV_FACE_CASCADE_SOURCE "")
set(_ev_face_candidates
    "${CMAKE_BINARY_DIR}/_deps/opencv-src/data/haarcascades/haarcascade_frontalface_default.xml"
    "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
)
foreach(_face IN LISTS _ev_face_candidates)
    if(EXISTS "${_face}")
        set(EV_FACE_CASCADE_SOURCE "${_face}")
        break()
    endif()
endforeach()

set(EV_FIXUP_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/FixupPortable.cmake")
set(EV_PORTABLE_BINDIR ".")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/FixupPortable.cmake.in"
               "${EV_FIXUP_SCRIPT}" @ONLY)

set(EV_INSTALL_COMPONENT "portable_runtime")
install(TARGETS emergency_video
    RUNTIME DESTINATION "${EV_PORTABLE_BINDIR}"
    COMPONENT "${EV_INSTALL_COMPONENT}"
)
if(EV_FACE_CASCADE_SOURCE)
    install(FILES "${EV_FACE_CASCADE_SOURCE}"
        DESTINATION "${EV_PORTABLE_BINDIR}"
        COMPONENT "${EV_INSTALL_COMPONENT}"
    )
endif()
install(SCRIPT "${EV_FIXUP_SCRIPT}" COMPONENT "${EV_INSTALL_COMPONENT}")

set(EV_PORTABLE_DIR "${CMAKE_BINARY_DIR}/portable")
add_custom_target(portable
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${EV_PORTABLE_DIR}"
    COMMAND ${CMAKE_COMMAND} --install "${CMAKE_BINARY_DIR}" --prefix "${EV_PORTABLE_DIR}" --component "${EV_INSTALL_COMPONENT}"
    COMMENT "Building portable package in ${EV_PORTABLE_DIR}"
)

option(EV_BUILD_LEGACY_OPENCV_UI "Build legacy OpenCV-window UI target (main.cpp)" OFF)
if(EV_BUILD_LEGACY_OPENCV_UI)
    add_executable(emergency_video_legacy
        main.cpp
        bitstream.cpp
        codec.cpp
        encoder.cpp
        decoder.cpp
        metrics.cpp
        util.cpp
    )
    target_link_libraries(emergency_video_legacy PRIVATE ${OpenCV_LIBS})
    target_include_directories(emergency_video_legacy PRIVATE ${OpenCV_INCLUDE_DIRS})
    if(MSVC)
        target_compile_options(emergency_video_legacy PRIVATE /W4)
    else()
        target_compile_options(emergency_video_legacy PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

if(EV_BUILD_TESTS)
    enable_testing()

    add_executable(emergency_video_tests
        tests/test_codec.cpp
        bitstream.cpp
        codec.cpp
        encoder.cpp
        decoder.cpp
        communicator_protocol.cpp
        queue_manager.cpp
        router.cpp
        persistent_store.cpp
        acoustic_link.cpp
        acoustic_modem.cpp
        fallback_controller.cpp
        crypto.cpp
    )
    target_include_directories(emergency_video_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    if(MSVC)
        target_compile_options(emergency_video_tests PRIVATE /W4)
    else()
        target_compile_options(emergency_video_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME codec_unit_tests COMMAND emergency_video_tests)
endif()
