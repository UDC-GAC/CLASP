set(FORMAT_SOURCES
	format.cpp
	format.hpp
	format_cxx.hpp
	format_cxx.cpp
	format_csr.cpp
	format_csr.hpp
	format_cvs.cpp
	format_cvs.hpp)

set_source_files_properties(${FORMAT_SOURCES} PROPERTIES LANGUAGE CUDA)

add_library(formats STATIC
	${FORMAT_SOURCES})

target_include_directories(formats
	PUBLIC .)

