# include directory
set(LIBTORCHSUM_INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")

# lib path
# link to .so in Linux
if (UNIX)
    set(LIBTORCHSUM_LIB "${CMAKE_CURRENT_LIST_DIR}/../lib/libtorch-summary.so")
endif (UNIX)
# link to .lib in Windows
if (WIN32)
    set(LIBTORCHSUM_LIB "${CMAKE_CURRENT_LIST_DIR}/../lib/libtorch-summary.lib")
endif (WIN32)
