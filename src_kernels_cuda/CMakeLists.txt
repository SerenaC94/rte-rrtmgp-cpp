FILE(GLOB sourcefiles
    "../src_kernels_cuda/gas_optics_kernel_launchers.cu"
    "../src_kernels_cuda/optical_props_kernel_launchers.cu"
    "../src_kernels_cuda/rte_solver_kernel_launchers.cu"
    "../src_kernels_cuda/subset_kernel_launchers.cu")

include_directories("../include" SYSTEM ${INCLUDE_DIRS})
add_library(rte_rrtmgp_kernels_cuda STATIC ${sourcefiles})

#find_package(OpenMP)
#if (OpenMP_CXX_FOUND)
#    #set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    #set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#		add_compile_options(-fopenmp)
#		target_link_libraries(rte_rrtmgp_kernels_cuda PUBLIC OpenMP::OpenMP_CXX)
#endif()

