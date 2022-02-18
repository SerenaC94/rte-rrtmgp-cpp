#include "Raytracer.h"
#include "Array.h"
#include <curand_kernel.h>
#include "rrtmgp_kernel_launcher_cuda.h"
#include "raytracer_kernels.h"
#include "Optical_props.h"
namespace
{
    inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
    {
        if (code != cudaSuccess)
        {
            fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }
    
    template<typename T>
    T* allocate_gpu(const int length)
    {
        T* data_ptr = Tools_gpu::allocate_gpu<T>(length);
    
        return data_ptr;
    }
    template<typename T>
    void copy_to_gpu(T* gpu_data, const T* cpu_data, const int length)
    {
        cuda_safe_call(cudaMemcpy(gpu_data, cpu_data, length*sizeof(T), cudaMemcpyHostToDevice));
    }
    
    
    template<typename T>
    void copy_from_gpu(T* cpu_data, const T* gpu_data, const int length)
    {
        cuda_safe_call(cudaMemcpy(cpu_data, gpu_data, length*sizeof(T), cudaMemcpyDeviceToHost));
    }
    
    template<typename TF>__global__
    void create_knull_grid(
            const int ncol_x, const int ncol_y, const int nlay, const TF k_ext_null_min,
            const Optics_ext* __restrict__ k_ext, TF* __restrict__ k_null_grid)
    {   
        const int grid_x = blockIdx.x*blockDim.x + threadIdx.x;
        const int grid_y = blockIdx.y*blockDim.y + threadIdx.y;
        const int grid_z = blockIdx.z*blockDim.z + threadIdx.z;
        if ( ( grid_x < ngrid_h) && ( grid_y < ngrid_h) && ( grid_z < ngrid_v))
        {
            const TF fx = TF(ncol_x) / TF(ngrid_h);
            const TF fy = TF(ncol_y) / TF(ngrid_h);
            const TF fz = TF(nlay) / TF(ngrid_v);

            const int x0 = grid_x*fx;
            const int x1 = floor((grid_x+1)*fx);
            const int y0 = grid_y*fy;
            const int y1 = floor((grid_y+1)*fy);
            const int z0 = grid_z*fz;
            const int z1 = floor((grid_z+1)*fz);
            
            const int ijk_grid = grid_x +grid_y*ngrid_h + grid_z*ngrid_h*ngrid_h;
            TF k_null = k_ext_null_min;
            
            for (int k=z0; k<z1; ++k)
                for (int j=y0; j<y1; ++j)
                    for (int i=x0; i<x1; ++i)
                    {
                        const int ijk_in = i + j*ncol_x + k*ncol_x*ncol_y;
                        const TF k_ext_tot = k_ext[ijk_in].gas + k_ext[ijk_in].cloud;
                        k_null = max(k_null, k_ext_tot);
                    }
            k_null_grid[ijk_grid] = k_null;
        }
    }


    template<typename TF>__global__
    void bundles_optical_props(
            const int ncol_x, const int ncol_y, const int nlay, const TF dz_grid,
            const TF* __restrict__ tau_tot, const TF* __restrict__ ssa,
            const TF* __restrict__ asy, const TF* __restrict__ tau_cld,
            Optics_ext* __restrict__ k_ext, Optics_scat* __restrict__ ssa_asy)
    {
        const int icol_x = blockIdx.x*blockDim.x + threadIdx.x;
        const int icol_y = blockIdx.y*blockDim.y + threadIdx.y;
        const int ilay = blockIdx.z*blockDim.z + threadIdx.z;
        if ( ( icol_x < ncol_x) && ( icol_y < ncol_y) && ( ilay < nlay))
        {
            const int idx = icol_x + icol_y*ncol_x + ilay*ncol_y*ncol_x;  
            const TF kext_cld = tau_cld[idx] / dz_grid;
            const TF kext_gas = tau_tot[idx] / dz_grid - kext_cld;
            k_ext[idx].cloud = kext_cld;
            k_ext[idx].gas = kext_gas;
            ssa_asy[idx].ssa = ssa[idx];
            ssa_asy[idx].asy = asy[idx];
        }
    }

    template<typename TF>__global__
    void count_to_flux_2d(
            const int ncol_x, const int ncol_y, const TF flux_per_ray,
            const TF* __restrict__ count_1, const TF* __restrict__ count_2, const TF* __restrict__ count_3, const TF* __restrict__ count_4,
            TF* __restrict__ flux_1, TF* __restrict__ flux_2, TF* __restrict__ flux_3, TF* __restrict__ flux_4)
    {
        const int icol_x = blockIdx.x*blockDim.x + threadIdx.x;
        const int icol_y = blockIdx.y*blockDim.y + threadIdx.y;

        if ( ( icol_x < ncol_x) && ( icol_y < ncol_y) )
        {
            const int idx = icol_x + icol_y*ncol_x;
            flux_1[idx] = count_1[idx] * flux_per_ray;
            flux_2[idx] = count_2[idx] * flux_per_ray;
            flux_3[idx] = count_3[idx] * flux_per_ray;
            flux_4[idx] = count_4[idx] * flux_per_ray;
        }
    }

    template<typename TF>__global__
    void count_to_flux_3d(
            const int ncol_x, const int ncol_y, const int nlay, const TF flux_per_ray,
            const TF* __restrict__ count_1, const TF* __restrict__ count_2,
            TF* __restrict__ flux_1, TF* __restrict__ flux_2)
    {
        const int icol_x = blockIdx.x*blockDim.x + threadIdx.x;
        const int icol_y = blockIdx.y*blockDim.y + threadIdx.y;
        const int ilay = blockIdx.z*blockDim.z + threadIdx.z;

        if ( ( icol_x < ncol_x) && ( icol_y < ncol_y) && ( ilay < nlay))
        {
            const int idx = icol_x + icol_y*ncol_x + ilay*ncol_x*ncol_y;
            flux_1[idx] = count_1[idx] * flux_per_ray;
            flux_2[idx] = count_2[idx] * flux_per_ray;
        }
    }
}

template<typename TF>
Raytracer_gpu<TF>::Raytracer_gpu()
{
    curandDirectionVectors32_t* qrng_vectors;
    curandGetDirectionVectors32(
                &qrng_vectors,
                CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6);
    unsigned int* qrng_constants;
    curandGetScrambleConstants32(&qrng_constants);

    this->qrng_vectors_gpu = allocate_gpu<curandDirectionVectors32_t>(2);
    this->qrng_constants_gpu = allocate_gpu<unsigned int>(2);
    
    copy_to_gpu(qrng_vectors_gpu, qrng_vectors, 2);
    copy_to_gpu(qrng_constants_gpu, qrng_constants, 2);
}

template<typename TF>
void Raytracer_gpu<TF>::trace_rays(
        const Int photons_to_shoot,
        const int ncol_x, const int ncol_y, const int nlay,
        const TF dx_grid, const TF dy_grid, const TF dz_grid,
        const Optical_props_2str_gpu<TF>& optical_props,
        const Optical_props_2str_gpu<TF>& cloud_optical_props,
        const TF surface_albedo,
        const TF zenith_angle,
        const TF azimuth_angle,
        const TF flux_tod_dir,
        const TF flux_tod_dif,//        Array_gpu<TF,2>& background_profiles,
        Array_gpu<TF,2>& flux_toa_up,
        Array_gpu<TF,2>& flux_sfc_dir,
        Array_gpu<TF,2>& flux_sfc_dif,
        Array_gpu<TF,2>& flux_sfc_up,
        Array_gpu<TF,3>& flux_abs_dir,
        Array_gpu<TF,3>& flux_abs_dif)
{
    // set of block and grid dimensions used in data processing kernels - requires some proper tuning later
    const int block_col_x = 8;
    const int block_col_y = 8;
    const int block_lay = 4;

    const int grid_col_x  = ncol_x/block_col_x + (ncol_x%block_col_x > 0);
    const int grid_col_y  = ncol_y/block_col_y + (ncol_y%block_col_y > 0);
    const int grid_lay  = nlay/block_lay + (nlay%block_lay > 0);

    dim3 grid_2d(grid_col_x, grid_col_y);
    dim3 block_2d(block_col_x, block_col_y);
    dim3 grid_3d(grid_col_x, grid_col_y, grid_lay);
    dim3 block_3d(block_col_x, block_col_y, block_lay);

    // bundle optical properties in struct
    Array_gpu<Optics_ext,3> k_ext({ncol_x, ncol_y, nlay});
    Array_gpu<Optics_scat,3> ssa_asy({ncol_x, ncol_y, nlay});
    
    bundles_optical_props<<<grid_3d, block_3d>>>(
            ncol_x, ncol_y, nlay, dz_grid,
            optical_props.get_tau().ptr(), optical_props.get_ssa().ptr(),
            optical_props.get_g().ptr(), cloud_optical_props.get_tau().ptr(),
            k_ext.ptr(), ssa_asy.ptr());
                                                                                
    // create k_null_grid
    const int block_kn_x = 8;
    const int block_kn_y = 8;
    const int block_kn_z = 4;

    const int grid_kn_x  = ngrid_h/block_kn_x + (ngrid_h%block_kn_x > 0);
    const int grid_kn_y  = ngrid_h/block_kn_y + (ngrid_h%block_kn_y > 0);
    const int grid_kn_z  = ngrid_v/block_kn_z + (ngrid_v%block_kn_z > 0);

    dim3 grid_kn(grid_kn_x, grid_kn_y, grid_kn_z);
    dim3 block_kn(block_kn_x, block_kn_y, block_kn_z);
    
    Array_gpu<TF,3> k_null_grid({ngrid_h, ngrid_h, ngrid_v});
    const TF k_ext_null_min = TF(1e-3);
    
    create_knull_grid<<<grid_kn, block_kn>>>(
            ncol_x, ncol_y, nlay, k_ext_null_min,
            k_ext.ptr(), k_null_grid.ptr());
    
    // initialise output arrays and set to 0
    Array_gpu<TF,2> toa_down_count({ncol_x, ncol_y});
    Array_gpu<TF,2> toa_up_count({ncol_x, ncol_y});
    Array_gpu<TF,2> surface_down_direct_count({ncol_x, ncol_y});
    Array_gpu<TF,2> surface_down_diffuse_count({ncol_x, ncol_y});
    Array_gpu<TF,2> surface_up_count({ncol_x, ncol_y});
    Array_gpu<TF,3> atmos_direct_count({ncol_x, ncol_y, nlay});
    Array_gpu<TF,3> atmos_diffuse_count({ncol_x, ncol_y, nlay});
    
    rrtmgp_kernel_launcher_cuda::zero_array(ncol_x, ncol_y, toa_down_count);
    rrtmgp_kernel_launcher_cuda::zero_array(ncol_x, ncol_y, toa_up_count);
    rrtmgp_kernel_launcher_cuda::zero_array(ncol_x, ncol_y, surface_down_direct_count);
    rrtmgp_kernel_launcher_cuda::zero_array(ncol_x, ncol_y, surface_down_diffuse_count);
    rrtmgp_kernel_launcher_cuda::zero_array(ncol_x, ncol_y, surface_up_count);
    rrtmgp_kernel_launcher_cuda::zero_array(ncol_x, ncol_y, nlay, atmos_direct_count);
    rrtmgp_kernel_launcher_cuda::zero_array(ncol_x, ncol_y, nlay, atmos_diffuse_count);
    
    // domain sizes
    const TF x_size = ncol_x * dx_grid;
    const TF y_size = ncol_y * dy_grid;
    const TF z_size = nlay * dz_grid;

    // direction of direct rays
    const TF dir_x = -std::sin(zenith_angle) * std::cos(azimuth_angle);
    const TF dir_y = -std::sin(zenith_angle) * std::sin(azimuth_angle);
    const TF dir_z = -std::cos(zenith_angle);


    dim3 grid{grid_size}, block{block_size};

    const Int photons_per_thread = photons_to_shoot / (grid_size * block_size);
    
    const TF flux_tod_tot = flux_tod_dir + flux_tod_dif;
    const TF diffuse_fraction = flux_tod_dif / flux_tod_tot;
    ray_tracer_kernel<<<grid, block>>>(
            photons_per_thread, k_null_grid.ptr(),
            toa_down_count.ptr(),
            toa_up_count.ptr(),
            surface_down_direct_count.ptr(),
            surface_down_diffuse_count.ptr(),
            surface_up_count.ptr(),
            atmos_direct_count.ptr(),
            atmos_diffuse_count.ptr(),
            k_ext.ptr(), ssa_asy.ptr(),
            surface_albedo,
            diffuse_fraction,
            x_size, y_size, z_size,
            dx_grid, dy_grid, dz_grid,
            dir_x, dir_y, dir_z,
            ncol_x, ncol_y, nlay,
            this->qrng_vectors_gpu, this->qrng_constants_gpu); 
    
    // convert counts to fluxes
    const TF flux_per_ray = flux_tod_tot / (photons_to_shoot / (ncol_x * ncol_y));
    
    count_to_flux_2d<<<grid_2d, block_2d>>>(
            ncol_x, ncol_y, flux_per_ray,
            toa_up_count.ptr(), 
            surface_down_direct_count.ptr(),
            surface_down_diffuse_count.ptr(),
            surface_up_count.ptr(),
            flux_toa_up.ptr(),
            flux_sfc_dir.ptr(),
            flux_sfc_dif.ptr(),
            flux_sfc_up.ptr());
    
    count_to_flux_3d<<<grid_3d, block_3d>>>(
            ncol_x, ncol_y, nlay, flux_per_ray,
            atmos_direct_count.ptr(),
            atmos_diffuse_count.ptr(),
            flux_abs_dir.ptr(),
            flux_abs_dif.ptr());
}

#ifdef RTE_RRTMGP_SINGLE_PRECISION
template class Raytracer_gpu<float>;
#else
template class Raytracer_gpu<double>;
#endif