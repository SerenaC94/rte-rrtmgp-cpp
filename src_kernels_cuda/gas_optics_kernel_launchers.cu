#include <chrono>
#include <functional>
#include <iostream>
#include <iomanip>

#include "rrtmgp_kernel_launcher_cuda.h"
#include "tools_gpu.h"
#include "Array.h"
#include "tuner.h"


namespace
{
    #include "gas_optics_kernels.cu"
    #include "gas_optical_depths_minor_cpu.cpp"
}

#include <cassert>
#include <algorithm>
#include <chrono>

namespace rrtmgp_kernel_launcher_cuda
{
    void reorder123x321(
            const int ni, const int nj, const int nk,
            const Float* arr_in, Float* arr_out)
    {
        Tuner_map& tunings = Tuner::get_map();

        dim3 grid(ni, nj, nk);
        dim3 block;

        if (tunings.count("reorder123x321_kernel") == 0)
        {
            std::tie(grid, block) = tune_kernel(
                "reorder123x321_kernel",
                dim3(ni, nj, nk),
                {1, 2, 4, 8, 16, 24, 32, 48, 64, 96},
                {1, 2, 4, 8, 16, 24, 32, 48, 64, 96},
                {1, 2, 4, 8, 16, 24, 32, 48, 64, 96},
                reorder123x321_kernel,
                ni, nj, nk, arr_in, arr_out);

            tunings["reorder123x321_kernel"].first = grid;
            tunings["reorder123x321_kernel"].second = block;
        }
        else
        {
            grid = tunings["reorder123x321_kernel"].first;
            block = tunings["reorder123x321_kernel"].second;
        }

        reorder123x321_kernel<<<grid, block>>>(
                ni, nj, nk, arr_in, arr_out);
    }


    void reorder12x21(
            const int ni, const int nj,
            const Float* arr_in, Float* arr_out)
    {
        const int block_i = 32;
        const int block_j = 16;

        const int grid_i = ni/block_i + (ni%block_i > 0);
        const int grid_j = nj/block_j + (nj%block_j > 0);

        dim3 grid_gpu(grid_i, grid_j);
        dim3 block_gpu(block_i, block_j);

        reorder12x21_kernel<<<grid_gpu, block_gpu>>>(
                ni, nj, arr_in, arr_out);
    }


    void zero_array(const int ni, const int nj, const int nk, Float* arr)
    {
        const int block_i = 32;
        const int block_j = 16;
        const int block_k = 1;

        const int grid_i = ni/block_i + (ni%block_i > 0);
        const int grid_j = nj/block_j + (nj%block_j > 0);
        const int grid_k = nk/block_k + (nk%block_k > 0);

        dim3 grid_gpu(grid_i, grid_j, grid_k);
        dim3 block_gpu(block_i, block_j, block_k);

        zero_array_kernel<<<grid_gpu, block_gpu>>>(
                ni, nj, nk, arr);

    }


    void interpolation(
            const int ncol, const int nlay,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const int* flavor,
            const Float* press_ref_log,
            const Float* temp_ref,
            Float press_ref_log_delta,
            Float temp_ref_min,
            Float temp_ref_delta,
            Float press_ref_trop_log,
            const Float* vmr_ref,
            const Float* play,
            const Float* tlay,
            Float* col_gas,
            int* jtemp,
            Float* fmajor, Float* fminor,
            Float* col_mix,
            Bool* tropo,
            int* jeta,
            int* jpress)
    {
        const int block_col  = 4;
        const int block_lay  = 2;
        const int block_flav = 16;

        const int grid_col  = ncol /block_col  + (ncol%block_col   > 0);
        const int grid_lay  = nlay /block_lay  + (nlay%block_lay   > 0);
        const int grid_flav = nflav/block_flav + (nflav%block_flav > 0);

        dim3 grid_gpu(grid_col, grid_lay, grid_flav);
        dim3 block_gpu(block_col, block_lay, block_flav);

        Float tmin = std::numeric_limits<Float>::min();
        interpolation_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, ngas, nflav, neta, npres, ntemp, tmin,
                flavor, press_ref_log, temp_ref,
                press_ref_log_delta, temp_ref_min,
                temp_ref_delta, press_ref_trop_log,
                vmr_ref, play, tlay,
                col_gas, jtemp, fmajor,
                fminor, col_mix, tropo,
                jeta, jpress);
    }


    void combine_abs_and_rayleigh(
            const int ncol, const int nlay, const int ngpt,
            const Float* tau_abs, const Float* tau_rayleigh,
            Float* tau, Float* ssa, Float* g)
    {
        Tuner_map& tunings = Tuner::get_map();

        Float tmin = std::numeric_limits<Float>::min();

        dim3 grid(ncol, nlay, ngpt);
        dim3 block;

        if (tunings.count("combine_abs_and_rayleigh_kernel") == 0)
        {
            std::tie(grid, block) = tune_kernel(
                "combine_abs_and_rayleigh_kernel",
                dim3(ncol, nlay, ngpt),
                {1, 2, 4, 8, 16, 24, 32, 48, 64, 96}, {1, 2, 4}, {1, 2, 4, 8, 16, 24, 32, 48, 64, 96},
                combine_abs_and_rayleigh_kernel,
                ncol, nlay, ngpt, tmin,
                tau_abs, tau_rayleigh,
                tau, ssa, g);

            tunings["combine_abs_and_rayleigh_kernel"].first = grid;
            tunings["combine_abs_and_rayleigh_kernel"].second = block;
        }
        else
        {
            grid = tunings["combine_abs_and_rayleigh_kernel"].first;
            block = tunings["combine_abs_and_rayleigh_kernel"].second;
        }

        combine_abs_and_rayleigh_kernel<<<grid, block>>>(
                ncol, nlay, ngpt, tmin,
                tau_abs, tau_rayleigh,
                tau, ssa, g);
    }


    void compute_tau_rayleigh(
            const int ncol, const int nlay, const int nbnd, const int ngpt,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const int* gpoint_flavor,
            const int* gpoint_bands,
            const int* band_lims_gpt,
            const Float* krayl,
            int idx_h2o, const Float* col_dry, const Float* col_gas,
            const Float* fminor, const int* jeta,
            const Bool* tropo, const int* jtemp,
            Float* tau_rayleigh)
    {
        Tuner_map& tunings = Tuner::get_map();

        dim3 grid(ncol, nlay, ngpt);
        dim3 block;

        if (tunings.count("compute_tau_rayleigh_kernel") == 0)
        {
            std::tie(grid, block) = tune_kernel(
                "compute_tau_rayleigh_kernel",
                dim3(ncol, nlay, ngpt),
                {1, 2, 4, 16, 24, 32}, {1, 2, 4}, {1, 2, 4, 8, 16},
                compute_tau_rayleigh_kernel,
                ncol, nlay, nbnd, ngpt,
                ngas, nflav, neta, npres, ntemp,
                gpoint_flavor,
                gpoint_bands,
                band_lims_gpt,
                krayl,
                idx_h2o, col_dry, col_gas,
                fminor, jeta,
                tropo, jtemp,
                tau_rayleigh);

            tunings["compute_tau_rayleigh_kernel"].first = grid;
            tunings["compute_tau_rayleigh_kernel"].second = block;
        }
        else
        {
            grid = tunings["compute_tau_rayleigh_kernel"].first;
            block = tunings["compute_tau_rayleigh_kernel"].second;
        }

        compute_tau_rayleigh_kernel<<<grid, block>>>(
                ncol, nlay, nbnd, ngpt,
                ngas, nflav, neta, npres, ntemp,
                gpoint_flavor,
                gpoint_bands,
                band_lims_gpt,
                krayl,
                idx_h2o, col_dry, col_gas,
                fminor, jeta,
                tropo, jtemp,
                tau_rayleigh);
    }


    struct Gas_optical_depths_major_kernel
    {
        template<unsigned int I, unsigned int J, unsigned int K, class... Args>
        static void launch(dim3 grid, dim3 block, Args... args)
        {
            gas_optical_depths_major_kernel<I, J, K><<<grid, block>>>(args...);
        }
    };


    //the kernel is called here, but vscode finds no references
    //commenting out to be sure
    /*
    struct Gas_optical_depths_minor_kernel
    {
        template<unsigned int I, unsigned int J, unsigned int K, class... Args>
        static void launch(dim3 grid, dim3 block, Args... args)
        {
            gas_optical_depths_minor_kernel<I, J, K><<<grid, block>>>(args...);
        }
    };
    */


    void compute_tau_absorption(
            const int ncol, const int nlay, const int nband, const int ngpt,
            const int ngas, const int nflav, const int neta, const int npres, const int ntemp,
            const int nminorlower, const int nminorklower,
            const int nminorupper, const int nminorkupper,
            const int idx_h2o,
            const int* gpoint_flavor, const int* gpoint_flavor_cpu,
            const int* band_lims_gpt, const int* band_lims_gpt_cpu,
            const Float* kmajor, const Float* kmajor_cpu,
            const Float* kminor_lower, const Float* kminor_lower_cpu,
            const Float* kminor_upper, const Float* kminor_upper_cpu,
            const int* minor_limits_gpt_lower, const int* minor_limits_gpt_lower_cpu,
            const int* minor_limits_gpt_upper, const int* minor_limits_gpt_upper_cpu,
            const Bool* minor_scales_with_density_lower, const Bool* minor_scales_with_density_lower_cpu,
            const Bool* minor_scales_with_density_upper, const Bool* minor_scales_with_density_upper_cpu,
            const Bool* scale_by_complement_lower, const Bool* scale_by_complement_lower_cpu,
            const Bool* scale_by_complement_upper, const Bool* scale_by_complement_upper_cpu,
            const int* idx_minor_lower, const int* idx_minor_lower_cpu,
            const int* idx_minor_upper, const int* idx_minor_upper_cpu,
            const int* idx_minor_scaling_lower, const int* idx_minor_scaling_lower_cpu,
            const int* idx_minor_scaling_upper, const int* idx_minor_scaling_upper_cpu,
            const int* kminor_start_lower, const int* kminor_start_lower_cpu,
            const int* kminor_start_upper, const int* kminor_start_upper_cpu,
            const Bool* tropo, const Bool* tropo_cpu,
            const Float* col_mix, const Float* col_mix_cpu,
            const Float* fmajor, const Float* fmajor_cpu,
            const Float* fminor, const Float* fminor_cpu,
            const Float* play, const Float* play_cpu,
            const Float* tlay, const Float* tlay_cpu,
            const Float* col_gas, const Float* col_gas_cpu,
            const int* jeta, const int* jeta_cpu,
            const int* jtemp, const int* jtemp_cpu,
            const int* jpress, const int* jpress_cpu,
            Float* tau, Array_gpu<Float,3> &tau_array,
            int execution_mode)
    {
        Tuner_map& tunings = Tuner::get_map();

        dim3 grid_gpu_maj(ngpt, nlay, ncol);
        dim3 block_gpu_maj;

        if (tunings.count("gas_optical_depths_major_kernel") == 0)
        {
            Float* tau_tmp = Tools_gpu::allocate_gpu<Float>(ngpt*nlay*ncol);
            std::tie(grid_gpu_maj, block_gpu_maj) =
                tune_kernel_compile_time<Gas_optical_depths_major_kernel>(
                    "gas_optical_depths_major_kernel",
                    dim3(ngpt, nlay, ncol),
                    std::integer_sequence<unsigned int, 1, 2, 4, 8, 16, 24, 32, 48, 64>{},
                    std::integer_sequence<unsigned int, 1, 2, 4>{},
                    std::integer_sequence<unsigned int, 8, 16, 24, 32, 48, 64, 96, 128, 256>{},
                    ncol, nlay, nband, ngpt,
                    nflav, neta, npres, ntemp,
                    gpoint_flavor, band_lims_gpt,
                    kmajor, col_mix, fmajor, jeta,
                    tropo, jtemp, jpress,
                    tau_tmp);

            Tools_gpu::free_gpu<Float>(tau_tmp);

            tunings["gas_optical_depths_major_kernel"].first = grid_gpu_maj;
            tunings["gas_optical_depths_major_kernel"].second = block_gpu_maj;
        }
        else
        {
            grid_gpu_maj = tunings["gas_optical_depths_major_kernel"].first;
            block_gpu_maj = tunings["gas_optical_depths_major_kernel"].second;
        }

        //only tau changes
        run_kernel_compile_time<Gas_optical_depths_major_kernel>(
                std::integer_sequence<unsigned int, 1, 2, 4, 8, 16, 24, 32, 48, 64>{},
                std::integer_sequence<unsigned int, 1, 2, 4>{},
                std::integer_sequence<unsigned int, 8, 16, 24, 32, 48, 64, 96, 128, 256>{},
                grid_gpu_maj, block_gpu_maj,
                ncol, nlay, nband, ngpt,
                nflav, neta, npres, ntemp,
                gpoint_flavor, band_lims_gpt,
                kmajor, col_mix, fmajor, jeta,
                tropo, jtemp, jpress,
                tau);



        if (execution_mode == 0) {
            // Lower
            cudaEvent_t start1, stop1;
            cudaEventCreate(&start1);
            cudaEventCreate(&stop1);

            int idx_tropo = 1;

            dim3 grid_gpu_min_1(1, 42, 8);
            dim3 block_gpu_min_1(8,1,16);

            std::cout << "Running GPU code: lower call" << std::endl;
            cudaEventRecord(start1);
            gas_optical_depths_minor_kernel<8,1,16><<<grid_gpu_min_1, block_gpu_min_1>>>(
                                            ncol, nlay, ngpt,
                                            ngas, nflav, ntemp, neta,
                                            nminorlower,
                                            nminorklower,
                                            idx_h2o, idx_tropo,
                                            gpoint_flavor,
                                            kminor_lower,
                                            minor_limits_gpt_lower,
                                            minor_scales_with_density_lower,
                                            scale_by_complement_lower,
                                            idx_minor_lower,
                                            idx_minor_scaling_lower,
                                            kminor_start_lower,
                                            play, tlay, col_gas,
                                            fminor, jeta, jtemp,
                                            tropo, tau, nullptr);
            cudaEventRecord(stop1);
            cudaEventSynchronize(stop1);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start1, stop1);
            std::cout << "Time elapsed: " << milliseconds << "ms" << std::endl;

            // Upper
            cudaEvent_t start2, stop2;
            cudaEventCreate(&start2);
            cudaEventCreate(&stop2);

            idx_tropo = 0;

            dim3 grid_gpu_min_2(1, 42, 4);
            dim3 block_gpu_min_2(8,1,32);

            std::cout << "Running GPU code: upper call" << std::endl;
            cudaEventRecord(start2);
            gas_optical_depths_minor_kernel<8,1,32><<<grid_gpu_min_2, block_gpu_min_2>>>(
                                            ncol, nlay, ngpt,
                                            ngas, nflav, ntemp, neta,
                                            nminorupper,
                                            nminorkupper,
                                            idx_h2o, idx_tropo,
                                            gpoint_flavor,
                                            kminor_upper,
                                            minor_limits_gpt_upper,
                                            minor_scales_with_density_upper,
                                            scale_by_complement_upper,
                                            idx_minor_upper,
                                            idx_minor_scaling_upper,
                                            kminor_start_upper,
                                            play, tlay, col_gas,
                                            fminor, jeta, jtemp,
                                            tropo, tau, nullptr);
            cudaEventRecord(stop2);
            cudaEventSynchronize(stop2);
            cudaEventElapsedTime(&milliseconds, start2, stop2);
            std::cout << "Time elapsed: " << milliseconds << "ms" << std::endl;
        } else {
            //transfer data from GPU to CPU
            Array<Float,3> tau_cpu_array(tau_array);
            Float* tau_cpu = tau_cpu_array.ptr();

            // Lower
            int idx_tropo = 1;
            
            if(execution_mode == 1) {
                std::cout << "Running serial CPU code: lower call" << std::endl;
                auto t0 = std::chrono::high_resolution_clock::now();
                gas_optical_depths_minor_cpu_serial(
                                                ncol, nlay, ngpt,
                                                ngas, nflav, ntemp, neta,
                                                nminorlower,
                                                nminorklower,
                                                idx_h2o, idx_tropo,
                                                gpoint_flavor_cpu,
                                                kminor_lower_cpu,
                                                minor_limits_gpt_lower_cpu,
                                                minor_scales_with_density_lower_cpu,
                                                scale_by_complement_lower_cpu,
                                                idx_minor_lower_cpu,
                                                idx_minor_scaling_lower_cpu,
                                                kminor_start_lower_cpu,
                                                play_cpu, tlay_cpu, col_gas_cpu,
                                                fminor_cpu, jeta_cpu, jtemp_cpu,
                                                tropo_cpu, tau_cpu, nullptr);
                auto t1 = std::chrono::high_resolution_clock::now();
                auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                double milliseconds = static_cast<double>(dt)/1000;
                std::cout << "Time elapsed: " << milliseconds << "ms" << std::endl;
            } else {
                std::cout << "Running parallel CPU code: lower call" << std::endl;
                auto t0 = std::chrono::high_resolution_clock::now();
                gas_optical_depths_minor_cpu_parallel(
                                                ncol, nlay,
                                                ntemp, neta,
                                                nminorlower,
                                                idx_h2o, idx_tropo,
                                                gpoint_flavor_cpu,
                                                kminor_lower_cpu,
                                                minor_limits_gpt_lower_cpu,
                                                minor_scales_with_density_lower_cpu,
                                                scale_by_complement_lower_cpu,
                                                idx_minor_lower_cpu,
                                                idx_minor_scaling_lower_cpu,
                                                kminor_start_lower_cpu,
                                                play_cpu, tlay_cpu, col_gas_cpu,
                                                fminor_cpu, jeta_cpu, jtemp_cpu,
                                                tropo_cpu, tau_cpu);
                auto t1 = std::chrono::high_resolution_clock::now();
                auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                double milliseconds = static_cast<double>(dt)/1000;
                std::cout << "Time elapsed: " << milliseconds << "ms" << std::endl;
            }

            // Upper
            idx_tropo = 0;

            if(execution_mode == 1) {
                std::cout << "Running serial CPU code: upper call" << std::endl;
                auto t0 = std::chrono::high_resolution_clock::now();
                gas_optical_depths_minor_cpu_serial(
                                                ncol, nlay, ngpt,
                                                ngas, nflav, ntemp, neta,
                                                nminorupper,
                                                nminorkupper,
                                                idx_h2o, idx_tropo,
                                                gpoint_flavor_cpu,
                                                kminor_upper_cpu,
                                                minor_limits_gpt_upper_cpu,
                                                minor_scales_with_density_upper_cpu,
                                                scale_by_complement_upper_cpu,
                                                idx_minor_upper_cpu,
                                                idx_minor_scaling_upper_cpu,
                                                kminor_start_upper_cpu,
                                                play_cpu, tlay_cpu, col_gas_cpu,
                                                fminor_cpu, jeta_cpu, jtemp_cpu,
                                                tropo_cpu, tau_cpu, nullptr);
                auto t1 = std::chrono::high_resolution_clock::now();
                auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                double milliseconds = static_cast<double>(dt)/1000;
                std::cout << "Time elapsed: " << milliseconds << "ms" << std::endl;
            } else {
                std::cout << "Running parallel CPU code: upper call" << std::endl;
                auto t0 = std::chrono::high_resolution_clock::now();
                gas_optical_depths_minor_cpu_parallel(
                                                ncol, nlay,
                                                ntemp, neta,
                                                nminorupper,
                                                idx_h2o, idx_tropo,
                                                gpoint_flavor_cpu,
                                                kminor_upper_cpu,
                                                minor_limits_gpt_upper_cpu,
                                                minor_scales_with_density_upper_cpu,
                                                scale_by_complement_upper_cpu,
                                                idx_minor_upper_cpu,
                                                idx_minor_scaling_upper_cpu,
                                                kminor_start_upper_cpu,
                                                play_cpu, tlay_cpu, col_gas_cpu,
                                                fminor_cpu, jeta_cpu, jtemp_cpu,
                                                tropo_cpu, tau_cpu);
                auto t1 = std::chrono::high_resolution_clock::now();
                auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                double milliseconds = static_cast<double>(dt)/1000;
                std::cout << "Time elapsed: " << milliseconds << "ms" << std::endl;
            }

            //transfer data from CPU to GPU
            auto t0 = std::chrono::high_resolution_clock::now();
            Array_gpu<Float,3> tau_gpu(tau_cpu_array);
            tau_array.copy_from_other_array(tau_gpu);
            auto t1 = std::chrono::high_resolution_clock::now();
            auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            double milliseconds = static_cast<double>(dt)/1000;
            std::cout << "tau transfer time: " << milliseconds << "ms" << std::endl;
        }
    }


    void Planck_source(
            const int ncol, const int nlay, const int nbnd, const int ngpt,
            const int nflav, const int neta, const int npres, const int ntemp,
            const int nPlanckTemp,
            const Float* tlay,
            const Float* tlev,
            const Float* tsfc,
            const int sfc_lay,
            const Float* fmajor,
            const int* jeta,
            const Bool* tropo,
            const int* jtemp,
            const int* jpress,
            const int* gpoint_bands,
            const int* band_lims_gpt,
            const Float* pfracin,
            const Float temp_ref_min, const Float totplnk_delta,
            const Float* totplnk,
            const int* gpoint_flavor,
            Float* sfc_src,
            Float* lay_src,
            Float* lev_src_inc,
            Float* lev_src_dec,
            Float* sfc_src_jac)
    {
        Tuner_map& tunings = Tuner::get_map();

        const Float delta_Tsurf = Float(1.);

        const int block_gpt = 16;
        const int block_lay = 4;
        const int block_col = 2;

        const int grid_gpt = ngpt/block_gpt + (ngpt%block_gpt > 0);
        const int grid_lay = nlay/block_lay + (nlay%block_lay > 0);
        const int grid_col = ncol/block_col + (ncol%block_col > 0);

        dim3 grid_gpu(grid_gpt, grid_lay, grid_col);
        dim3 block_gpu(block_gpt, block_lay, block_col);
        
        if (tunings.count("Planck_source_kernel") == 0)
        {
            std::tie(grid_gpu, block_gpu) = tune_kernel(
                    "Planck_source_kernel",
                    dim3(ngpt, nlay, ncol),
                    {1, 2, 4},
                    {1, 2},
                    {1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 256},
                    Planck_source_kernel,
                    ncol, nlay, nbnd, ngpt,
                    nflav, neta, npres, ntemp, nPlanckTemp,
                    tlay, tlev, tsfc, sfc_lay,
                    fmajor, jeta, tropo, jtemp,
                    jpress, gpoint_bands, band_lims_gpt,
                    pfracin, temp_ref_min, totplnk_delta,
                    totplnk, gpoint_flavor,
                    delta_Tsurf, sfc_src, lay_src,
                    lev_src_inc, lev_src_dec,
                    sfc_src_jac);
            
            tunings["Planck_source_kernel"].first = grid_gpu;
            tunings["Planck_source_kernel"].second = block_gpu;
        }
        else
        {
            grid_gpu = tunings["Planck_source_kernel"].first;
            block_gpu = tunings["Planck_source_kernel"].second;
        }

        Planck_source_kernel<<<grid_gpu, block_gpu>>>(
                ncol, nlay, nbnd, ngpt,
                nflav, neta, npres, ntemp, nPlanckTemp,
                tlay, tlev, tsfc, sfc_lay,
                fmajor, jeta, tropo, jtemp,
                jpress, gpoint_bands, band_lims_gpt,
                pfracin, temp_ref_min, totplnk_delta,
                totplnk, gpoint_flavor,
                delta_Tsurf,
                sfc_src, lay_src,
                lev_src_inc, lev_src_dec,
                sfc_src_jac);
    }
}
