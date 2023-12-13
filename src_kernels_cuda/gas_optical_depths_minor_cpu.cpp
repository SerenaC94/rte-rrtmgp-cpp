#include "Types.h"
#include <iostream>
#include <omp.h>

void gas_optical_depths_minor_cpu_serial(
        const int ncol, const int nlay, const int ngpt, //ngpt: NOT USED
        const int ngas, const int nflav, const int ntemp, const int neta, //ngas, nflav: NOT USED
        const int nminor,
        const int nminork, //NOT USED
        const int idx_h2o, const int idx_tropo,
        const int* const gpoint_flavor,
        const Float* const kminor,
        const int* const minor_limits_gpt, 
        const Bool* const minor_scales_with_density,
        const Bool* const scale_by_complement, 
        const int* const idx_minor,
        const int* const idx_minor_scaling,
        const int* const kminor_start,
        const Float* const play,
        const Float* const tlay,
        const Float* const col_gas,
        const Float* const fminor,
        const int* const jeta,
        const int* const jtemp,
        const Bool* const tropo,
        Float* const tau,
        Float* const tau_minor) //NOT USED
{
    for(int icol=0; icol<ncol; icol++) //loop over y axis
    {
        for(int ilay=0; ilay<nlay; ilay++) //loop over z axis: could be swapped with previous loop
        {
            const int idx_collay = icol + ilay*ncol;

            if (tropo[idx_collay] == idx_tropo) {
                for (int imnr=0; imnr<nminor; ++imnr) //might be parallelizable
                {
                    //scaling computation (instead of shared memory initialization)
                    const int ncl = ncol * nlay;
                    Float scaling = col_gas[idx_collay + idx_minor[imnr] * ncl];

                    if (minor_scales_with_density[imnr])
                    {
                        const Float PaTohPa = 0.01;
                        scaling *= PaTohPa * play[idx_collay] / tlay[idx_collay];

                        if (idx_minor_scaling[imnr] > 0)
                        {
                            const int idx_collaywv = icol + ilay*ncol + idx_h2o*ncl;
                            Float vmr_fact = Float(1.) / col_gas[idx_collay];
                            Float dry_fact = Float(1.) / (Float(1.) + col_gas[idx_collaywv] * vmr_fact);

                            if (scale_by_complement[imnr])
                                scaling *= (Float(1.) - col_gas[idx_collay + idx_minor_scaling[imnr] * ncl] * vmr_fact * dry_fact);
                            else
                                scaling *= col_gas[idx_collay + idx_minor_scaling[imnr] * ncl] * vmr_fact * dry_fact;
                        }
                    }

                    //actual computation
                    const int gpt_start = minor_limits_gpt[2*imnr]-1;
                    const int gpt_end = minor_limits_gpt[2*imnr+1];
                    const int gpt_offs = 1-idx_tropo;
                    const int iflav = gpoint_flavor[2*gpt_start + gpt_offs]-1;

                    const int idx_fcl2 = 2 * 2 * (icol + ilay*ncol + iflav*ncol*nlay);
                    const int idx_fcl1 = 2 * (icol + ilay*ncol + iflav*ncol*nlay);

                    const Float* kfminor = &fminor[idx_fcl2];
                    const Float* kin = &kminor[0];

                    const int j0 = jeta[idx_fcl1];
                    const int j1 = jeta[idx_fcl1+1];
                    const int kjtemp = jtemp[idx_collay];
                    const int band_gpt = gpt_end-gpt_start;
                    const int gpt_offset = kminor_start[imnr]-1;

                    for (int igpt=0; igpt<band_gpt; igpt++)
                    {
                        Float ltau_minor = kfminor[0] * kin[(kjtemp-1) + (j0-1)*ntemp + (igpt+gpt_offset)*ntemp*neta] +
                                        kfminor[1] * kin[(kjtemp-1) +  j0   *ntemp + (igpt+gpt_offset)*ntemp*neta] +
                                        kfminor[2] * kin[kjtemp     + (j1-1)*ntemp + (igpt+gpt_offset)*ntemp*neta] +
                                        kfminor[3] * kin[kjtemp     +  j1   *ntemp + (igpt+gpt_offset)*ntemp*neta];

                        const int idx_out = icol + ilay*ncol + (igpt+gpt_start)*ncol*nlay;
                        tau[idx_out] += ltau_minor * scaling; //set the output
                    }
                }
            }
        }
    }
}

void gas_optical_depths_minor_cpu_parallel(
        const int ncol, const int nlay,
        const int ntemp, const int neta,
        const int nminor,
        const int idx_h2o, const int idx_tropo,
        const int* const gpoint_flavor,
        const Float* const kminor,
        const int* const minor_limits_gpt, 
        const Bool* const minor_scales_with_density,
        const Bool* const scale_by_complement, 
        const int* const idx_minor,
        const int* const idx_minor_scaling,
        const int* const kminor_start,
        const Float* const play,
        const Float* const tlay,
        const Float* const col_gas,
        const Float* const fminor,
        const int* const jeta,
        const int* const jtemp,
        const Bool* const tropo,
        Float* const tau)
{
    constexpr Float PaTohPa = 0.01;
    const int ncl = ncol * nlay;
    const int ntemp_times_neta = ntemp * neta;

    #pragma omp parallel for \
    default(none) firstprivate(ncol, nlay, ntemp, neta, nminor, idx_h2o, idx_tropo, \
    gpoint_flavor, kminor, minor_limits_gpt, minor_scales_with_density, scale_by_complement, idx_minor, \
    idx_minor_scaling, kminor_start, play, tlay, col_gas, fminor, jeta, jtemp, tropo, tau, ncl, ntemp_times_neta) \
    collapse(2) schedule(static, 1)
    for(int ilay=0; ilay<nlay; ilay++) //loop over y axis
    {
        for(int icol=0; icol<ncol; icol++) //loop over z axis: could be swapped with previous loop
        {
            const int ilay_times_ncol = ilay*ncol;
            const int idx_collay = icol + ilay_times_ncol;

            if (tropo[idx_collay] == idx_tropo) {
                const int icol_plus_ilay_times_ncol = icol + ilay_times_ncol;
                const int idx_collaywv = icol_plus_ilay_times_ncol + idx_h2o*ncl;
                Float vmr_fact = Float(1.) / col_gas[idx_collay];
                Float dry_fact = Float(1.) / (Float(1.) + col_gas[idx_collaywv] * vmr_fact);
                Float vmr_times_dry_fact = vmr_fact * dry_fact;

                for (int imnr=0; imnr<nminor; ++imnr) //might be parallelizable
                {
                    //scaling computation (instead of shared memory initialization)
                    Float scaling = col_gas[idx_collay + idx_minor[imnr] * ncl];

                    if (minor_scales_with_density[imnr])
                    {
                        scaling *= PaTohPa * play[idx_collay] / tlay[idx_collay];

                        if (idx_minor_scaling[imnr] > 0)
                        {
                            Float base_scaling = col_gas[idx_collay + idx_minor_scaling[imnr] * ncl] * vmr_times_dry_fact;
                            if (scale_by_complement[imnr])
                                scaling *= Float(1.) - base_scaling;
                            else
                                scaling *= base_scaling;
                        }
                    }

                    //actual computation
                    const int twice_imnr = 2*imnr;
                    const int gpt_start = minor_limits_gpt[twice_imnr]-1;
                    const int gpt_end = minor_limits_gpt[twice_imnr+1];
                    const int gpt_offs = 1-idx_tropo;
                    const int iflav = gpoint_flavor[2*gpt_start + gpt_offs]-1;

                    const int base_idx_fcl = icol_plus_ilay_times_ncol + iflav*ncl;
                    const int idx_fcl2 = base_idx_fcl << 2;
                    const int idx_fcl1 = base_idx_fcl << 1;

                    const Float* kfminor = &fminor[idx_fcl2];
                    const Float* kin = kminor;

                    const int j0 = jeta[idx_fcl1];
                    const int j1 = jeta[idx_fcl1+1];
                    const int kjtemp = jtemp[idx_collay];
                    const int band_gpt = gpt_end-gpt_start;
                    const int gpt_offset = kminor_start[imnr]-1;

                    const int start_addr_1 = (kjtemp-1) +  j0*ntemp;
                    const int start_addr_0 = start_addr_1 - ntemp;
                    const int start_addr_3 = kjtemp +  j1*ntemp;
                    const int start_addr_2 = start_addr_3 - ntemp;

                    //can be parallelized
                    if(band_gpt == 16) {
                        #pragma unroll
                        for (int igpt=0; igpt<16; igpt++)
                        {
                            const int end_addr = (igpt+gpt_offset)*ntemp_times_neta;
                            Float ltau_minor = kfminor[0] * kin[start_addr_0 + end_addr] +
                                            kfminor[1] * kin[start_addr_1 + end_addr] +
                                            kfminor[2] * kin[start_addr_2 + end_addr] +
                                            kfminor[3] * kin[start_addr_3 + end_addr];

                            const int idx_out = icol_plus_ilay_times_ncol + (igpt+gpt_start)*ncl;
                            tau[idx_out] += ltau_minor * scaling; //set the output
                        }
                    } else {
                        for (int igpt=0; igpt<band_gpt; igpt++)
                        {
                            const int end_addr = (igpt+gpt_offset)*ntemp_times_neta;
                            Float ltau_minor = kfminor[0] * kin[start_addr_0 + end_addr] +
                                            kfminor[1] * kin[start_addr_1 + end_addr] +
                                            kfminor[2] * kin[start_addr_2 + end_addr] +
                                            kfminor[3] * kin[start_addr_3 + end_addr];

                            const int idx_out = icol_plus_ilay_times_ncol + (igpt+gpt_start)*ncl;
                            tau[idx_out] += ltau_minor * scaling; //set the output
                        }
                    }
                }
            }
        }
    }
}