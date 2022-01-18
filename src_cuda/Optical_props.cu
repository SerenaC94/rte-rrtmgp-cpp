/*
 * This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
 * and Rapid Radiative Transfer Model for GCM applications Parallel (RRTMGP).
 *
 * The original code is found at https://github.com/earth-system-radiation/rte-rrtmgp.
 *
 * Contacts: Robert Pincus and Eli Mlawer
 * email: rrtmgp@aer.com
 *
 * Copyright 2015-2020,  Atmospheric and Environmental Research and
 * Regents of the University of Colorado.  All right reserved.
 *
 * This C++ interface can be downloaded from https://github.com/earth-system-radiation/rte-rrtmgp-cpp
 *
 * Contact: Chiel van Heerwaarden
 * email: chiel.vanheerwaarden@wur.nl
 *
 * Copyright 2020, Wageningen University & Research.
 *
 * Use and duplication is permitted under the terms of the
 * BSD 3-clause license, see http://opensource.org/licenses/BSD-3-Clause
 *
 */

#include "Optical_props.h"
#include "Array.h"


#include "optical_props_kernel_launcher_cuda.h"





// Optical properties per gpoint.
template<typename TF>
Optical_props_gpu<TF>::Optical_props_gpu(
        const Array<TF,2>& band_lims_wvn,
        const Array<int,2>& band_lims_gpt)
{
    Array<int,2> band_lims_gpt_lcl(band_lims_gpt);
    Array_gpu<int,2> band_lims_gpt_lcl_gpu(band_lims_gpt);

    this->band2gpt = band_lims_gpt_lcl;
    this->band2gpt_gpu = this->band2gpt;
    this->band_lims_wvn = band_lims_wvn;

    // Make a map between g-points and bands.
    this->gpt2band.set_dims({band_lims_gpt_lcl.max()});
    for (int iband=1; iband<=band_lims_gpt_lcl.dim(2); ++iband)
    {
        for (int i=band_lims_gpt_lcl({1,iband}); i<=band_lims_gpt_lcl({2,iband}); ++i)
            this->gpt2band({i}) =  iband;
    }
    this->gpt2band_gpu = this->gpt2band;
}


// Optical properties per band.
template<typename TF>
Optical_props_gpu<TF>::Optical_props_gpu(
        const Array<TF,2>& band_lims_wvn)
{
    Array<int,2> band_lims_gpt_lcl({2, band_lims_wvn.dim(2)});

    for (int iband=1; iband<=band_lims_wvn.dim(2); ++iband)
    {
        band_lims_gpt_lcl({1, iband}) = iband;
        band_lims_gpt_lcl({2, iband}) = iband;
    }

    this->band2gpt = band_lims_gpt_lcl;
    this->band2gpt_gpu = this->band2gpt;
    this->band_lims_wvn = band_lims_wvn;

    // Make a map between g-points and bands.
    this->gpt2band.set_dims({band_lims_gpt_lcl.max()});
    for (int iband=1; iband<=band_lims_gpt_lcl.dim(2); ++iband)
    {
        for (int i=band_lims_gpt_lcl({1,iband}); i<=band_lims_gpt_lcl({2,iband}); ++i)
            this->gpt2band({i}) =  iband;
    }
    this->gpt2band_gpu = this->gpt2band;
}


template<typename TF>
Optical_props_1scl_gpu<TF>::Optical_props_1scl_gpu(
        const int ncol,
        const int nlay,
        const Optical_props_gpu<TF>& optical_props_gpu) :
    Optical_props_arry_gpu<TF>(optical_props_gpu),
    tau({ncol, nlay})
{}



template<typename TF>
Optical_props_2str_gpu<TF>::Optical_props_2str_gpu(
        const int ncol,
        const int nlay,
        const Optical_props_gpu<TF>& optical_props_gpu) :
    Optical_props_arry_gpu<TF>(optical_props_gpu),
    tau({ncol, nlay}),
    ssa({ncol, nlay}),
    g  ({ncol, nlay})
{}



template<typename TF>
void Optical_props_2str_gpu<TF>::delta_scale(const Array_gpu<TF,3>& forward_frac)
{
    const int ncol = this->get_ncol();
    const int nlay = this->get_nlay();
    const int ngpt = this->get_ngpt();

    optical_props_kernel_launcher_cuda::delta_scale_2str_k(
            ncol, nlay, ngpt,
            this->get_tau(), this->get_ssa(), this->get_g());
}


template<typename TF>
void add_to(Optical_props_1scl_gpu<TF>& op_inout, const Optical_props_1scl_gpu<TF>& op_in)
{
    const int ncol = op_inout.get_ncol();
    const int nlay = op_inout.get_nlay();

    optical_props_kernel_launcher_cuda::increment_1scalar_by_1scalar(
            ncol, nlay,
            op_inout.get_tau(), op_in.get_tau());
}


template<typename TF>
void add_to(Optical_props_2str_gpu<TF>& op_inout, const Optical_props_2str_gpu<TF>& op_in)
{
    const int ncol = op_inout.get_ncol();
    const int nlay = op_inout.get_nlay();
    optical_props_kernel_launcher_cuda::increment_2stream_by_2stream(
            ncol, nlay,
            op_inout.get_tau(), op_inout.get_ssa(), op_inout.get_g(),
            op_in   .get_tau(), op_in   .get_ssa(), op_in   .get_g());    
}


#ifdef RTE_RRTMGP_SINGLE_PRECISION
template class Optical_props_gpu<float>;
template class Optical_props_1scl_gpu<float>;
template class Optical_props_2str_gpu<float>;
template void add_to(Optical_props_2str_gpu<float>&, const Optical_props_2str_gpu<float>&);
template void add_to(Optical_props_1scl_gpu<float>&, const Optical_props_1scl_gpu<float>&);
#else
template class Optical_props_gpu<double>;
template class Optical_props_1scl_gpu<double>;
template class Optical_props_2str_gpu<double>;
template void add_to(Optical_props_2str_gpu<double>&, const Optical_props_2str_gpu<double>&);
template void add_to(Optical_props_1scl_gpu<double>&, const Optical_props_1scl_gpu<double>&);
#endif
