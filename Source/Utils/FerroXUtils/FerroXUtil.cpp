/*
 * This file is part of FerroX.
 *
 * Contributor: Prabhat Kumar
 *
 */
#include <FerroXUtil.H>

using namespace amrex;


void FerroX_Util::Contains_sc(MultiFab& MaterialMask, bool& contains_SC)
{

	int has_SC = 0;

        for ( MFIter mfi(MaterialMask, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

            const Box& bx = mfi.validbox();
            const auto lo = amrex::lbound(bx);
            const auto hi = amrex::ubound(bx);

            const Array4<Real>& mask = MaterialMask.array(mfi);

            for (auto k = lo.z; k <= hi.z; ++k) {
            for (auto j = lo.y; j <= hi.y; ++j) {
            for (auto i = lo.x; i <= hi.x; ++i) {
                  if (mask(i,j,k) >= 2.0) {
                          has_SC = 1;
                  }
            }
            }
            }

        } // end MFIter

       // parallel reduce max has_SC
       ParallelDescriptor::ReduceIntMax(has_SC);
 
       if(has_SC == 1) contains_SC = true;
}

void FerroX_Util::AverageFaceCenteredMultiFabToCellCenters(const std::array< amrex::MultiFab, AMREX_SPACEDIM >& face_arr, 
                                                             amrex::MultiFab& cc_arr)
{
    for (MFIter mfi(cc_arr, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {   
        const Array4<Real> & cc = cc_arr.array(mfi);
        const amrex::Box& bx = mfi.validbox();

        // Get the face-centered data for each direction
        std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> face_arr_data;
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            face_arr_data[dir] = face_arr[dir].array(mfi);
        }   

        // Iterate over the cells in the valid region and compute the average
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            amrex::Real sum = 0.0;
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                sum += 0.5 * (face_arr_data[dir](i,j,k) + face_arr_data[dir](i-1,j,k)); // X-dir
                sum += 0.5 * (face_arr_data[dir](i,j,k) + face_arr_data[dir](i,j-1,k)); // Y-dir
#if (AMREX_SPACEDIM == 3)
                sum += 0.5 * (face_arr_data[dir](i,j,k) + face_arr_data[dir](i,j,k-1)); // Z-dir
#endif
            }   
            cc(i,j,k) = sum / AMREX_SPACEDIM;
        }); 

    }   

}


