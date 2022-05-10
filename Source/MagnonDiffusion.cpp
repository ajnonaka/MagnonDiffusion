#include "MagnonDiffusion.H"

#include <AMReX_ParmParse.H>

AMREX_GPU_MANAGED int MagnonDiffusion::max_grid_size;
AMREX_GPU_MANAGED int MagnonDiffusion::nsteps;
AMREX_GPU_MANAGED int MagnonDiffusion::plot_int;
    
// time step
AMREX_GPU_MANAGED amrex::Real MagnonDiffusion::dt;

AMREX_GPU_MANAGED amrex::Vector<int> MagnonDiffusion::bc_lo;
AMREX_GPU_MANAGED amrex::Vector<int> MagnonDiffusion::bc_hi;

AMREX_GPU_MANAGED amrex::GpuArray<int, AMREX_SPACEDIM> MagnonDiffusion::n_cell; // number of cells in each direction
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> MagnonDiffusion::prob_lo; // physical lo coordinate
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> MagnonDiffusion::prob_hi; // physical hi coordinate

// General BC parameters for Dirichlet, Neumann, or Robin
// Dirichlet: phi = f
// Neumann: d(phi)/dn = f
// Robin: a*phi+b*d(phi)/dn = f
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> MagnonDiffusion::bc_lo_f; // BC rhs.
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> MagnonDiffusion::bc_hi_f; // BC rhs.
    
// Robin BC parameters
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> MagnonDiffusion::bc_lo_a; // robin BC coeffs. 
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> MagnonDiffusion::bc_hi_a; // robin BC coeffs. 
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> MagnonDiffusion::bc_lo_b; // robin BC coeffs. 
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> MagnonDiffusion::bc_hi_b; // robin BC coeffs. 

// magnon diffusion parameters
AMREX_GPU_MANAGED amrex::Real MagnonDiffusion::D_const;
AMREX_GPU_MANAGED amrex::Real MagnonDiffusion::tau_p;

void InitializeMagnonDiffusionNamespace() {
    
    // ParmParse is way of reading inputs from the inputs file
    amrex::ParmParse pp;

    amrex::Vector<int> temp_int(AMREX_SPACEDIM);     // temporary for parsing
    amrex::Vector<amrex::Real> temp(AMREX_SPACEDIM); // temporary for parsing

    // We need to get n_cell from the inputs file - this is the number of cells on each side of
    //   a square (or cubic) domain.
    pp.getarr("n_cell",temp_int);
    for (int i=0; i<AMREX_SPACEDIM; ++i) {
        n_cell[i] = temp_int[i];
    }
        
    // parse in material properties
    pp.get("D_const", D_const);
    pp.get("tau_p", tau_p);

    // The domain is broken into boxes of size max_grid_size
    pp.get("max_grid_size",max_grid_size);

    // Default plot_int to -1, allow us to set it to something else in the inputs file
    //  If plot_int < 0 then no plot files will be writtenq
    plot_int = -1;
    pp.query("plot_int",plot_int);

    // nsteps must be specified in the inputs file
    pp.get("nsteps",nsteps);

    // time step
    pp.get("dt",dt);

    pp.getarr("prob_lo",temp);
    for (int i=0; i<AMREX_SPACEDIM; ++i) {
        prob_lo[i] = temp[i];
    }

    pp.getarr("prob_hi",temp);
    for (int i=0; i<AMREX_SPACEDIM; ++i) {
        prob_hi[i] = temp[i];
    }

    // default values for bc parameters
    for (int i=0; i<AMREX_SPACEDIM; ++i) {
        bc_lo_f[i] = 0.;
        bc_hi_f[i] = 0.;
        bc_lo_a[i] = 0.;
        bc_hi_a[i] = 0.;
        bc_lo_b[i] = 0.;
        bc_hi_b[i] = 0.;
    }

    // read in bc parameters
    if (pp.queryarr("bc_lo_f",temp)) {
        for (int i=0; i<AMREX_SPACEDIM; ++i) {
            bc_lo_f[i] = temp[i];
        }
    }
    if (pp.queryarr("bc_hi_f",temp)) {
        for (int i=0; i<AMREX_SPACEDIM; ++i) {
            bc_hi_f[i] = temp[i];
        }
    }
    if (pp.queryarr("bc_lo_a",temp)) {
        for (int i=0; i<AMREX_SPACEDIM; ++i) {
            bc_lo_a[i] = temp[i];
        }
    }
    if (pp.queryarr("bc_hi_a",temp)) {
        for (int i=0; i<AMREX_SPACEDIM; ++i) {
            bc_hi_a[i] = temp[i];
        }
    }
    if (pp.queryarr("bc_lo_b",temp)) {
        for (int i=0; i<AMREX_SPACEDIM; ++i) {
            bc_lo_b[i] = temp[i];
        }
    }
    if (pp.queryarr("bc_hi_b",temp)) {
        for (int i=0; i<AMREX_SPACEDIM; ++i) {
            bc_hi_b[i] = temp[i];
        }
    }

    // read in BC; see Src/Base/AMReX_BC_TYPES.H for supported types
    bc_lo.resize(AMREX_SPACEDIM);
    bc_hi.resize(AMREX_SPACEDIM);
    pp.getarr("bc_lo", bc_lo);
    pp.getarr("bc_hi", bc_hi);
}

void FillBoundaryPhysical (MultiFab& phi, const Geometry& geom) {

    // Physical Domain
    Box dom(geom.Domain());
    
    for (MFIter mfi(phi, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        // one ghost cell
        Box bx = mfi.growntilebox(1);

        const Array4<Real>& data = phi.array(mfi);

        // x ghost cells
        int lo = dom.smallEnd(0);
        int hi = dom.bigEnd(0);
        
        if (bx.smallEnd(0) < lo) {
            if (bc_lo[0] == BCType::foextrap || bc_lo[0] == BCType::ext_dir) {
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (i<lo) {
                        data(i,j,k) = bc_lo_f[0];
                    }
                });
            }
        }

        if (bx.bigEnd(0) > hi) {
            if (bc_hi[0] == BCType::foextrap || bc_hi[0] == BCType::ext_dir) {
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (i>hi) {
                        data(i,j,k) = bc_hi_f[0];
                    }
                });
            }
        }
        
        // y ghost cells
        lo = dom.smallEnd(1);
        hi = dom.bigEnd(1);
        
        if (bx.smallEnd(1) < lo) {
            if (bc_lo[1] == BCType::foextrap || bc_lo[1] == BCType::ext_dir) {
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (j<lo) {
                        data(i,j,k) = bc_lo_f[1];
                    }
                });
            }
        }

        if (bx.bigEnd(1) > hi) {
            if (bc_hi[1] == BCType::foextrap || bc_hi[1] == BCType::ext_dir) {
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (j>hi) {
                        data(i,j,k) = bc_hi_f[1];
                    }
                });
            }
        }
        
        // z ghost cells
        lo = dom.smallEnd(2);
        hi = dom.bigEnd(2);
        
        if (bx.smallEnd(2) < lo) {
            if (bc_lo[2] == BCType::foextrap || bc_lo[2] == BCType::ext_dir) {
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (k<lo) {
                        data(i,j,k) = bc_lo_f[2];
                    }
                });
            }
        }

        if (bx.bigEnd(2) > hi) {
            if (bc_hi[2] == BCType::foextrap || bc_hi[2] == BCType::ext_dir) {
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (k>hi) {
                        data(i,j,k) = bc_hi_f[2];
                    }
                });
            }
        }
        
    } // end MFIter
}

void FillBoundaryRobin (MultiFab& robin_a,
                        MultiFab& robin_b,
                        MultiFab& robin_f,
                        const Geometry& geom) {

    // Physical Domain
    Box dom(geom.Domain());
    
    for (MFIter mfi(robin_a, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        // one ghost cell
        Box bx = mfi.growntilebox(1);

        const Array4<Real>& data_a = robin_a.array(mfi);
        const Array4<Real>& data_b = robin_b.array(mfi);
        const Array4<Real>& data_f = robin_f.array(mfi);

        // x ghost cells
        int lo = dom.smallEnd(0);
        int hi = dom.bigEnd(0);
        
        if (bx.smallEnd(0) < lo) {
            if (bc_lo[0] == 6) { // 6 = Robin
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (i<lo) {
                        data_a(i,j,k) = bc_lo_a[0];
                        data_b(i,j,k) = bc_lo_b[0];
                        data_f(i,j,k) = bc_lo_f[0];
                    }
                });
            }
        }

        if (bx.bigEnd(0) > hi) {
            if (bc_hi[0] == 6) { // 6 = Robin
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (i>hi) {
                        data_a(i,j,k) = bc_hi_a[0];
                        data_b(i,j,k) = bc_hi_b[0];
                        data_f(i,j,k) = bc_hi_f[0];
                    }
                });
            }
        }
        
        // y ghost cells
        lo = dom.smallEnd(1);
        hi = dom.bigEnd(1);
        
        if (bx.smallEnd(1) < lo) {
            if (bc_lo[1] == 6) { // 6 = Robin
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (j<lo) {
                        data_a(i,j,k) = bc_lo_a[1];
                        data_b(i,j,k) = bc_lo_b[1];
                        data_f(i,j,k) = bc_lo_f[1];
                    }
                });
            }
        }

        if (bx.bigEnd(1) > hi) {
            if (bc_hi[1] == 6) { // 6 = Robin
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (j>hi) {
                        data_a(i,j,k) = bc_hi_a[1];
                        data_b(i,j,k) = bc_hi_b[1];
                        data_f(i,j,k) = bc_hi_f[1];
                    }
                });
            }
        }
        
        // z ghost cells
        lo = dom.smallEnd(2);
        hi = dom.bigEnd(2);
        
        if (bx.smallEnd(2) < lo) {
            if (bc_lo[2] == 6) { // 6 = Robin
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (k<lo) {
                        data_a(i,j,k) = bc_lo_a[2];
                        data_b(i,j,k) = bc_lo_b[2];
                        data_f(i,j,k) = bc_lo_f[2];
                    }
                });
            }
        }

        if (bx.bigEnd(2) > hi) {
            if (bc_hi[2] == 6) { // 6 = Robin
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (k>hi) {
                        data_a(i,j,k) = bc_hi_a[2];
                        data_b(i,j,k) = bc_hi_b[2];
                        data_f(i,j,k) = bc_hi_f[2];
                    }
                });
            }
        }
        
    } // end MFIter

}
