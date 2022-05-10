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
