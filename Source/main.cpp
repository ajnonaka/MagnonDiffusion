
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include "myfunc.H"
#include "myfunc_F.H"  // includes advance.cpp

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
    return 0;
}

void main_main ()
{
    // What time is it now?  We'll use this to compute total run time.
    auto strt_time = amrex::second();

    // AMREX_SPACEDIM: number of dimensions
    int max_grid_size, nsteps, plot_int;
    // time step
    Real dt;

    Vector<int> bc_lo(AMREX_SPACEDIM,0);
    Vector<int> bc_hi(AMREX_SPACEDIM,0);

    amrex::GpuArray<int, AMREX_SPACEDIM> n_cell; // number of cells in each direction
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo; // physical lo coordinate
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi; // physical hi coordinate

    // General BC parameters for Dirichlet, Neumann, or Robin
    // Dirichlet: phi = f
    // Neumann: d(phi)/dn = f
    // Robin: a*phi+b*d(phi)/dn = f
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> bc_lo_f; // BC rhs.
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> bc_hi_f; // BC rhs.
    
    // Robin BC parameters
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> bc_lo_a; // robin BC coeffs. 
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> bc_hi_a; // robin BC coeffs. 
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> bc_lo_b; // robin BC coeffs. 
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> bc_hi_b; // robin BC coeffs. 

    // magnon diffusion parameters
    Real D_const, tau_p;

    // inputs parameters
    {
        // ParmParse is way of reading inputs from the inputs file
        ParmParse pp;

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
        pp.getarr("bc_lo", bc_lo);
        pp.getarr("bc_hi", bc_hi);
    }

    // determine whether boundary conditions are periodic
    Vector<int> is_periodic(AMREX_SPACEDIM,0);
    for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
        if (bc_lo[idim] == BCType::int_dir && bc_hi[idim] == BCType::int_dir) {
            is_periodic[idim] = 1;
        }
    }

    // make BoxArray and Geometry
    BoxArray ba;
    Geometry geom;
    {
        IntVect dom_lo(AMREX_D_DECL(          0,           0,           0));
        IntVect dom_hi(AMREX_D_DECL(n_cell[0]-1, n_cell[1]-1, n_cell[2]-1));
        Box domain(dom_lo, dom_hi);

        // Initialize the boxarray "ba" from the single box "bx"
        ba.define(domain);
        // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
        ba.maxSize(max_grid_size);

        // This defines the physical box in each direction.
        RealBox real_box({AMREX_D_DECL( prob_lo[0], prob_lo[1], prob_lo[2])},
                         {AMREX_D_DECL( prob_hi[0], prob_hi[1], prob_hi[2])});

        // This defines a Geometry object
        geom.define(domain,&real_box,CoordSys::cartesian,is_periodic.data());
    }

    // Nghost = number of ghost cells for each array
    int Nghost = 1;

    // Ncomp = number of components for each array
    int Ncomp  = 1;

    // time = starting time in the simulation
    Real time = 0.0;

    // How Boxes are distrubuted among MPI processes
    DistributionMapping dm(ba);

    // we allocate two phi multifabs; one will store the old state, the other the new.
    MultiFab phi_old(ba, dm, Ncomp, Nghost);
    MultiFab phi_new(ba, dm, Ncomp, Nghost);

    // Initialize phi_new by calling a Fortran routine.
    // MFIter = MultiFab Iterator
    for ( MFIter mfi(phi_new); mfi.isValid(); ++mfi )
    {
        const Box& bx = mfi.validbox();

        init_phi(BL_TO_FORTRAN_BOX(bx),
                 BL_TO_FORTRAN_ANYD(phi_new[mfi]),
                 geom.CellSize(), geom.ProbLo(), geom.ProbHi());
    }

    // Set up BCRec; see Src/Base/AMReX_BC_TYPES.H for supported types
    Vector<BCRec> bc(phi_old.nComp());
    for (int n = 0; n < phi_old.nComp(); ++n)
    {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {

            // lo-side BCs
            if (bc_lo[idim] == BCType::int_dir) {
                bc[n].setLo(idim, BCType::int_dir);  // periodic uses "internal Dirichlet"
            }
            else if (bc_lo[idim] == BCType::foextrap) {
                bc[n].setLo(idim, BCType::foextrap); // first-order extrapolation
            }
            else if (bc_lo[idim] == BCType::ext_dir) {
                bc[n].setLo(idim, BCType::ext_dir);  // external Dirichlet
            }
            else {
                amrex::Abort("Invalid bc_lo");
            }

            // hi-side BCs
            if (bc_hi[idim] == BCType::int_dir) {
                bc[n].setHi(idim, BCType::int_dir);  // periodic uses "internal Dirichlet"
            }
            else if (bc_hi[idim] == BCType::foextrap) {
                bc[n].setHi(idim, BCType::foextrap); // first-order extrapolation (homogeneous Neumann)
            }
            else if (bc_hi[idim] == BCType::ext_dir) {
                bc[n].setHi(idim, BCType::ext_dir);  // external Dirichlet
            }
            else {
                amrex::Abort("Invalid bc_hi");
            }

        }
    }

    // Write a plotfile of the initial data if plot_int > 0 (plot_int was defined in the inputs file)
    if (plot_int > 0)
    {
        int n = 0;
        const std::string& pltfile = amrex::Concatenate("plt",n,5);
        WriteSingleLevelPlotfile(pltfile, phi_new, {"phi"}, geom, time, 0);
    }


    for (int n = 1; n <= nsteps; ++n)
    {
        MultiFab::Copy(phi_old, phi_new, 0, 0, 1, 0);

        // new_phi = (I-dt)^{-1} * old_phi + dt
        // magnon diffusion case has updated alpha and beta coeffs
        // (a * alpha * I - b del*beta del ) phi = RHS
        advance(phi_old, phi_new, dt, D_const, tau_p, geom, ba, dm, bc); 
        time = time + dt;

        // Tell the I/O Processor to write out which step we're doing
        amrex::Print() << "Advanced step " << n << "\n";

        // Write a plotfile of the current data (plot_int was defined in the inputs file)
        if (plot_int > 0 && n%plot_int == 0)
        {
            const std::string& pltfile = amrex::Concatenate("plt",n,5);
            WriteSingleLevelPlotfile(pltfile, phi_new, {"phi"}, geom, time, n);
        }
    }

    // Call the timer again and compute the maximum difference between the start time and stop time
    //   over all processors
    auto stop_time = amrex::second() - strt_time;
    const int IOProc = ParallelDescriptor::IOProcessorNumber();
    ParallelDescriptor::ReduceRealMax(stop_time,IOProc);

    // Tell the I/O Processor to write out the "run time"
    amrex::Print() << "Run time = " << stop_time << std::endl;
}
