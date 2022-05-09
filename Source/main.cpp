#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H> 
#include <AMReX_MultiFab.H> 
#include <AMReX_VisMF.H>

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
    Real total_step_strt_time = ParallelDescriptor::second();

    // size of each box
    int max_grid_size;

    // total steps in the simulation
    int nsteps;

    // Default plot_int to -1, allow us to set it to something else in the inputs file
    //  If plot_int < 0 then no plot files will be writtenq
    int plot_int;

    // time step
    Real dt; 

    amrex::GpuArray<amrex::Real, 3> prob_lo; // physical lo coordinate
    amrex::GpuArray<amrex::Real, 3> prob_hi; // physical hi coordinate

    int phi_bc_flag_hi;
    int phi_bc_flag_lo;

    // Robin BC parameters
    Real robin_bc_lo_a;
    Real robin_bc_hi_a;
    Real robin_bc_lo_b;
    Real robin_bc_hi_b;
    Real robin_bc_lo_f;
    Real robin_bc_hi_f;

    int TimeIntegratorOrder;

    // magnon diffusion parameters
    Real D_const, tau_p;

    // inputs parameters
    {
        // ParmParse is way of reading inputs from the inputs file
        ParmParse pp;

        // We need to get n_cell from the inputs file - this is the number of cells on each side of
        amrex::Vector<int> temp_int(AMREX_SPACEDIM);
        if (pp.queryarr("n_cell",temp_int)) {
            for (int i=0; i<AMREX_SPACEDIM; ++i) {
                n_cell[i] = temp_int[i];
            }
        }

        // The domain is broken into boxes of size max_grid_size
        pp.get("max_grid_size",max_grid_size);

        pp.get("phi_bc_flag_hi",phi_bc_flag_hi); // 0 : P = 0, 1 : dp/dz = p/lambda, 2 : dp/dz = 0
        pp.get("phi_bc_flag_lo",phi_bc_flag_hi); // 0 : P = 0, 1 : dp/dz = p/lambda, 2 : dp/dz = 0
        pp.get("robin_bc_lo_a",robin_bc_lo_a);
        pp.get("robin_bc_hi_a",robin_bc_hi_a);
        pp.get("robin_bc_lo_b",robin_bc_lo_b);
        pp.get("robin_bc_hi_b",robin_bc_hi_b);
        pp.get("robin_bc_lo_f",robin_bc_lo_f);
        pp.get("robin_bc_hi_f",robin_bc_hi_f);

	    pp.get("TimeIntegratorOrder",TimeIntegratorOrder);

        // Material Properties
        pp.get("D_const",D_const); // diffusion constant
        pp.get("tau_p",  tau_p); // relaxation time constant

        // Default plot_int to -1, allow us to set it to something else in the inputs file
        //  If plot_int < 0 then no plot files will be writtenq
        plot_int = -1;
        pp.query("plot_int",plot_int);

        // Default nsteps to 0, allow us to set it to something else in the inputs file
        nsteps = 10;
        pp.query("nsteps",nsteps);

        // time step
        pp.get("dt",dt);

        amrex::Vector<amrex::Real> temp(AMREX_SPACEDIM);
        if (pp.queryarr("prob_lo",temp)) {
            for (int i=0; i<AMREX_SPACEDIM; ++i) {
                prob_lo[i] = temp[i];
            }
        }
        if (pp.queryarr("prob_hi",temp)) {
            for (int i=0; i<AMREX_SPACEDIM; ++i) {
                prob_hi[i] = temp[i];
            }
        }
    }

    // make BoxArray and Geometry
    BoxArray ba;
    Geometry geom;
    {
        IntVect dom_lo(AMREX_D_DECL(       0,        0,        0));
        IntVect dom_hi(AMREX_D_DECL(n_cell-1, n_cell-1, n_cell-1));
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

    // Compute the time step
    // Implicit time step is imFactor*(explicit time step)
    const Real* dx = geom.CellSize();

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
        advance(phi_old, phi_new, dt, geom, ba, dm, bc);
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
