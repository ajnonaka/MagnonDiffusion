
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include "MagnonDiffusion.H"
#include "Input/BoundaryConditions/BoundaryConditions.H"
#include "Input/GeometryProperties/GeometryProperties.H"
#include "Utils/SelectWarpXUtils/WarpXUtil.H"
#include "Utils/SelectWarpXUtils/WarpXProfilerWrapper.H"
#include "Utils/eXstaticUtils/eXstaticUtil.H"
#include "Utils/FerroXUtils/FerroXUtil.H"

using namespace amrex;
using namespace MagnonDiffusion;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        c_MagnonDiffusion pMagnonDiffusion;
        pMagnonDiffusion.InitData();
        main_main(pMagnonDiffusion);
    }

    amrex::Finalize();
    return 0;
}

void main_main (c_MagnonDiffusion& rMagnonDiffusion)
{
    // What time is it now?  We'll use this to compute total run time.
    auto strt_time = amrex::second();
    
    auto& rGprop = rMagnonDiffusion.get_GeometryProperties();
    auto& geom = rGprop.geom;
    auto& ba = rGprop.ba;
    auto& dm = rGprop.dm;
    auto& is_periodic = rGprop.is_periodic;
    auto& prob_lo = rGprop.prob_lo;
    auto& prob_hi = rGprop.prob_hi;
    auto& n_cell = rGprop.n_cell;

    // read in inputs file
    InitializeMagnonDiffusionNamespace(prob_lo, prob_hi);

    // Nghost = number of ghost cells for each array
    int Nghost = 1;

    // Ncomp = number of components for each array
    int Ncomp  = 1;

    // time = starting time in the simulation
    Real time = 0.0;

    // we allocate two phi multifabs; one will store the old state, the other the new.
    MultiFab phi_old(ba, dm, Ncomp, Nghost);
    MultiFab phi_new(ba, dm, Ncomp, Nghost);

    //MultiFabs to parse robin BC coefficients
    MultiFab robin_hi_a(ba, dm, Ncomp, Nghost);
    MultiFab robin_hi_b(ba, dm, Ncomp, Nghost);
    MultiFab robin_hi_f(ba, dm, Ncomp, Nghost);
    robin_hi_a.setVal(0.);
    robin_hi_b.setVal(0.);
    robin_hi_f.setVal(0.);

    //MultiFabs to store position dependent D and tau parameters
    MultiFab tau_mf(ba, dm, 1, 1); 
    MultiFab D_mf(ba, dm, 1, 1); 
    tau_mf.setVal(tau_p); //initialize with comstants. will be filled using parser
    D_mf.setVal(D_const);
    
    Initialize_Robin_Coefs(rMagnonDiffusion, geom, robin_hi_a, robin_hi_b, robin_hi_f);
    initialize_mf_using_parser(rMagnonDiffusion, geom, tau_mf, D_mf); //read position dependent tau and and D using parser 

#ifdef AMREX_USE_EB
    MultiFab Plt(ba, dm, 5, 0,  MFInfo(), *rGprop.pEB->p_factory_union);
#else    
    MultiFab Plt(ba, dm, 5, 0);
#endif

    // Initialize phi_new here
    phi_new.setVal(0.);

    MultiFab::Copy(Plt, phi_new, 0, 0, 1, 0);
    MultiFab::Copy(Plt, robin_hi_a, 0, 1, 1, 0);
    MultiFab::Copy(Plt, robin_hi_b, 0, 2, 1, 0);
    MultiFab::Copy(Plt, robin_hi_f, 0, 3, 1, 0);
    MultiFab::Copy(Plt, D_mf, 0, 4, 1, 0);

    // Write a plotfile of the initial data if plot_int > 0 (plot_int was defined in the inputs file)
    if (plot_int > 0)
    {
        int n = 0;
        const std::string& pltfile = amrex::Concatenate("plt",n,5);
#ifdef AMREX_USE_EB
        EB_WriteSingleLevelPlotfile(pltfile, Plt, {"phi", "robin_a", "robin_b", "robin_f", "tau"}, geom, time, n);
#else    
        WriteSingleLevelPlotfile(pltfile, Plt, {"phi", "robin_a", "robin_b", "robin_f", "tau"}, geom, time, n);
#endif
    }

    D_mf.mult(dt, 0, 1, 1); //We have D_const(i,j,k) in D_mf. Multiply it by dt to set D_mf = dt*D_const

    fill_acoef(tau_mf); //modify tau to fill a coef. tau_mf(i,j,k) = 1 + dt/tau_mf(i,j,k)

    for (int n = 1; n <= nsteps; ++n)
    {
        MultiFab::Copy(phi_old, phi_new, 0, 0, 1, 0);

        // new_phi = (I-dt)^{-1} * old_phi + dt
        // magnon diffusion case has updated alpha and beta coeffs
        // (a * alpha * I - b del*beta del ) phi = RHS
        advance(phi_old, phi_new, robin_hi_a, robin_hi_b, robin_hi_f, tau_mf, D_mf, rMagnonDiffusion, geom);
        time = time + dt;

        // Tell the I/O Processor to write out which step we're doing
        amrex::Print() << "Advanced step " << n << "\n";

        // Write a plotfile of the current data (plot_int was defined in the inputs file)
        MultiFab::Copy(Plt, phi_new, 0, 0, 1, 0);
        MultiFab::Copy(Plt, D_mf, 0, 4, 1, 0);

        if (plot_int > 0 && n%plot_int == 0)
        {
            const std::string& pltfile = amrex::Concatenate("plt",n,5);
#ifdef AMREX_USE_EB
            EB_WriteSingleLevelPlotfile(pltfile, Plt, {"phi", "robin_a", "robin_b", "robin_f","tau"}, geom, time, n);
#else    
            WriteSingleLevelPlotfile(pltfile, Plt, {"phi", "robin_a", "robin_b", "robin_f", "tau"}, geom, time, n);
#endif
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
