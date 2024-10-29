
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
    MultiFab mu_old(ba, dm, Ncomp, Nghost);
    MultiFab mu_new(ba, dm, Ncomp, Nghost);

    MultiFab T_old(ba, dm, Ncomp, Nghost);
    MultiFab T_new(ba, dm, Ncomp, Nghost);

    //MultiFab to store currents. Only used for diagnostics
    Array<MultiFab, AMREX_SPACEDIM> J_mu;
    Array<MultiFab, AMREX_SPACEDIM> J_T;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        J_mu[dir].define(ba, dm, Ncomp, 1);
        J_T[dir].define(ba, dm, Ncomp, 1);
    }

    //MultiFabs to parse robin BC coefficients for mu
    MultiFab robin_hi_a(ba, dm, Ncomp, Nghost);
    MultiFab robin_hi_b(ba, dm, Ncomp, Nghost);
    MultiFab robin_hi_f(ba, dm, Ncomp, Nghost);
    robin_hi_a.setVal(0.);
    robin_hi_b.setVal(0.);
    robin_hi_f.setVal(0.);

    //MultiFabs to parse robin BC coefficients for T
    MultiFab T_robin_hi_a(ba, dm, Ncomp, Nghost);
    MultiFab T_robin_hi_b(ba, dm, Ncomp, Nghost);
    MultiFab T_robin_hi_f(ba, dm, Ncomp, Nghost);
    T_robin_hi_a.setVal(0.);
    T_robin_hi_b.setVal(0.);
    T_robin_hi_f.setVal(0.);

    //MultiFabs to store position dependent D and tau parameters
    MultiFab sigma(ba, dm, 1, 1); 
    MultiFab lambda(ba, dm, 1, 1); 
    MultiFab kappa(ba, dm, 1, 1); 
    MultiFab Cv_mf(ba, dm, 1, 1); 
    sigma.setVal(0.); //initialize with zero. will be filled using parser
    lambda.setVal(0.);
    kappa.setVal(0.);
    Cv_mf.setVal(2.6e6);
   
    MultiFab acoef_mf(ba, dm, 1, 1);
    MultiFab bcoef_mf(ba, dm, 1, 1);
    MultiFab T_bcoef_mf(ba, dm, 1, 1);
    acoef_mf.setVal(0.);
    bcoef_mf.setVal(0.);
    T_bcoef_mf.setVal(0.);
    
    // Set up RHS
    MultiFab Jc_mf(ba, dm, 1, 1); 
    Jc_mf.setVal(0.);

    Initialize_Robin_Coefs(rMagnonDiffusion, geom, robin_hi_a, robin_hi_b, robin_hi_f);
    Initialize_Robin_Coefs_T(rMagnonDiffusion, geom, T_robin_hi_a, T_robin_hi_b, T_robin_hi_f);
    initialize_sigma_using_parser(rMagnonDiffusion, geom, sigma); //read position dependent sigma, lambda, and kappa using parser 
    initialize_lambda_using_parser(rMagnonDiffusion, geom, lambda); //read position dependent sigma, lambda, and kappa using parser 
    initialize_kappa_using_parser(rMagnonDiffusion, geom, kappa); //read position dependent sigma, lambda, and kappa using parser 
    initialize_jc_using_parser(rMagnonDiffusion, geom, Jc_mf); //read position dependent sigma, lambda, and kappa using parser 
    initialize_Cv_using_parser(rMagnonDiffusion, geom, Cv_mf); //read position dependent sigma, lambda, and kappa using parser 

#ifdef AMREX_USE_EB
    MultiFab Plt(ba, dm, 14, 0,  MFInfo(), *rGprop.pEB->p_factory_union);
#else    
    MultiFab Plt(ba, dm, 14, 0);
#endif

    // Initialize mu_new here
    mu_new.setVal(0.);
    T_new.setVal(298.);

    MultiFab::Copy(Plt, mu_new, 0, 0, 1, 0);
    MultiFab::Copy(Plt, T_new, 0, 1, 1, 0);
    MultiFab::Copy(Plt, T_robin_hi_a, 0, 2, 1, 0);
    MultiFab::Copy(Plt, T_robin_hi_b, 0, 3, 1, 0);
    MultiFab::Copy(Plt, T_robin_hi_f, 0, 4, 1, 0);
    MultiFab::Copy(Plt, sigma, 0, 5, 1, 0);
    MultiFab::Copy(Plt, Jc_mf, 0, 6, 1, 0);
    MultiFab::Copy(Plt, kappa, 0, 7, 1, 0);

    // Write a plotfile of the initial data if plot_int > 0 (plot_int was defined in the inputs file)
    if (plot_int > 0)
    {
        int n = 0;
        const std::string& pltfile = amrex::Concatenate("plt",n,5);
#ifdef AMREX_USE_EB
            EB_WriteSingleLevelPlotfile(pltfile, Plt, {"mu", "T", "robin_a", "robin_b", "robin_f","sigma", "Jc", "kappa", "J_mux", "J_muy", "J_muz", "J_Tx", "J_Ty", "J_Tz"}, geom, time, n);
#else    
            WriteSingleLevelPlotfile(pltfile, Plt, {"mu", "T", "robin_a", "robin_b", "robin_f", "sigma", "Jc", "kappa", "J_mux", "J_muy", "J_muz", "J_Tx"    , "J_Ty", "J_Tz"}, geom, time, n);
#endif
    }

    fill_acoef(acoef_mf, sigma, lambda); //fill a coef acoef_mf(i,j,k) = 1 + sigma(i,j,k)*dt/lambda(i,j,k)^2
    fill_bcoef(bcoef_mf, sigma); //bcoef_mf(i,j,k) = dt*sigma(i,j,k)
    fill_bcoef_T(T_bcoef_mf, kappa, Cv_mf); //T_bcoef_mf(i,j,k) = dt*kappa(i,j,k)/Cv(i,j,k)
    
    for (int n = 1; n <= nsteps; ++n)
    {
        MultiFab::Copy(mu_old, mu_new, 0, 0, 1, 0);
        MultiFab::Copy(T_old, T_new, 0, 0, 1, 1);

        // new_mu = (I-dt)^{-1} * old_mu + dt
        // magnon diffusion case has updated alpha and beta coeffs
        // (a * alpha * I - b del*beta del ) mu = RHS
        advance_T(T_old, T_new, mu_new, J_T, T_robin_hi_a, T_robin_hi_b, T_robin_hi_f, T_bcoef_mf, sigma, kappa, Jc_mf, Cv_mf, rMagnonDiffusion, geom);
        advance_mu(mu_old, mu_new, T_new, J_mu, robin_hi_a, robin_hi_b, robin_hi_f, sigma, acoef_mf, bcoef_mf, rMagnonDiffusion, geom);
        
        //Now J_mu and J_T contain grad(mu) and grad(T) respectively. modify in place to get J_mu = -sigma*grad(mu) - sigma*S*grad(T)
        //J_mu = J_m,diff + j_m,q: 2016 VanBees PRB 94, 174437 (2016)
        //J_T = -kappa*grad(T)
        compute_currents(J_mu, J_T, sigma, kappa); 
        time = time + dt;

        // Tell the I/O Processor to write out which step we're doing
        amrex::Print() << "Advanced step " << n << "\n";

        if (plot_int > 0 && n%plot_int == 0)
        {

            // Write a plotfile of the current data (plot_int was defined in the inputs file)
            MultiFab::Copy(Plt, mu_new, 0, 0, 1, 0);
            MultiFab::Copy(Plt, T_new, 0, 1, 1, 0);
            MultiFab::Copy(Plt, J_mu[0], 0, 8, 1, 0);
            MultiFab::Copy(Plt, J_mu[1], 0, 9, 1, 0);
            MultiFab::Copy(Plt, J_mu[2], 0, 10, 1, 0);
            MultiFab::Copy(Plt, J_T[0], 0, 11, 1, 0);
            MultiFab::Copy(Plt, J_T[1], 0, 12, 1, 0);
            MultiFab::Copy(Plt, J_T[2], 0, 13, 1, 0);

            const std::string& pltfile = amrex::Concatenate("plt",n,5);
#ifdef AMREX_USE_EB
            EB_WriteSingleLevelPlotfile(pltfile, Plt, {"mu", "T", "robin_a", "robin_b", "robin_f","sigma", "Jc", "kappa", "J_mux", "J_muy", "J_muz", "J_Tx", "J_Ty", "J_Tz"}, geom, time, n);
#else    
            WriteSingleLevelPlotfile(pltfile, Plt, {"mu", "T", "robin_a", "robin_b", "robin_f", "sigma", "Jc", "kappa", "J_mux", "J_muy", "J_muz", "J_Tx"    , "J_Ty", "J_Tz"}, geom, time, n);
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
