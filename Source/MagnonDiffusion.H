#ifndef _magnondiffusion_H_
#define _magnondiffusion_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_LO_BCTYPES.H>

#include "Input/GeometryProperties/GeometryProperties_fwd.H"
#include "Input/BoundaryConditions/BoundaryConditions_fwd.H"
#include "Utils/SelectWarpXUtils/WarnManager_fwd.H"

#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_Geometry.H>

#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>           
             
#ifdef AMREX_USE_EB
#include <AMReX_MLEBABecLap.H>    
#endif    

#include "MagnonDiffusion_namespace.H"

using namespace MagnonDiffusion;
using namespace amrex;

enum class WarnPriority
{
    /** Low priority warning:
     * essentially an informative message
     */
    low,
    /** Medium priority warning:
     * a bug or a performance issue may affect the simulation
     */
    medium,
    /** High priority warning:
     * a very serious bug or performance issue
     * almost certainly affects the simulation
     */
    high
};


class c_MagnonDiffusion
{

public:



    c_MagnonDiffusion (); //constructor
    ~c_MagnonDiffusion (); //destructor


    /********************* FUNCTIONS/PARAMETERS TAKEN FROM WARPX ****************************/

    static c_MagnonDiffusion& GetInstance ();
    static void ResetInstance ();

    void RecordWarning(
        std::string topic,
        std::string text,
        WarnPriority priority = WarnPriority::medium);

    void PrintLocalWarnings(const std::string& when);

    void PrintGlobalWarnings(const std::string& when);

    std::unique_ptr<Utils::WarnManager> m_p_warn_manager; //Warning manager: used to record and print error msgs.

    bool m_always_warn_immediately = false; //flag to emit control emitting warning as soon as it is recorded

    void ReadData ();
    void InitData ();

    static bool do_device_synchronize;

    c_GeometryProperties& get_GeometryProperties () { return *m_pGeometryProperties;}
    c_BoundaryConditions& get_BoundaryConditions () { return *m_pBoundaryConditions;}
    const amrex::Real get_time() { return m_time_instant;}
    const amrex::Real set_time(int n) { m_time_instant = n*m_timestep; return m_time_instant;}


private:

    static c_MagnonDiffusion* m_instance;

    amrex::Real m_timestep;
    int m_total_steps;

    amrex::Real m_time_instant = 0;

    std::unique_ptr<c_GeometryProperties> m_pGeometryProperties;
    std::unique_ptr<c_BoundaryConditions> m_pBoundaryConditions;

};

void main_main (c_MagnonDiffusion& rMagnonDiffusion);

void InitializeMagnonDiffusionNamespace (const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
                                         const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_hi);

void advance (MultiFab& phi_old,
              MultiFab& phi_new,
              MultiFab& robin_hi_a,
              MultiFab& robin_hi_b,
              MultiFab& robin_hi_f,
              c_MagnonDiffusion& rMagnonDiffusion,
              const Geometry& geom);

void FillBoundaryPhysical (MultiFab& phi, const Geometry& geom);

void FillBoundaryRobin (MultiFab& robin_a,
                        MultiFab& robin_b,
                        MultiFab& robin_f,
                        MultiFab& robin_hi_a,
                        MultiFab& robin_hi_b,
                        MultiFab& robin_hi_f,
                        const Geometry& geom);

void Initialize_Robin_Coefs(c_MagnonDiffusion& rMagnonDiffusion, const Geometry& geom, 
                            MultiFab& robin_a,
                            MultiFab& robin_b,
                            MultiFab& robin_f);
#endif
