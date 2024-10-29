#include "MagnonDiffusion.H"
#include "Utils/eXstaticUtils/eXstaticUtil.H"
#include "Utils/SelectWarpXUtils/MsgLogger/MsgLogger.H"
#include "Utils/SelectWarpXUtils/WarnManager.H"
#include "Utils/SelectWarpXUtils/WarpXUtil.H"
#include "Utils/SelectWarpXUtils/WarpXProfilerWrapper.H"
#include "../../Utils/SelectWarpXUtils/WarpXUtil.H"
     
#include "Input/GeometryProperties/GeometryProperties.H"
#include "Input/BoundaryConditions/BoundaryConditions.H"

#include <AMReX_ParmParse.H>

c_MagnonDiffusion* c_MagnonDiffusion::m_instance = nullptr;
#ifdef AMREX_USE_GPU
bool c_MagnonDiffusion::do_device_synchronize = true;
#else
bool c_MagnonDiffusion::do_device_synchronize = false;
#endif


c_MagnonDiffusion& c_MagnonDiffusion::GetInstance() 
{    
     
    if (!m_instance) {
        m_instance = new c_MagnonDiffusion();
    }
    return *m_instance;
     
}    
     
     
void 
c_MagnonDiffusion::ResetInstance ()
{    
    delete m_instance;
    m_instance = nullptr;
}    
     

c_MagnonDiffusion::c_MagnonDiffusion ()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t{************************c_MagnonDiffusion Constructor()************************\n";
#endif
    m_instance = this;
    m_p_warn_manager = std::make_unique<Utils::WarnManager>();

    ReadData();

#ifdef PRINT_NAME
    amrex::Print() << "\t}************************c_MagnonDiffusion Constructor()************************\n";
#endif
}

c_MagnonDiffusion::~c_MagnonDiffusion ()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t{************************c_MagnonDiffusion Destructor()************************\n";
#endif

#ifdef PRINT_NAME
    amrex::Print() << "\t}************************c_MagnonDiffusion Destructor()************************\n";
#endif
}

void
c_MagnonDiffusion::RecordWarning(
        std::string topic,
        std::string text,
        WarnPriority priority)
{
    WARPX_PROFILE("WarpX::RecordWarning");

    auto msg_priority = Utils::MsgLogger::Priority::high;
    if(priority == WarnPriority::low)
        msg_priority = Utils::MsgLogger::Priority::low;
    else if(priority == WarnPriority::medium)
        msg_priority = Utils::MsgLogger::Priority::medium;

    if(m_always_warn_immediately){
        amrex::Warning(
            "!!!!!! WARNING: ["
            + std::string(Utils::MsgLogger::PriorityToString(msg_priority))
            + "][" + topic + "] " + text);
    }

#ifdef AMREX_USE_OMP
    #pragma omp critical
#endif
    {
        m_p_warn_manager->record_warning(topic, text, msg_priority);
    }
}

void
c_MagnonDiffusion::PrintLocalWarnings(const std::string& when)
{

    WARPX_PROFILE("WarpX::PrintLocalWarnings");
    const std::string warn_string = m_p_warn_manager->print_local_warnings(when);
    amrex::AllPrint() << warn_string;

}


void
c_MagnonDiffusion::PrintGlobalWarnings(const std::string& when)
{

    WARPX_PROFILE("WarpX::PrintGlobalWarnings");
    const std::string warn_string = m_p_warn_manager->print_global_warnings(when);
    amrex::Print() << warn_string;

}

void
c_MagnonDiffusion::ReadData ()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t{************************c_MagnonDiffusion::ReadData()************************\n";
    amrex::Print() << "\t\tin file: " << __FILE__ << " at line: " << __LINE__ << "\n";
#endif

    m_timestep = 0;
    m_total_steps = 1;
    amrex::ParmParse pp;

    #ifdef TIME_DEPENDENT
        queryWithParser(pp,"timestep", m_timestep);
        queryWithParser(pp,"steps", m_total_steps);
    #endif

    m_pGeometryProperties = std::make_unique<c_GeometryProperties>();

    m_pBoundaryConditions = std::make_unique<c_BoundaryConditions>();

#ifdef PRINT_NAME
    amrex::Print() << "\t\t}************************c_MagnonDiffusion::ReadData()************************\n";
#endif
}


void
c_MagnonDiffusion::InitData ()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t{************************c_MagnonDiffusion::InitData()************************\n";
    amrex::Print() << "\tin file: " << __FILE__ << " at line: " << __LINE__ << "\n";
#endif

    m_pGeometryProperties->InitData();

#ifdef PRINT_NAME
    amrex::Print() << "\t}************************c_MagnonDiffusion::InitData()************************\n";
#endif
}


AMREX_GPU_MANAGED int MagnonDiffusion::nsteps;
AMREX_GPU_MANAGED int MagnonDiffusion::plot_int;
    
// time step
AMREX_GPU_MANAGED amrex::Real MagnonDiffusion::dt;

AMREX_GPU_MANAGED amrex::Vector<int> MagnonDiffusion::bc_lo;
AMREX_GPU_MANAGED amrex::Vector<int> MagnonDiffusion::bc_hi;

AMREX_GPU_MANAGED amrex::Vector<int> MagnonDiffusion::T_bc_lo;
AMREX_GPU_MANAGED amrex::Vector<int> MagnonDiffusion::T_bc_hi;

// General BC parameters for Dirichlet, Neumann, or Robin
// Dirichlet: phi = f
// Neumann: d(phi)/dn = f
// Robin: a*phi+b*d(phi)/dn = f
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> MagnonDiffusion::bc_lo_f; // BC rhs.
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> MagnonDiffusion::bc_hi_f; // BC rhs.
    
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> MagnonDiffusion::T_bc_lo_f; // BC rhs.
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> MagnonDiffusion::T_bc_hi_f; // BC rhs.
    
// Robin BC parameters
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> MagnonDiffusion::bc_lo_a; // robin BC coeffs. 
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> MagnonDiffusion::bc_hi_a; // robin BC coeffs. 
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> MagnonDiffusion::bc_lo_b; // robin BC coeffs. 
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> MagnonDiffusion::bc_hi_b; // robin BC coeffs. 

AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> MagnonDiffusion::T_bc_lo_a; // robin BC coeffs. 
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> MagnonDiffusion::T_bc_hi_a; // robin BC coeffs. 
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> MagnonDiffusion::T_bc_lo_b; // robin BC coeffs. 
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> MagnonDiffusion::T_bc_hi_b; // robin BC coeffs. 

// magnon diffusion parameters
AMREX_GPU_MANAGED amrex::Real MagnonDiffusion::J_c;
AMREX_GPU_MANAGED amrex::Real MagnonDiffusion::sigma_Pt;
AMREX_GPU_MANAGED amrex::Real MagnonDiffusion::S_s;

void InitializeMagnonDiffusionNamespace(const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
                                        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_hi) {
    
    // ParmParse is way of reading inputs from the inputs file
    amrex::ParmParse pp;

    amrex::Vector<int> temp_int(AMREX_SPACEDIM);     // temporary for parsing
    amrex::Vector<amrex::Real> temp(AMREX_SPACEDIM); // temporary for parsing

    // parse in material properties
    pp.get("J_c", J_c);
    pp.get("sigma_Pt", sigma_Pt);
    pp.get("S_s", S_s);

    // Default plot_int to -1, allow us to set it to something else in the inputs file
    //  If plot_int < 0 then no plot files will be writtenq
    plot_int = -1;
    pp.query("plot_int",plot_int);

    // nsteps must be specified in the inputs file
    pp.get("nsteps",nsteps);

    // time step
    pp.get("dt",dt);

    // default values for bc parameters
    for (int i=0; i<AMREX_SPACEDIM; ++i) {
        bc_lo_f[i] = 0.;
        bc_hi_f[i] = 0.;
        bc_lo_a[i] = 0.;
        bc_hi_a[i] = 0.;
        bc_lo_b[i] = 0.;
        bc_hi_b[i] = 0.;
        T_bc_lo_f[i] = 0.;
        T_bc_hi_f[i] = 0.;
        T_bc_lo_a[i] = 0.;
        T_bc_hi_a[i] = 0.;
        T_bc_lo_b[i] = 0.;
        T_bc_hi_b[i] = 0.;
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

    // read in temperature bc parameter
    if (pp.queryarr("T_bc_lo_f",temp)) {
        for (int i=0; i<AMREX_SPACEDIM; ++i) {
            T_bc_lo_f[i] = temp[i];
        }
    }
    if (pp.queryarr("T_bc_hi_f",temp)) {
        for (int i=0; i<AMREX_SPACEDIM; ++i) {
            T_bc_hi_f[i] = temp[i];
        }
    }
    if (pp.queryarr("T_bc_lo_a",temp)) {
        for (int i=0; i<AMREX_SPACEDIM; ++i) {
            T_bc_lo_a[i] = temp[i];
        }
    }
    if (pp.queryarr("T_bc_hi_a",temp)) {
        for (int i=0; i<AMREX_SPACEDIM; ++i) {
            T_bc_hi_a[i] = temp[i];
        }
    }
    if (pp.queryarr("T_bc_lo_b",temp)) {
        for (int i=0; i<AMREX_SPACEDIM; ++i) {
            T_bc_lo_b[i] = temp[i];
        }
    }
    if (pp.queryarr("T_bc_hi_b",temp)) {
        for (int i=0; i<AMREX_SPACEDIM; ++i) {
            T_bc_hi_b[i] = temp[i];
        }
    }

    // read in BC; see Src/Base/AMReX_BC_TYPES.H for supported types
    bc_lo.resize(AMREX_SPACEDIM);
    bc_hi.resize(AMREX_SPACEDIM);
    pp.getarr("bc_lo", bc_lo);
    pp.getarr("bc_hi", bc_hi);

    T_bc_lo.resize(AMREX_SPACEDIM);
    T_bc_hi.resize(AMREX_SPACEDIM);
    pp.getarr("T_bc_lo", T_bc_lo);
    pp.getarr("T_bc_hi", T_bc_hi);
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

void FillBoundaryPhysical_T (MultiFab& phi, const Geometry& geom) {

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
            if (T_bc_lo[0] == BCType::foextrap || T_bc_lo[0] == BCType::ext_dir) {
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (i<lo) {
                        data(i,j,k) = T_bc_lo_f[0];
                    }
                });
            }
        }

        if (bx.bigEnd(0) > hi) {
            if (T_bc_hi[0] == BCType::foextrap || T_bc_hi[0] == BCType::ext_dir) {
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (i>hi) {
                        data(i,j,k) = T_bc_hi_f[0];
                    }
                });
            }
        }
        
        // y ghost cells
        lo = dom.smallEnd(1);
        hi = dom.bigEnd(1);
        
        if (bx.smallEnd(1) < lo) {
            if (T_bc_lo[1] == BCType::foextrap || T_bc_lo[1] == BCType::ext_dir) {
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (j<lo) {
                        data(i,j,k) = T_bc_lo_f[1];
                    }
                });
            }
        }

        if (bx.bigEnd(1) > hi) {
            if (T_bc_hi[1] == BCType::foextrap || T_bc_hi[1] == BCType::ext_dir) {
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (j>hi) {
                        data(i,j,k) = T_bc_hi_f[1];
                    }
                });
            }
        }
        
        // z ghost cells
        lo = dom.smallEnd(2);
        hi = dom.bigEnd(2);
        
        if (bx.smallEnd(2) < lo) {
            if (T_bc_lo[2] == BCType::foextrap || T_bc_lo[2] == BCType::ext_dir) {
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (k<lo) {
                        data(i,j,k) = T_bc_lo_f[2];
                    }
                });
            }
        }

        if (bx.bigEnd(2) > hi) {
            if (T_bc_hi[2] == BCType::foextrap || T_bc_hi[2] == BCType::ext_dir) {
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (k>hi) {
                        data(i,j,k) = T_bc_hi_f[2];
                    }
                });
            }
        }
        
    } // end MFIter
}

void FillBoundaryRobin (MultiFab& robin_a,
                        MultiFab& robin_b,
                        MultiFab& robin_f,
                        MultiFab& robin_hi_a,                       
                        MultiFab& robin_hi_b,
                        MultiFab& robin_hi_f,
                        const Geometry& geom) {

    // Physical Domain
    Box dom(geom.Domain());
    
    for (MFIter mfi(robin_a, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        // one ghost cell
        Box bx = mfi.growntilebox(1);

        const Array4<Real>& data_a = robin_a.array(mfi);
        const Array4<Real>& data_b = robin_b.array(mfi);
        const Array4<Real>& data_f = robin_f.array(mfi);

        const Array4<Real>& robin_bc_hi_a = robin_hi_a.array(mfi);
        const Array4<Real>& robin_bc_hi_b = robin_hi_b.array(mfi);
        const Array4<Real>& robin_bc_hi_f = robin_hi_f.array(mfi);

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
                        data_a(i,j,k) = robin_bc_hi_a(i,j,k);
                        data_b(i,j,k) = robin_bc_hi_b(i,j,k);
                        data_f(i,j,k) = robin_bc_hi_f(i,j,k);
                    }
                });
            }
        }
        
    } // end MFIter

}

void Initialize_Robin_Coefs(c_MagnonDiffusion& rMagnonDiffusion, const Geometry& geom, MultiFab& robin_a, MultiFab& robin_b, MultiFab& robin_f)
{ 
    auto& rGprop = rMagnonDiffusion.get_GeometryProperties();
    Box const& domain = rGprop.geom.Domain();

    const auto dx = rGprop.geom.CellSizeArray();
    const auto& real_box = rGprop.geom.ProbDomain();
    const auto iv_robin_a = robin_a.ixType().toIntVect();
    const auto iv_robin_b = robin_b.ixType().toIntVect();
    const auto iv_robin_f = robin_f.ixType().toIntVect();

    for (MFIter mfi(robin_a, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const auto& robin_a_arr = robin_a.array(mfi);
        const auto& robin_b_arr = robin_b.array(mfi);
        const auto& robin_f_arr = robin_f.array(mfi);
       
        // one ghost cell 
        Box bx = mfi.growntilebox(1);

	std::string robin_a_s;
	std::unique_ptr<amrex::Parser> robin_a_parser;
        std::string m_str_robin_a_function;

	std::string robin_b_s;
	std::unique_ptr<amrex::Parser> robin_b_parser;
        std::string m_str_robin_b_function;

	std::string robin_f_s;
	std::unique_ptr<amrex::Parser> robin_f_parser;
        std::string m_str_robin_f_function;

	ParmParse pp_robin_a("robin_a");


	if (pp_robin_a.query("robin_a_function(x,y,z)", m_str_robin_a_function) ) {
            robin_a_s = "parse_robin_a_function";
        }

        if (robin_a_s == "parse_robin_a_function") {
            Store_parserString(pp_robin_a, "robin_a_function(x,y,z)", m_str_robin_a_function);
            robin_a_parser = std::make_unique<amrex::Parser>(
                                     makeParser(m_str_robin_a_function,{"x","y","z"}));
        }

	ParmParse pp_robin_b("robin_b");


	if (pp_robin_b.query("robin_b_function(x,y,z)", m_str_robin_b_function) ) {
            robin_b_s = "parse_robin_b_function";
        }

        if (robin_b_s == "parse_robin_b_function") {
            Store_parserString(pp_robin_b, "robin_b_function(x,y,z)", m_str_robin_b_function);
            robin_b_parser = std::make_unique<amrex::Parser>(
                                     makeParser(m_str_robin_b_function,{"x","y","z"}));
        }

	ParmParse pp_robin_f("robin_f");


	if (pp_robin_f.query("robin_f_function(x,y,z)", m_str_robin_f_function) ) {
            robin_f_s = "parse_robin_f_function";
        }

        if (robin_f_s == "parse_robin_f_function") {
            Store_parserString(pp_robin_f, "robin_f_function(x,y,z)", m_str_robin_f_function);
            robin_f_parser = std::make_unique<amrex::Parser>(
                                     makeParser(m_str_robin_f_function,{"x","y","z"}));
        }

        const auto& macro_parser_robin_a = robin_a_parser->compile<3>();
        const auto& macro_parser_robin_b = robin_b_parser->compile<3>();
        const auto& macro_parser_robin_f = robin_f_parser->compile<3>();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_robin_a,macro_parser_robin_a,robin_a_arr);
            eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_robin_b, macro_parser_robin_b, robin_b_arr );
            eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_robin_f,macro_parser_robin_f,robin_f_arr);
        });

    }
	robin_a.FillBoundary(geom.periodicity());
	robin_b.FillBoundary(geom.periodicity());
	robin_f.FillBoundary(geom.periodicity());
}

void Initialize_Robin_Coefs_T(c_MagnonDiffusion& rMagnonDiffusion, const Geometry& geom, MultiFab& robin_a, MultiFab& robin_b, MultiFab& robin_f)
{ 
    auto& rGprop = rMagnonDiffusion.get_GeometryProperties();
    Box const& domain = rGprop.geom.Domain();

    const auto dx = rGprop.geom.CellSizeArray();
    const auto& real_box = rGprop.geom.ProbDomain();
    const auto iv_robin_a = robin_a.ixType().toIntVect();
    const auto iv_robin_b = robin_b.ixType().toIntVect();
    const auto iv_robin_f = robin_f.ixType().toIntVect();

    for (MFIter mfi(robin_a, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const auto& robin_a_arr = robin_a.array(mfi);
        const auto& robin_b_arr = robin_b.array(mfi);
        const auto& robin_f_arr = robin_f.array(mfi);
       
        // one ghost cell 
        Box bx = mfi.growntilebox(1);

	std::string robin_a_s;
	std::unique_ptr<amrex::Parser> robin_a_parser;
        std::string m_str_robin_a_function;

	std::string robin_b_s;
	std::unique_ptr<amrex::Parser> robin_b_parser;
        std::string m_str_robin_b_function;

	std::string robin_f_s;
	std::unique_ptr<amrex::Parser> robin_f_parser;
        std::string m_str_robin_f_function;

	ParmParse pp_robin_a("T_robin_a");


	if (pp_robin_a.query("T_robin_a_function(x,y,z)", m_str_robin_a_function) ) {
            robin_a_s = "parse_robin_a_function";
        }

        if (robin_a_s == "parse_robin_a_function") {
            Store_parserString(pp_robin_a, "T_robin_a_function(x,y,z)", m_str_robin_a_function);
            robin_a_parser = std::make_unique<amrex::Parser>(
                                     makeParser(m_str_robin_a_function,{"x","y","z"}));
        }

	ParmParse pp_robin_b("T_robin_b");


	if (pp_robin_b.query("T_robin_b_function(x,y,z)", m_str_robin_b_function) ) {
            robin_b_s = "parse_robin_b_function";
        }

        if (robin_b_s == "parse_robin_b_function") {
            Store_parserString(pp_robin_b, "T_robin_b_function(x,y,z)", m_str_robin_b_function);
            robin_b_parser = std::make_unique<amrex::Parser>(
                                     makeParser(m_str_robin_b_function,{"x","y","z"}));
        }

	ParmParse pp_robin_f("T_robin_f");


	if (pp_robin_f.query("T_robin_f_function(x,y,z)", m_str_robin_f_function) ) {
            robin_f_s = "parse_robin_f_function";
        }

        if (robin_f_s == "parse_robin_f_function") {
            Store_parserString(pp_robin_f, "T_robin_f_function(x,y,z)", m_str_robin_f_function);
            robin_f_parser = std::make_unique<amrex::Parser>(
                                     makeParser(m_str_robin_f_function,{"x","y","z"}));
        }

        const auto& macro_parser_robin_a = robin_a_parser->compile<3>();
        const auto& macro_parser_robin_b = robin_b_parser->compile<3>();
        const auto& macro_parser_robin_f = robin_f_parser->compile<3>();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_robin_a,macro_parser_robin_a,robin_a_arr);
            eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_robin_b, macro_parser_robin_b, robin_b_arr );
            eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_robin_f,macro_parser_robin_f,robin_f_arr);
        });

    }
	robin_a.FillBoundary(geom.periodicity());
	robin_b.FillBoundary(geom.periodicity());
	robin_f.FillBoundary(geom.periodicity());
}

void initialize_sigma_using_parser(c_MagnonDiffusion& rMagnonDiffusion, const Geometry& geom, MultiFab& sigma)
{ 
    auto& rGprop = rMagnonDiffusion.get_GeometryProperties();
    Box const& domain = rGprop.geom.Domain();

    const auto dx = rGprop.geom.CellSizeArray();
    const auto& real_box = rGprop.geom.ProbDomain();
    const auto iv_sigma = sigma.ixType().toIntVect();

    for (MFIter mfi(sigma, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const auto& sigma_arr = sigma.array(mfi);
       
        // one ghost cell 
	//Box bx = mfi.validbox();
        const Box& bx = mfi.growntilebox(1);

	std::string sigma_s;
	std::unique_ptr<amrex::Parser> sigma_parser;
        std::string m_str_sigma_function;

	ParmParse pp_sigma("mf_sigma");

	if (pp_sigma.query("sigma_function(x,y,z)", m_str_sigma_function) ) {
            sigma_s = "parse_sigma_function";
        }

        if (sigma_s == "parse_sigma_function") {
            Store_parserString(pp_sigma, "sigma_function(x,y,z)", m_str_sigma_function);
            sigma_parser = std::make_unique<amrex::Parser>(
                                     makeParser(m_str_sigma_function,{"x","y","z"}));
        }

        const auto& macro_parser_sigma = sigma_parser->compile<3>();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_sigma,macro_parser_sigma,sigma_arr);
        });

    }
	sigma.FillBoundary(geom.periodicity());
}

void initialize_lambda_using_parser(c_MagnonDiffusion& rMagnonDiffusion, const Geometry& geom, MultiFab& lambda)
{ 
    auto& rGprop = rMagnonDiffusion.get_GeometryProperties();
    Box const& domain = rGprop.geom.Domain();

    const auto dx = rGprop.geom.CellSizeArray();
    const auto& real_box = rGprop.geom.ProbDomain();
    const auto iv_lambda = lambda.ixType().toIntVect();

    for (MFIter mfi(lambda, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const auto& lambda_arr = lambda.array(mfi);
       
        // one ghost cell 
        //Box bx = mfi.validbox();
        const Box& bx = mfi.growntilebox(1);

	std::string lambda_s;
	std::unique_ptr<amrex::Parser> lambda_parser;
        std::string m_str_lambda_function;
	
        ParmParse pp_lambda("mf_lambda");

	if (pp_lambda.query("lambda_function(x,y,z)", m_str_lambda_function) ) {
            lambda_s = "parse_lambda_function";
        }

        if (lambda_s == "parse_lambda_function") {
            Store_parserString(pp_lambda, "lambda_function(x,y,z)", m_str_lambda_function);
            lambda_parser = std::make_unique<amrex::Parser>(
                                     makeParser(m_str_lambda_function,{"x","y","z"}));
        }

        const auto& macro_parser_lambda = lambda_parser->compile<3>();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_lambda,macro_parser_lambda,lambda_arr);
        });

    }
	lambda.FillBoundary(geom.periodicity());
}

void initialize_kappa_using_parser(c_MagnonDiffusion& rMagnonDiffusion, const Geometry& geom, MultiFab& kappa)
{ 
    auto& rGprop = rMagnonDiffusion.get_GeometryProperties();
    Box const& domain = rGprop.geom.Domain();

    const auto dx = rGprop.geom.CellSizeArray();
    const auto& real_box = rGprop.geom.ProbDomain();
    const auto iv_kappa = kappa.ixType().toIntVect();

    for (MFIter mfi(kappa, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const auto& kappa_arr = kappa.array(mfi);
       
        // one ghost cell 
        //Box bx = mfi.validbox();
        const Box& bx = mfi.growntilebox(1);

	std::string kappa_s;
	std::unique_ptr<amrex::Parser> kappa_parser;
        std::string m_str_kappa_function;
	
        ParmParse pp_kappa("mf_kappa");

	if (pp_kappa.query("kappa_function(x,y,z)", m_str_kappa_function) ) {
            kappa_s = "parse_kappa_function";
        }

        if (kappa_s == "parse_kappa_function") {
            Store_parserString(pp_kappa, "kappa_function(x,y,z)", m_str_kappa_function);
            kappa_parser = std::make_unique<amrex::Parser>(
                                     makeParser(m_str_kappa_function,{"x","y","z"}));
        }

        const auto& macro_parser_kappa = kappa_parser->compile<3>();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_kappa,macro_parser_kappa,kappa_arr);
        });

    }
	kappa.FillBoundary(geom.periodicity());
}


void initialize_jc_using_parser(c_MagnonDiffusion& rMagnonDiffusion, const Geometry& geom, MultiFab& Jc_mf)
{ 
    auto& rGprop = rMagnonDiffusion.get_GeometryProperties();
    Box const& domain = rGprop.geom.Domain();

    const auto dx = rGprop.geom.CellSizeArray();
    const auto& real_box = rGprop.geom.ProbDomain();
    const auto iv_jc = Jc_mf.ixType().toIntVect();

    for (MFIter mfi(Jc_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const auto& jc_arr = Jc_mf.array(mfi);
       
        // one ghost cell 
        //Box bx = mfi.validbox();
        const Box& bx = mfi.growntilebox(1);

	std::string jc_s;
	std::unique_ptr<amrex::Parser> jc_parser;
        std::string m_str_jc_function;
	
        ParmParse pp_jc("mf_jc");

	if (pp_jc.query("jc_function(x,y,z)", m_str_jc_function) ) {
            jc_s = "parse_jc_function";
        }

        if (jc_s == "parse_jc_function") {
            Store_parserString(pp_jc, "jc_function(x,y,z)", m_str_jc_function);
            jc_parser = std::make_unique<amrex::Parser>(
                                     makeParser(m_str_jc_function,{"x","y","z"}));
        }

        const auto& macro_parser_jc = jc_parser->compile<3>();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_jc,macro_parser_jc,jc_arr);
        });

    }
	Jc_mf.FillBoundary(geom.periodicity());
}

void initialize_Cv_using_parser(c_MagnonDiffusion& rMagnonDiffusion, const Geometry& geom, MultiFab& Cv_mf)
{ 
    auto& rGprop = rMagnonDiffusion.get_GeometryProperties();
    Box const& domain = rGprop.geom.Domain();

    const auto dx = rGprop.geom.CellSizeArray();
    const auto& real_box = rGprop.geom.ProbDomain();
    const auto iv_cv = Cv_mf.ixType().toIntVect();

    for (MFIter mfi(Cv_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const auto& cv_arr = Cv_mf.array(mfi);
       
        // one ghost cell 
        //Box bx = mfi.validbox();
        const Box& bx = mfi.growntilebox(1);

	std::string cv_s;
	std::unique_ptr<amrex::Parser> cv_parser;
        std::string m_str_cv_function;
	
        ParmParse pp_cv("mf_cv");

	if (pp_cv.query("cv_function(x,y,z)", m_str_cv_function) ) {
            cv_s = "parse_cv_function";
        }

        if (cv_s == "parse_cv_function") {
            Store_parserString(pp_cv, "cv_function(x,y,z)", m_str_cv_function);
            cv_parser = std::make_unique<amrex::Parser>(
                                     makeParser(m_str_cv_function,{"x","y","z"}));
        }

        const auto& macro_parser_cv = cv_parser->compile<3>();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_cv,macro_parser_cv,cv_arr);
        });

    }
	Cv_mf.FillBoundary(geom.periodicity());
}

void fill_acoef(MultiFab& acoef_mf, MultiFab& sigma, MultiFab& lambda){

    // loop over boxes
    for (MFIter mfi(acoef_mf); mfi.isValid(); ++mfi)
    {   
        //const Box& bx = mfi.validbox();
        const Box& bx = mfi.growntilebox(1);

        const Array4<Real>& acoef_mf_arr = acoef_mf.array(mfi);
        const Array4<Real>& sigma_arr = sigma.array(mfi);
        const Array4<Real>& lambda_arr = lambda.array(mfi);


        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {   
            acoef_mf_arr(i,j,k) = 1.0 + dt*( sigma_arr(i,j,k) / ( lambda_arr(i,j,k)*lambda_arr(i,j,k) ) );

        }); 
    }   
}


void fill_bcoef(MultiFab& bcoef_mf, MultiFab& sigma){

    // loop over boxes
    for (MFIter mfi(bcoef_mf); mfi.isValid(); ++mfi)
    {   
        //const Box& bx = mfi.validbox();
        const Box& bx = mfi.growntilebox(1);

        const Array4<Real>& bcoef_mf_arr = bcoef_mf.array(mfi);
        const Array4<Real>& sigma_arr = sigma.array(mfi);


        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {   
            bcoef_mf_arr(i,j,k) = dt*sigma_arr(i,j,k);

        }); 
    }   
}

void fill_bcoef_T(MultiFab& bcoef_mf, MultiFab& kappa, MultiFab& Cv_mf){

    // loop over boxes
    for (MFIter mfi(bcoef_mf); mfi.isValid(); ++mfi)
    {   
        //const Box& bx = mfi.validbox();
        const Box& bx = mfi.growntilebox(1);

        const Array4<Real>& bcoef_mf_arr = bcoef_mf.array(mfi);
        const Array4<Real>& kappa_arr = kappa.array(mfi);
        const Array4<Real>& Cv_arr = Cv_mf.array(mfi);


        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {   
            bcoef_mf_arr(i,j,k) = dt*kappa_arr(i,j,k)/Cv_arr(i,j,k);

        }); 
    }   
}

void compute_currents(Array<MultiFab, AMREX_SPACEDIM> &J_mu,
                       Array<MultiFab, AMREX_SPACEDIM> &J_T,
                       MultiFab& sigma,
                       MultiFab& kappa){

    // loop over boxes
    for (MFIter mfi(J_mu[0]); mfi.isValid(); ++mfi)
    {   
        //const Box& bx = mfi.validbox();
        const Box& bx = mfi.growntilebox(1);

        const Array4<Real>& J_mux = J_mu[0].array(mfi);
        const Array4<Real>& J_muy = J_mu[1].array(mfi);
        const Array4<Real>& J_muz = J_mu[2].array(mfi);
        const Array4<Real>& J_Tx = J_T[0].array(mfi);
        const Array4<Real>& J_Ty = J_T[1].array(mfi);
        const Array4<Real>& J_Tz = J_T[2].array(mfi);
        const Array4<Real>& sigma_arr = sigma.array(mfi);
        const Array4<Real>& kappa_arr = kappa.array(mfi);


        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {   
            J_mux(i,j,k) = -1.0*sigma_arr(i,j,k)*J_mux(i,j,k) - 1.0*sigma_arr(i,j,k)*S_s*J_Tx(i,j,k);
            J_muy(i,j,k) = -1.0*sigma_arr(i,j,k)*J_muy(i,j,k) - 1.0*sigma_arr(i,j,k)*S_s*J_Ty(i,j,k);
            J_muz(i,j,k) = -1.0*sigma_arr(i,j,k)*J_muz(i,j,k) - 1.0*sigma_arr(i,j,k)*S_s*J_Tz(i,j,k);

            J_Tx(i,j,k) = -1.0*kappa_arr(i,j,k)*J_Tx(i,j,k);
            J_Ty(i,j,k) = -1.0*kappa_arr(i,j,k)*J_Ty(i,j,k);
            J_Tz(i,j,k) = -1.0*kappa_arr(i,j,k)*J_Tz(i,j,k);

        }); 
    }   
}
