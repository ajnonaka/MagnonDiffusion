namespace MagnonDiffusion {

    extern AMREX_GPU_MANAGED int nsteps;
    extern AMREX_GPU_MANAGED int plot_int;
    
    // time step
    extern AMREX_GPU_MANAGED amrex::Real dt;

    extern AMREX_GPU_MANAGED amrex::Vector<int> bc_lo;
    extern AMREX_GPU_MANAGED amrex::Vector<int> bc_hi;

    extern AMREX_GPU_MANAGED amrex::Vector<int> T_bc_lo;
    extern AMREX_GPU_MANAGED amrex::Vector<int> T_bc_hi;

    // General BC parameters for Dirichlet, Neumann, or Robin
    // Dirichlet: phi = f
    // Neumann: d(phi)/dn = f
    // Robin: a*phi+b*d(phi)/dn = f
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> bc_lo_f; // BC rhs.
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> bc_hi_f; // BC rhs.
    
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> T_bc_lo_f; // BC rhs.
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> T_bc_hi_f; // BC rhs.
    
    // Robin BC parameters
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> bc_lo_a; // robin BC coeffs. 
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> bc_hi_a; // robin BC coeffs. 
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> bc_lo_b; // robin BC coeffs. 
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> bc_hi_b; // robin BC coeffs. 

    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> T_bc_lo_a; // robin BC coeffs. 
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> T_bc_hi_a; // robin BC coeffs. 
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> T_bc_lo_b; // robin BC coeffs. 
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> T_bc_hi_b; // robin BC coeffs. 

    // magnon diffusion parameters
    extern AMREX_GPU_MANAGED amrex::Real J_c;
    extern AMREX_GPU_MANAGED amrex::Real sigma_Pt;
    extern AMREX_GPU_MANAGED amrex::Real S_s;
    
}
