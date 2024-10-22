#include <AMReX_Gpu.H>
// Algorithm to calculate Laplacian of temperature

/**
 * Perform derivative along x on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real UpwardDx (
    amrex::Array4<amrex::Real const> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    amrex::Real const inv_dx = 1./dx[0];
    return inv_dx*( F(i+1,j,k) - F(i,j,k) );
}


/**
 * Perform derivative along x on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DownwardDx (
    amrex::Array4<amrex::Real const> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    amrex::Real const inv_dx = 1./dx[0];
    return inv_dx*( F(i,j,k) - F(i-1,j,k) );
}


/**
 * Perform derivative along y on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real UpwardDy (
    amrex::Array4<amrex::Real const> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    amrex::Real const inv_dy = 1./dx[1];
    return inv_dy*( F(i,j+1,k) - F(i,j,k) );
}


/**
 * Perform derivative along y on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DownwardDy (
    amrex::Array4<amrex::Real const> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    amrex::Real const inv_dy = 1./dx[1];
    return inv_dy*( F(i,j,k) - F(i,j-1,k) );
}



/**
 * Perform derivative along z on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real UpwardDz (
    amrex::Array4<amrex::Real const> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    amrex::Real const inv_dz = 1./dx[2];
    return inv_dz*( F(i,j,k+1) - F(i,j,k) );
}


/**
 * Perform derivative along z on a nodal grid, from a cell-centered field `F`*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real DownwardDz (
    amrex::Array4<amrex::Real const> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    amrex::Real const inv_dz = 1./dx[2];
    return inv_dz*( F(i,j,k) - F(i,j,k-1) );
}

/**
  * Perform divergence of gradient along x on F field */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real LaplacianDx (
    amrex::Array4<amrex::Real const> const& F, int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    amrex::Real const inv_dx = 1./dx[0];

    return inv_dx*(UpwardDx(F, i, j, k, dx) - DownwardDx(F, i, j, k, dx));

 }


/**
  * Perform divergence of gradient along y on F field */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real LaplacianDy (
    amrex::Array4<amrex::Real const> const& F, int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    amrex::Real const inv_dy = 1./dx[1];

    return inv_dy*(UpwardDy(F, i, j, k, dx) - DownwardDy(F, i, j, k, dx));

 }


/**
  * Perform divergence of gradient along z on F field */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real LaplacianDz (
    amrex::Array4<amrex::Real const> const& F, int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

    amrex::Real const inv_dz = 1./dx[2];

    return inv_dz*(UpwardDz(F, i, j, k, dx) - DownwardDz(F, i, j, k, dx));

 }

/**
  * Compute the sum to get Laplacian of M field */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real Laplacian(amrex::Array4<amrex::Real const> const& F, int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

        //amrex::Print() << "This is called " << Ms_lo_x << "\n";
        return LaplacianDx(F, i, j, k, dx) + LaplacianDy(F, i, j, k, dx) + LaplacianDz(F, i, j, k, dx);
 }

