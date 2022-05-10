#ifndef _magnondiffusion_H_
#define _magnondiffusion_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_LO_BCTYPES.H>

#include "MagnonDiffusion_namespace.H"

using namespace MagnonDiffusion;
using namespace amrex;

void main_main ();

void InitializeMagnonDiffusionNamespace ();

void advance (MultiFab& phi_old,
              MultiFab& phi_new,
              const Geometry& geom);

void FillBoundaryPhysical (MultiFab& phi, const Geometry& geom);

void FillBoundaryRobin (MultiFab& robin_a,
                        MultiFab& robin_b,
                        MultiFab& robin_f,
                        const Geometry& geom);

#endif
