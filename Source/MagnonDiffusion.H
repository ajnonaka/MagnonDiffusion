#ifndef _magnondiffusion_H_
#define _magnondiffusion_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_LO_BCTYPES.H>

#include "MagnonDiffusion_namespace.H"

using namespace MagnonDiffusion;

void main_main ();

void InitializeMagnonDiffusionNamespace ();

void advance (amrex::MultiFab& phi_old,
              amrex::MultiFab& phi_new,
              const amrex::Geometry& geom);

#endif
