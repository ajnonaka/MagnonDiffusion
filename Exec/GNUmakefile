# AMREX_HOME defines the directory in which we will find all the AMReX code.
AMREX_HOME ?= ../../amrex
CODE_HOME := ..

DEBUG     = FALSE
USE_MPI   = TRUE
USE_OMP   = FALSE
COMP      = gnu
DIM       = 3
USE_EB    = FALSE
CXXSTD    = c++17

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ../Source/Make.package
VPATH_LOCATIONS  += ../Source
INCLUDE_LOCATIONS += ../Source

Pdirs 	:= Base Boundary LinearSolvers/MLMG AmrCore

ifeq ($(USE_EB),TRUE)
Pdirs += EB
USERSuffix := $(USERSuffix).EB
endif

Ppack	+= $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)
include $(Ppack)

Code_dirs := Input Utils
Code_pack   += $(foreach dir, $(Code_dirs), $(CODE_HOME)/Source/$(dir)/Make.package)
include $(Code_pack)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules

