// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "mfem.hpp"
#include "linalg/tlayouts.hpp"
#include "linalg/tassign_ops.hpp"
#include "linalg/tsmall_matrix_ops.hpp"
#include "linalg/ttensor_ops.hpp"
#include "linalg/ttensor_types.hpp"
#include "linalg/tmatrix_products.hpp"
#include "linalg/ttensor_products.hpp"
#include "linalg/tvector_layouts.hpp"
#include "mesh/tmesh.hpp"
#include "fem/tfinite_elements_h1.hpp"
#include "fem/tintrules.hpp"
#include "fem/tshape_evaluators.hpp"
#include "fem/tfespace_h1.hpp"
#include "fem/tfespace_l2.hpp"
#include "fem/tmass_kernel.hpp"
#include "fem/tdiffusion_kernel.hpp"
