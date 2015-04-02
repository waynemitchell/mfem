// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "_hypre_parcsr_mv.h"

namespace mfem
{

// This module contains functions that are logically part of HYPRE.
// It can be thought of a an extension of HYPRE.

namespace internal
{

/** Parallel essential BC elimination from the system A*X = B.
    Adapted from hypre_ParCSRMatrixEliminateRowsCols. */
void hypre_ParCSRMatrixEliminateAXB(hypre_ParCSRMatrix *A,
                                    HYPRE_Int num_rowscols_to_elim,
                                    HYPRE_Int *rowscols_to_elim,
                                    hypre_ParVector *X,
                                    hypre_ParVector *B);

/** Parallel essential BC elimination from matrix A only. The eliminated
    elements are stored in a new matrix Ae, so that the modified A matrix and
    the Ae matrix sum to the original A matrix. */
void hypre_ParCSRMatrixEliminate(hypre_ParCSRMatrix *A,
                                 hypre_ParCSRMatrix **Ae,
                                 HYPRE_Int num_rowscols_to_elim,
                                 HYPRE_Int *rowscols_to_elim);

}

} // namespace mfem::internal

#endif // MFEM_USE_MPI
