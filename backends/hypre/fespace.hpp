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

#ifndef MFEM_BACKENDS_HYPRE_FESPACE_HPP
#define MFEM_BACKENDS_HYPRE_FESPACE_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_MPI)

#include "../../fem/fem.hpp"
#include "layout.hpp"
#include "parmatrix.hpp"

namespace mfem
{

namespace hypre
{

/// TODO: doxygen
class FiniteElementSpace
{
protected:
   SharedPtr<const Engine> e;
   mfem::ParFiniteElementSpace *pfespace;

   SharedPtr<Layout> t_layout;
   SharedPtr<Layout> l_layout;
   SharedPtr<Layout> e_layout;

   mutable ParMatrix *prolong;

   ParMatrix *MakeProlongation() const;

public:
   /// TODO: doxygen
   FiniteElementSpace(const Engine &e_, mfem::ParFiniteElementSpace &fespace) :
      pfespace(&fespace), t_layout(NULL), l_layout(NULL), e_layout(NULL), prolong(NULL)
   {
      e.Reset(&e_);

      HYPRE_Int *t_dof_offsets = fespace.GetTrueDofOffsets();
      HYPRE_Int *l_dof_offsets = fespace.GetDofOffsets();

      const mfem::Table &e2d_table = fespace.GetElementToDofTable();

      int rank = -1, size = -1;
      MPI_Comm_rank(fespace.GetComm(), &rank);
      MPI_Comm_size(fespace.GetComm(), &size);
      MFEM_ASSERT((rank >= 0) && (size > 0), "");

      mfem::Array<int> eoffsets(size+1);
      eoffsets[0] = 0;
      const std::size_t esize = e2d_table.Size_of_connections();
      MPI_Allgather(&esize, 1, MPI_UNSIGNED, &eoffsets[1], 1, MPI_UNSIGNED, fespace.GetComm());

      e_layout.Reset(new Layout(*e, eoffsets.GetData(), rank));
      l_layout.Reset(new Layout(*e, l_dof_offsets, rank));
      t_layout.Reset(new Layout(*e, t_dof_offsets, rank));
   }

   /// Virtual destructor
   virtual ~FiniteElementSpace() {
      delete prolong;
   }

   Layout &GetTLayout() const { return *t_layout; }
   Layout &GetLLayout() const { return *l_layout; }
   Layout &GetELayout() const { return *e_layout; }

   virtual const mfem::hypre::ParMatrix& GetProlongation() const {
      if (!prolong) prolong = MakeProlongation();
      return *prolong;
   }
};

} // namespace mfem::hypre

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_MPI)

#endif // MFEM_BACKENDS_HYPRE_FESPACE_HPP
