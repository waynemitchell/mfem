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

#ifndef MFEM_BACKENDS_HYPRE_LAYOUT_HPP
#define MFEM_BACKENDS_HYPRE_LAYOUT_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_MPI)

#include "../../general/array.hpp"
#include "../base/layout.hpp"
#include "HYPRE_utilities.h"

namespace mfem
{

namespace hypre
{

class Layout : public PLayout
{
protected:
   //
   // Inherited fields
   //
   // SharedPtr<const mfem::Engine> engine;
   // std::size_t size;

   mfem::Array<HYPRE_Int> offsets;
   int rank;
   size_t total_size;

   DLayout base;

public:
   Layout(const Engine &e, HYPRE_Int *offsets_, int rank_)
      : PLayout(e, 0), offsets(), rank(rank_), base(NULL) // size (set to 0 here) is initialized later in the constructor
   {
      /*
        if HYPRE_AssumedPartitionCheck() returns true: then len(offsets_) = 3
        offsets_[0] = beginning of internal
        offsets_[1] = end of internal (not inclusive)
        offsets_[2] = global total size

        else: then len(offsets_) = len(comm) + 1 where offsets_[rank] = beginning of interval
       */
      if (HYPRE_AssumedPartitionCheck())
      {
         offsets.SetSize(3);
         size = offsets_[1] - offsets_[0];
      }
      else
      {
         int comm_size, comm_rank;
         MPI_Comm_size(e.GetComm(), &comm_size);
         MPI_Comm_rank(e.GetComm(), &comm_rank);
         offsets.SetSize(comm_size + 1);
         size = offsets_[comm_rank+1] - offsets_[comm_rank];
      }
      for (int i = 0; i < offsets.Size(); i++) offsets[i] = offsets_[i];

      total_size = offsets[offsets.Size() - 1];

      base = engine->MakeLayout(size);
   }

   virtual ~Layout() { }

   /// Resize the layout
   virtual void Resize(std::size_t new_size) {
      if (new_size != size) mfem_error("NOT SUPPORTED");
   }

   PLayout& Base() { return *base; }

   std::size_t GlobalSize() const { return total_size; }

   // NOTE: the HYPRE routines do not use const HYPRE_Int *, so we
   // need to perform a const crime here.
   HYPRE_Int *Offsets() const { return const_cast<HYPRE_Int*>(offsets.GetData()); }
};

} // namespace mfem::occa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_MPI)

#endif // MFEM_BACKENDS_HYPRE_LAYOUT_HPP
