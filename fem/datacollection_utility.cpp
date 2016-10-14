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

#include "fem.hpp"

#include <cstdio>      // snprintf


namespace mfem
{


// -----------------------------------------------
//    Utility functions to make it easier to allocate and register data
//    in a data collection
// -----------------------------------------------

void DataCollectionHelper::GenerateFieldName(const char* inputFldName, int arr_idx,
                       char* outputFldName, const int sz)
{
   if (arr_idx >= 0)
   {
      // Use a 1-based indexing for compatibility with Lua
      std::snprintf(outputFldName, sz, "%s_%03d", inputFldName,arr_idx + 1);
   }
   else
   {
      std::snprintf(outputFldName, sz, "%s", inputFldName);
   }
}


void DataCollectionHelper::AllocateVector( Vector* vec,
                                int sz,
                                DataCollection* dc,
                                const char* fldName,
                                int arr_idx)
{
   if ( dc )
   {
      char name[NAME_SZ];
      GenerateFieldName(fldName, arr_idx, name, NAME_SZ);

      vec->NewDataAndSize(dc->GetFieldData(name, sz), sz);
   }
   else
   {
      vec->SetSize(sz);
   }
}


void DataCollectionHelper::AllocateGridFunc( GridFunction* gf,
                                         FiniteElementSpace* fes,
                                         DataCollection* dc,
                                         const char* fldName,
                                         int arr_idx,
                                         bool registerGF)
{
   if ( dc )
   {
      char name[NAME_SZ];
      GenerateFieldName(fldName, arr_idx, name, NAME_SZ);

      const int sz = fes->GetVSize();
      Vector v(dc->GetFieldData(name, sz), sz);
      gf->MakeRef(fes, v, 0);

      if (registerGF)
      {
         dc->RegisterField(name, gf);
      }
   }
   else
   {
      gf->SetSpace(fes);
   }
}

#ifdef MFEM_USE_MPI
void DataCollectionHelper::AllocateGridFunc( ParGridFunction* pgf,
                                         ParFiniteElementSpace* pfes,
                                         DataCollection* dc,
                                         const char* fldName,
                                         int arr_idx,
                                         bool registerGF)
{
   if ( dc )
   {
      char name[NAME_SZ];
      GenerateFieldName(fldName, arr_idx, name, NAME_SZ);

      const int sz = pfes->GetVSize();
      Vector v(dc->GetFieldData(name, sz), sz);
      pgf->MakeRef(pfes, v, 0);

      if (registerGF)
      {
         dc->RegisterField(name, pgf);
      }
   }
   else
   {
      pgf->SetSpace(pfes);
   }
}
#endif

}
