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

#include <cstdio>      // snprintf, sscanf
#include <sstream>

namespace mfem
{


// -----------------------------------------------
//    Utility functions to make it easier to allocate and register data
//    in a data collection
// -----------------------------------------------

void DataCollectionUtility::GenerateFieldName(const char* inputFldName,
                                              int arr_idx,
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


std::string DataCollectionUtility::DecomposeFieldName(const std::string&
                                                      inputFldName, int& arr_idx)
{
   int idx = arr_idx = -1;

   std::string::size_type pos = inputFldName.find_last_of('_');
   if (pos != std::string::npos)
   {
      std::istringstream isstr( inputFldName.substr(pos+1, inputFldName.size()));
      if ( !( isstr  >> idx).fail() )
      {
         arr_idx = idx-1;
         return inputFldName.substr(0,pos);
      }
   }
   return inputFldName;
}



void DataCollectionUtility::AllocateVector( Vector* vec,
                                            int sz,
                                            DataCollection* dc,
                                            const char* fldName,
                                            int arr_idx)
{
   MFEM_ASSERT(vec != NULL, "Attempted to allocate into a null vector pointer");

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

   MFEM_ASSERT( vec->Size() == sz, "Vector only has storage for " << vec->Size()
                <<" elements but " << sz << " was requested.");
}


namespace
{

/**
 * Templated implementation function to workaround the non-virtual
 * function calls of GridFunction / ParGridFunction.
 * \see DataCollectionUtility::AllocateGridFunc()
 */
template<typename GF, typename FES>
void AllocGridFuncImpl(GF* gf, FES* fes, DataCollection* dc,
                       const char* fldName, int arr_idx, bool registerGF)
{
   if ( dc )
   {
      const int NAME_SZ = DataCollectionUtility::NAME_SZ;
      char name[NAME_SZ];
      DataCollectionUtility::GenerateFieldName(fldName, arr_idx, name, NAME_SZ);

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

/**
 * Templated implementation function to workaround the non-virtual
 * function calls of GridFunction / ParGridFunction.
 * \see DataCollectionUtility::AllocateGridFuncInPlace()
 */
template<typename GF, typename FES>
void AllocGridFuncInPlaceImpl(GF* gf, FES* fes, const char* srcField,
                              Vector* srcVec, int offset,
                              DataCollection* dc,const char* fldName,int arr_idx)
{
   // Compose the field name
   const int NAME_SZ = DataCollectionUtility::NAME_SZ;
   char name[NAME_SZ];
   DataCollectionUtility::GenerateFieldName(fldName, arr_idx, name, NAME_SZ);

   // The data for gf will be offset from that of the srcField data
   if (dc)
   {
      const int sz = fes->GetVSize();
      dc->GetFieldData(name, sz, srcField, offset);
   }

   gf->MakeRef(fes, *srcVec, offset);

   // Register this grid function as a field in the DataCollection
   if (dc)
   {
      dc->RegisterField(name, gf);
   }
}

}


void DataCollectionUtility::AllocateGridFunc( GridFunction* gf,
                                              FiniteElementSpace* fes,
                                              DataCollection* dc,
                                              const char* fldName,
                                              int arr_idx,
                                              bool registerGF)
{
   // Note: GridFunction::MakeRef(), GridFunction::SetSpace()
   //       are not virtual functions, so we must explicitly dynamic_cast
   //       to a parallel version when the actual type is virtual.
   //       Since the actual code for both cases is identical,
   //       we are using a templated 'implementation' function AllocGridFuncImpl

   MFEM_ASSERT(gf != NULL, "Attempted to dereference a null grid function");
   MFEM_ASSERT(fes != NULL,
               "Attempted to dereference a null finite element space");


#ifdef MFEM_USE_MPI
   ParGridFunction* pgf = dynamic_cast<ParGridFunction*>(gf);
   ParFiniteElementSpace* pfes = dynamic_cast<ParFiniteElementSpace*>(fes);

   if (pgf && pfes)
   {
      AllocGridFuncImpl(pgf,pfes, dc,fldName,arr_idx,registerGF);
   }
   else
   {
      AllocGridFuncImpl(gf,fes, dc,fldName,arr_idx,registerGF);
   }

#else
   AllocGridFuncImpl(gf,fes, dc,fldName,arr_idx,registerGF);
#endif

   MFEM_ASSERT( gf->Size() == fes->GetVSize(), "GridFunction only has storage for "
                << gf->Size() <<" dofs but its finite element space requires "
                << fes->GetVSize() <<" dofs.");
}

void DataCollectionUtility::AllocateGridFuncInPlace( GridFunction* gf,
                                                     FiniteElementSpace* fes,
                                                     const char* srcField, Vector* srcVec, int offset,
                                                     DataCollection* dc,
                                                     const char* fldName,
                                                     int arr_idx)
{
   // Note: GridFunction::MakeRef(), GridFunction::SetSpace()
   //       are not virtual functions, so we must explicitly dynamic_cast
   //       to a parallel version when the actual type is virtual.
   //       Since the actual code for both cases is identical,
   //       we are using a templated 'implementation' function AllocGridFuncImpl

   MFEM_ASSERT(gf != NULL, "Attempted to dereference a null grid function");
   MFEM_ASSERT(fes != NULL,
               "Attempted to dereference a null finite element space");

#ifdef MFEM_USE_MPI
   ParGridFunction* pgf = dynamic_cast<ParGridFunction*>(gf);
   ParFiniteElementSpace* pfes = dynamic_cast<ParFiniteElementSpace*>(fes);

   if (pgf && pfes)
   {
      AllocGridFuncInPlaceImpl(pgf,pfes,srcField,srcVec,offset, dc,fldName,arr_idx);
   }
   else
   {
      AllocGridFuncInPlaceImpl(gf,fes,srcField, srcVec,offset, dc,fldName,arr_idx);
   }

#else
   AllocGridFuncInPlaceImpl(gf,fes,srcField, srcVec, offset, dc,fldName,arr_idx);
#endif

   MFEM_ASSERT( gf->Size() == fes->GetVSize(), "GridFunction only has storage for "
                << gf->Size() <<" dofs but its finite element space requires "
                << fes->GetVSize() <<" dofs.");
}


} // end namespace mfem

