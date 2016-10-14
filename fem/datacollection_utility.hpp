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

#ifndef MFEM_DATACOLLECTION_UTILITY
#define MFEM_DATACOLLECTION_UTILITY

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI
#include "pgridfunc.hpp"
#include "pfespace.hpp"
#else
#include "gridfunc.hpp"
#include "fespace.hpp"
#endif

#include "datacollection.hpp"

#ifdef MFEM_USE_SIDRE
#include "../external/sidredatacollection.hpp"
#endif

namespace mfem
{

/**
 * \class DataCollectionHelper
 * This class contains several utility functions to simplify usage
 * of a DataCollection with GridFunctions, Vectors and Arrays.
 */
class DataCollectionHelper
{
public:

    static const int NAME_SZ = 50;

    /**
     * Generates a field name, including indexing for fields within an array
     * \param [in] inputFldName The base name of the field
     * \param [in] arr_idx The index of the field within an array when arr_idx != -1
     * \param [out] outputFldName A null terminated string containing the output name
     * \param [in] sz The size of the buffer containing the outputFldName string
     */
    static void GenerateFieldName(const char* inputFldName, int arr_idx,
                           char* outputFldName, const int sz);



    /**
     *  Utility function to allocate an mfem Vector within a DataCollection dc
     *  \note Only allocates from the datacollection when dc is non-null.
     *        Otherwise, it sets the size of the vector to sz using the Vector's default allocator (e.g. std::new)
     */
    static void AllocateVector( Vector* vec,
                             int sz,
                             DataCollection* dc,
                             const char* fldName,
                             int arr_idx = -1);


    /** Utility function to allocate and register a grid function
     *  with a DataCollection dc
     *  \note Only allocates from the datacollection when dc is non-null.
     *  \note Registers the grid function when registerGF == true (default)
     *  \param arr_idx When gf is in an array of gridfunctions this parameter provides an index.
     *         A value of -1 (default) indicates that it is not in an array.
     */
    static void AllocateGridFunc( GridFunction* gf,
                                      FiniteElementSpace* fes,
                                      DataCollection* dc,
                                      const char* fldName,
                                      int arr_idx = -1,
                                      bool registerGF = true);

#ifdef MFEM_USE_MPI
    /** Utility function to allocate and register a parallel grid function
     *  with a DataCollection dc
     *  \note Only allocates from the datacollection when dc is non-null.
     *  \note Registers the grid function when registerGF == true (default)
     *  \param arr_idx When pgf is in an array of gridfunctions this parameter provides an index.
     *  A value of -1 (default) indicates that it is not in an array.
     */
    static void AllocateGridFunc( ParGridFunction* pgf,
                                      ParFiniteElementSpace* pfes,
                                      DataCollection* dc,
                                      const char* fldName,
                                      int arr_idx = -1,
                                      bool registerGF = true);
#endif

    /**
     *  Utility function to allocate and register an mfem Array (templated on type T)
     *  with a DataCollection dc.
     *  \note This function is meant to be used with a SidreDataCollection.
     *        It only allocates from dc when it is non-null and the dc is of type SidreDataCollection*.
     *        Otherwise, it allocates and sets the size using mfem's default allocator (e.g. std::new)
     */
    template<typename T>
    static void AllocateArray( Array<T>* arr
                                   , int sz
                                   , DataCollection* dc
                                   , const char* fldName
                                   , int arr_idx = -1)
    {
    #ifdef MFEM_USE_SIDRE
       if ( dc && dynamic_cast<SidreDataCollection*>(dc))
       {
          char name[NAME_SZ];
          GenerateFieldName(fldName, arr_idx, name, NAME_SZ);

          T* data = dynamic_cast<SidreDataCollection*>(dc)->GetArrayData<T>(name, sz);
          arr->MakeRef( data, sz);
       }
       else
    #endif
       {
          arr->SetSize(sz);
       }
    }
};




} // end namespace mfem

#endif  // MFEM_DATACOLLECTION_UTILITY
