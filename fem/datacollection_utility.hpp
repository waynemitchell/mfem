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

#include "gridfunc.hpp"
#include "fespace.hpp"

#include "datacollection.hpp"

#ifdef MFEM_USE_SIDRE
#include "sidredatacollection.hpp"
#endif

namespace mfem
{

/**
 * \class DataCollectionUtility
 * This class contains several utility functions to simplify usage
 * of a DataCollection with GridFunctions, Vectors and Arrays.
 * Its functions access names arrays from DataCollections,
 * allocate data within a DataCollection and register GridFunctions
 * with a DataCollection.
 */
class DataCollectionUtility
{
public:

    static const int NAME_SZ = 50;

    /**
     * Generates a field name, including indexing for fields within an array
     * \param [in] inputFldName The base name of the field
     * \param [in] arr_idx The index of the field within an array when arr_idx != -1
     * \param [out] outputFldName A null terminated string containing the output name
     * \param [in] sz The size of the buffer containing the outputFldName string
     * \note For arrays, the naming convention is to use 1-based indexing
     *       e.g. if inputFldName is 'fld' and arr_idx is 0,
     *            the outputFldName will be 'fld_1'
     */
    static void GenerateFieldName(const char* inputFldName, int arr_idx,
                           char* outputFldName, const int sz);



    /**
     *  Utility function to associate a named data array from a DataCollection dc
     *  with an mfem Vector vec. If dc does not already contain the named array,
     *  it will allocate storage for sz doubles.
     *  \note if dc is null, the size of the vector is set to sz
     *        using the Vector's default allocator (e.g. std::new)
     *  \param vec A (non-null) pointer to a Vector
     *  \param sz The size of the array
     *  \param dc A pointer to the data collection instance (may be null)
     *  \param fldName The name of the Vector's array in dc
     *  \param arr_idx An optional index for fldName, appended when arr_idx != 1
     *  \pre vec is not null
     *  \post vec's underlying array has space for sz doubles
     *  \see GenerateFieldName
     */
    static void AllocateVector( Vector* vec,
                             int sz,
                             DataCollection* dc,
                             const char* fldName,
                             int arr_idx = -1);

    /**
     *  Utility function to associate a named data array from a DataCollection dc
     *  with an mfem GridFunction gf and to optionally register gf with dc.
     *  If dc does not already contain the named array, it will allocate storage
     *  (sufficient for the FiniteElementSpace fes)
     *  \note if dc is null, storage for the GridFunction will be allocated
     *        based on the FiniteElementSpace using the GridFunction's default allocator (e.g. std::new)
     *  \param gf A (non-null) pointer to a GridFunction or ParGridFunction
     *  \param fes A (non-null) pointer to a FiniteElementSpace or ParFiniteElementSpace
     *  \param dc A pointer to the data collection instance (may be null)
     *  \param fldName The name of the GridFunction (an its data) in dc
     *  \param arr_idx An optional index for fldName, appended when arr_idx != 1
     *  \param registerGF When true (default) and dc is non-null, register the gf with dc
     *  \pre gf and fes are not null
     *  \post gf's underlying array has the right amount of space for the fes type
     *  \see GenerateFieldName
     */
    static void AllocateGridFunc( GridFunction* gf,
                                      FiniteElementSpace* fes,
                                      DataCollection* dc,
                                      const char* fldName,
                                      int arr_idx = -1,
                                      bool registerGF = true);


    /**
     *  Utility function to set an mfem GridFunction gf's data
     *  to an offset within the memory space of an existing mfem Vector srcVec
     *  and to register gf with the provided DataCollection dc
     *  \note only registers gf when dc is not null
     *  \param gf A (non-null) pointer to a GridFunction or ParGridFunction
     *  \param fes A (non-null) pointer to a FiniteElementSpace or ParFiniteElementSpace
     *  \param srcVec The (non-null) source vector to which gf's data will point
     *  \parem offset The offset within srcVec for gf's data
     *  \param dc A pointer to the data collection instance (may be null)
     *  \param fldName The name of the GridFunction (an its data) in dc
     *  \param arr_idx An optional index for fldName, appended when arr_idx != 1
     *  \pre gf and fes are not null
     *  \post gf's underlying array pointer is set
     *  \see GenerateFieldName
     */
    static void AllocateGridFuncInPlace( GridFunction* gf,
                                             FiniteElementSpace* fes,
                                             const char* srcField, Vector* srcVec, int offset,
                                             DataCollection* dc,
                                             const char* fldName,
                                             int arr_idx = -1);


    /**
     *  Utility function to associate a named data array (of type T*) from a DataCollection dc
     *  with an mfem Array<T>. If dc does not already contain the named array, it will allocate storage.
     *  \param arr A (non-null) pointer to an Array
     *  \param sz The size of the array
     *  \param dc A pointer to the data collection instance (may be null)
     *  \param fldName The name of the Vector's array in dc
     *  \param arr_idx An optional index for fldName, appended when arr_idx != 1
     *  \pre arr is not null
     *  \post arr's underlying array has space for sz elements of type T
     *  \note if dc is null, or dc is not an instance of SidreDataCollection
     *        we only set the size of the array to sz
     *        using the Array's default allocator (e.g. std::new)
     *  \note When dc is an instance of SidreDataCollection, this function is only
     *        defined for types supported by Sidre (e.g. signed and unsigned integers
     *        and floats of various bit-widths)
     *  \see GenerateFieldName
     */
    template<typename T>
    static void AllocateArray( Array<T>* arr
                                   , int sz
                                   , DataCollection* dc
                                   , const char* fldName
                                   , int arr_idx = -1)
    {
         MFEM_ASSERT(arr != NULL, "Attempted to allocate into a null array pointer");

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

       MFEM_ASSERT( arr->Size() == sz, "Array only has storage for " << arr->Size()
                    <<" elements but " << sz << " was requested.");

    }
};




} // end namespace mfem

#endif  // MFEM_DATACOLLECTION_UTILITY
