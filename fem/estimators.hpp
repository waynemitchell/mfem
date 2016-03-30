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

#ifndef MFEM_ERROR_ESTIMATORS
#define MFEM_ERROR_ESTIMATORS

#include "../config/config.hpp"
#include "../linalg/vector.hpp"
#include "bilinearform.hpp"

namespace mfem
{

/** @brief The ErrorEstimator class is the base class for all error estimators.
 */
class ErrorEstimator
{
public:
   virtual ~ErrorEstimator() { }
};


/** @brief The IsotropicErrorEstimator class is the base class for all error
    estimators that compute one non-negative real (double) number for every
    element in the Mesh.
 */
class IsotropicErrorEstimator : public ErrorEstimator
{
public:
   /// @brief Get a Vector with all element errors.
   virtual const Vector &GetLocalErrors() = 0;

   virtual ~IsotropicErrorEstimator() { }
};


/** @brief The ZienkiewiczZhuEstimator class implements the Zienkiewicz-Zhu
    error estimation procedure.

    The required BilinearFormIntegrator must implement the methods
    ComputeElementFlux() and ComputeFluxEnergy().
 */
class ZienkiewiczZhuEstimator : public IsotropicErrorEstimator
{
protected:
   long current_sequence;
   Vector error_estimates;
   double total_error;

   BilinearFormIntegrator *integ; ///< Not owned.
   GridFunction *solution; ///< Not owned.

   FiniteElementSpace *flux_space; /**< @brief
      Owned. Its Update() method is called when needed. */

   /// @brief Check if the mesh of the solution was modified.
   bool MeshIsModified()
   {
      long mesh_sequence = solution->FESpace()->GetMesh()->GetSequence();
      MFEM_ASSERT(mesh_sequence >= current_sequence, "");
      return (mesh_sequence > current_sequence);
   }

   /// @brief Compute the element error estimates.
   void ComputeEstimates();

public:
   /** @brief Construct a new ZienkiewiczZhuEstimator object.
       @param integ    This BilinearFormIntegrator must implement the methods
                       ComputeElementFlux() and ComputeFluxEnergy().
       @param sol      The solution field whose error is to be estimated.
       @param flux_fes The ZienkiewiczZhuEstimator assumes ownership of this
                       FiniteElementSpace and will call its Update() method when
                       needed. */
   ZienkiewiczZhuEstimator(BilinearFormIntegrator &integ, GridFunction &sol,
                           FiniteElementSpace *flux_fes)
      : current_sequence(-1),
        total_error(),
        integ(&integ),
        solution(&sol),
        flux_space(flux_fes)
   { }

   /// @brief Get a Vector with all element errors.
   virtual const Vector &GetLocalErrors()
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return error_estimates;
   }

   /** @brief Destroy a ZienkiewiczZhuEstimator object. Destroys the owned
       FiniteElementSpace, flux_space. */
   virtual ~ZienkiewiczZhuEstimator()
   {
      delete flux_space;
   }
};

} // namespace mfem

#endif // MFEM_ERROR_ESTIMATORS
