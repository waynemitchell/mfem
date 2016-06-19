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


/** @brief The AnisotropicErrorEstimator class is the base class for all error
    estimators that compute one non-negative real (double) number and
    anisotropic flag for every element in the Mesh.
 */
class AnisotropicErrorEstimator : public IsotropicErrorEstimator
{
public:
   /// @brief Get a Vector with all element errors.
   virtual const Vector &GetLocalErrors() = 0;

   /** @brief Get an Array<int> with anisotropic flags for all mesh elements.
       Return an empty array when anisotropic estimates are not available or
       enabled. */
   virtual const Array<int> &GetAnisotropicFlags() = 0;

   virtual ~AnisotropicErrorEstimator() { }
};


/** @brief The ZienkiewiczZhuEstimator class implements the Zienkiewicz-Zhu
    error estimation procedure.

    The required BilinearFormIntegrator must implement the methods
    ComputeElementFlux() and ComputeFluxEnergy().
 */
class ZienkiewiczZhuEstimator : public AnisotropicErrorEstimator
{
protected:
   long current_sequence;
   Vector error_estimates;
   double total_error;
   bool anisotropic;
   Array<int> aniso_flags;

   BilinearFormIntegrator *integ; ///< Not owned.
   GridFunction *solution; ///< Not owned.

   FiniteElementSpace *flux_space; /**< @brief Ownership based on own_flux_fes.
      Its Update() method is called automatically by this class when needed. */
   bool own_flux_fes; ///< Ownership flag for flux_space.

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
        anisotropic(false),
        integ(&integ),
        solution(&sol),
        flux_space(flux_fes),
        own_flux_fes(true)
   { }

   /** @brief Construct a new ZienkiewiczZhuEstimator object.
       @param integ    This BilinearFormIntegrator must implement the methods
                       ComputeElementFlux() and ComputeFluxEnergy().
       @param sol      The solution field whose error is to be estimated.
       @param flux_fes The ZienkiewiczZhuEstimator does NOT assume ownership of
                       this FiniteElementSpace; will call its Update() method
                       when needed. */
   ZienkiewiczZhuEstimator(BilinearFormIntegrator &integ, GridFunction &sol,
                           FiniteElementSpace &flux_fes)
      : current_sequence(-1),
        total_error(),
        anisotropic(false),
        integ(&integ),
        solution(&sol),
        flux_space(&flux_fes),
        own_flux_fes(false)
   { }

   /** @brief Enable/disable anisotropic estimates. To enable this option, the
       BilinearFormIntegrator must support the 'd_energy' parameter in its
       ComputeFluxEnergy() method. */
   void SetAnisotropic(bool aniso = true) { anisotropic = aniso; }

   /// @brief Get a Vector with all element errors.
   virtual const Vector &GetLocalErrors()
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return error_estimates;
   }

   /** @brief Get an Array<int> with anisotropic flags for all mesh elements.
       Return an empty array when anisotropic estimates are not available or
       enabled. */
   virtual const Array<int> &GetAnisotropicFlags()
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return aniso_flags;
   }

   /** @brief Destroy a ZienkiewiczZhuEstimator object. Destroys, if owned, the
       FiniteElementSpace, flux_space. */
   virtual ~ZienkiewiczZhuEstimator()
   {
      if (own_flux_fes) { delete flux_space; }
   }
};

} // namespace mfem

#endif // MFEM_ERROR_ESTIMATORS
