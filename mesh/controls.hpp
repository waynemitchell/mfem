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

#ifndef MFEM_MESH_CONTROLS
#define MFEM_MESH_CONTROLS

#include "../config/config.hpp"
#include "../general/array.hpp"
#include "mesh.hpp"
#include "../fem/estimators.hpp"

#include <limits>

namespace mfem
{

class MeshMarker
{
 protected:
   long num_marked_elements;

public:
   /// @brief Construct a MeshMarker.
   MeshMarker() : num_marked_elements(0) { }

   /// @brief Return a list with all marked elements.
   virtual const Array<Refinement> &GetMarkedElements() = 0;

   /// @brief Reset the marked elements.
   virtual void Reset() = 0;

   /// @brief Destroy a MeshMarker object.
   virtual ~MeshMarker() { }

   /// @brief Get the global number of marked elements.
   long GetNumMarkedElements() const { return num_marked_elements; }
};


/** @brief MeshMarker based on IsotropicErrorEstimator using an error threshold.

    This class uses the given IsotropicErrorEstimator to estimate the local
    element errors and then marks for refinement all elements i such that
    loc_err_i > threshold. The threshold is computed as
    \code
       threshold = max(total_err * total_fraction * pow(num_elements,-1.0/p),
                       local_err_goal);
    \endcode
    where p (=total_norm_p), total_fraction, and local_err_goal are settable
    parameters, total_err = (sum_i local_err_i^p)^{1/p}, when p < inf,
    or total_err = max_i local_err_i, when p = inf. */
class ThresholdAMRMarker : public MeshMarker
{
protected:
   Mesh &mesh;
   IsotropicErrorEstimator &estimator;
   AnisotropicErrorEstimator *aniso_estimator;

   double total_norm_p;
   double total_err_goal;
   double total_fraction;
   double local_err_goal;
   long   max_elements;

   double threshold;
   Array<Refinement> marked_elements;
   long current_sequence;

   double GetNorm(const Vector &local_err) const;

   void MarkElements();

public:
   /** @brief Construct a ThresholdAMRMarker using the given
       IsotropicErrorEstimator. */
   ThresholdAMRMarker(Mesh &m, IsotropicErrorEstimator &est);

   // default destructor (virtual)

   /** @brief Set the exponent, p, of the discrete p-norm used to compute the
       total error from the local element errors. */
   void SetTotalErrorNormP(double norm_p =
                              std::numeric_limits<double>::infinity())
   { total_norm_p = norm_p; }

   /** @brief Set the total error stopping criterion: stop when
       total_err <= total_err_goal. The default value is zero. */
   void SetTotalErrorGoal(double err_goal) { total_err_goal = err_goal; }

   /** @brief Set the total fraction used in the computation of the threshold.
       The default value is 1/2.
       @note If fraction == 0, total_err is essentially ignored in the threshold
       computation, i.e. threshold = local error goal. */
   void SetTotalErrorFraction(double fraction) { total_fraction = fraction; }

   /** @brief Set the local stopping criterion: stop when
       local_err_i <= local_err_goal. The default value is zero.
       @note If local_err_goal == 0, it is essentially ignored in the threshold
       computation. */
   void SetLocalErrorGoal(double err_goal) { local_err_goal = err_goal; }

   /** @brief Set the maximum number of elements stoppin criterion: stop when
       the input mesh has num_elements >= max_elem. The default value is
       LONG_MAX. */
   void SetMaxElements(long max_elem) { max_elements = max_elem; }

   /// @brief Return a list with all marked elements.
   virtual const Array<Refinement> &GetMarkedElements()
   {
      MFEM_ASSERT(current_sequence <= mesh.GetSequence(), "");
      if (current_sequence < mesh.GetSequence()) { MarkElements(); }
      return marked_elements;
   }

   /// @brief Get the last threshold used for marking.
   double GetThreshold() const { return threshold; }

   /// @brief Reset the marked elements and the estimator.
   void Reset();
};


/** @brief The MeshControl class serves as base for Mesh manipulation classes.

    The main purpose of the class is to provide a common abstraction
    for various AMR mesh control schemes. The typical use in an AMR loop is:
    \code
       for (...)
       {
          // computations ...
          while (control->Apply(mesh))
          {
             // update FiniteElementSpaces and GridFunctions
             if (control->Continue()) { break; }
          }
          if (control->Stop()) { break; }
       }
    \endcode
 */
class MeshControl
{
private:
   int mod;

protected:
   friend class MeshControlSequence;

   /** @brief Implementation of the mesh operation. Invoked by the Apply()
              public method.
       @return Combination of ActionInfo constants. */
   virtual int ApplyImpl(Mesh &mesh) = 0;

   /// @brief Constructor to be used by derived classes.
   MeshControl() : mod(NONE) { }

public:
   /** @brief Action and information constants and masks.

       Combinations of constants are returned by the Apply() virtual method and
       can be accessed directly with GetActionInfo() or indirectly with methods
       like Stop(), Continue(), etc. The information bits (INFO mask) can be set
       only when the UPDATE bit is set. */
   enum Action
   {
      NONE        = 0, /**< continue with computations without updating spaces
                            or grid-functions, i.e. the mesh was not modified */
      CONTINUE    = 1, /**< update spaces and grid-functions and continue
                            computations with the new mesh */
      STOP        = 2, ///< a stopping criterion was satisfied
      AGAIN       = 3, /**< update spaces and grid-functions and call the
                            control Update() method again */
      MASK_UPDATE = 1, ///< bit mask for the "update" bit
      MASK_ACTION = 3  ///< bit mask for all "action" bits
   };

   enum Info
   {
      REFINED     = 4*1, ///< the mesh was refined
      DEREFINED   = 4*2, ///< the mesh was de-refined
      REBALANCED  = 4*3, ///< the mesh was rebalanced
      MASK_INFO   = ~3   ///< bit mask for all "info" bits
   };

   /** @brief Perform the mesh operation.
       @return true if FiniteElementSpaces and GridFunctions need to be updated.
   */
   bool Apply(Mesh &mesh) { return ((mod = ApplyImpl(mesh)) & MASK_UPDATE); }

   /** @brief Check if STOP action is requested, e.g. stopping criterion is
       satisfied. */
   bool Stop() const { return ((mod & MASK_ACTION) == STOP); }
   /** @brief Check if AGAIN action is requested, i.e. FiniteElementSpaces and
       GridFunctions need to be updated, and Update() must be called again. */
   bool Again() const { return ((mod & MASK_ACTION) == AGAIN); }
   /** @brief Check if CONTINUE action is requested, i.e. FiniteElementSpaces
       and GridFunctions need to be updated and computations should continue. */
   bool Continue() const { return ((mod & MASK_ACTION) == CONTINUE); }

   /// @brief Check if the mesh was refined.
   bool Refined() const { return ((mod & MASK_INFO) == REFINED); }
   /// @brief Check if the mesh was de-refined.
   bool Derefined() const { return ((mod & MASK_INFO) == DEREFINED); }
   /// @brief Check if the mesh was rebalanced.
   bool Rebalanced() const { return ((mod & MASK_INFO) == REBALANCED); }

   /** @brief Get the full ActionInfo value generated by the last call to
       Update(). */
   int GetActionInfo() const { return mod; }

   /// @brief Reset the MeshControl.
   virtual void Reset() = 0;

   /// @brief The destructor is virtual.
   virtual ~MeshControl() { }
};


/** Composition of MeshControls into a sequence. Use the Append() method to
    create the sequence. */
class MeshControlSequence : public MeshControl
{
protected:
   int step;
   Array<MeshControl*> sequence; ///< MeshControls sequence, owned by us.

   /// @brief Do not allow copy construction, due to assumed ownership.
   MeshControlSequence(const MeshControlSequence &) { }

   /** @brief Apply the MeshControlSequence.
       @return ActionInfo value corresponding to the last applied control from
       the sequence. */
   virtual int ApplyImpl(Mesh &mesh);

public:
   /// @brief Constructor. Use the Append() method to create the sequence.
   MeshControlSequence() : step(-1) { }

   /// @brief Delete all controls from the sequence.
   virtual ~MeshControlSequence();

   /** @brief Add a control to the end of the sequence. The MeshControlSequence
       assumes ownership of the control. */
   void Append(MeshControl *mc) { sequence.Append(mc); }

   /// @brief Access the underlying sequence.
   Array<MeshControl*> &GetSequence() { return sequence; }

   /// @brief Reset all MeshControls in the sequence.
   virtual void Reset();
};


/** @brief Refinement control using a MeshMarker.

    This class uses the given MeshMarker to mark elements and then calls the
    mesh method GeneralRefinement() to perform the refinements. */
class RefinementControl : public MeshControl
{
protected:
   MeshMarker &marker;
   int non_conforming;
   int nc_limit;

   /** @brief Apply the RefinementControl.
       @return STOP if a stopping criterion is satisfied or no elements were
       marked for refinement; REFINED + CONTINUE otherwise. */
   virtual int ApplyImpl(Mesh &mesh);

public:
   /** @brief Construct a ThresholdAMRControl using the given
       IsotropicErrorEstimator. */
   RefinementControl(MeshMarker &mm);

   // default destructor (virtual)

   /// @brief Use nonconforming refinement, if possible.
   void SetNonconformingRefinement(int nc_limit = 0)
   {
      non_conforming = 1;
      this->nc_limit = nc_limit;
   }

   /// @brief Use conforming refinement, if possible (this is the default).
   void SetConformingRefinement(int nc_limit = 0)
   {
      non_conforming = -1;
      this->nc_limit = nc_limit;
   }

   /// @brief Reset the associated MeshMarker.
   virtual void Reset() { marker.Reset(); }
};


/** @brief De-refinement control using an error threshold.

    This de-refinement control marks elements in the hierarchy whose children
    are leaves and their combined error is below a given threshold. The
    errors of the children are combined by one of the following operations:
    - op = 0: minimum of the errors
    - op = 1: sum of the errors (default)
    - op = 2: maximum of the errors. */
class ThresholdDerefineControl : public MeshControl
{
protected:
   IsotropicErrorEstimator *estimator; ///< Not owned.

   double threshold;
   int nc_limit, op;

   /** @brief Apply the ThresholdDerefineControl.
       @return DEREFINED + CONTINUE if some elements were de-refined; NONE
       otherwise. */
   virtual int ApplyImpl(Mesh &mesh);

public:
   /** @brief Construct a ThresholdDerefineControl using the given
       IsotropicErrorEstimator. */
   ThresholdDerefineControl(IsotropicErrorEstimator *est)
      : estimator(est)
   {
      threshold = 0.0;
      nc_limit = 0;
      op = 1;
   }

   // default destructor (virtual)

   /// @brief Set the de-refinement threshold. The default value is zero.
   void SetThreshold(double thresh) { threshold = thresh; }
   void SetOp(int op) { this->op = op; }
   void SetNCLimit(int nc_lim) { nc_limit = nc_lim; }

   /// @brief Reset the associated estimator.
   virtual void Reset() { estimator->Reset(); }
};


/** @brief De-refinement control using an error threshold.

    Similar to class ThresholdDerefineControl, the only difference is
    the way the 'nc_limit' is enforced: this control performs all marked de-
    refinements followed by refinements to ensure the required 'nc_limit'.
*/
class ThresholdDerefineControl2 : public ThresholdDerefineControl
{
protected:
   int stage; // 0 - de-refine, 1 - limit NC level

   /** @brief Apply the ThresholdDerefineControl.
       @return DEREFINE + CONTINUE if some elements were de-refined; NONE
       otherwise. */
   virtual int ApplyImpl(Mesh &mesh);

public:
   /** @brief Construct a ThresholdDerefineControl2 using the given
       IsotropicErrorEstimator. */
   ThresholdDerefineControl2(IsotropicErrorEstimator *est)
      : ThresholdDerefineControl(est), stage(0)
   { }

   // default destructor (virtual)

   /// @brief Reset the associated estimator.
   virtual void Reset() { estimator->Reset(); stage = 0; }
};


/** @brief ParMesh rebalancing control.

    If the mesh is a parallel mesh, perform rebalancing; otherwise, do nothing.
*/
class RebalanceControl : public MeshControl
{
protected:
   /** @brief Rebalance a parallel mesh (only non-conforming parallel meshes are
       supported).
       @return CONTINUE + REBALANCE on success, NONE otherwise. */
   virtual int ApplyImpl(Mesh &mesh);

public:
   /// @brief Empty.
   virtual void Reset() { }
};

} // namespace mfem

#endif // MFEM_MESH_CONTROLS
