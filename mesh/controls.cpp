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

#include "controls.hpp"
#include "pmesh.hpp"

namespace mfem
{

ThresholdAMRMarker::ThresholdAMRMarker(Mesh &m, IsotropicErrorEstimator &est)
   : mesh (m), estimator(est)
{
   aniso_estimator = dynamic_cast<AnisotropicErrorEstimator*>(&estimator);
   total_norm_p = std::numeric_limits<double>::infinity();
   total_err_goal = 0.0;
   total_fraction = 0.5;
   local_err_goal = 0.0;
   max_elements = std::numeric_limits<long>::max();

   threshold = 0.0;
   num_marked_elements = 0L;
   current_sequence = -1;
}

// protected method
double ThresholdAMRMarker::GetNorm(const Vector &local_err) const
{
#ifndef MFEM_USE_MPI
   return local_err.Normlp(total_norm_p);
#else
   ParMesh *pmesh = dynamic_cast<ParMesh*>(&mesh);
   return pmesh ? local_err.ParNormlp(total_norm_p, pmesh->GetComm()) :
          local_err.Normlp(total_norm_p);
#endif
}

// protected method
void ThresholdAMRMarker::MarkElements()
{
   threshold = 0.0;
   num_marked_elements = 0;
   marked_elements.SetSize(0);
   current_sequence = mesh.GetSequence();

   const long num_elements = mesh.GetGlobalNE();
   if (num_elements >= max_elements) { return; }

   const int NE = mesh.GetNE();
   const Vector &local_err = estimator.GetLocalErrors();
   MFEM_ASSERT(local_err.Size() == NE, "invalid size of local_err");

   double total_err = GetNorm(local_err);
   if (total_err <= total_err_goal) { return; }

   threshold = std::max(total_err * total_fraction *
                        std::pow(num_elements, -1.0/total_norm_p),
                        local_err_goal);

   for (int el = 0; el < NE; el++)
   {
      if (local_err(el) > threshold)
      {
         marked_elements.Append(Refinement(el));
      }
   }
   if (aniso_estimator)
   {
      const Array<int> &aniso_flags = aniso_estimator->GetAnisotropicFlags();
      if (aniso_flags.Size() > 0)
      {
         for (int i = 0; i < marked_elements.Size(); i++)
         {
            Refinement &ref = marked_elements[i];
            ref.ref_type = aniso_flags[ref.index];
         }
      }
   }

   num_marked_elements = mesh.ReduceInt(marked_elements.Size());
}

void ThresholdAMRMarker::Reset()
{
   estimator.Reset();
   current_sequence = -1;
   num_marked_elements = 0;
   // marked_elements.SetSize(0); // not necessary
}


MeshControlSequence::~MeshControlSequence()
{
   // delete in reverse order
   for (int i = sequence.Size()-1; i >= 0; i--)
   {
      delete sequence[i];
   }
}

int MeshControlSequence::ApplyImpl(Mesh &mesh)
{
   if (sequence.Size() == 0) { return NONE; }
next_step:
   step = (step + 1) % sequence.Size();
   bool last = (step == sequence.Size() - 1);
   int mod = sequence[step]->ApplyImpl(mesh);
   switch (mod & MASK_ACTION)
   {
      case NONE:     if (last) { return NONE; } goto next_step;
      case CONTINUE: return last ? mod : (AGAIN | (mod & MASK_INFO));
      case STOP:     return STOP;
      case AGAIN:    --step; return mod;
   }
   return NONE;
}

void MeshControlSequence::Reset()
{
   for (int i = 0; i < sequence.Size(); i++)
   {
      sequence[i]->Reset();
   }
}


RefinementControl::RefinementControl(MeshMarker &mm)
   : marker(mm)
{
   non_conforming = -1;
   nc_limit = 0;
}

int RefinementControl::ApplyImpl(Mesh &mesh)
{
   const Array<Refinement> &marked_el = marker.GetMarkedElements();

   if (marker.GetNumMarkedElements() == 0) { return STOP; }

   mesh.GeneralRefinement(marked_el, non_conforming, nc_limit);
   return CONTINUE + REFINED;
}


int ThresholdDerefineControl::ApplyImpl(Mesh &mesh)
{
   if (mesh.Conforming()) { return NONE; }

   const Vector &local_err = estimator->GetLocalErrors();
   bool derefs = mesh.DerefineByError(local_err, threshold, nc_limit, op);

   return derefs ? CONTINUE + DEREFINED : NONE;
}


int ThresholdDerefineControl2::ApplyImpl(Mesh &mesh)
{
   if (mesh.Conforming()) { return NONE; }

   if (stage == 0)
   {
      const Vector &local_err = estimator->GetLocalErrors();
      // use nc_limit = 0
      bool derefs = mesh.DerefineByError(local_err, threshold, 0, op);

      return derefs ?
             (nc_limit > 0 ? (++stage, AGAIN) : CONTINUE) + DEREFINED : NONE;
   }
   else
   {
      // TODO: this needs to be done through the Mesh class to make sure Mesh
      //       generates the information for Update(). Is one Update() call
      //       sufficeint here?
      // mesh.ncmesh->LimitNCLevel(nc_limit);

      long total_refined = 0; // TODO
      return total_refined ?
             CONTINUE /* or AGAIN? */ + REFINED : (stage = 0, NONE);
   }
}


int RebalanceControl::ApplyImpl(Mesh &mesh)
{
#ifdef MFEM_USE_MPI
   ParMesh *pmesh = dynamic_cast<ParMesh*>(&mesh);
   return (pmesh && pmesh->Nonconforming()) ?
          (pmesh->Rebalance(), CONTINUE + REBALANCED) : NONE;
#else
   return NONE;
#endif
}


} // namespace mfem
