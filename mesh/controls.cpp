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

MeshControlSequence::~MeshControlSequence()
{
   // delete in reverse order
   for (int i = sequence.Size()-1; i >= 0; i--)
   {
      delete sequence[i];
   }
}

int MeshControlSequence::Apply(Mesh &mesh)
{
   if (sequence.Size() == 0) { return NONE; }
next_step:
   step = (step + 1) % sequence.Size();
   bool last = (step == sequence.Size() - 1);
   int mod = sequence[step]->Apply(mesh);
   switch (mod & ACTION)
   {
      case NONE:     if (last) { return NONE; } goto next_step;
      case CONTINUE: return last ? mod : (AGAIN | (mod & INFO));
      case STOP:     return STOP;
      case AGAIN:    --step; return mod;
   }
   return NONE;
}


ThresholdAMRControl::ThresholdAMRControl(IsotropicErrorEstimator *est)
   : estimator(est)
{
   total_norm_p = std::numeric_limits<double>::infinity();
   total_err_goal = 0.0;
   total_fraction = 0.5;
   local_err_goal = 0.0;
   max_elements = std::numeric_limits<long>::max();

   threshold = 0.0;
   num_marked_elements = 0L;

   non_conforming = -1;
   nc_limit = 0;
}

double ThresholdAMRControl::GetNorm(const Vector &local_err, Mesh &mesh) const
{
#ifndef MFEM_USE_MPI
   return local_err.Normlp(total_norm_p);
#else
   ParMesh *pmesh = dynamic_cast<ParMesh*>(&mesh);
   return pmesh ? local_err.ParNormlp(total_norm_p, pmesh->GetComm()) :
          local_err.Normlp(total_norm_p);
#endif
}

int ThresholdAMRControl::Apply(Mesh &mesh)
{
   threshold = 0.0;
   num_marked_elements = 0;

   const long num_elements = mesh.GetGlobalNE();
   if (num_elements >= max_elements) { return STOP; }

   const int NE = mesh.GetNE();
   const Vector &local_err = estimator->GetLocalErrors();
   MFEM_ASSERT(local_err.Size() == NE, "invalid size of local_err");

   double total_err = GetNorm(local_err, mesh);
   if (total_err <= total_err_goal) { return STOP; }

   threshold = std::max(total_err * total_fraction *
                        std::pow(num_elements, -1.0/total_norm_p),
                        local_err_goal);

   marked_elements.SetSize(0);
   for (int el = 0; el < NE; el++)
   {
      if (local_err(el) > threshold)
      {
         marked_elements.Append(el);
      }
   }

   num_marked_elements = mesh.ReduceInt(marked_elements.Size());
   if (num_marked_elements == 0) { return STOP; }

   mesh.GeneralRefinement(marked_elements, non_conforming, nc_limit);
   return CONTINUE + REFINE;
}


int ThresholdDerefineControl::Apply(Mesh &mesh)
{
   if (mesh.Conforming()) { return NONE; }

   const Vector &local_err = estimator->GetLocalErrors();
   bool derefs = mesh.DerefineByError(local_err, threshold, nc_limit, op);

   return derefs ? CONTINUE + DEREFINE : NONE;
}


int ThresholdDerefineControl2::Apply(Mesh &mesh)
{
   if (mesh.Conforming()) { return NONE; }

   if (stage == 0)
   {
      const Vector &local_err = estimator->GetLocalErrors();
      // use nc_limit = 0
      bool derefs = mesh.DerefineByError(local_err, threshold, 0, op);

      return derefs ?
             (nc_limit > 0 ? (++stage, AGAIN) : CONTINUE) + DEREFINE : NONE;
   }
   else
   {
      // TODO: this needs to be done through the Mesh class to make sure Mesh
      //       generates the information for Update(). Is one Update() call
      //       sufficeint here?
      // mesh.ncmesh->LimitNCLevel(nc_limit);

      long total_refined = 0; // TODO
      return total_refined ?
             CONTINUE /* or AGAIN? */ + REFINE : (stage = 0, NONE);
   }
}


int RebalanceControl::Apply(Mesh &mesh)
{
#ifdef MFEM_USE_MPI
   ParMesh *pmesh = dynamic_cast<ParMesh*>(&mesh);
   return (pmesh && pmesh->Nonconforming()) ?
          (pmesh->Rebalance(), CONTINUE + REBALANCE) : NONE;
#else
   return NONE;
#endif
}


// TODO: delete this
void Test(MeshControl *control, Mesh &mesh)
{
   for (int i = 0; i < 100; i++)
   {
      // computations ...
      while (control->Update(mesh))
      {
         // update FiniteElementSpaces and GridFunctions
         if (control->Continue()) { break; }
      }
      if (control->Stop()) { break; }
   }
}

} // namespace mfem
