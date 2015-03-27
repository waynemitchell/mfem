// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "fem.hpp"
#include "../general/sort_pairs.hpp"

namespace mfem
{

ParFiniteElementSpace::ParFiniteElementSpace(ParFiniteElementSpace &pf)
   : FiniteElementSpace(pf)
{
   MyComm = pf.MyComm;
   NRanks = pf.NRanks;
   MyRank = pf.MyRank;
   pmesh = pf.pmesh;
   gcomm = pf.gcomm;
   pf.gcomm = NULL;
   ltdof_size = pf.ltdof_size;
   ltexdof_size = pf.ltexdof_size;
   Swap(ldof_group, pf.ldof_group);
   Swap(lexdof_group, pf.lexdof_group);
   Swap(ldof_ltdof, pf.ldof_ltdof);
   Swap(lexdof_ltexdof, pf.lexdof_ltexdof);
   Swap(dof_offsets, pf.dof_offsets);
   Swap(tdof_offsets, pf.tdof_offsets);
   Swap(exdof_offsets, pf.exdof_offsets);
   Swap(texdof_offsets, pf.texdof_offsets);
   Swap(tdof_nb_offsets, pf.tdof_nb_offsets);
   Swap(texdof_nb_offsets, pf.texdof_nb_offsets);
   Swap(ldof_sign, pf.ldof_sign);
   P = pf.P;
   pf.P = NULL;
   num_face_nbr_dofs = pf.num_face_nbr_dofs;
   pf.num_face_nbr_dofs = -1;
   Swap<Table>(face_nbr_element_dof, pf.face_nbr_element_dof);
   Swap<Table>(face_nbr_ldof, pf.face_nbr_ldof);
   Swap(face_nbr_glob_dof_map, pf.face_nbr_glob_dof_map);
   Swap<Table>(send_face_nbr_ldof, pf.send_face_nbr_ldof);
}

ParFiniteElementSpace::ParFiniteElementSpace(ParMesh *pm,
					     const FiniteElementCollection *f,
					     int dim, int order, bool pr_dofs)
  : FiniteElementSpace(pm, f, dim, order, pr_dofs)
{
   mesh = pmesh = pm;

   MyComm = pmesh->GetComm();
   MPI_Comm_size(MyComm, &NRanks);
   MPI_Comm_rank(MyComm, &MyRank);

   P = NULL;

   if (NURBSext)
   {
      if (own_ext)
      {
         // the FiniteElementSpace constructor created a serial
         // NURBSExtension of higher order than the mesh NURBSExtension

         ParNURBSExtension *pNe = new ParNURBSExtension(
            NURBSext, dynamic_cast<ParNURBSExtension *>(pmesh->NURBSext));
         // serial NURBSext is destroyed by the above constructor
         NURBSext = pNe;
         UpdateNURBS();
      }

      ConstructTrueNURBSDofs();
   }
   else
   {
      ConstructTrueDofs();
   }

   GenerateGlobalOffsets();

   num_face_nbr_dofs = -1;
}

void ParFiniteElementSpace::GetGroupComm(
   GroupCommunicator &gc, int ldof_type, Array<int> *ldof_sign)
{
   int gr;
   int ng = pmesh->GetNGroups();
   int nvd, ned, nfd;
   Array<int> dofs;

   int group_ldof_counter;
   Table &group_ldof = gc.GroupLDofTable();

   nvd = fec->DofForGeometry(Geometry::POINT);
   ned = fec->DofForGeometry(Geometry::SEGMENT);
   nfd = (fdofs) ? (fdofs[1]-fdofs[0]) : (0);

   if (ldof_sign)
   {
      ldof_sign->SetSize(GetNDofs());
      *ldof_sign = 1;
   }

   // count the number of ldofs in all groups (excluding the local group 0)
   group_ldof_counter = 0;
   for (gr = 1; gr < ng; gr++)
   {
      group_ldof_counter += nvd * pmesh->GroupNVertices(gr);
      group_ldof_counter += ned * pmesh->GroupNEdges(gr);
      group_ldof_counter += nfd * pmesh->GroupNFaces(gr);
   }
   if (ldof_type)
   {
      group_ldof_counter *= vdim;
   }
   // allocate the I and J arrays in group_ldof
   group_ldof.SetDims(ng, group_ldof_counter);

   // build the full group_ldof table
   group_ldof_counter = 0;
   group_ldof.GetI()[0] = group_ldof.GetI()[1] = 0;
   for (gr = 1; gr < ng; gr++)
   {
      int j, k, l, m, o, nv, ne, nf;
      const int *ind;

      nv = pmesh->GroupNVertices(gr);
      ne = pmesh->GroupNEdges(gr);
      nf = pmesh->GroupNFaces(gr);

      // vertices
      if (nvd > 0)
         for (j = 0; j < nv; j++)
         {
            k = pmesh->GroupVertex(gr, j);

            dofs.SetSize(nvd);
            m = nvd * k;
            for (l = 0; l < nvd; l++, m++)
            {
               dofs[l] = m;
            }

            if (ldof_type)
            {
               DofsToVDofs(dofs);
            }

            for (l = 0; l < dofs.Size(); l++)
            {
               group_ldof.GetJ()[group_ldof_counter++] = dofs[l];
            }
         }

      // edges
      if (ned > 0)
         for (j = 0; j < ne; j++)
         {
            pmesh->GroupEdge(gr, j, k, o);

            dofs.SetSize(ned);
            m = nvdofs+k*ned;
            ind = fec->DofOrderForOrientation(Geometry::SEGMENT, o);
            for (l = 0; l < ned; l++)
               if (ind[l] < 0)
               {
                  dofs[l] = m + (-1-ind[l]);
                  if (ldof_sign)
                  {
                     (*ldof_sign)[dofs[l]] = -1;
                  }
               }
               else
               {
                  dofs[l] = m + ind[l];
               }

            if (ldof_type)
            {
               DofsToVDofs(dofs);
            }

            for (l = 0; l < dofs.Size(); l++)
            {
               group_ldof.GetJ()[group_ldof_counter++] = dofs[l];
            }
         }

      // faces
      if (nfd > 0)
         for (j = 0; j < nf; j++)
         {
            pmesh->GroupFace(gr, j, k, o);

            dofs.SetSize(nfd);
            m = nvdofs+nedofs+fdofs[k];
            ind = fec->DofOrderForOrientation(
                     mesh->GetFaceBaseGeometry(k), o);
            for (l = 0; l < nfd; l++)
               if (ind[l] < 0)
               {
                  dofs[l] = m + (-1-ind[l]);
                  if (ldof_sign)
                  {
                     (*ldof_sign)[dofs[l]] = -1;
                  }
               }
               else
               {
                  dofs[l] = m + ind[l];
               }

            if (ldof_type)
            {
               DofsToVDofs(dofs);
            }

            for (l = 0; l < dofs.Size(); l++)
            {
               group_ldof.GetJ()[group_ldof_counter++] = dofs[l];
            }
         }

      group_ldof.GetI()[gr+1] = group_ldof_counter;
   }

   gc.Finalize();
}

void ParFiniteElementSpace::ApplyLDofSigns(Array<int> &dofs) const
{
   for (int i = 0; i < dofs.Size(); i++)
      if (dofs[i] < 0)
      {
         if (ldof_sign[-1-dofs[i]] < 0)
         {
            dofs[i] = -1-dofs[i];
         }
      }
      else
      {
         if (ldof_sign[dofs[i]] < 0)
         {
            dofs[i] = -1-dofs[i];
         }
      }
}

void ParFiniteElementSpace::GetElementDofs(int i, Array<int> &dofs) const
{
   if (elem_dof)
   {
      elem_dof->GetRow(i, dofs);
      return;
   }
   FiniteElementSpace::GetElementDofs(i, dofs);
   ApplyLDofSigns(dofs);
}

void ParFiniteElementSpace::GetBdrElementDofs(int i, Array<int> &dofs) const
{
   if (bdrElem_dof)
   {
      bdrElem_dof->GetRow(i, dofs);
      return;
   }
   FiniteElementSpace::GetBdrElementDofs(i, dofs);
   ApplyLDofSigns(dofs);
}

void ParFiniteElementSpace::GetFaceDofs(int i, Array<int> &dofs) const
{
   FiniteElementSpace::GetFaceDofs(i, dofs);
   ApplyLDofSigns(dofs);
}

void ParFiniteElementSpace::GenerateGlobalOffsets()
{
   if (HYPRE_AssumedPartitionCheck())
   {
      HYPRE_Int ldof[4];

      ldof[0] = GetVSize();
      ldof[1] = TrueVSize();
      ldof[2] = GetNPrDofs();

      dof_offsets.SetSize(3);
      tdof_offsets.SetSize(3);

      exdof_offsets.SetSize(3);
      texdof_offsets.SetSize(3);

      MPI_Scan(ldof, &dof_offsets[0], 3, HYPRE_MPI_INT, MPI_SUM, MyComm);

      tdof_offsets[1] = dof_offsets[1];
      tdof_offsets[0] = tdof_offsets[1] - ldof[1];

      dof_offsets[1] = dof_offsets[0];
      dof_offsets[0] = dof_offsets[1] - ldof[0];

      texdof_offsets[1] = tdof_offsets[1] - dof_offsets[2];
      texdof_offsets[0] = texdof_offsets[1] - (ldof[1] - vdim*ldof[2]);

      exdof_offsets[1] = dof_offsets[1] - dof_offsets[2];
      exdof_offsets[0] = exdof_offsets[1] - (ldof[0] - vdim*ldof[2]);

      // get the global sizes in (t)dof_offsets[2]
      if (MyRank == NRanks-1)
      {
         ldof[0] = dof_offsets[1];
         ldof[1] = tdof_offsets[1];
         ldof[2] = exdof_offsets[1];
         ldof[3] = texdof_offsets[1];
      }

      MPI_Bcast(ldof, 4, HYPRE_MPI_INT, NRanks-1, MyComm);
      dof_offsets[2] = ldof[0];
      tdof_offsets[2] = ldof[1];
      exdof_offsets[2] = ldof[2];
      texdof_offsets[2] = ldof[3];

      // Check for overflow
      MFEM_VERIFY(dof_offsets[0] >= 0 && dof_offsets[1] >= 0,
                  "overflow in global dof_offsets");
      MFEM_VERIFY(tdof_offsets[0] >= 0 && tdof_offsets[1] >= 0,
                  "overflow in global tdof_offsets");
      MFEM_VERIFY(exdof_offsets[0] >= 0 && exdof_offsets[1] >= 0,
                  "overflow in global exdof_offsets");
      MFEM_VERIFY(texdof_offsets[0] >= 0 && texdof_offsets[1] >= 0,
                  "overflow in global texdof_offsets");

      // Communicate the neighbor offsets in tdof_nb_offsets
      GroupTopology &gt = GetGroupTopo();
      int nsize = gt.GetNumNeighbors()-1;
      MPI_Request *requests = new MPI_Request[4*nsize];
      MPI_Status  *statuses = new MPI_Status[4*nsize];
      tdof_nb_offsets.SetSize(nsize+1);
      tdof_nb_offsets[0] = tdof_offsets[0];

      texdof_nb_offsets.SetSize(nsize+1);
      texdof_nb_offsets[0] = texdof_offsets[0];

      int request_counter = 0;
      // send and receive neighbors' local tdof offsets
      for (int i = 1; i <= nsize; i++)
         MPI_Irecv(&tdof_nb_offsets[i], 1, HYPRE_MPI_INT, gt.GetNeighborRank(i),
                   5365, MyComm, &requests[request_counter++]);

      for (int i = 1; i <= nsize; i++)
         MPI_Isend(&tdof_nb_offsets[0], 1, HYPRE_MPI_INT, gt.GetNeighborRank(i),
                   5365, MyComm, &requests[request_counter++]);

      // send and receive neighbors' local texdof offsets
      for (int i = 1; i <= nsize; i++)
         MPI_Irecv(&texdof_nb_offsets[i], 1, HYPRE_MPI_INT, gt.GetNeighborRank(i),
                   5365, MyComm, &requests[request_counter++]);

      for (int i = 1; i <= nsize; i++)
         MPI_Isend(&texdof_nb_offsets[0], 1, HYPRE_MPI_INT, gt.GetNeighborRank(i),
                   5365, MyComm, &requests[request_counter++]);

      MPI_Waitall(request_counter, requests, statuses);

      delete [] statuses;
      delete [] requests;
   }
   else
   {
      int i;
      HYPRE_Int ldof  = GetVSize();
      HYPRE_Int ltdof = TrueVSize();
      HYPRE_Int lexdof  = GetExVSize();
      HYPRE_Int ltexdof = TrueExVSize();

      dof_offsets.SetSize (NRanks+1);
      tdof_offsets.SetSize(NRanks+1);
      exdof_offsets.SetSize (NRanks+1);
      texdof_offsets.SetSize(NRanks+1);

      MPI_Allgather(&ldof, 1, HYPRE_MPI_INT,
                    &dof_offsets[1], 1, HYPRE_MPI_INT, MyComm);
      MPI_Allgather(&ltdof, 1, HYPRE_MPI_INT,
                    &tdof_offsets[1], 1, HYPRE_MPI_INT, MyComm);
      MPI_Allgather(&lexdof, 1, HYPRE_MPI_INT,
                    &exdof_offsets[1], 1, HYPRE_MPI_INT, MyComm);
      MPI_Allgather(&ltexdof, 1, HYPRE_MPI_INT,
                    &texdof_offsets[1], 1, HYPRE_MPI_INT, MyComm);

      dof_offsets[0] = tdof_offsets[0] = 0;
      exdof_offsets[0] = texdof_offsets[0] = 0;
      for (i = 1; i < NRanks; i++)
      {
         dof_offsets [i+1] += dof_offsets [i];
         tdof_offsets[i+1] += tdof_offsets[i];
         exdof_offsets [i+1] += exdof_offsets [i];
         texdof_offsets[i+1] += texdof_offsets[i];
      }

      // Check for overflow
      MFEM_VERIFY(dof_offsets[MyRank] >= 0 && dof_offsets[MyRank+1] >= 0,
                  "overflow in global dof_offsets");
      MFEM_VERIFY(tdof_offsets[MyRank] >= 0 && tdof_offsets[MyRank+1] >= 0,
                  "overflow in global tdof_offsets");
      MFEM_VERIFY(exdof_offsets[MyRank] >= 0 && exdof_offsets[MyRank+1] >= 0,
                  "overflow in global exdof_offsets");
      MFEM_VERIFY(texdof_offsets[MyRank] >= 0 && texdof_offsets[MyRank+1] >= 0,
                  "overflow in global texdof_offsets");
   }
}

HypreParMatrix *ParFiniteElementSpace::Dof_TrueDof_Matrix() // matrix P
{
   if (P)
   {
      return P;
   }

   int ldof  = GetVSize();
   int ltdof = TrueVSize();

   HYPRE_Int *i_diag = new HYPRE_Int[ldof+1];
   HYPRE_Int *j_diag = new HYPRE_Int[ltdof];
   int diag_counter;

   HYPRE_Int *i_offd = new HYPRE_Int[ldof+1];
   HYPRE_Int *j_offd = new HYPRE_Int[ldof-ltdof];
   int offd_counter;

   HYPRE_Int *cmap   = new HYPRE_Int[ldof-ltdof];

   HYPRE_Int *col_starts = GetTrueDofOffsets();
   HYPRE_Int *row_starts = GetDofOffsets();

   Array<Pair<HYPRE_Int, int> > cmap_j_offd(ldof-ltdof);

   i_diag[0] = i_offd[0] = 0;
   diag_counter = offd_counter = 0;
   for (int i = 0; i < ldof; i++)
   {
      int ltdof = GetLocalTDofNumber(i);
      if (ltdof >= 0)
      {
         j_diag[diag_counter++] = ltdof;
      }
      else
      {
         cmap_j_offd[offd_counter].one = GetGlobalTDofNumber(i);
         cmap_j_offd[offd_counter].two = offd_counter;
         offd_counter++;
      }
      i_diag[i+1] = diag_counter;
      i_offd[i+1] = offd_counter;
   }

   SortPairs<HYPRE_Int, int>(cmap_j_offd, offd_counter);

   for (int i = 0; i < offd_counter; i++)
   {
      cmap[i] = cmap_j_offd[i].one;
      j_offd[cmap_j_offd[i].two] = i;
   }

   P = new HypreParMatrix(MyComm, MyRank, NRanks, row_starts, col_starts,
                          i_diag, j_diag, i_offd, j_offd, cmap, offd_counter);

   return P;
}

HypreParMatrix *ParFiniteElementSpace::ExDof_TrueExDof_Matrix() // matrix Pex
{
   if (Pex)
   {
      return Pex;
   }

   int ldof  = GetExVSize();
   int ltdof = TrueExVSize();

   HYPRE_Int *i_diag = new HYPRE_Int[ldof+1];
   HYPRE_Int *j_diag = new HYPRE_Int[ltdof];
   int diag_counter;

   HYPRE_Int *i_offd = new HYPRE_Int[ldof+1];
   HYPRE_Int *j_offd = new HYPRE_Int[ldof-ltdof];
   int offd_counter;

   HYPRE_Int *cmap   = new HYPRE_Int[ldof-ltdof];

   HYPRE_Int *col_starts = GetTrueExDofOffsets();
   HYPRE_Int *row_starts = GetExDofOffsets();

   Array<Pair<HYPRE_Int, int> > cmap_j_offd(ldof-ltdof);

   i_diag[0] = i_offd[0] = 0;
   diag_counter = offd_counter = 0;
   for (int i = 0; i < ldof; i++)
   {
      int ltdof = GetLocalTExDofNumber(i);
      if (ltdof >= 0)
      {
         j_diag[diag_counter++] = ltdof;
      }
      else
      {
         cmap_j_offd[offd_counter].one = GetGlobalTExDofNumber(i);
         cmap_j_offd[offd_counter].two = offd_counter;
         offd_counter++;
      }
      i_diag[i+1] = diag_counter;
      i_offd[i+1] = offd_counter;
   }

   SortPairs<HYPRE_Int, int>(cmap_j_offd, offd_counter);

   for (int i = 0; i < offd_counter; i++)
   {
      cmap[i] = cmap_j_offd[i].one;
      j_offd[cmap_j_offd[i].two] = i;
   }

   Pex = new HypreParMatrix(MyComm, MyRank, NRanks, row_starts, col_starts,
			    i_diag, j_diag, i_offd, j_offd, cmap,
			    offd_counter);

   return Pex;
}

void ParFiniteElementSpace::DivideByGroupSize(double *vec)
{
   GroupTopology &gt = GetGroupTopo();

   for (int i = 0; i < ldof_group.Size(); i++)
      if (gt.IAmMaster(ldof_group[i])) // we are the master
      {
         vec[ldof_ltdof[i]] /= gt.GetGroupSize(ldof_group[i]);
      }
}

GroupCommunicator *ParFiniteElementSpace::ScalarGroupComm()
{
   GroupCommunicator *gc = new GroupCommunicator(GetGroupTopo());
   if (NURBSext)
   {
      gc->Create(pNURBSext()->ldof_group);
   }
   else
   {
      GetGroupComm(*gc, 0);
   }
   return gc;
}

void ParFiniteElementSpace::Synchronize(Array<int> &ldof_marker) const
{
   if (ldof_marker.Size() != GetVSize())
   {
      mfem_error("ParFiniteElementSpace::Synchronize");
   }

   // implement allreduce(|) as reduce(|) + broadcast
   gcomm->Reduce<int>(ldof_marker, GroupCommunicator::BitOR);
   gcomm->Bcast(ldof_marker);
}

void ParFiniteElementSpace::GetEssentialVDofs(const Array<int> &bdr_attr_is_ess,
                                              Array<int> &ess_dofs) const
{
   FiniteElementSpace::GetEssentialVDofs(bdr_attr_is_ess, ess_dofs);

   // Make sure that processors without boundary elements mark
   // their boundary dofs (if they have any).
   Synchronize(ess_dofs);
}

int ParFiniteElementSpace::GetLocalTDofNumber(int ldof)
{
   if (GetGroupTopo().IAmMaster(ldof_group[ldof]))
   {
      return ldof_ltdof[ldof];
   }
   else
   {
      return -1;
   }
}

HYPRE_Int ParFiniteElementSpace::GetGlobalTDofNumber(int ldof)
{
   if (HYPRE_AssumedPartitionCheck())
   {
      return ldof_ltdof[ldof] +
             tdof_nb_offsets[GetGroupTopo().GetGroupMaster(ldof_group[ldof])];
   }

   return ldof_ltdof[ldof] +
          tdof_offsets[GetGroupTopo().GetGroupMasterRank(ldof_group[ldof])];
}

HYPRE_Int ParFiniteElementSpace::GetGlobalScalarTDofNumber(int sldof)
{
   if (HYPRE_AssumedPartitionCheck())
   {
      if (ordering == Ordering::byNODES)
         return ldof_ltdof[sldof] +
                tdof_nb_offsets[GetGroupTopo().GetGroupMaster(
                                   ldof_group[sldof])] / vdim;
      else
         return (ldof_ltdof[sldof*vdim] +
                 tdof_nb_offsets[GetGroupTopo().GetGroupMaster(
                                    ldof_group[sldof*vdim])]) / vdim;
   }

   if (ordering == Ordering::byNODES)
      return ldof_ltdof[sldof] +
             tdof_offsets[GetGroupTopo().GetGroupMasterRank(
                             ldof_group[sldof])] / vdim;
   else
      return (ldof_ltdof[sldof*vdim] +
              tdof_offsets[GetGroupTopo().GetGroupMasterRank(
                              ldof_group[sldof*vdim])]) / vdim;
}

HYPRE_Int ParFiniteElementSpace::GetMyDofOffset()
{
   if (HYPRE_AssumedPartitionCheck())
   {
      return dof_offsets[0];
   }
   else
   {
      return dof_offsets[MyRank];
   }
}

int ParFiniteElementSpace::GetLocalTExDofNumber(int ldof)
{
   if (GetGroupTopo().IAmMaster(lexdof_group[ldof]))
   {
      return lexdof_ltexdof[ldof];
   }
   else
   {
      return -1;
   }
}

HYPRE_Int ParFiniteElementSpace::GetGlobalTExDofNumber(int ldof)
{
   if (HYPRE_AssumedPartitionCheck())
   {
      return lexdof_ltexdof[ldof] +
             texdof_nb_offsets[GetGroupTopo().GetGroupMaster(lexdof_group[ldof])];
   }

   return lexdof_ltexdof[ldof] +
          texdof_offsets[GetGroupTopo().GetGroupMasterRank(lexdof_group[ldof])];
}

HYPRE_Int ParFiniteElementSpace::GetGlobalScalarTExDofNumber(int sldof)
{
   if (HYPRE_AssumedPartitionCheck())
   {
      if (ordering == Ordering::byNODES)
         return lexdof_ltexdof[sldof] +
                texdof_nb_offsets[GetGroupTopo().GetGroupMaster(
                                   lexdof_group[sldof])] / vdim;
      else
         return (lexdof_ltexdof[sldof*vdim] +
                 texdof_nb_offsets[GetGroupTopo().GetGroupMaster(
                                    lexdof_group[sldof*vdim])]) / vdim;
   }

   if (ordering == Ordering::byNODES)
      return lexdof_ltexdof[sldof] +
             texdof_offsets[GetGroupTopo().GetGroupMasterRank(
                             lexdof_group[sldof])] / vdim;
   else
      return (lexdof_ltexdof[sldof*vdim] +
              texdof_offsets[GetGroupTopo().GetGroupMasterRank(
                              lexdof_group[sldof*vdim])]) / vdim;
}

HYPRE_Int ParFiniteElementSpace::GetMyExDofOffset()
{
   if (HYPRE_AssumedPartitionCheck())
   {
      return exdof_offsets[0];
   }
   else
   {
      return exdof_offsets[MyRank];
   }
}

void ParFiniteElementSpace::ExchangeFaceNbrData()
{
   if (num_face_nbr_dofs >= 0)
   {
      return;
   }

   pmesh->ExchangeFaceNbrData();

   int num_face_nbrs = pmesh->GetNFaceNeighbors();

   if (num_face_nbrs == 0)
   {
      num_face_nbr_dofs = 0;
      return;
   }

   MPI_Request *requests = new MPI_Request[2*num_face_nbrs];
   MPI_Request *send_requests = requests;
   MPI_Request *recv_requests = requests + num_face_nbrs;
   MPI_Status  *statuses = new MPI_Status[num_face_nbrs];

   Array<int> ldofs;
   Array<int> ldof_marker(GetVSize());
   ldof_marker = -1;

   Table send_nbr_elem_dof;

   send_nbr_elem_dof.MakeI(pmesh->send_face_nbr_elements.Size_of_connections());
   send_face_nbr_ldof.MakeI(num_face_nbrs);
   face_nbr_ldof.MakeI(num_face_nbrs);
   int *send_el_off = pmesh->send_face_nbr_elements.GetI();
   int *recv_el_off = pmesh->face_nbr_elements_offset;
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int *my_elems = pmesh->send_face_nbr_elements.GetRow(fn);
      int  num_my_elems = pmesh->send_face_nbr_elements.RowSize(fn);

      for (int i = 0; i < num_my_elems; i++)
      {
         GetElementVDofs(my_elems[i], ldofs);
         for (int j = 0; j < ldofs.Size(); j++)
            if (ldof_marker[ldofs[j]] != fn)
            {
               ldof_marker[ldofs[j]] = fn;
               send_face_nbr_ldof.AddAColumnInRow(fn);
            }
         send_nbr_elem_dof.AddColumnsInRow(send_el_off[fn] + i, ldofs.Size());
      }

      int nbr_rank = pmesh->GetFaceNbrRank(fn);
      int tag = 0;
      MPI_Isend(&send_face_nbr_ldof.GetI()[fn], 1, MPI_INT, nbr_rank, tag,
                MyComm, &send_requests[fn]);

      MPI_Irecv(&face_nbr_ldof.GetI()[fn], 1, MPI_INT, nbr_rank, tag,
                MyComm, &recv_requests[fn]);
   }

   MPI_Waitall(num_face_nbrs, recv_requests, statuses);
   face_nbr_ldof.MakeJ();

   num_face_nbr_dofs = face_nbr_ldof.Size_of_connections();

   MPI_Waitall(num_face_nbrs, send_requests, statuses);
   send_face_nbr_ldof.MakeJ();

   // send/receive the I arrays of send_nbr_elem_dof/face_nbr_element_dof,
   // respectively (they contain the number of dofs for each face-neighbor
   // element)
   face_nbr_element_dof.MakeI(recv_el_off[num_face_nbrs]);

   int *send_I = send_nbr_elem_dof.GetI();
   int *recv_I = face_nbr_element_dof.GetI();
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int nbr_rank = pmesh->GetFaceNbrRank(fn);
      int tag = 0;
      MPI_Isend(send_I + send_el_off[fn], send_el_off[fn+1] - send_el_off[fn],
                MPI_INT, nbr_rank, tag, MyComm, &send_requests[fn]);

      MPI_Irecv(recv_I + recv_el_off[fn], recv_el_off[fn+1] - recv_el_off[fn],
                MPI_INT, nbr_rank, tag, MyComm, &recv_requests[fn]);
   }

   MPI_Waitall(num_face_nbrs, send_requests, statuses);
   send_nbr_elem_dof.MakeJ();

   ldof_marker = -1;

   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int *my_elems = pmesh->send_face_nbr_elements.GetRow(fn);
      int  num_my_elems = pmesh->send_face_nbr_elements.RowSize(fn);

      for (int i = 0; i < num_my_elems; i++)
      {
         GetElementVDofs(my_elems[i], ldofs);
         for (int j = 0; j < ldofs.Size(); j++)
            if (ldof_marker[ldofs[j]] != fn)
            {
               ldof_marker[ldofs[j]] = fn;
               send_face_nbr_ldof.AddConnection(fn, ldofs[j]);
            }
         send_nbr_elem_dof.AddConnections(
            send_el_off[fn] + i, ldofs, ldofs.Size());
      }
   }
   send_face_nbr_ldof.ShiftUpI();
   send_nbr_elem_dof.ShiftUpI();

   // convert the ldof indices in send_nbr_elem_dof
   int *send_J = send_nbr_elem_dof.GetJ();
   for (int fn = 0, j = 0; fn < num_face_nbrs; fn++)
   {
      int  num_ldofs = send_face_nbr_ldof.RowSize(fn);
      int *ldofs     = send_face_nbr_ldof.GetRow(fn);
      int  j_end     = send_I[send_el_off[fn+1]];

      for (int i = 0; i < num_ldofs; i++)
      {
         ldof_marker[ldofs[i]] = i;
      }

      for ( ; j < j_end; j++)
      {
         send_J[j] = ldof_marker[send_J[j]];
      }
   }

   MPI_Waitall(num_face_nbrs, recv_requests, statuses);
   face_nbr_element_dof.MakeJ();

   // send/receive the J arrays of send_nbr_elem_dof/face_nbr_element_dof,
   // respectively (they contain the element dofs in enumeration local for
   // the face-neighbor pair)
   int *recv_J = face_nbr_element_dof.GetJ();
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int nbr_rank = pmesh->GetFaceNbrRank(fn);
      int tag = 0;

      MPI_Isend(send_J + send_I[send_el_off[fn]],
                send_I[send_el_off[fn+1]] - send_I[send_el_off[fn]],
                MPI_INT, nbr_rank, tag, MyComm, &send_requests[fn]);

      MPI_Irecv(recv_J + recv_I[recv_el_off[fn]],
                recv_I[recv_el_off[fn+1]] - recv_I[recv_el_off[fn]],
                MPI_INT, nbr_rank, tag, MyComm, &recv_requests[fn]);
   }

   MPI_Waitall(num_face_nbrs, recv_requests, statuses);

   // shift the J array of face_nbr_element_dof
   for (int fn = 0, j = 0; fn < num_face_nbrs; fn++)
   {
      int shift = face_nbr_ldof.GetI()[fn];
      int j_end = recv_I[recv_el_off[fn+1]];

      for ( ; j < j_end; j++)
      {
         recv_J[j] += shift;
      }
   }

   MPI_Waitall(num_face_nbrs, send_requests, statuses);

   // send/receive the J arrays of send_face_nbr_ldof/face_nbr_ldof,
   // respectively
   send_J = send_face_nbr_ldof.GetJ();
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int nbr_rank = pmesh->GetFaceNbrRank(fn);
      int tag = 0;

      MPI_Isend(send_face_nbr_ldof.GetRow(fn),
                send_face_nbr_ldof.RowSize(fn),
                MPI_INT, nbr_rank, tag, MyComm, &send_requests[fn]);

      MPI_Irecv(face_nbr_ldof.GetRow(fn),
                face_nbr_ldof.RowSize(fn),
                MPI_INT, nbr_rank, tag, MyComm, &recv_requests[fn]);
   }

   MPI_Waitall(num_face_nbrs, recv_requests, statuses);
   MPI_Waitall(num_face_nbrs, send_requests, statuses);

   // send my_dof_offset (i.e. my_ldof_offset) to face neighbors and receive
   // their offset in dof_face_nbr_offsets, used to define face_nbr_glob_dof_map
   face_nbr_glob_dof_map.SetSize(num_face_nbr_dofs);
   Array<HYPRE_Int> dof_face_nbr_offsets(num_face_nbrs);
   HYPRE_Int my_dof_offset = GetMyDofOffset();
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int nbr_rank = pmesh->GetFaceNbrRank(fn);
      int tag = 0;

      MPI_Isend(&my_dof_offset, 1, HYPRE_MPI_INT, nbr_rank, tag,
                MyComm, &send_requests[fn]);

      MPI_Irecv(&dof_face_nbr_offsets[fn], 1, HYPRE_MPI_INT, nbr_rank, tag,
                MyComm, &recv_requests[fn]);
   }

   MPI_Waitall(num_face_nbrs, recv_requests, statuses);

   // set the array face_nbr_glob_dof_map which holds the global ldof indices of
   // the face-neighbor dofs
   for (int fn = 0, j = 0; fn < num_face_nbrs; fn++)
   {
      for (int j_end = face_nbr_ldof.GetI()[fn+1]; j < j_end; j++)
         face_nbr_glob_dof_map[j] =
            dof_face_nbr_offsets[fn] + face_nbr_ldof.GetJ()[j];
   }

   MPI_Waitall(num_face_nbrs, send_requests, statuses);

   delete [] statuses;
   delete [] requests;
}

void ParFiniteElementSpace::GetFaceNbrElementVDofs(
   int i, Array<int> &vdofs) const
{
   face_nbr_element_dof.GetRow(i, vdofs);
}

const FiniteElement *ParFiniteElementSpace::GetFaceNbrFE(int i) const
{
   const FiniteElement *FE =
      fec->FiniteElementForGeometry(
         pmesh->face_nbr_elements[i]->GetGeometryType());

   if (NURBSext)
      mfem_error("ParFiniteElementSpace::GetFaceNbrFE"
                 " does not support NURBS!");

   return FE;
}

void ParFiniteElementSpace::Lose_Dof_TrueDof_Matrix()
{
   hypre_ParCSRMatrix *csrP = (hypre_ParCSRMatrix*)(*P);
   hypre_ParCSRMatrixOwnsRowStarts(csrP) = 1;
   hypre_ParCSRMatrixOwnsColStarts(csrP) = 1;
   P -> StealData();
   dof_offsets.LoseData();
   tdof_offsets.LoseData();
}

void ParFiniteElementSpace::ConstructTrueDofs()
{
   int i, gr, n = GetVSize(), nex = GetExVSize();
   GroupTopology &gt = pmesh->gtopo;
   gcomm = new GroupCommunicator(gt);
   Table &group_ldof = gcomm->GroupLDofTable();

   GetGroupComm(*gcomm, 1, &ldof_sign);

   // Define ldof_group and mark ldof_ltdof with
   //   -1 for ldof that is ours
   //   -2 for ldof that is in a group with another master
   ldof_group.SetSize(n);
   ldof_ltdof.SetSize(n);
   lexdof_group.SetSize(n);
   lexdof_ltexdof.SetSize(n);
   ldof_group = 0;
   ldof_ltdof = -1;
   lexdof_group = 0;
   lexdof_ltexdof = -1;

   for (gr = 1; gr < group_ldof.Size(); gr++)
   {
      const int *ldofs = group_ldof.GetRow(gr);
      const int nldofs = group_ldof.RowSize(gr);
      for (i = 0; i < nldofs; i++)
      {
         ldof_group[ldofs[i]] = gr;
         lexdof_group[ldofs[i]] = gr;
      }

      if (!gt.IAmMaster(gr)) // we are not the master
         for (i = 0; i < nldofs; i++)
         {
            ldof_ltdof[ldofs[i]] = -2;
            lexdof_ltexdof[ldofs[i]] = -2;
         }
   }

   // count ltdof_size
   ltdof_size = 0;
   for (i = 0; i < n; i++)
      if (ldof_ltdof[i] == -1)
      {
         ldof_ltdof[i] = ltdof_size++;
      }

   ltexdof_size = 0;
   for (i = 0; i < nex; i++)
      if (lexdof_ltexdof[i] == -1)
      {
         lexdof_ltexdof[i] = ltexdof_size++;
      }

   // Check for consistency
   MFEM_VERIFY(ltdof_size == ltexdof_size + this->GetNPrDofs(),
	       "mismatch in number of true dofs and true exposed plus private dofs");

   // have the group masters broadcast their ltdofs to the rest of the group
   gcomm->Bcast(ldof_ltdof);

   // have the group masters broadcast their ltexdofs to the rest of the group
   gcomm->Bcast(lexdof_ltexdof);
}

void ParFiniteElementSpace::ConstructTrueNURBSDofs()
{
   int n = GetVSize();
   GroupTopology &gt = pNURBSext()->gtopo;
   gcomm = new GroupCommunicator(gt);

   // pNURBSext()->ldof_group is for scalar space!
   if (vdim == 1)
   {
      ldof_group.MakeRef(pNURBSext()->ldof_group);
   }
   else
   {
      const int *scalar_ldof_group = pNURBSext()->ldof_group;
      ldof_group.SetSize(n);
      for (int i = 0; i < n; i++)
      {
         ldof_group[i] = scalar_ldof_group[VDofToDof(i)];
      }
   }

   gcomm->Create(ldof_group);

   // ldof_sign.SetSize(n);
   // ldof_sign = 1;
   ldof_sign.DeleteAll();

   ltdof_size = 0;
   ldof_ltdof.SetSize(n);
   for (int i = 0; i < n; i++)
   {
      if (gt.IAmMaster(ldof_group[i]))
      {
         ldof_ltdof[i] = ltdof_size;
         ltdof_size++;
      }
      else
      {
         ldof_ltdof[i] = -2;
      }
   }

   ltexdof_size = ltdof_size - this->GetNPrDofs();

   // have the group masters broadcast their ltdofs to the rest of the group
   gcomm->Bcast(ldof_ltdof);
}

void ParFiniteElementSpace::Update()
{
   FiniteElementSpace::Update();

   ldof_group.DeleteAll();
   ldof_ltdof.DeleteAll();
   dof_offsets.DeleteAll();
   tdof_offsets.DeleteAll();
   tdof_nb_offsets.DeleteAll();
   ldof_sign.DeleteAll();
   delete P;
   P = NULL;
   delete gcomm;
   gcomm = NULL;
   num_face_nbr_dofs = -1;
   face_nbr_element_dof.Clear();
   face_nbr_ldof.Clear();
   face_nbr_glob_dof_map.DeleteAll();
   send_face_nbr_ldof.Clear();
   ConstructTrueDofs();
   GenerateGlobalOffsets();
}

FiniteElementSpace *ParFiniteElementSpace::SaveUpdate()
{
   ParFiniteElementSpace *cpfes = new ParFiniteElementSpace(*this);
   Constructor();
   ConstructTrueDofs();
   GenerateGlobalOffsets();
   return cpfes;
}

}

#endif
