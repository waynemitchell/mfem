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

#include "../config.hpp"

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
   Swap(ldof_group, pf.ldof_group);
   Swap(ldof_ltdof, pf.ldof_ltdof);
   Swap(dof_offsets, pf.dof_offsets);
   Swap(tdof_offsets, pf.tdof_offsets);
   Swap(tdof_nb_offsets, pf.tdof_nb_offsets);
   Swap(ldof_sign, pf.ldof_sign);
   P = pf.P;
   pf.P = NULL;
   num_face_nbr_dofs = pf.num_face_nbr_dofs;
   pf.num_face_nbr_dofs = -1;
   Swap<Table>(face_nbr_element_dof, pf.face_nbr_element_dof);
   Swap<Table>(face_nbr_gdof, pf.face_nbr_gdof);
   Swap<Table>(send_face_nbr_ldof, pf.send_face_nbr_ldof);
}

ParFiniteElementSpace::ParFiniteElementSpace(
   ParMesh *pm, const FiniteElementCollection *f, int dim, int order)
   : FiniteElementSpace(pm, f, dim, order)
{
   mesh = pmesh = pm;

   MyComm = pmesh->GetComm();
   MPI_Comm_size(MyComm, &NRanks);
   MPI_Comm_rank(MyComm, &MyRank);

   P = NULL;

   if (pmesh->pncmesh)
   {
      gcomm = NULL;
      return;
   }

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
      group_ldof_counter *= vdim;
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
               dofs[l] = m;

            if (ldof_type)
               DofsToVDofs(dofs);

            for (l = 0; l < dofs.Size(); l++)
               group_ldof.GetJ()[group_ldof_counter++] = dofs[l];
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
                     (*ldof_sign)[dofs[l]] = -1;
               }
               else
                  dofs[l] = m + ind[l];

            if (ldof_type)
               DofsToVDofs(dofs);

            for (l = 0; l < dofs.Size(); l++)
               group_ldof.GetJ()[group_ldof_counter++] = dofs[l];
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
                     (*ldof_sign)[dofs[l]] = -1;
               }
               else
                  dofs[l] = m + ind[l];

            if (ldof_type)
               DofsToVDofs(dofs);

            for (l = 0; l < dofs.Size(); l++)
               group_ldof.GetJ()[group_ldof_counter++] = dofs[l];
         }

      group_ldof.GetI()[gr+1] = group_ldof_counter;
   }

   gc.Finalize();
}

void ParFiniteElementSpace::GetElementDofs(int i, Array<int> &dofs) const
{
   if (elem_dof)
   {
      elem_dof->GetRow(i, dofs);
      return;
   }
   FiniteElementSpace::GetElementDofs(i, dofs);
   for (i = 0; i < dofs.Size(); i++)
      if (dofs[i] < 0)
      {
         if (ldof_sign[-1-dofs[i]] < 0)
            dofs[i] = -1-dofs[i];
      }
      else
      {
         if (ldof_sign[dofs[i]] < 0)
            dofs[i] = -1-dofs[i];
      }
}

void ParFiniteElementSpace::GetBdrElementDofs(int i, Array<int> &dofs) const
{
   if (bdrElem_dof)
   {
      bdrElem_dof->GetRow(i, dofs);
      return;
   }
   FiniteElementSpace::GetBdrElementDofs(i, dofs);
   for (i = 0; i < dofs.Size(); i++)
      if (dofs[i] < 0)
      {
         if (ldof_sign[-1-dofs[i]] < 0)
            dofs[i] = -1-dofs[i];
      }
      else
      {
         if (ldof_sign[dofs[i]] < 0)
            dofs[i] = -1-dofs[i];
      }
}

void ParFiniteElementSpace::GenerateGlobalOffsets()
{
   if (HYPRE_AssumedPartitionCheck())
   {
      int ldof[2];

      ldof[0] = GetVSize();
      ldof[1] = TrueVSize();

      dof_offsets.SetSize(3);
      tdof_offsets.SetSize(3);

      MPI_Scan(ldof, &dof_offsets[0], 2, MPI_INT, MPI_SUM, MyComm);

      tdof_offsets[1] = dof_offsets[1];
      tdof_offsets[0] = tdof_offsets[1] - ldof[1];

      dof_offsets[1] = dof_offsets[0];
      dof_offsets[0] = dof_offsets[1] - ldof[0];

      // get the global sizes in (t)dof_offsets[2]
      if (MyRank == NRanks-1)
      {
         ldof[0] = dof_offsets[1];
         ldof[1] = tdof_offsets[1];
      }

      MPI_Bcast(ldof, 2, MPI_INT, NRanks-1, MyComm);
      dof_offsets[2] = ldof[0];
      tdof_offsets[2] = ldof[1];
   }
   else
   {
      int i;
      int ldof  = GetVSize();
      int ltdof = TrueVSize();

      dof_offsets.SetSize (NRanks+1);
      tdof_offsets.SetSize(NRanks+1);

      MPI_Allgather(&ldof, 1, MPI_INT, &dof_offsets[1], 1, MPI_INT, MyComm);
      MPI_Allgather(&ltdof, 1, MPI_INT, &tdof_offsets[1], 1, MPI_INT, MyComm);

      dof_offsets[0] = tdof_offsets[0] = 0;
      for (i = 1; i < NRanks; i++)
      {
         dof_offsets [i+1] += dof_offsets [i];
         tdof_offsets[i+1] += tdof_offsets[i];
      }
   }
}

HypreParMatrix *ParFiniteElementSpace::Dof_TrueDof_Matrix() // matrix P
{
   int  i;

   if (P)
      return P;

   if (pmesh->pncmesh)
   {
      GetConformingInterpolation();
      return P;
   }

   int  ldof = GetVSize();
   int  ltdof = TrueVSize();

   GroupTopology &gt = GetGroupTopo();

   int *i_diag;
   int *j_diag;
   int  diag_counter;

   int *i_offd;
   int *j_offd;
   int  offd_counter;

   int *cmap;
   int *col_starts;
   int *row_starts;

   col_starts = GetTrueDofOffsets();
   row_starts = GetDofOffsets();

   i_diag = hypre_TAlloc(HYPRE_Int, ldof+1);
   j_diag = hypre_TAlloc(HYPRE_Int, ltdof);

   i_offd = hypre_TAlloc(HYPRE_Int, ldof+1);
   j_offd = hypre_TAlloc(HYPRE_Int, ldof-ltdof);

   cmap   = hypre_TAlloc(HYPRE_Int, ldof-ltdof);

   Array<Pair<int, int> > cmap_j_offd(ldof-ltdof);

   if (HYPRE_AssumedPartitionCheck())
   {
      int nsize = gt.GetNumNeighbors()-1;
      MPI_Request *requests = new MPI_Request[2*nsize];
      MPI_Status  *statuses = new MPI_Status[2*nsize];
      tdof_nb_offsets.SetSize(nsize+1);
      tdof_nb_offsets[0] = col_starts[0];

      int request_counter = 0;
      // send and receive neighbors' local tdof offsets
      for (i = 1; i <= nsize; i++)
         MPI_Irecv(&tdof_nb_offsets[i], 1, MPI_INT, gt.GetNeighborRank(i), 5365,
                   MyComm, &requests[request_counter++]);

      for (i = 1; i <= nsize; i++)
         MPI_Isend(&tdof_nb_offsets[0], 1, MPI_INT, gt.GetNeighborRank(i), 5365,
                   MyComm, &requests[request_counter++]);

      MPI_Waitall(request_counter, requests, statuses);

      delete [] statuses;
      delete [] requests;
   }

   i_diag[0] = i_offd[0] = 0;
   diag_counter = offd_counter = 0;
   for (i = 0; i < ldof; i++)
   {
      int proc = gt.GetGroupMasterRank(ldof_group[i]);
      if (proc == MyRank)
      {
         j_diag[diag_counter++] = ldof_ltdof[i];
      }
      else
      {
         if (HYPRE_AssumedPartitionCheck())
            cmap_j_offd[offd_counter].one =
               tdof_nb_offsets[gt.GetGroupMaster(ldof_group[i])] + ldof_ltdof[i];
         else
            cmap_j_offd[offd_counter].one = col_starts[proc] + ldof_ltdof[i];
         cmap_j_offd[offd_counter].two = offd_counter;
         offd_counter++;
      }
      i_diag[i+1] = diag_counter;
      i_offd[i+1] = offd_counter;
   }

   SortPairs<int, int>(cmap_j_offd, offd_counter);

   for (i = 0; i < offd_counter; i++)
   {
      cmap[i] = cmap_j_offd[i].one;
      j_offd[cmap_j_offd[i].two] = i;
   }

   P = new HypreParMatrix(MyComm, MyRank, NRanks, row_starts, col_starts,
                          i_diag, j_diag, i_offd, j_offd, cmap, offd_counter);

   return P;
}

void ParFiniteElementSpace::DivideByGroupSize(double *vec)
{
   GroupTopology &gt = GetGroupTopo();

   for (int i = 0; i < ldof_group.Size(); i++)
      if (gt.IAmMaster(ldof_group[i])) // we are the master
         vec[ldof_ltdof[i]] /= gt.GetGroupSize(ldof_group[i]);
}

GroupCommunicator *ParFiniteElementSpace::ScalarGroupComm()
{
   GroupCommunicator *gc = new GroupCommunicator(GetGroupTopo());
   if (NURBSext)
      gc->Create(pNURBSext()->ldof_group);
   else
      GetGroupComm(*gc, 0);
   return gc;
}

void ParFiniteElementSpace::Synchronize(Array<int> &ldof_marker) const
{
   if (ldof_marker.Size() != GetVSize())
      mfem_error("ParFiniteElementSpace::Synchronize");

   if (pmesh->pncmesh)
   {
      std::cout << "FIXME!!! ParFiniteElementSpace::Synchronize for PNCMesh.\n";
      return;
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
      return ldof_ltdof[ldof];
   else
      return -1;
}

int ParFiniteElementSpace::GetGlobalTDofNumber(int ldof)
{
   if (HYPRE_AssumedPartitionCheck())
   {
      if (!P)
         Dof_TrueDof_Matrix();
      return ldof_ltdof[ldof] +
         tdof_nb_offsets[GetGroupTopo().GetGroupMaster(ldof_group[ldof])];
   }

   return ldof_ltdof[ldof] +
      tdof_offsets[GetGroupTopo().GetGroupMasterRank(ldof_group[ldof])];
}

int ParFiniteElementSpace::GetGlobalScalarTDofNumber(int sldof)
{
   if (HYPRE_AssumedPartitionCheck())
   {
      if (!P)
         Dof_TrueDof_Matrix();
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

int ParFiniteElementSpace::GetMyDofOffset()
{
   if (HYPRE_AssumedPartitionCheck())
      return dof_offsets[0];
   else
      return dof_offsets[MyRank];
}

void ParFiniteElementSpace::ExchangeFaceNbrData()
{
   if (num_face_nbr_dofs >= 0)
      return;

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
   face_nbr_gdof.MakeI(num_face_nbrs);
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

      MPI_Irecv(&face_nbr_gdof.GetI()[fn], 1, MPI_INT, nbr_rank, tag,
                MyComm, &recv_requests[fn]);
   }

   MPI_Waitall(num_face_nbrs, recv_requests, statuses);
   face_nbr_gdof.MakeJ();

   num_face_nbr_dofs = face_nbr_gdof.Size_of_connections();

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
         ldof_marker[ldofs[i]] = i;

      for ( ; j < j_end; j++)
         send_J[j] = ldof_marker[send_J[j]];
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
      int shift = face_nbr_gdof.GetI()[fn];
      int j_end = recv_I[recv_el_off[fn+1]];

      for ( ; j < j_end; j++)
         recv_J[j] += shift;
   }

   MPI_Waitall(num_face_nbrs, send_requests, statuses);

   // send/receive the J arrays of send_face_nbr_ldof/face_nbr_gdof,
   // respectively
   send_J = send_face_nbr_ldof.GetJ();
   // switch to global dof numbers
   int my_dof_offset = GetMyDofOffset();
   int tot_send_dofs = send_face_nbr_ldof.Size_of_connections();
   for (int i = 0; i < tot_send_dofs; i++)
      send_J[i] += my_dof_offset;
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int nbr_rank = pmesh->GetFaceNbrRank(fn);
      int tag = 0;

      MPI_Isend(send_face_nbr_ldof.GetRow(fn),
                send_face_nbr_ldof.RowSize(fn),
                MPI_INT, nbr_rank, tag, MyComm, &send_requests[fn]);

      MPI_Irecv(face_nbr_gdof.GetRow(fn),
                face_nbr_gdof.RowSize(fn),
                MPI_INT, nbr_rank, tag, MyComm, &recv_requests[fn]);
   }

   MPI_Waitall(num_face_nbrs, send_requests, statuses);

   // switch back to local dof numbers
   for (int i = 0; i < tot_send_dofs; i++)
      send_J[i] -= my_dof_offset;

   MPI_Waitall(num_face_nbrs, recv_requests, statuses);

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
   int i, gr, n = GetVSize();
   GroupTopology &gt = pmesh->gtopo;
   gcomm = new GroupCommunicator(gt);
   Table &group_ldof = gcomm->GroupLDofTable();

   GetGroupComm(*gcomm, 1, &ldof_sign);

   // Define ldof_group and mark ldof_ltdof with
   //   -1 for ldof that is ours
   //   -2 for ldof that is in a group with another master
   ldof_group.SetSize(n);
   ldof_ltdof.SetSize(n);
   ldof_group = 0;
   ldof_ltdof = -1;

   for (gr = 1; gr < group_ldof.Size(); gr++)
   {
      const int *ldofs = group_ldof.GetRow(gr);
      const int nldofs = group_ldof.RowSize(gr);
      for (i = 0; i < nldofs; i++)
         ldof_group[ldofs[i]] = gr;

      if (!gt.IAmMaster(gr)) // we are not the master
         for (i = 0; i < nldofs; i++)
            ldof_ltdof[ldofs[i]] = -2;
   }

   // count ltdof_size
   ltdof_size = 0;
   for (i = 0; i < n; i++)
      if (ldof_ltdof[i] == -1)
         ldof_ltdof[i] = ltdof_size++;

   // have the group masters broadcast their ltdofs to the rest of the group
   gcomm->Bcast(ldof_ltdof);
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
         ldof_group[i] = scalar_ldof_group[VDofToDof(i)];
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

   // have the group masters broadcast their ltdofs to the rest of the group
   gcomm->Bcast(ldof_ltdof);
}

struct Dependency
{
   int rank, dof;
   double coef;

   typedef Array<Dependency> List;
   Dependency(int r, int d, double c) : rank(r), dof(d), coef(c) {}
};

inline int DecodeDof(int dof, double& sign)
{
   if (dof >= 0)
      return (sign = 1.0, dof);
   else
      return (sign = -1.0, -1 - dof);
}

inline bool Is1To1(const Dependency::List &list)
{
   return !list.Size() || (list.Size() == 1 && std::abs(list[0].coef) == 1.0);
}

static void AddSlaveDependencies(Dependency::List deps[],
   int master_rank, Array<int>& master_dofs,
   int slave_rank, Array<int>& slave_dofs, DenseMatrix& I)
{
   // make each slave DOF dependent on all master DOFs
   for (int i = 0; i < slave_dofs.Size(); i++)
   {
      double ssign;
      int sdof = DecodeDof(slave_dofs[i], ssign);

      Dependency::List &list = deps[sdof];
      if (Is1To1(list)) // slave dependencies override 1-to-1 dependencies
      {
         list.DeleteAll();
         list.Reserve(master_dofs.Size());

         for (int j = 0; j < master_dofs.Size(); j++)
         {
            double coef = I(i, j);
            if (std::abs(coef) > 1e-12)
            {
               double msign;
               int mdof = DecodeDof(master_dofs[j], msign);
               if (mdof != sdof || master_rank != slave_rank)
                 list.Append(Dependency(master_rank, mdof, coef*ssign*msign));
            }
         }
      }
   }
}

static void Add1To1Dependencies(Dependency::List deps[], int owner_rank,
   Array<int>& owner_dofs, Array<int>& dependent_dofs)
{
   MFEM_ASSERT(owner_dofs.Size() == dependent_dofs.Size(), "");
   for (int i = 0; i < owner_dofs.Size(); i++)
   {
      double osign, dsign;
      int odof = DecodeDof(owner_dofs[i], osign);
      int ddof = DecodeDof(dependent_dofs[i], dsign);

      Dependency::List &list = deps[ddof];
      if (Is1To1(list)) // don't touch existing slave dependencies
      {
         if (list.Size() == 0)
         {
            list.Append(Dependency(owner_rank, odof, osign*dsign));
         }
         else if (list[0].rank > owner_rank)
         {
            // 1-to-1 dependency already exists but lower rank takes precedence
            list[0] = Dependency(owner_rank, odof, osign*dsign);
         }
      }
   }
}

void ParFiniteElementSpace::GetConformingDofs
(int type, int index, Array<int>& dofs)
{
//   Array<int> dofs;
   switch (type)
   {
   case 0: GetVertexDofs(index, dofs); break;
   case 1: GetEdgeDofs(index, dofs); break;
   case 2: GetFaceDofs(index, dofs); break;
   }

//   ConvertToConformingVDofs(dofs, cdofs);
   // TODO: remove invalid (non-conforming) DOFs
}

static bool IsTrueDof(const Dependency::List &list, int my_rank)
{
   return !list.Size() || (list.Size() == 1 && list[0].rank == my_rank);
}


void ParFiniteElementSpace::GetConformingInterpolation()
{
   ParNCMesh* pncmesh = pmesh->pncmesh;

   // *** STEP 1: exchange shared vertex/edge/face DOFs with neighbors ***

   NeighborDofMessage::Map send_dofs, recv_dofs;

   // prepare neighbor DOF messages for shared vertices/edges/faces
   for (int type = 0; type < 3; type++)
   {
      const NCMesh::NCList &list = pncmesh->GetSharedList(type);
      Array<int> cdofs;

      int cs = list.conforming.size(), ms = list.masters.size();
      for (int i = 0; i < cs+ms; i++)
      {
         // loop through all (shared) conforming+master vertices/edges/faces
         const NCMesh::MeshId& id =
            (i < cs) ? (const NCMesh::MeshId&) list.conforming[i]
                     : (const NCMesh::MeshId&) list.masters[i-cs];

         int owner = pncmesh->GetOwner(type, id.index), gsize;
         if (owner == MyRank)
         {
            // we own a shared v/e/f, send its DOFs to others in group
            GetConformingDofs(type, id.index, cdofs);
            const int* group = pncmesh->GetGroup(type, id.index, gsize);
            for (int j = 0; j < gsize; j++)
               if (group[j] != MyRank)
                  send_dofs[group[j]].AddDofs(type, id, cdofs, pncmesh);
         }
         else
         {
            // we don't own this v/e/f and expect to receive DOFs for it
            recv_dofs[owner].SetNCMesh(pncmesh);
         }
      }
   }

   // send/receive all DOF messages
   NeighborDofMessage::IsendAll(send_dofs, MyComm);
   NeighborDofMessage::RecvAll(recv_dofs, MyComm);

   // *** STEP 2: build dependency lists ***

   int num_cdofs = ndofs;//GetNConformingDofs();
   Dependency::List* deps = new Dependency::List[num_cdofs];

   Array<int> master_dofs, slave_dofs;
   Array<int> owner_dofs, my_dofs;

   // loop through *all* master edges/faces, constrain their slaves
   for (int type = 1; type < 3; type++)
   {
      const NCMesh::NCList &list = (type > 1) ? pncmesh->GetFaceList()
                                              : pncmesh->GetEdgeList();
      if (!list.masters.size()) continue;

      IsoparametricTransformation T;
      if (type > 1) T.SetFE(&QuadrilateralFE); else T.SetFE(&SegmentFE);

      const FiniteElement* fe = fec->FiniteElementForGeometry(
         ((type > 1) ? Geometry::SQUARE : Geometry::SEGMENT));

      DenseMatrix I(fe->GetDof());

      // process masters that we own or that have slaves we own
      for (int mi = 0; mi < list.masters.size(); mi++)
      {
         const NCMesh::Master &mf = list.masters[mi];
         if (!pncmesh->RankInGroup(type, mf.index, MyRank)) continue;

         // get master DOFs
         int master_rank = pncmesh->GetOwner(type, mf.index);
         if (master_rank == MyRank)
            GetConformingDofs(type, mf.index, master_dofs);
         else
            recv_dofs[master_rank].GetDofs(type, mf, master_dofs);

         if (!master_dofs.Size()) continue;

         // constrain slaves that we own
         for (int si = mf.slaves_begin; si < mf.slaves_end; si++)
         {
            const NCMesh::Slave &sf = list.slaves[si];
            int slave_rank = pncmesh->GetOwner(type, sf.index);
            if (slave_rank != MyRank) continue;

            GetConformingDofs(type, sf.index, slave_dofs);
            if (!slave_dofs.Size()) continue;

            T.GetPointMat() = sf.point_matrix;
            fe->GetLocalInterpolation(T, I);

            // make each slave DOF dependent on all master DOFs
            AddSlaveDependencies(deps, master_rank, master_dofs,
                                 slave_rank, slave_dofs, I);
         }
      }
   }

   // add one-to-one dependencies between shared conforming vertices/edges/faces
   for (int type = 0; type < 3; type++)
   {
      const NCMesh::NCList &list = pncmesh->GetSharedList(type);
      for (int i = 0; i < list.conforming.size(); i++)
      {
         const NCMesh::MeshId &id = list.conforming[i];
         int owner = pncmesh->GetOwner(type, id.index);
         if (owner == MyRank)
         {
            GetConformingDofs(type, id.index, my_dofs);
            Add1To1Dependencies(deps, owner, my_dofs, my_dofs);
         }
         else
         {
            recv_dofs[owner].GetDofs(type, id, owner_dofs);
            GetConformingDofs(type, id.index, my_dofs);
            Add1To1Dependencies(deps, owner, owner_dofs, my_dofs);
         }
      }
   }

   // *** STEP 3: request P matrix rows that we need from neighbors ***

   NeighborRowRequest::Map send_requests, recv_requests;

   // copy communication topology from the DOF messages
   NeighborDofMessage::Map::iterator it;
   for (it = send_dofs.begin(); it != send_dofs.end(); ++it)
      recv_requests[it->first];
   for (it = recv_dofs.begin(); it != recv_dofs.end(); ++it)
      send_requests[it->first];

   // request rows we depend on
   for (int i = 0; i < num_cdofs; i++)
   {
      const Dependency::List &list = deps[i];
      for (int j = 0; j < list.Size(); j++)
      {
         const Dependency &dep = list[j];
         if (dep.rank != MyRank)
            send_requests[dep.rank].RequestRow(dep.dof);
      }
   }

   NeighborRowRequest::IsendAll(send_requests, MyComm);
   NeighborRowRequest::RecvAll(recv_requests, MyComm);

   // *** STEP 4: iteratively build the P matrix ***

   // DOFs that stayed independent or are ours are true DOFs
   ltdof_size = 0;
   for (int i = 0; i < num_cdofs; i++)
      if (IsTrueDof(deps[i], MyRank))
         ltdof_size++;

   // FIXME: vdim
   GenerateGlobalOffsets();
   int glob_true_dofs = tdof_offsets[NRanks];
   int glob_cdofs = dof_offsets[NRanks];

   // create the local part (local rows) of the P matrix
   SparseMatrix localP(num_cdofs, glob_true_dofs);

   Array<bool> finalized(num_cdofs);
   finalized = false;

   // put identity in P for true DOFs
   for (int i = 0, true_dof = 0; i < num_cdofs; i++)
      if (IsTrueDof(deps[i], MyRank))
      {
         localP.Add(i, tdof_offsets[MyRank] + true_dof++, 1.0);
         finalized[i] = true;
      }

   Array<int> cols;
   Vector srow;

   NeighborRowReply::Map recv_replies;
   std::vector<NeighborRowReply::Map> send_replies;

   int num_finalized = ltdof_size;
   while (1)
   {
      // finalize what can currently be finalized
      for (int dof = 0, i; dof < num_cdofs; dof++)
      {
         if (finalized[dof]) continue;

         const Dependency::List &list = deps[dof];

         // check that rows of all constraining DOFs are available
         for (i = 0; i < list.Size(); i++)
         {
            const Dependency &dep = list[i];
            if (dep.rank == MyRank)
               { if (!finalized[dep.dof]) break; }
            else
               if (!recv_replies[dep.rank].HaveRow(dep.dof)) break;
         }
         if (i < list.Size()) continue;

         // form a linear combination of rows that 'dof' depends on
         for (i = 0; i < list.Size(); i++)
         {
            const Dependency &dep = list[i];
            if (dep.rank == MyRank)
               localP.GetRow(dep.dof, cols, srow);
            else
               recv_replies[dep.rank].GetRow(dep.dof, cols, srow);

            srow *= dep.coef;
            localP.AddRow(dof, cols, srow);
         }

         finalized[dof] = true;
         num_finalized++;
      }

      // send rows that are requested by neighbors and are available
      send_replies.push_back(NeighborRowReply::Map());

      NeighborRowRequest::Map::iterator it;
      for (it = recv_requests.begin(); it != recv_requests.end(); ++it)
      {
         NeighborRowRequest &req = it->second;
         std::set<int>::iterator row;
         for (row = req.rows.begin(); row != req.rows.end(); )
            if (finalized[*row])
            {
               localP.GetRow(*row, cols, srow);
               send_replies.back()[it->first].AddRow(*row, cols, srow);
               req.rows.erase(row++);
            }
            else
               ++row;
      }
      NeighborRowReply::IsendAll(send_replies.back(), MyComm);

      // are we finished?
      if (num_finalized >= num_cdofs)
         break;

      // wait for a reply from neighbors
      int rank, size;
      NeighborRowReply::Probe(rank, size, MyComm);
      recv_replies[rank].Recv(rank, size, MyComm);

      // there may be more, receive all replies available
      while (NeighborRowReply::IProbe(rank, size, MyComm))
         recv_replies[rank].Recv(rank, size, MyComm);
   }

   delete [] deps;
   localP.Finalize();

   std::cout << "MyRank = " << MyRank << std::endl;
   localP.Print();

   // create the parallel matrix P
   P = new HypreParMatrix(MyComm, num_cdofs, glob_cdofs, glob_true_dofs,
                          localP.GetI(), localP.GetJ(), localP.GetData(),
                          dof_offsets.GetData(), tdof_offsets.GetData());

   // make sure we can discard all send buffers
   NeighborDofMessage::WaitAllSent(send_dofs);
   NeighborRowRequest::WaitAllSent(send_requests);
   for (int i = 0; i < send_replies.size(); i++)
      NeighborRowReply::WaitAllSent(send_replies[i]);

   ldof_sign.SetSize(num_cdofs);
   ldof_sign = 1;
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
   face_nbr_gdof.Clear();
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
