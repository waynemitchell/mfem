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

#include "mesh_headers.hpp"
#include "pncmesh.hpp"

namespace mfem
{

void ElementId::ToBuffer(uint8_t*& buffer) const
{
   MFEM_ASSERT(root_index < (1 << 24) && length <= 0xff, "");

   int root = root_index;
   *buffer++ = (uint8_t) root;
   root >>= 8;
   *buffer++ = (uint8_t) root;
   root >>= 8;
   *buffer++ = (uint8_t) root;

   *buffer++ = (uint8_t) length;
   int bytes = (length + 1) & 0x1;
   for (int i = 0; i < bytes; i++)
      *buffer++ = path[i];
}

void ElementId::FromBuffer(uint8_t*& buffer)
{
   root_index = *buffer++;
   root_index <= 8;
   root_index |= *buffer++;
   root_index << 8;
   root_index |= *buffer++;

   length = *buffer++;
   int bytes = (length + 1) & 0x1;
   for (int i = 0; i < bytes; i++)
      path[i] = *buffer++;
}


ParNCMesh::ParNCMesh(MPI_Comm comm, const Mesh *coarse_mesh)
   : NCMesh(coarse_mesh)
{
   MyComm = comm;
   MPI_Comm_size(MyComm, &NRanks);
   MPI_Comm_rank(MyComm, &MyRank);

   InitialPartition();
   AssignLeafIndices();
}

void ParNCMesh::InitialPartition()
{
   // assign the 'n' leaves to the 'NRanks' processors
   int n = leaf_elements.Size();
   for (int i = 0; i < n; i++)
      leaf_elements[i]->rank = i * NRanks / n;
}

void ParNCMesh::CollectLeafElements(Element* elem)
{
   // This function is an override of its serial version, called from
   // UpdateLeafElements. The only difference is that this version doesn't
   // assign elem->index to ghost elements, which will prevent them from getting
   // into the Mesh (see NCMesh::GetMeshComponents).

   if (!elem->ref_type)
   {
      if (elem->rank != MyRank)
         elem->index = -1;
      else
         elem->index = leaf_elements.Size();

      leaf_elements.Append(elem);
   }
   else
   {
      elem->index = -1;
      for (int i = 0; i < 8; i++)
         if (elem->child[i])
            CollectLeafElements(elem->child[i]);
   }
}

void ParNCMesh::AssignLeafIndices()
{
   // This is an override of NCMesh::AssignLeafIndices(). The difference is
   // that we don't assign a Mesh index to ghost elements. This will make them
   // skipped in NCMesh::GetMeshComponents.

   for (int i = 0, index = 0; i < leaf_elements.Size(); i++)
   {
      Element* leaf = leaf_elements[i];
      if (leaf->rank == MyRank)
         leaf->index = index++;
      else
         leaf->index = -1;
   }
}




} // namespace mfem

#endif // MFEM_USE_MPI
