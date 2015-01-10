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

#include <map>
#include <limits>

namespace mfem
{

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

void ParNCMesh::OnMeshUpdated(Mesh *mesh)
{
   // This is an override (or extension of) NCMesh::OnMeshUpdated().
   // In addition to getting edge/face indices from 'mesh', we also
   // assign indices to ghost edges/faces that don't exist in the 'mesh'.

   // clear Edge:: and Face::index
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
      if (it->edge) it->edge->index = -1;
   for (HashTable<Face>::Iterator it(faces); it; ++it)
      it->index = -1;

   // go assign existing edge/face indices
   NCMesh::OnMeshUpdated(mesh);

   // assign ghost edge indices
   NEdges = mesh->GetNEdges();
   NGhostEdges = 0;
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
      if (it->edge && it->edge->index < 0)
         it->edge->index = NEdges + (NGhostEdges++);

   // assign ghost face indices
   NFaces = mesh->GetNFaces();
   NGhostFaces = 0;
   for (HashTable<Face>::Iterator it(faces); it; ++it)
      if (it->index < 0)
         it->index = NFaces + (NGhostFaces++);
}

void ParNCMesh::ElementHasEdge(Element *elem, Edge *edge)
{
   // Called by NCMesh::BuildEdgeList when an edge is visited in a leaf element.
   // This allows us to determine edge ownership and processors that share it
   // without duplicating all the HashTable lookups in NCMesh::BuildEdgeList().

   int &owner = edge_owner[edge->index];
   owner = std::min(owner, elem->rank);

   tmp_edge_ranks.Append(IndexRank(edge->index, elem->rank));
}

void ParNCMesh::BuildEdgeList()
{
   edge_owner = std::numeric_limits<int>::max();
   tmp_edge_ranks.SetSize(12*leaf_elements.Size() * 3/2);

   NCMesh::BuildEdgeList();


   // conforming faces/edges: group contains the participating ranks, lowest it the master

   // master faces/edges: initially the group contains ranks that share the master e/f and
   // the lowest is the master, but after that the ranks of the slaves are added to the group

   shared_edges.Clear();

   tmp_edge_ranks.DeleteAll();
}

void ParNCMesh::ElementHasFace(Element* elem, Face* face)
{
   // Called by NCMesh::BuildFaceList when a face is visited in a leaf element.
   // This allows us to determine face ownership and processors that share it
   // without duplicating all the HashTable lookups in NCMesh::BuildFaceList().

   int &owner = face_owner[face->index];
   owner = std::min(owner, elem->rank);

   tmp_face_ranks.Append(IndexRank(face->index, elem->rank));
}

void ParNCMesh::BuildFaceList()
{
   face_owner = std::numeric_limits<int>::max();
   tmp_face_ranks.SetSize(6*leaf_elements.Size() * 3/2);

   NCMesh::BuildFaceList();

   shared_faces.Clear();


   tmp_face_ranks.Sort();

   tmp_face_ranks.DeleteAll();
}


//// Message encoding //////////////////////////////////////////////////////////

void ParNCMesh::ElementSet::SetInt(int pos, int value)
{
   // helper to put an int to the data array
   data[pos] = value & 0xff;
   data[pos+1] = (value >> 8) & 0xff;
   data[pos+2] = (value >> 16) & 0xff;
   data[pos+3] = (value >> 24) & 0xff;
}

int ParNCMesh::ElementSet::GetInt(int pos) const
{
   // helper to get an int from the data array
   return (int) data[pos] +
          ((int) data[pos+1] << 8) +
          ((int) data[pos+2] << 16) +
          ((int) data[pos+3] << 24);
}

bool ParNCMesh::ElementSet::EncodeTree
(Element* elem, const std::set<Element*> &elements)
{
   // is 'elem' in the set?
   if (elements.find(elem) != elements.end())
   {
      // we reached a 'leaf' of our subtree, mark this as zero child mask
      data.Append(0);
      return true;
   }
   else if (elem->ref_type)
   {
      // write a bit mask telling what subtrees contain elements from the set
      data.Append(0);
      unsigned char& mask = data.Last();

      // check the subtrees
      for (int i = 0; i < 8; i++)
         if (elem->child[i])
            if (EncodeTree(elem->child[i], elements))
               mask |= (unsigned char) 1 << i;

      // if we found no elements don't write anything
      if (!mask)
         data.DeleteLast();

      return mask != 0;
   }
   return false;
}

ParNCMesh::ElementSet::ElementSet
(const std::set<Element*> &elements, const Array<Element*> &ncmesh_roots)
{
   int header_pos = 0;
   data.SetSize(4);

   // Each refinement tree that contains at least one element from the set
   // is encoded as HEADER + TREE, where HEADER is the root element number and
   // TREE is the output of EncodeTree().
   for (int i = 0; i < ncmesh_roots.Size(); i++)
   {
      if (EncodeTree(ncmesh_roots[i], elements))
      {
         SetInt(header_pos, i);

         // make room for the next header
         header_pos = data.Size();
         data.SetSize(header_pos + 4);
      }
   }

   // mark end of data
   SetInt(header_pos, -1);
}

void ParNCMesh::ElementSet::DecodeTree
(Element* elem, int &pos, Array<Element*> &elements) const
{
   int mask = data[pos++];
   if (!mask)
   {
      elements.Append(elem);
   }
   else
   {
      for (int i = 0; i < 8; i++)
         if (mask & (1 << i))
            DecodeTree(elem->child[i], pos, elements);
   }
}

void ParNCMesh::ElementSet::Get
(Array<Element*> &elements, const Array<Element*> &ncmesh_roots) const
{
   // TODO: read from stream directly
   int root, pos = 0;
   while ((root = GetInt(pos)) >= 0)
   {
      pos += 4;
      DecodeTree(ncmesh_roots[root], pos, elements);
   }
}

template<typename T>
static inline void write(std::ostream& os, T value)
{
   os.write((char*) &value, sizeof(T));
}

template<typename T>
static inline T read(std::istream& is)
{
   T value;
   is.read((char*) &value, sizeof(T));
   return value;
}

void ParNCMesh::ElementSet::Dump(std::ostream &os) const
{
   write<int>(os, data.Size()); // TODO: remove
   os.write((char*) data.GetData(), data.Size());
}

void ParNCMesh::ElementSet::Load(std::istream &is)
{
   data.SetSize(read<int>(is)); // TODO: remove
   is.read((char*) data.GetData(), data.Size());
}

void ParNCMesh::EncodeEdgesFaces
(const Array<EdgeId> &edges, const Array<FaceId> &faces, std::ostream &os) const
{
   std::map<Element*, int> element_id;

   // get a list of elements involved, dump them to 'os' and create the mapping
   // element_id: (Element* -> stream ID)
   {
      std::set<Element*> elements;
      for (int i = 0; i < edges.Size(); i++)
         elements.insert(edges[i].element);
      for (int i = 0; i < faces.Size(); i++)
         elements.insert(faces[i].element);

      ElementSet eset(elements, root_elements);
      eset.Dump(os);

      Array<Element*> decoded;
      eset.Get(decoded, root_elements);

      for (int i = 0; i < decoded.Size(); i++)
         element_id[decoded[i]] = i;
   }

   // write edges
   write<int>(os, edges.Size());
   for (int i = 0; i < edges.Size(); i++)
   {
      write<int>(os, element_id[edges[i].element]);
      write<char>(os, edges[i].local);
   }

   // write faces
   write<int>(os, faces.Size());
   for (int i = 0; i < faces.Size(); i++)
   {
      write<int>(os, element_id[faces[i].element]);
      write<char>(os, faces[i].local);
   }
}

void ParNCMesh::DecodeEdgesFaces
(Array<EdgeId> &edges, Array<FaceId> &faces, std::istream &is) const
{
   // read the list of elements
   ElementSet eset(is);

   Array<Element*> elements;
   eset.Get(elements, root_elements);

/*   // read edges
   int ne = read<int>(is);
   edges.SetSize(ne);
   for (int i = 0; i < ne; i++)
   {
      Element* elem = elements[read<int>(is)];
      MFEM_ASSERT(!elem->ref_type, "Not a leaf element.");

      EdgeId &eid = edges[i];
      eid.element = elem;
      eid.local = read<char>(is);

      const int* ev = GI[elem->geom].edges[eid.local];
      Node* node = nodes.Peek(elem->node[ev[0]], elem->node[ev[1]]);

      MFEM_ASSERT(node && node->edge, "Edge not found.");
      eid.index = node->edge->index;
   }

   // read faces
   int nf = read<int>(is);
   faces.SetSize(nf);
   for (int i = 0; i < nf; i++)
   {
      Element* elem = elements[read<int>(is)];
      MFEM_ASSERT(!elem->ref_type, "Not a leaf element.");

      FaceId &fid = faces[i];
      fid.element = elem;
      fid.local = read<char>(is);

      const int* fv = GI[elem->geom].faces[fid.local];
      Face* face = this->faces.Peek(elem->node[fv[0]], elem->node[fv[1]],
                                    elem->node[fv[2]], elem->node[fv[3]]);

      MFEM_ASSERT(face, "Face not found.");
      fid.index = face->index;
   }*/
   // TODO: GI
}

MPI_Request ParNCMesh::NeighborDofMessage::Isend
(int rank, MPI_Comm comm, const ParNCMesh &pncmesh)
{
   Array<EdgeId> edges;
   {
   }

   Array<FaceId> faces;
   {
      /*faces.Reserve(face_dofs.size());
      IdToDof::const_iterator it;
      for (it = face_dofs.begin(); it != face_dofs.end(); ++it)
         faces.Append(it->first);*/
   }

   std::ostringstream stream(data);
   pncmesh.EncodeEdgesFaces(edges, faces, stream);

   // dump the DOFs

   face_dofs.clear();
   edge_dofs.clear();

   return Base::Isend(rank, comm);
}

void ParNCMesh::NeighborDofMessage::Recv
(int rank, int size, MPI_Comm comm, const ParNCMesh &pncmesh)
{
   Base::Recv(rank, size, comm);

   // decode

   data.clear();
}

} // namespace mfem

#endif // MFEM_USE_MPI
