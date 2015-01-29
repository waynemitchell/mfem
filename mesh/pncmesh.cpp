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

ParNCMesh::ParNCMesh(MPI_Comm comm, const NCMesh &ncmesh)
   : NCMesh(ncmesh)
{
   MyComm = comm;
   MPI_Comm_size(MyComm, &NRanks);
   MPI_Comm_rank(MyComm, &MyRank);

   InitialPartition();
   AssignLeafIndices();
   UpdateVertices();
   //PruneGhosts();
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

void ParNCMesh::UpdateVertices()
{
   // This is an override of NCMesh::UpdateVertices. This version first
   // assigns Vertex::index to vertices of elements of our rank. Only these
   // vertices then make it to the Mesh in NCMesh::GetMeshComponents.
   // The remaining (ghost) vertices are assigned indices greater or equal to
   // Mesh::GetNV().

   for (HashTable<Node>::Iterator it(nodes); it; ++it)
      if (it->vertex)
         it->vertex->index = -1;

   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      Element* elem = leaf_elements[i];
      if (elem->rank == MyRank)
         for (int j = 0; j < GI[elem->geom].nv; j++)
            elem->node[j]->vertex->index = 0; // mark vertices that we need
   }

   NVertices = 0;
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
      if (it->vertex && it->vertex->index >= 0)
         it->vertex->index = NVertices++;

   vertex_nodeId.SetSize(NVertices);
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
      if (it->vertex && it->vertex->index >= 0)
         vertex_nodeId[it->vertex->index] = it->id;

   NGhostVertices = 0;
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
      if (it->vertex && it->vertex->index < 0)
         it->vertex->index = NVertices + (NGhostVertices++);
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

void ParNCMesh::ElementSharesEdge(Element *elem, Edge *edge)
{
   // Called by NCMesh::BuildEdgeList when an edge is visited in a leaf element.
   // This allows us to determine edge ownership and processors that share it
   // without duplicating all the HashTable lookups in NCMesh::BuildEdgeList().

   int &owner = edge_owner[edge->index];
   owner = std::min(owner, elem->rank);

   tmp_ranks.Append(IndexRank(edge->index, elem->rank));
}

void ParNCMesh::ElementSharesFace(Element* elem, Face* face)
{
   // Analogous to ElementHasEdge.

   int &owner = face_owner[face->index];
   owner = std::min(owner, elem->rank);

   tmp_ranks.Append(IndexRank(face->index, elem->rank));
}

void ParNCMesh::BuildEdgeList()
{
   // This is an extension of NCMesh::BuildEdgeList() which also determines
   // edge ownership, creates edge processor groups and lists shared edges.

   int nedges = NEdges + NGhostEdges;
   edge_owner.SetSize(nedges);
   edge_owner = std::numeric_limits<int>::max();

   tmp_ranks.SetSize(12*leaf_elements.Size() * 3/2);
   tmp_ranks.SetSize(0);

   NCMesh::BuildEdgeList();

   AddSlaveRanks(nedges, edge_list);
   MakeGroups(nedges, edge_group);
   MakeShared(edge_group, edge_list, shared_edges);
}

void ParNCMesh::BuildFaceList()
{
   // This is an extension of NCMesh::BuildFaceList() which also determines
   // face ownership, creates face processor groups and lists shared faces.

   int nfaces = NFaces + NGhostFaces;
   face_owner.SetSize(nfaces);
   face_owner = std::numeric_limits<int>::max();

   tmp_ranks.SetSize(6*leaf_elements.Size() * 3/2);
   tmp_ranks.SetSize(0);

   NCMesh::BuildFaceList();

   AddSlaveRanks(nfaces, face_list);
   MakeGroups(nfaces, face_group);
   MakeShared(face_group, face_list, shared_faces);
}

void ParNCMesh::AddSlaveRanks(int nitems, const NCList& list)
{
   // create a mapping from slave face index to master face index
   Array<int> slave_to_master(nitems);
   slave_to_master = -1;

   for (int i = 0; i < list.slaves.size(); i++)
   {
      const Slave& sf = list.slaves[i];
      slave_to_master[sf.index] = sf.master;
   }

   // We need the groups of master edges/faces to contain the ranks of their
   // slaves (so that master DOFs get sent to those who share the slaves).
   // This can be done by appending more items to 'tmp_ranks' for the masters.
   // (Note that a slave edge can be shared by more than one element/processor.)

   int size = tmp_ranks.Size();
   for (int i = 0; i < size; i++)
   {
      int master = slave_to_master[tmp_ranks[i].index];
      if (master >= 0)
         tmp_ranks.Append(IndexRank(master, tmp_ranks[i].rank));
   }
}

void ParNCMesh::MakeGroups(int nitems, Table &groups)
{
   // The list of processors for each edge/face is obtained by simply sorting
   // the 'tmp_ranks' array, removing duplicities and converting to a Table.

   tmp_ranks.Sort();
   tmp_ranks.Unique();
   int size = tmp_ranks.Size();

   // create CSR array I of row beginnings
   int* I = new int[nitems+1];
   int next_I = 0, last_index = -1;
   for (int i = 0; i < size; i++)
   {
      if (tmp_ranks[i].index != last_index)
      {
         I[next_I++] = i;
         last_index = tmp_ranks[i].index;
      }
   }
   I[next_I] = size;
   MFEM_ASSERT(next_I == nitems, "");

   // J array is ready-made
   int* J = new int[size];
   for (int i = 0; i < size; i++)
      J[i] = tmp_ranks[i].rank;

   // we have a CSR table of ranks for each edge/face
   groups.SetIJ(I, J, nitems);

   tmp_ranks.DeleteAll();
}

static bool is_shared(const Table& groups, int index, int MyRank)
{
   // An edge/face is shared if its group contains more than one processor and
   // at the same time one of them is ourselves.

   int size = groups.RowSize(index);
   if (size <= 1)
      return false;

   const int* group = groups.GetRow(index);
   for (int i = 0; i < size; i++)
      if (group[i] == MyRank)
         return true;

   return false;
}

void ParNCMesh::MakeShared
(const Table &groups, const NCList &list, NCList &shared)
{
   // Filter the full lists, only output items that are shared.
   // Slaves are omitted as they need to be obtained from the full list.

   shared.Clear();

   for (int i = 0; i < list.conforming.size(); i++)
      if (is_shared(groups, list.conforming[i].index, MyRank))
         shared.conforming.push_back(list.conforming[i]);

   for (int i = 0; i < list.masters.size(); i++)
      if (is_shared(groups, list.masters[i].index, MyRank))
         shared.masters.push_back(list.masters[i]);
}

void ParNCMesh::BuildSharedVertices()
{
   int nvertices = NVertices + NGhostVertices;
   vertex_owner.SetSize(nvertices);
   vertex_owner = std::numeric_limits<int>::max();

   tmp_ranks.SetSize(8*leaf_elements.Size());
   tmp_ranks.SetSize(0);

   Array<MeshId> vertex_id(nvertices);

   // similarly to edges/faces, we loop over the vertices of all leaf elements
   // to determine which processors share each vertex
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      Element* elem = leaf_elements[i];
      for (int j = 0; j < GI[elem->geom].nv; j++)
      {
         Node* node = elem->node[j];
         int index = node->vertex->index;

         int &owner = vertex_owner[index];
         owner = std::min(owner, elem->rank);

         tmp_ranks.Append(IndexRank(index, elem->rank));

         MeshId &id = vertex_id[index];
         id.index = (node->edge ? -1 : index);
         id.element = elem;
         id.local = j;
      }
   }

   MakeGroups(nvertices, vertex_group);

   // create a list of shared vertices, skip obviously slave vertices
   // (for simplicity, we don't guarantee to skip all slave vertices)
   shared_vertices.Clear();
   for (int i = 0; i < nvertices; i++)
   {
      if (is_shared(vertex_group, i, MyRank) && vertex_id[i].index >= 0)
         shared_vertices.conforming.push_back(vertex_id[i]);
   }
}


//// ElementSet ////////////////////////////////////////////////////////////////

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
      int mpos = data.Size();
      data.Append(0);

      // check the subtrees
      int mask = 0;
      for (int i = 0; i < 8; i++)
         if (elem->child[i])
            if (EncodeTree(elem->child[i], elements))
               mask |= (unsigned char) 1 << i;

      if (mask)
         data[mpos] = mask;
      else
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

void ParNCMesh::ElementSet::Decode
(Array<Element*> &elements, const Array<Element*> &ncmesh_roots) const
{
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
   write<int>(os, data.Size());
   os.write((const char*) data.GetData(), data.Size());
}

void ParNCMesh::ElementSet::Load(std::istream &is)
{
   data.SetSize(read<int>(is));
   is.read((char*) data.GetData(), data.Size());
}


//// EncodeMeshIds/DecodeMeshIds ///////////////////////////////////////////////

void ParNCMesh::EncodeMeshIds(std::ostream &os, Array<MeshId> ids[3]) const
{
   std::map<Element*, int> element_id;

   // get a list of elements involved, dump them to 'os' and create the mapping
   // element_id: (Element* -> stream ID)
   {
      std::set<Element*> elements;
      for (int type = 0; type < 3; type++)
         for (int i = 0; i < ids[type].Size(); i++)
            elements.insert(ids[type][i].element);

      ElementSet eset(elements, root_elements);
      eset.Dump(os);

      Array<Element*> decoded;
      eset.Decode(decoded, root_elements);

      for (int i = 0; i < decoded.Size(); i++)
         element_id[decoded[i]] = i;
   }

   // write the IDs as element/local pairs
   for (int type = 0; type < 3; type++)
   {
      write<int>(os, ids[type].Size());
      for (int i = 0; i < ids[type].Size(); i++)
      {
         const MeshId& id = ids[type][i];
         write<int>(os, element_id[id.element]); // TODO: variable 1-4 bytes
         write<char>(os, id.local);
      }
   }
}

void ParNCMesh::DecodeMeshIds(std::istream &is, Array<MeshId> ids[3]) const
{
   // read the list of elements
   ElementSet eset(is);

   Array<Element*> elements;
   eset.Decode(elements, root_elements);

   // read vertex/edge/face IDs
   for (int type = 0; type < 3; type++)
   {
      int ne = read<int>(is);
      ids[type].SetSize(ne);

      for (int i = 0; i < ne; i++)
      {
         Element* elem = elements[read<int>(is)];
         MFEM_ASSERT(!elem->ref_type, "Not a leaf element.");

         MeshId &id = ids[type][i];
         id.element = elem;
         id.local = read<char>(is);

         // find vertex/edge/face index
         GeomInfo &gi = GI[elem->geom];
         switch (type)
         {
            case 0:
            {
               id.index = elem->node[id.local]->vertex->index;
               break;
            }
            case 1:
            {
               const int* ev = gi.edges[id.local];
               Node* node = nodes.Peek(elem->node[ev[0]], elem->node[ev[1]]);
               MFEM_ASSERT(node && node->edge, "Edge not found.");
               id.index = node->edge->index;
               break;
            }
            default:
            {
               const int* fv = gi.faces[id.local];
               Face* face = faces.Peek(elem->node[fv[0]], elem->node[fv[1]],
                                       elem->node[fv[2]], elem->node[fv[3]]);
               MFEM_ASSERT(face, "Face not found.");
               id.index = face->index;
            }
         }
      }
   }
}


//// Messages //////////////////////////////////////////////////////////////////

void NeighborDofMessage::AddDofs
(int type, const NCMesh::MeshId &id, const Array<int> &dofs, ParNCMesh* pncmesh)
{
   MFEM_ASSERT(type >= 0 && type < 3, "");
   id_dofs[type][id].assign(dofs.GetData(), dofs.GetData() + dofs.Size());
   this->pncmesh = pncmesh;
}

void NeighborDofMessage::GetDofs
(int type, const NCMesh::MeshId& id, Array<int>& dofs)
{
   MFEM_ASSERT(type >= 0 && type < 3, "");
#ifdef MFEM_DEBUG
   if (id_dofs[type].find(id) == id_dofs[type].end())
      MFEM_ABORT("Type/ID " << type << "/" << id.index
                 << " not found in neighbor message.");
#endif
   std::vector<int> &vec = id_dofs[type][id];
   dofs.MakeRef(vec.data(), vec.size());
}

static void write_dofs(std::ostream &os, const std::vector<int> &dofs)
{
   write<int>(os, dofs.size());
   // TODO: we should compress the ints, mostly they are contiguous ranges
   os.write((const char*) dofs.data(), dofs.size() * sizeof(int));
}

static void read_dofs(std::istream &is, std::vector<int> &dofs)
{
   dofs.resize(read<int>(is));
   is.read((char*) dofs.data(), dofs.size() * sizeof(int));
}

void NeighborDofMessage::Encode()
{
   IdToDofs::const_iterator it;

   // collect vertex/edge/face IDs
   Array<NCMesh::MeshId> ids[3];
   for (int type = 0; type < 3; type++)
   {
      ids[type].Reserve(id_dofs[type].size());
      for (it = id_dofs[type].begin(); it != id_dofs[type].end(); ++it)
         ids[type].Append(it->first);
   }

   // encode the IDs
   std::ostringstream stream;
   pncmesh->EncodeMeshIds(stream, ids);

   // dump the DOFs
   for (int type = 0; type < 3; type++)
   {
      for (it = id_dofs[type].begin(); it != id_dofs[type].end(); ++it)
         write_dofs(stream, it->second);

      // no longer need the original data
      id_dofs[type].clear();
   }

   stream.str().swap(data);
}

void NeighborDofMessage::Decode()
{
   std::istringstream stream(data);

   // decode vertex/edge/face IDs
   Array<NCMesh::MeshId> ids[3];
   pncmesh->DecodeMeshIds(stream, ids);

   // load DOFs
   for (int type = 0; type < 3; type++)
   {
      id_dofs[type].clear();
      for (int i = 0; i < ids[type].Size(); i++)
         read_dofs(stream, id_dofs[type][ids[type][i]]);
   }

   // no longer need the raw data
   data.clear();
}

void NeighborRowRequest::Encode()
{
   std::ostringstream stream;

   // write the int set to the stream
   write<int>(stream, rows.size());
   for (std::set<int>::iterator it = rows.begin(); it != rows.end(); ++it)
      write<int>(stream, *it);

   rows.clear();
   stream.str().swap(data);
}

void NeighborRowRequest::Decode()
{
   std::istringstream stream(data);

   // read the int set from the stream
   rows.clear();
   int size = read<int>(stream);
   for (int i = 0; i < size; i++)
      rows.insert(rows.end(), read<int>(stream));

   data.clear();
}

void NeighborRowReply::AddRow
(int row, const Array<int> &cols, const Vector &srow)
{
   Row& row_data = rows[row];
   row_data.cols.assign(cols.GetData(), cols.GetData() + cols.Size());
   row_data.srow = srow;
}

void NeighborRowReply::GetRow(int row, Array<int> &cols, Vector &srow)
{
#ifdef MFEM_DEBUG
   if (rows.find(row) == rows.end())
      MFEM_ABORT("Row " << row << " not found in neighbor message.");
#endif
   Row& row_data = rows[row];
   cols.MakeRef(row_data.cols.data(), row_data.cols.size());
   srow.SetDataAndSize(row_data.srow.GetData(), row_data.srow.Size());
}

void NeighborRowReply::Encode()
{
   std::ostringstream stream;

   // dump the rows to the stream
   write<int>(stream, rows.size());
   for (std::map<int, Row>::iterator it = rows.begin(); it != rows.end(); ++it)
   {
      write<int>(stream, it->first); // row number
      Row& row_data = it->second;
      MFEM_ASSERT(row_data.cols.size() == row_data.srow.Size(), "");
      write_dofs(stream, row_data.cols);
      stream.write((const char*) row_data.srow.GetData(),
                   sizeof(double) * row_data.srow.Size());
   }

   rows.clear();
   stream.str().swap(data);
}

void NeighborRowReply::Decode()
{
   std::istringstream stream(data);

   // NOTE: there is no rows.clear() since a row reply can be received
   // repeatedly and the received rows accumulate.

   // read the rows
   int size = read<int>(stream);
   for (int i = 0; i < size; i++)
   {
      Row& row_data = rows[read<int>(stream)];
      read_dofs(stream, row_data.cols);
      row_data.srow.SetSize(row_data.cols.size());
      stream.read((char*) row_data.srow.GetData(),
                  sizeof(double) * row_data.srow.Size());
   }

   data.clear();
}

} // namespace mfem

#endif // MFEM_USE_MPI
