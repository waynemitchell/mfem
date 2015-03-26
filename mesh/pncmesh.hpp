#ifndef MFEM_PNCMESH
#define MFEM_PNCMESH

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include <map>
#include <set>

#include "ncmesh.hpp"
#include "../general/communication.hpp"
#include "../general/sort_pairs.hpp"

namespace mfem
{

/** TODO: explain
 *
 *  - leaf_elements vs. Mesh elements
 *  - ghost numbering
 *  - face/edge orientation
 *  - who is a neighbor
 *
 */
class ParNCMesh : public NCMesh
{
public:
   ParNCMesh(MPI_Comm comm, const NCMesh& ncmesh);

   /** */
   virtual void Refine(const Array<Refinement> &refinements);

   /** Return a list of vertices shared by this processor and at least one other
       processor. (NOTE: only NCList::conforming will be set.) */
   const NCList& GetSharedVertices()
   {
      if (shared_vertices.Empty()) { BuildSharedVertices(); }
      return shared_vertices;
   }

   /** Return a list of edges shared by this processor and at least one other
       processor. (NOTE: this is a subset of the NCMesh::edge_list; slaves are
       empty.) */
   const NCList& GetSharedEdges()
   {
      if (edge_list.Empty()) { BuildEdgeList(); }
      return shared_edges;
   }

   /** Return a list of faces shared by this processor and another processor.
       (NOTE: this is a subset of NCMesh::face_list; slaves are empty.) */
   const NCList& GetSharedFaces()
   {
      if (face_list.Empty()) { BuildFaceList(); }
      return shared_faces;
   }

   /// Helper to get shared vertices/edges/faces ('type' == 0/1/2 resp.).
   const NCList& GetSharedList(int type)
   {
      switch (type)
      {
         case 0: return GetSharedVertices();
         case 1: return GetSharedEdges();
         default: return GetSharedFaces();
      }
   }

   /// Return (shared) face orientation relative to the owner element.
   int GetFaceOrientation(int index) const
   {
      return face_orient[index];
   }

   /// Return vertex/edge/face ('type' == 0/1/2, resp.) owner.
   int GetOwner(int type, int index) const
   {
      switch (type)
      {
         case 0: return vertex_owner[index];
         case 1: return edge_owner[index];
         default: return face_owner[index];
      }
   }

   /** Return a list of processors sharing a vertex/edge/face
       ('type' == 0/1/2, resp.) and the size of the list. */
   const int* GetGroup(int type, int index, int &size) const
   {
      const Table* table;
      switch (type)
      {
         case 0: table = &vertex_group; break;
         case 1: table = &edge_group; break;
         default: table = &face_group;
      }
      size = table->RowSize(index);
      return table->GetRow(index);
   }

   /** Returns true if 'rank' is in the processor group of a vertex/edge/face
       ('type' == 0/1/2, resp.). */
   bool RankInGroup(int type, int index, int rank) const
   {
      int size;
      const int* group = GetGroup(type, index, size);
      for (int i = 0; i < size; i++)
         if (group[i] == rank) { return true; }
      return false;
   }

   /// Returns true if the specified vertex/edge/face is a ghost.
   bool IsGhost(int type, int index) const
   {
      switch (type)
      {
         case 0: return index >= NVertices;
         case 1: return index >= NEdges;
         default: return index >= NFaces;
      }
   }

   /** Extension of NCMesh::GetBoundaryClosure. Filters out ghost vertices and
       ghost edges from 'bdr_vertices' and 'bdr_edges'. */
   virtual void GetBoundaryClosure(const Array<int> &bdr_attr_is_ess,
                                   Array<int> &bdr_vertices,
                                   Array<int> &bdr_edges);

   /** Extract a debugging Mesh containing all leaf elements, including ghosts.
       The debug mesh will have element attributes set to 1 for real elements
       and to 2 for ghost elements. */
   void GetDebugMesh(Mesh &debug_mesh) const;


protected:
   MPI_Comm MyComm;
   int NRanks, MyRank;

   int NVertices, NGhostVertices;
   int NEdges, NGhostEdges;
   int NFaces, NGhostFaces;

   // lists of vertices/edges/faces shared by us and at least one more processor
   NCList shared_vertices;
   NCList shared_edges;
   NCList shared_faces;

   // owner processor for each vertex/edge/face
   Array<int> vertex_owner;
   Array<int> edge_owner;
   Array<int> face_owner;

   // list of processors sharing each vertex/edge/face
   Table vertex_group;
   Table edge_group;
   Table face_group;

   Array<char> face_orient;
   Array<Element*> index_leaf;

   virtual void Update();

   /// Assigns elements to processors at the initial stage (ParMesh creation).
   int InitialPartition(int index) const
   { return index * NRanks / leaf_elements.Size(); }

   virtual void UpdateVertices();
   virtual void AssignLeafIndices();

   virtual void OnMeshUpdated(Mesh *mesh);

   virtual void BuildEdgeList();
   virtual void BuildFaceList();

   virtual void ElementSharesEdge(Element* elem, Edge* edge);
   virtual void ElementSharesFace(Element* elem, Face* face);

   void BuildSharedVertices();

   void CalcFaceOrientations();

   Array<Connection> index_rank; // temporary

   void AddMasterSlaveRanks(int nitems, const NCList& list);
   void MakeShared(const Table &groups, const NCList &list, NCList &shared);

   /** Uniquely encodes a set of elements in the erefinement hierarchy of an
       NCMesh. Can be dumped to a stream, sent to another processor, loaded,
       and decoded to identify the same set of elements (refinements) in a
       different but compatible NCMesh. The elements don't have to be leaves,
       but they must represent subtrees of 'ncmesh_roots'. */
   class ElementSet
   {
   public:
      ElementSet(const std::set<Element*> &elements,
                 const Array<Element*> &ncmesh_roots);
      void Dump(std::ostream &os) const;

      ElementSet() {}
      ElementSet(std::istream &is) { Load(is); }
      void Load(std::istream &is);
      void Decode(Array<Element*> &elements,
                  const Array<Element*> &ncmesh_roots) const;

   protected:
      Array<unsigned char> data; ///< encoded refinement (sub-)trees

      bool EncodeTree(Element* elem, const std::set<Element*> &elements);
      void DecodeTree(Element* elem, int &pos, Array<Element*> &elements) const;

      void SetInt(int pos, int value);
      int GetInt(int pos) const;
   };

   /// Write to 'os' a processor-independent encoding of vertex/edge/face IDs.
   void EncodeMeshIds(std::ostream &os, Array<MeshId> ids[], int dim) const;

   /// Read from 'is' a processor-independent encoding of vetex/edge/face IDs.
   void DecodeMeshIds(std::istream &is, Array<MeshId> ids[], int dim,
                      bool decode_indices) const;

   /** Return true if an element is on a processor boundary, i.e., if at least
       one of its vertices, edges or faces is shared. */
   bool OnProcessorBoundary(Element* elem) const;

   /** Return a list of processors that own elements in the immediate
       neighborhood of 'elem' (i.e., vertex, edge and face neighbors),
       and are not 'MyRank'. */
   void ElementNeighborProcessors(Element* elem, Array<int> &ranks) const;

   /** Get a list of ranks that own elements in the neighborhood of our region.
       NOTE: MyRank is not included. */
   void GetNeighbors(Array<int> &neighbors);

   /** Traverse the (local) refinement tree and determine which subtrees are
       no longer needed, i.e., their leaves are not owned by us nor are they our
       ghosts. These subtrees are then derefined. */
   void Prune();

   /// Internal. Recursive part of Prune().
   bool PruneTree(Element* elem);


   /**  */
   class NeighborRefinementMessage : public VarMessage<289>
   {
   public:
      std::vector<ElemRefType> refinements;

      void AddRefinement(Element* elem, int ref_type)
      { refinements.push_back(ElemRefType(elem, ref_type)); }

      /// Set pointer to ParNCMesh (needed to encode the message).
      void SetNCMesh(ParNCMesh* pncmesh) { this->pncmesh = pncmesh; }

      typedef std::map<int, NeighborRefinementMessage> Map;

   protected:
      ParNCMesh* pncmesh;

      virtual void Encode();
      virtual void Decode();
   };


   friend class ParMesh;
   friend class NeighborDofMessage;
};

/*
TODO
+ R matrix
+ nonzero essential BC
+ vdim, vdofs
- conforming case R matrix
+ assumed partition
- cP + P
- big-int merge
- big-int P matrix
- ProjectBdrCoefficient
- curved/two-level parmesh
- hcurl/hdiv
- hilbert ordering
- rebalance
- parallel aniso refine
*/

/** */
class NeighborDofMessage : public VarMessage<135>
{
public:
   /// Add vertex/edge/face DOFs to an outgoing message.
   void AddDofs(int type, const NCMesh::MeshId &id, const Array<int> &dofs,
                ParNCMesh* pncmesh);

   /// Set pointer to ParNCMesh (needed to encode the message).
   void SetNCMesh(ParNCMesh* pncmesh) { this->pncmesh = pncmesh; }

   /// Get vertex/edge/face DOFs from a received message.
   void GetDofs(int type, const NCMesh::MeshId& id, Array<int>& dofs);

   typedef std::map<int, NeighborDofMessage> Map;

protected:
   // TODO: we need a Dictionary class if we want to avoid STL
   typedef std::map<NCMesh::MeshId, std::vector<int> > IdToDofs;
   IdToDofs id_dofs[3];

   ParNCMesh* pncmesh;

   virtual void Encode();
   virtual void Decode();

   void ReorderEdgeDofs(const NCMesh::MeshId &id, std::vector<int> &dofs);
};

/** */
class NeighborRowRequest: public VarMessage<312>
{
public:
   std::set<int> rows;

   void RequestRow(int row) { rows.insert(row); }
   void RemoveRequest(int row) { rows.erase(row); }

   typedef std::map<int, NeighborRowRequest> Map;

protected:
   virtual void Encode();
   virtual void Decode();
};

/** */
class NeighborRowReply: public VarMessage<313>
{
public:
   void AddRow(int row, const Array<int> &cols, const Vector &srow);

   bool HaveRow(int row) const { return rows.find(row) != rows.end(); }
   void GetRow(int row, Array<int> &cols, Vector &srow);

   typedef std::map<int, NeighborRowReply> Map;

protected:
   struct Row { std::vector<int> cols; Vector srow; };
   std::map<int, Row> rows;

   virtual void Encode();
   virtual void Decode();
};


// comparison operator so that MeshId can be used as key in std::map
inline bool operator< (const NCMesh::MeshId &a, const NCMesh::MeshId &b)
{
   return a.index < b.index;
}


} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_PNCMESH
