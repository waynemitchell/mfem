#ifndef MFEM_PNCMESH
#define MFEM_PNCMESH

#include "../config.hpp"

#ifdef MFEM_USE_MPI

#include <map>
#include <set>

#include "ncmesh.hpp"
#include "../general/communication.hpp"

namespace mfem
{

/// Variable-length MPI message containing unspecific binary data.
template<int Tag>
struct VarMessage
{
   std::string data;

   /// Non-blocking send to processor 'rank'.
   MPI_Request Isend(int rank, MPI_Comm comm)
   {
      MPI_Request request;
      MPI_Isend(data.data(), data.length(), MPI_BYTE, rank, Tag, comm, &request);
      std::cout << "Sent message length " << data.length() << std::endl;
      return request;
   }

   /** Blocking probe for incoming message of this type from any rank.
       Returns the rank and message size. */
   static void Probe(int &rank, int &size, MPI_Comm comm)
   {
      MPI_Status status;
      MPI_Probe(MPI_ANY_SOURCE, Tag, comm, &status);
      rank = status.MPI_SOURCE;
      MPI_Get_count(&status, MPI_BYTE, &size);
   }

   /// Post-probe receive from processor 'rank' of message size 'size'.
   void Recv(int rank, int size, MPI_Comm comm)
   {
      data.resize(size);
      MPI_Status status;
      MPI_Recv((void*) data.data(), size, MPI_BYTE, rank, Tag, comm, &status);
   }
};




/** \brief
 *
 */
class ParNCMesh : public NCMesh
{
public:
   ParNCMesh(MPI_Comm comm, const NCMesh& ncmesh);

   /** Return a list of edges shared by our processor and at least one other
       processor. This is just a subset of the regular EdgeList. */
   const EdgeList& GetSharedEdges()
   {
      if (edge_list.Empty()) BuildEdgeList();
      return shared_edges;
   }

   /** Return a list of faces shared by our processor and another processor.
       This is just a subset of the regular FaceList. */
   const FaceList& GetSharedFaces()
   {
      if (face_list.Empty()) BuildFaceList();
      return shared_faces;
   }

   /// Return processor owning an edge. (Note: any edge in the Mesh.)
   int EdgeOwner(int index) const { return edge_owner[index]; }

   /// Return processor owning a face. (Note: any face in the Mesh.)
   int FaceOwner(int index) const { return face_owner[index]; }

   /// Return a list of processors sharing an edge, and the list size.
   const int* EdgeGroup(int index, int &size) const
   {
      size = edge_group.RowSize(index);
      return edge_group.GetRow(index);
   }

   /// Return a list of processors sharing a face, and the list size.
   const int* FaceGroup(int index, int &size) const
   {
      size = face_group.RowSize(index);
      return face_group.GetRow(index);
   }

   /// Helper to get edge (type == 0) or face (type == 1) owner.
   int GetOwner(int type, int index)
   {
      return type ? FaceOwner(index) : EdgeOwner(index);
   }

   /// Helper to get edge (type == 0) or face (type == 1) group and its size.
   const int* GetGroup(int type, int index, int &size)
   {
      return type ? FaceGroup(index, size) : EdgeGroup(index, size);
   }

   /** */
   class NeighborDofMessage : public VarMessage<135>
   {
   public:
      typedef VarMessage<135> Base;

      void AddFaceDofs(const FaceId &fid, const Array<int> &dofs);
      void AddEdgeDofs(const EdgeId &eid, const Array<int> &dofs);

      MPI_Request Isend(int rank, MPI_Comm comm, const ParNCMesh &pncmesh);
      void Recv(int rank, int size, MPI_Comm comm, const ParNCMesh &pncmesh);

      void GetFaceDofs(const FaceId& fid, Array<int>& dofs);
      void GetEdgeDofs(const EdgeId& eid, Array<int>& dofs);

      /// Helper to call AddEdgeDofs (type == 0) or AddFaceDofs (type == 1)
      void AddDofs(int type, const FaceId& fid, const Array<int> &dofs)
      { if (type) AddFaceDofs(fid, dofs); else AddEdgeDofs(fid, dofs); }

      /// Helper to call GetEdgeDofs (type == 0) or GetFaceDofs (type == 1)
      void GetDofs(int type, const FaceId& fid, Array<int> &dofs)
      { if (type) GetFaceDofs(fid, dofs); else GetEdgeDofs(fid, dofs); }

   protected:
      // TODO: we need a Dictionary class if we want to avoid STL
      typedef std::map<FaceId, std::vector<int> > IdToDofs;
      IdToDofs face_dofs, edge_dofs;
   };

   class NeighborRowMessage : public VarMessage<312>
   {
      // TODO
   };

protected:

   MPI_Comm MyComm;
   int NRanks, MyRank;

   int NEdges, NGhostEdges;
   int NFaces, NGhostFaces;

   //
   EdgeList shared_edges;
   FaceList shared_faces;

   //
   Array<int> edge_owner;
   Array<int> face_owner;

   //
   Table edge_group;
   Table face_group;

   void InitialPartition();

   virtual void AssignLeafIndices();

   virtual void OnMeshUpdated(Mesh *mesh);

   virtual void BuildEdgeList();
   virtual void BuildFaceList();

   virtual void ElementSharesEdge(Element* elem, Edge* edge);
   virtual void ElementSharesFace(Element* elem, Face* face);

   /// Struct to help sorting edges/faces
   struct IndexRank
   {
      int index, rank;
      IndexRank(int index, int rank) : index(index), rank(rank) {}
      bool operator< (const IndexRank &rhs) const
      { return (index == rhs.index) ? (rank < rhs.rank) : (index < rhs.index); }
      bool operator== (const IndexRank &rhs) const
      { return (index == rhs.index) && (rank == rhs.rank); }
   };

   Array<IndexRank> tmp_ranks;

   void AddSlaveRanks(int nfaces, const FaceList& list);
   void MakeGroups(int nfaces, Table &groups);
   void MakeShared(const Table &groups, const FaceList &list, FaceList &shared);

   /** Uniquely encodes a set of elements in the refinement hierarchy of an
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

   /// Write to 'os' a processor-independent encoding of given edge and face IDs.
   void EncodeEdgesFaces(const Array<EdgeId> &edges, const Array<FaceId> &faces,
                         std::ostream &os) const;

   /// Read from 'is' a processor-independent encoding of edge and face IDs.
   void DecodeEdgesFaces(Array<EdgeId> &edges, Array<FaceId> &faces,
                         std::istream &is) const;

   friend class ParMesh;
   friend class NeighborDofMessage;

};

inline bool operator< (const NCMesh::FaceId &a, const NCMesh::FaceId &b)
{ return a.index < b.index; }

/*
NOTES

- parallel Dependency (ParFiniteElementSpace): {cdof, rank, coef}
- 1-to-1 dependencies, dep_list.Size() == 1, can be overruled by standard NC
  dependencies

P MATRIX ALGORITHM

Phase 1 -- create dependency lists
- for each master face/edge, send cdofs if slave rank different
  (cdofs to be sent collected in a message for each neighbor)
- for shared conforming edges/faces send cdofs if owner
- for each slave/conforming non-owner, mark processors to receive from
- actually send/receive the messages
- local depencency algo as in serial, get cdofs from nonlocal faces/edges
  from the messages
- collect 1-to-1 deps for conforming from the messages

Phase 2 -- construct P matrix
- mark cdofs with empty deps as tdofs
- create P matrix as diag, offd SparseMatrix
- schedule recv from neighbors that own P rows we need
- loop
  - finalize cdofs that can be finalized locally, send (combinations of) P rows
    to neighbors that need them
  - wait for rows from neighbors

MESSAGE (Phase 1)
- list of {edge/face id, cdofs} + ElementSet
- searchable by edge/face index

*/

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_PNCMESH
