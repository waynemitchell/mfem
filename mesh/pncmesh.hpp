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
   MPI_Request send_req;

   /// Non-blocking send to processor 'rank'.
   void Isend(int rank, MPI_Comm comm)
   {
      MPI_Isend(data.data(), data.length(), MPI_BYTE, rank, Tag, comm, &send_req);
   }

   /// Wait for a preceding Isend to finish.
   void WaitSent()
   {
      MPI_Wait(&send_req, MPI_STATUS_IGNORE);
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

   /** Return a list of vertices shared by this processor and at least one other
       processor. (NOTE: only NCList::conforming will be set.) */
   const NCList& GetSharedVertices()
   {
      if (shared_vertices.Empty()) BuildSharedVertices();
      return shared_vertices;
   }

   /** Return a list of edges shared by this processor and at least one other
       processor. (NOTE: this is a subset of the NCMesh::edge_list; slaves are
       empty.) */
   const NCList& GetSharedEdges()
   {
      if (edge_list.Empty()) BuildEdgeList();
      return shared_edges;
   }

   /** Return a list of faces shared by this processor and another processor.
       (NOTE: this is a subset of NCMesh::face_list; slaves are empty.) */
   const NCList& GetSharedFaces()
   {
      if (face_list.Empty()) BuildFaceList();
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

   /// Return vertex/edge/face ('type' == 0/1/2, resp.) owner.
   int GetOwner(int type, int index)
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
   const int* GetGroup(int type, int index, int &size)
   {
      switch (type)
      {
      case 0: size = vertex_group.RowSize(index); return vertex_group.GetRow(index);
      case 1: size = edge_group.RowSize(index); return edge_group.GetRow(index);
      default: size = face_group.RowSize(index); return face_group.GetRow(index);
      }
   }

   /** */
   class NeighborDofMessage : public VarMessage<135>
   {
   public:
      typedef VarMessage<135> Base;

      /// Add vertex/edge/face DOFs to an outgoing message.
      void AddDofs(int type, const MeshId &id, const Array<int> &dofs)
      {
         MFEM_ASSERT(type >= 0 && type < 3, "");
         id_dofs[type][id].assign(dofs.GetData(), dofs.GetData() + dofs.Size());
      }

      MPI_Request Isend(int rank, MPI_Comm comm, const ParNCMesh &pncmesh);
      void Recv(int rank, int size, MPI_Comm comm, const ParNCMesh &pncmesh);

      /// Get vertex/edge/face DOFs from a received message.
      void GetDofs(int type, const MeshId& id, Array<int>& dofs);

   protected:
      // TODO: we need a Dictionary class if we want to avoid STL
      typedef std::map<MeshId, std::vector<int> > IdToDofs;
      IdToDofs id_dofs[3];
   };

   class NeighborRowsRequest: public VarMessage<312>
   {
      // TODO
   };

   class NeighborRowsReply: public VarMessage<313>
   {
      // TODO
   };

protected:

   MPI_Comm MyComm;
   int NRanks, MyRank;

   int NVertices;
   int NEdges, NGhostEdges;
   int NFaces, NGhostFaces;

   //
   NCList shared_vertices;
   NCList shared_edges;
   NCList shared_faces;

   //
   Array<int> vertex_owner;
   Array<int> edge_owner;
   Array<int> face_owner;

   //
   Table vertex_group;
   Table edge_group;
   Table face_group;

   void InitialPartition();

   virtual void AssignLeafIndices();

   virtual void OnMeshUpdated(Mesh *mesh);

   virtual void BuildEdgeList();
   virtual void BuildFaceList();

   virtual void ElementSharesEdge(Element* elem, Edge* edge);
   virtual void ElementSharesFace(Element* elem, Face* face);

   void BuildSharedVertices();

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

   void AddSlaveRanks(int nitems, const NCList& list);
   void MakeGroups(int nitems, Table &groups);
   void MakeShared(const Table &groups, const NCList &list, NCList &shared);

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

   /// Write to 'os' a processor-independent encoding of vertex/edge/face IDs.
   void EncodeMeshIds(std::ostream &os, Array<MeshId> ids[3]) const;

   /// Read from 'is' a processor-independent encoding of vetex/edge/face IDs.
   void DecodeMeshIds(std::istream &is, Array<MeshId> ids[3]) const;

   friend class ParMesh;
   friend class NeighborDofMessage;

};

inline bool operator< (const NCMesh::MeshId &a, const NCMesh::MeshId &b)
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

TODO
+ shared vertices
- fix NCMesh::BuildEdgeList
- AddSlaveDependencies, AddOneOneDependencies
- Phase 2
- invalid CDOFs?

*/

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_PNCMESH
