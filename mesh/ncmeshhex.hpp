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

#ifndef MFEM_NCMESHHEX
#define MFEM_NCMESHHEX

#include "../fem/geom.hpp"
#include "../general/hash.hpp"

// TODO: these won't be needed once this module is purely geometric
class SparseMatrix;
class DenseMatrix;
class IsoparametricTransformation;
class FiniteElementSpace;


/** \brief A class for non-conforming AMR on higher-order hexahedral,
 *  quadrilateral or triangular meshes.
 *
 *  The class is used as follows:
 *
 *  1. NCMeshHex is constructed from elements of an existing Mesh. The elements
 *     are copied and become the roots of the refinement hierarchy.
 *
 *  2. Some elements are refined with the Refine() method. Both isotropic and
 *     anisotropic refinements of quads/hexes are supported.
 *
 *  3. A new Mesh is created from NCMeshHex containing the leaf elements.
 *     This new mesh may have non-conforming (hanging) edges and faces.
 *
 *  4. A conforming interpolation matrix is obtained using GetInterpolation().
 *     The matrix can be used to constrain the hanging DOFs so a continous
 *     solution is obtained.
 *
 *  5. Refine some more leaf elements, i.e., repeat from step 2.
 */
class NCMeshHex
{
protected:
   struct Node; // forward declaration, not for public use

public:
   NCMeshHex(const Mesh *mesh);

   int Dimension() const { return Dim; }

   /** This is an element in the refinement hierarchy. Each element has
       either been refined and points to its children, or is a leaf and points
       to its vertex nodes. */
   struct Element
   {
      int geom; // Geometry::Type of the element
      int attribute;
      int ref_type; // bit mask of X,Y,Z refinements (bits 0,1,2, respectively)
      union
      {
         Node* node[8]; // element corners (if ref_type == 0)
         Element* child[8]; // 2-8 children (if ref_type != 0)
      };

      Element(int geom, int attr);
   };

   /** Returns one of the elements that are based on the elements of the
       original mesh. These are the roots of the refinement trees. */
   Element* GetRootElement(int index) { return root_elements[index]; }

   /** Returns a leaf (unrefined) element. NOTE: leaf elements are enumerated
       when converting from NCMeshHex to Mesh. Only use this function after. */
   Element* GetLeafElement(int index) { return leaf_elements[index]; }

   int NRootElements() const { return root_elements.Size(); }
   int NLeafElements() const { return leaf_elements.Size(); }

   struct Refinement
   {
      int index; ///< leaf element number
      int ref_type; ///< refinement type (7 = full isotropic)

      Refinement(int index, int type = 7)
         : index(index), ref_type(type) {}
   };

   /** Perform the given batch of refinements. Please note that in the presence
       of anisotropic splits additional refinements may be necessary to keep
       the mesh consistent. However, the function always performas at least the
       requested refinements. */
   void Refine(Array<Refinement>& refinements);

   /** Derefine -- not implemented yet */
   //void Derefine(Element* elem);

   /** Check mesh and potentially refine some elements so that the maximum level
       of hanging nodes is not greater than 'max_level'. */
   void LimitNCLevel(int max_level);

   SparseMatrix* GetInterpolation(Mesh* mesh, FiniteElementSpace* space);

   /// Return total number of bytes allocated.
   long MemoryUsage();

   ~NCMeshHex();


protected: // interface for Mesh to be able to construct itself from us

   void GetVerticesElementsBoundary(Array< ::Vertex>& vertices,
                                    Array< ::Element*>& elements,
                                    Array< ::Element*>& boundary);
   friend class Mesh;


protected: // implementation

   int Dim;

   /** We want vertices and edges to autodestruct when elements stop using
       (i.e., referencing) them. This base class does the reference counting. */
   struct RefCount
   {
      int ref_count;

      RefCount() : ref_count(0) {}

      int Ref() {
         return ++ref_count;
      }
      int Unref() {
         int ret = --ref_count;
         if (!ret) delete this;
         return ret;
      }
   };

   /** A vertex in the NC mesh. Elements point to vertices indirectly through
       their Nodes. */
   struct Vertex : public RefCount
   {
      double pos[3]; ///< 3D position
      int index;     ///< vertex number in the Mesh

      Vertex() {}
      Vertex(double x, double y, double z) : index(-1)
         { pos[0] = x, pos[1] = y, pos[2] = z; }
   };

   /** An NC mesh edge. Edges don't do much more than just exist. */
   struct Edge : public RefCount
   {
      int attribute; ///< boundary element attribute, -1 if internal edge
      int index;     ///< edge number in the Mesh

      Edge() : attribute(-1), index(-1) {}
      bool Boundary() const { return attribute >= 0; }
   };

   /** A Node can hold a Vertex, an Edge, or both. Elements directly point to
       their corner nodes, but edge nodes also exist and can be accessed using
       a hash-table given their two end-point node IDs. All nodes can be
       accessed in this way, with the exception of top-level vertex nodes.
       When an element is being refined, the mid-edge nodes are readily
       available with this mechanism. The new elements "sign in" into the nodes
       to have vertices and edges created for them or to just have their
       reference counts increased. The parent element "signs off" its nodes,
       which decrements the vertex and edge reference counts. Vertices and edges
       are destroyed when their reference count drops to zero. */
   struct Node : public Hashed2<Node>
   {
      Vertex* vertex;
      Edge* edge;

      Node(int id) : Hashed2<Node>(id), vertex(NULL), edge(NULL) {}

      // Bump ref count on a vertex or an edge, or create them. Used when an
      // element starts using a vertex or an edge.
      void RefVertex();
      void RefEdge();

      // Decrement ref on vertex or edge when an element is not using them
      // anymore. The vertex, edge or the whole Node can autodestruct.
      // (The hash-table pointer needs to be known then to remove the node.)
      void UnrefVertex(HashTable<Node>& nodes);
      void UnrefEdge(HashTable<Node>& nodes);

      ~Node();
   };

   /** Similarly to nodes, faces can be accessed by hashing their four vertex
       node IDs. A face knows about the one or two elements that are using it.
       A face that is not on the boundary and only has one element referencing
       it is either a master or a slave face. */
   struct Face : public RefCount, public Hashed4<Face>
   {
      int attribute;    ///< boundary element attribute, -1 if internal face
      int index;        ///< face number in the Mesh
      Element* elem[2]; ///< up to 2 elements sharing the face

      Face(int id) : Hashed4<Face>(id), attribute(-1), index(-1)
         { elem[0] = elem[1] = NULL; }

      bool Boundary() const { return attribute >= 0; }

      // add or remove an element from the 'elem[2]' array
      void RegisterElement(Element* e);
      void ForgetElement(Element* e);

      // return one of elem[0] or elem[1] and make sure the other is NULL
      Element* GetSingleElement() const;

      // overloaded Unref without auto-destruction
      int Unref() { return --ref_count; }
   };

   Array<Element*> root_elements; // initialized by constructor
   Array<Element*> leaf_elements; // updated by UpdateLeafElements

   HashTable<Node> nodes; // associative container holding all Nodes
   HashTable<Face> faces; // associative container holding all Faces

   struct RefStackItem
   {
      Element* elem;
      int ref_type;

      RefStackItem(Element* elem, int type)
         : elem(elem), ref_type(type) {}
   };

   Array<RefStackItem> ref_stack; ///< stack of scheduled refinements

   void Refine(Element* elem, int ref_type);

   void GetLeafElements(Element* e);
   void UpdateLeafElements();

   void DeleteHierarchy(Element* elem);

   Element* NewHexahedron(Node* n0, Node* n1, Node* n2, Node* n3,
                          Node* n4, Node* n5, Node* n6, Node* n7,
                          int attr,
                          int fattr0, int fattr1, int fattr2,
                          int fattr3, int fattr4, int fattr5);

   Element* NewQuadrilateral(Node* n0, Node* n1, Node* n2, Node* n3,
                             int attr,
                             int eattr0, int eattr1, int eattr2, int eattr3);

   Element* NewTriangle(Node* n0, Node* n1, Node* n2,
                        int attr, int eattr0, int eattr1, int eattr2);

   Vertex* NewVertex(Node* v1, Node* v2);

   Node* GetMidEdgeVertex(Node* v1, Node* v2);
   Node* GetMidEdgeVertexSimple(Node* v1, Node* v2);
   Node* GetMidFaceVertex(Node* e1, Node* e2, Node* e3, Node* e4);

   int FaceSplitType(Node* v1, Node* v2, Node* v3, Node* v4,
                     Node* mid[4] = NULL /* optional output of mid-edge nodes*/);

   void ForceRefinement(Node* v1, Node* v2, Node* v3, Node* v4);

   void CheckAnisoFace(Node* v1, Node* v2, Node* v3, Node* v4,
                       Node* mid12, Node* mid34, int level = 0);

   void CheckIsoFace(Node* v1, Node* v2, Node* v3, Node* v4,
                     Node* e1, Node* e2, Node* e3, Node* e4, Node* midf);

   void RefElementNodes(Element *elem);
   void UnrefElementNodes(Element *elem);
   void RegisterFaces(Element* elem);

   Node* PeekAltParents(Node* v1, Node* v2);

   bool NodeSetX1(Node* node, Node** n);
   bool NodeSetX2(Node* node, Node** n);
   bool NodeSetY1(Node* node, Node** n);
   bool NodeSetY2(Node* node, Node** n);
   bool NodeSetZ1(Node* node, Node** n);
   bool NodeSetZ2(Node* node, Node** n);


   // interpolation

   struct Dependency
   {
      int dof;
      double coef;

      Dependency(int dof, double coef)
         : dof(dof), coef(coef) {}
   };

   typedef Array<Dependency> DepList;

   /** Holds temporary data for each nonconforming (FESpace-assigned) DOF
       during the interpolation algorithm. */
   struct DofData
   {
      bool finalized; ///< true if cP matrix row is known for this DOF
      DepList dep_list; ///< list of other DOFs this DOF depends on

      DofData() : finalized(false) {}
      bool Independent() const { return !dep_list.Size(); }
   };

   DofData* dof_data; ///< DOF temporary data

   FiniteElementSpace* space;

   static int find_node(Element* elem, Node* node);

   void ReorderFacePointMat(Node* v0, Node* v1, Node* v2, Node* v3,
                           Element* elem, DenseMatrix& pm);

   void AddDependencies(Array<int>& master_dofs, Array<int>& slave_dofs,
                        DenseMatrix& I);

   void ConstrainEdge(Node* v0, Node* v1,
                      IsoparametricTransformation& edge_T,
                      Array<int>& master_dofs, int level);

   void ConstrainFace(Node* v0, Node* v1, Node* v2, Node* v3,
                      IsoparametricTransformation& face_T,
                      Array<int>& master_dofs, int level);

   void ProcessMasterEdge(Node* node[2], Node* edge);
   void ProcessMasterFace(Node* node[4], Face* face);

   bool DofFinalizable(DofData& vd);


   // utility

   void FaceSplitLevel(Node* v1, Node* v2, Node* v3, Node* v4,
                       int& h_level, int& v_level);

   void CountSplits(Element* elem, int splits[3]);

   int CountElements(Element* elem);

};


#endif
