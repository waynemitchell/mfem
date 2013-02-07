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

#ifndef MFEM_NCMESH
#define MFEM_NCMESH

class Mesh;
class FiniteElementSpace;
class GridFunction;

class NonconformingMesh;
// Declare a short name for NonconformingMesh = NCMesh
typedef NonconformingMesh NCMesh;

/** A class for general nonconforming mesh (with hanging nodes), supporting
    local refinement and de-refinement. */
class NonconformingMesh
{
public:
   class Face;
   class Edge;
   class Vertex;

   /** Base class for managing data associated with vertices, edges, faces.
       This can be achieved by deriving new data-manager classes that define new
       vertex, edge, face sub-classes extended with relevant data. The role of
       this class is to maintain the extended data by implementing the virtual
       methods in the base class which will be called when the nonconforming
       mesh is refined or de-refined. The base class provides support for
       encoding and accessing vertex location relative to the coarse-level
       vertices, edges, faces. The NonconformingMesh class has a pointer to a
       Data_manager which is optional (i.e. it may be NULL) and should be used
       only when necessary. */
   class Data_manager
   {
   protected:
      int c_face_idx, c_face_nv;
      int *c_face_edges;
      Face *c_face;
      Vertex *c_face_verts[4];
      const IntegrationRule *ref_verts;

      // assumes that 'c_edge_idx' is an edge of the current coarse face
      int GetEdgeCoords(double a, int c_edge_idx, double *x);

      void Avg(const Vertex *v0, const Vertex *v1, Vertex *v);
      void Avg(const Edge *edge, Vertex *v)
      { Avg(edge->vertex[0], edge->vertex[1], v); }
      void Avg(const Face *quad, Vertex *v)
      { Avg(quad->edge[0]->child[0]->vertex[1],
            quad->edge[2]->child[0]->vertex[1], v); }

      const IntegrationPoint &GetCoarseVertexCoords(const Vertex *v);
      void GetVertexCoords(const Vertex *v, double *y);

   public:
      NonconformingMesh *ncmesh;
      Data_manager() { c_face_idx = -1; ncmesh = NULL; }
      virtual void SetCoarseFace(int _c_face_idx);

      // Adjust the interior data on the given edge assuming that (at least)
      // one of its vertices was adjusted (moved). It is also assumed that all
      // interior vertices are dependent, i.e. the edge is a real edge.
      virtual void AdjustRealEdgeData(Edge *edge) { }

      virtual void AdjustInteriorFaceData(Face *face) { }

      void GetFaceToCoarseFacePointMatrix(Face *face, DenseMatrix &pm);
      void GetFaceToCoarseFaceIPT(
         Face *face, IntegrationPointTransformation &ipT);
      const IntegrationRule &GetReferenceVertices() { return *ref_verts; }

      // Adjust the dependent interior data on the given edge assuming the
      // given vertices were adjusted (moved). The 'edge' must have only
      // truly-refined ancestors.
      void AdjustEdgeData(Edge *edge, bool v0, bool v1);

      // Adjust the dependent interior data of the given triangle face assuming
      // some of its edges were adjusted. A vertex is also assumed to have been
      // adjusted if both of its adjacent edges were adjusted.
      void AdjustTriangleData(Face *face, bool e0, bool e1, bool e2);

      // Adjust the dependent interior data of the given quad face assuming
      // some of its edges were adjusted. A vertex is also assumed to have been
      // adjusted if both of its adjacent edges were adjusted.
      void AdjustQuadData(Face *face, bool e0, bool e1, bool e2, bool e3);

      // Adjust the dependent interior data of the given face assuming
      // the given edge was adjusted (moved).
      void AdjustFaceData(Face *face, Edge *edge);

      virtual Vertex *NewCoarseVertex(int c_vert_idx);
      virtual Vertex *NewEdgeVertex(Edge *edge);
      virtual Vertex *NewQuadVertex(Face *quad);
      virtual Edge *NewEdge(Edge *parent) { return new Edge(parent); }
      virtual Face *NewFace(Face *parent) { return new Face(parent); }

      virtual void TrulyRefineEdge(Edge *edge, Face *newly_refined_face) { }
      virtual void DerefineEdge(Edge *edge, Face *derefined_face) { }
      virtual void RefineFace(Face *face) { }
      virtual void DerefineFace(Face *face) { }

      virtual void FreeVertex(Vertex *vtx) { delete vtx; }
      virtual void FreeEdge(Edge *edge) { delete edge; }
      virtual void FreeFace(Face *face) { delete face; }
   };

   /** A vertex with an integer 'id' and two double 'coordinates'. The base
       Data_manager class can be used to maintains the vertex coordinates in
       encoded form relative the coarse-level vertices, edges, faces. */
   class Vertex
   {
   public:
      int id;
      double x[2];

      Vertex() { id = -1; }
      Vertex(int c_vert_idx) { id = -1; x[0] = 0.0; x[1] = -c_vert_idx; }

      void Set(const double z0, const double z1)
      { x[0] = z0; x[1] = z1; }

      void Set(Vertex *parent0, Vertex *parent1)
      {
         x[0] = (parent0->x[0] + parent1->x[0])/2;
         x[1] = (parent0->x[1] + parent1->x[1])/2;
      }
   };

   /// Edge with at most 2 adjacent faces
   class Edge
   {
   public:
      int id;
      Edge *parent;
      Edge *child[2];
      Face *face[2];
      Vertex *vertex[2];

      Edge(Edge *my_parent);

      bool isRefined() const { return (child[0] != NULL); }

      /** Check to see if the edge and its adjacent faces are refined.
          Valid only if all the ancestors of 'this' are truly refined. */
      inline bool isTrulyRefined() const;

      /// Refine an edge when an adjacent face is being refined
      void Refine(Face *newly_refined_face, Face *child0, Face *child1,
                  Data_manager *man = NULL);

      /// De-refine an edge when an adjacent face is being de-refined
      void Derefine(Face *derefined_face, Data_manager *man = NULL);

      /** Set the coordinates, x, of the interior vertices on this edge
          based on the coordinates of the vertices of this edge. */
      void SetInteriorVertices();

      int Check() const;

      void FreeHierarchy(Data_manager *man = NULL, bool free_this = true);

      // ~Edge();
   };

   /// Triangular or quadrilateral face
   class Face
   {
   public:
      int id;
      Face *parent;
      Face *child[4];
      Edge *edge[4];

      Face(Face *my_parent);

      bool isRefined() const { return (child[0] != NULL); }

      bool isQuad() const { return (edge[3] != NULL); }

      /// Refine the face
      void Refine(Data_manager *man = NULL);

      /// De-refine the face
      void Derefine(Data_manager *man = NULL);

      /** Fill the 'face_vert' array with pointers to the vertices of
          the face and return number of vertices, 3 or 4. */
      int GetVertices(Vertex **face_vert);

      void SetReferenceVertices();

      void SetInteriorVertices();

      int Check() const;

      void FreeHierarchy(Data_manager *man = NULL, bool free_this = true);

      // ~Face();
   };

   class Face_iterator_base
   {
   protected:
      Face *face;

      Face_iterator_base() { }
      Face_iterator_base(Face *_face) { face = _face; }

   public:
      operator Face *() const { return face; }
      Face &operator*() const { return *face; }
      Face *operator->() const { return face; }
   };

   /// Iterator over the tree specified by the given root
   class SimpleFace_iterator : public Face_iterator_base
   {
   protected:
      Face *root;

      void find_leaf() { while (face->isRefined()) face = face->child[3]; }

   public:
      SimpleFace_iterator(Face *_root, bool forward = true)
      { root = _root; if (forward) start(); else end(); }

      void start() { face = root; }
      void end() { face = root; if (face) find_leaf(); }

      void next();
      void prev();

      /// Prefix increment
      SimpleFace_iterator &operator++() { next(); return *this; }
      /// Prefix decrement
      SimpleFace_iterator &operator--() { prev(); return *this; }
      /// Postfix decrement
      Face *operator--(int) { Face *f = face; prev(); return f; }
   };

   /// Iterator over all faces (all levels) of a nonconforming mesh
   class AllFace_iterator : public Face_iterator_base
   {
   protected:
      int c_face_idx;
      const Array<Face *> &c_faces;

      inline void next_coarse();
      void next_up();

   public:
      /// Construct an iterator to the first face of a nonconforming mesh
      AllFace_iterator(const NonconformingMesh &ncmesh, int c_face_start = 0);

      int CoarseFace() const { return c_face_idx; }

      inline void next()
      { if (face->isRefined()) face = face->child[0]; else next_up(); }

      /// Prefix increment
      AllFace_iterator &operator++() { next(); return *this; }
   };

   /// Iterator over the fine faces (the leaves) of a nonconforming mesh
   class Face_iterator : public AllFace_iterator
   {
   protected:
      inline void find_leaf()
      { while (face->isRefined()) face = face->child[0]; }

   public:
      /// Construct an iterator to the first face of a nonconforming mesh
      Face_iterator(const NonconformingMesh &ncmesh, int c_face_start = 0)
         : AllFace_iterator(ncmesh, c_face_start) { if (face) find_leaf(); }

      inline void next() { next_up(); if (face) find_leaf(); }

      /// Prefix increment
      Face_iterator &operator++() { next(); return *this; }
   };

   /// Iterator over the refined faces of a nonconforming mesh
   class RefinedFace_iterator : public AllFace_iterator
   {
   public:
      RefinedFace_iterator(const NonconformingMesh &ncmesh);
      RefinedFace_iterator(const RefinedFace_iterator &it)
         : AllFace_iterator(it) { }

      void next();

      /// Prefix increment
      RefinedFace_iterator &operator++() { next(); return *this; }
   };

   class Edge_iterator_base
   {
   protected:
      Edge *edge;

      Edge_iterator_base() { }
      Edge_iterator_base(Edge *e) { edge = e; }

   public:
      operator Edge *() const { return edge; }
      Edge &operator*() const { return *edge; }
      Edge *operator->() const { return edge; }
   };

   /// Iterator over the tree specified by the given root
   class SimpleEdge_iterator : public Edge_iterator_base
   {
   protected:
      Edge *root;

      void find_leaf() { while (edge->isRefined()) edge = edge->child[1]; }

   public:
      SimpleEdge_iterator(Edge *_root, bool forward = true)
      { root = _root; if (forward) start(); else end(); }

      void start() { edge = root; }
      void end() { edge = root; if (edge) find_leaf(); }

      void next_up();
      void next();
      void prev();

      /// Prefix increment
      SimpleEdge_iterator &operator++() { next(); return *this; }
      /// Prefix decrement
      SimpleEdge_iterator &operator--() { prev(); return *this; }
      /// Postfix decrement
      Edge *operator--(int) { Edge *e = edge; prev(); return e; }
   };

   class AllVertex_iterator;

   /// Iterator over all edges of a nonconforming mesh
   class AllEdge_iterator : public Edge_iterator_base
   {
   protected:
      RefinedFace_iterator face;
      int c_edge_idx;
      const Array<Edge *> &c_edges;

      friend class AllVertex_iterator;

      Vertex *next_up();

      inline Vertex *next_or_vertex();

   public:
      /// Construct an iterator to the first edge of a nonconforming mesh
      AllEdge_iterator(const NonconformingMesh &ncmesh, int c_edge_start = 0);

      int CoarseEdge() const { return c_edge_idx; }
      int CoarseFace() const { return face.CoarseFace(); }

      void next()
      { if (edge->isRefined()) edge = edge->child[0]; else next_up(); }

      /// Prefix increment
      AllEdge_iterator &operator++() { next(); return *this; }
   };

   class Vertex_iterator;

   /** Iterator over all truly refined edges of a nonconforming mesh.
       A truly refined edge is one that satisfies both:
       a) its parent (if not NULL) is truly refined, and
       b) 'isTrulyRefined' is true for that edge. */
   class TrulyRefinedEdge_iterator : public AllEdge_iterator
   {
   protected:
      friend class Vertex_iterator;

      Vertex *next_or_vertex();

   public:
      TrulyRefinedEdge_iterator(const NonconformingMesh &ncmesh);

      void next();

      /// Prefix increment
      TrulyRefinedEdge_iterator &operator++() { next(); return *this; }
   };

   /** Iterator over the real edges of a nonconforming mesh.
       A real edge is one that satifies:
       a) its parent (if not NULL) is truly refined, and
       b) the edge itself is not truly refined. */
   class Edge_iterator : public AllEdge_iterator
   {
      inline void find_real_edge()
      { while (edge->isTrulyRefined()) edge = edge->child[0]; }

   public:
      /// Construct an iterator to the first edge of a nonconforming mesh
      Edge_iterator(const NonconformingMesh &ncmesh, int c_edge_start = 0)
         : AllEdge_iterator(ncmesh, c_edge_start)
      { if (edge) find_real_edge(); }

      void next() { next_up(); if (edge) find_real_edge(); }

      /// Prefix increment
      Edge_iterator &operator++() { next(); return *this; }
   };

   class Vertex_iterator_base
   {
   protected:
      Vertex *vert;

      Vertex_iterator_base() { }

   public:
      operator Vertex *() const { return vert; }
      Vertex &operator*() const { return *vert; }
      Vertex *operator->() const { return vert; }
   };

   /// Iterator over all vertices of a nonconforming mesh
   class AllVertex_iterator : public Vertex_iterator_base
   {
   protected:
      int c_vert_idx;
      const Array<Vertex *> &c_verts;
      AllEdge_iterator edge;

   public:
      /// Construct an iterator to the first edge of a nonconforming mesh
      AllVertex_iterator(const NonconformingMesh &ncmesh)
         : c_verts(ncmesh.c_verts), edge(ncmesh)
      { vert = c_verts[c_vert_idx = 0]; }

      void next();

      void parent_vertices(Vertex **pv);

      /// Prefix increment
      AllVertex_iterator &operator++() { next(); return *this; }
   };

   /// Iterator over the independent vertices of a nonconforming mesh
   class Vertex_iterator : public Vertex_iterator_base
   {
   protected:
      int c_vert_idx;
      const Array<Vertex *> &c_verts;
      TrulyRefinedEdge_iterator edge;

   public:
      /// Construct an iterator to the first vertex of a nonconforming mesh
      Vertex_iterator(const NonconformingMesh &ncmesh)
         : c_verts(ncmesh.c_verts), edge(ncmesh)
      { vert = c_verts[c_vert_idx = 0]; }

      void next();

      /// Prefix increment
      Vertex_iterator &operator++() { next(); return *this; }
   };

   /// Iterator over the dependent vertices on a given real edge
   class DependentVertex_iterator : public Vertex_iterator_base
   {
   protected:
      SimpleEdge_iterator edge;

   public:
      DependentVertex_iterator(Edge *e) : edge(e)
      { vert = (edge->isRefined()) ? edge->child[0]->vertex[1] : NULL; }

      const Edge *parent() { return edge; }

      inline void next();

      /// Prefix increment
      DependentVertex_iterator &operator++() { next(); return *this; }
   };

   // NonconformingMesh members
   Array<Vertex *> c_verts;
   Array<Edge *> c_edges;
   Array<Face *> c_faces;

   // Coarse edge indices per coarse face (stride is 4). If an edge of a
   // coarse face is not a coarse edge, the index must be set to -1.
   // Needed only when data_manager is not NULL.
   Array<int> c_face_edge;

   Data_manager *data_manager;

   /** Create a nonconforming mesh with one coarse element: Geometry::TRIANGLE
       or Geometry::SQUARE. */
   NonconformingMesh(int geom);

   /// Create a nonconforming mesh from a 2D mesh
   NonconformingMesh(const Mesh *mesh, Data_manager *man = NULL);

   /// Refine a face
   void Refine(const AllFace_iterator &face)
   {
      if (data_manager)
         data_manager->SetCoarseFace(face.CoarseFace());
      face->Refine(data_manager);
   }
   /// De-refine a face
   void Derefine(const AllFace_iterator &face)
   {
      if (data_manager)
         data_manager->SetCoarseFace(face.CoarseFace());
      face->Derefine(data_manager);
   }

   /** Reconstruct a (refined) nonconforming mesh as a 2D mesh. The new mesh is
       cut along edges containing dependent nodes. The option for 'bdr_type'
       are:
       0) inherit the boundary from the original mesh
       1) generate the boundary from the new mesh topology
       2) generate boundary from the coarse edges. */
   Mesh *GetRefinedMesh(Mesh *c_mesh, bool remove_curv = false,
                        int bdr_type = 0) const;

   /** Interpolate a GridFunction on the refined mesh. The
       FiniteElementCollection of the fine GridFunction must be:
       1) the same as the FiniteElementCollection of the coarse GridFunction or
       2) a collection of nodal FiniteElements. */
   void GetRefinedGridFunction(Mesh *c_mesh, GridFunction *c_sol,
                               Mesh *f_mesh, GridFunction *f_sol) const;

   /** Interpolate a GridFunction on the refined mesh. If 'linearize' is true
       the new GridFunction uses LinearFECollection, otherwise it uses the
       same FiniteElementCollection as 'c_sol'. The new GridFunction owns the
       new FiniteElementCollection (LinearFECollection or a copy of the coarse
       one) and the new FiniteElementSpace. */
   GridFunction *GetRefinedGridFunction(Mesh *c_mesh, GridFunction *c_sol,
                                        Mesh *f_mesh, bool linearize) const;

   SparseMatrix *GetInterpolation(Mesh *f_mesh, FiniteElementSpace *f_fes);

   ~NonconformingMesh();
};


// Inline methods

inline bool NCMesh::Edge::isTrulyRefined() const
{
   // check to see if the edge and its adjacent faces are refined
   return (this->isRefined() &&
           (!face[0] || face[0]->isRefined()) &&
           (!face[1] || face[1]->isRefined()));
}

inline void NCMesh::AllFace_iterator::next_coarse()
{
   if (++c_face_idx < c_faces.Size())
      face = c_faces[c_face_idx];
   else
      face = NULL;
}

inline NCMesh::Vertex *NCMesh::AllEdge_iterator::next_or_vertex()
{
   if (edge->isRefined())
      edge = edge->child[0];
   else
      return next_up();
   return NULL;
}

inline void NCMesh::DependentVertex_iterator::next()
{
   for (++edge; edge; ++edge)
      if (edge->isRefined())
      {
         vert = edge->child[0]->vertex[1];
         return;
      }
   vert = NULL;
}

#endif
