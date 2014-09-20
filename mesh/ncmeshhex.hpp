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

#include "../general/hash.hpp"

class FiniteElementCollection;


/**
 *
 */
class NCMeshHex
{
protected:
   struct Node; // forward declaration, not for public use

public:
   NCMeshHex(const Mesh *mesh);

   ///
   struct Element
   {
      Node* node[8]; // element corners
      int ref_type; // bit mask of X,Y,Z refinements (bits 0,1,2, respectively)
      Element* child[8]; // children (if ref_type != 0)
      int attribute;

      Element(int attr);
   };

   /** Returns one of the elements that are based on the elements of the
       original mesh. These are the roots of the refinement trees. */
   Element* GetRootElement(int index) { return root_elements[index]; }

   int NRootElements() const { return root_elements.Size(); }
   int NLeafElements() const { return num_leaf_elements; }

   void Refine(Element* elem, int ref_type /*= 7*/);
   void Derefine(Element* elem);

   SparseMatrix *GetInterpolation(Mesh *f_mesh, FiniteElementSpace *f_fes);

   ~NCMeshHex();

protected: // interface for Mesh to be able to construct itself from us

   void GetVertices(Array< ::Vertex>& vertices);
   void GetElements(Array< ::Element*>& elements,
                    Array< ::Element*>& boundary);

   friend class Mesh;

protected: // implementation

   ///
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

   ///
   struct Vertex : public RefCount
   {
      double pos[3];
      int index; ///< internal numbering, set by IndexVertices to 0..NV-1

      Vertex() {}
      Vertex(double x, double y, double z)
         { pos[0] = x, pos[1] = y, pos[2] = z; }
   };

   ///
   struct Edge : public RefCount
   {
      int index; ///< internal numbering, set by IndexEdges to 0..NE-1
   };

   ///
   struct Node : public Hashed2
   {
      Vertex* vertex;
      Edge* edge;

      Node(int id) : Hashed2(id), vertex(NULL), edge(NULL) {}

      // Bump ref count on a vertex or an edge, or create them. Used when an
      // element starts using a vertex or an edge.
      void RefVertex();
      void RefEdge();

      // Decrement ref on vertex or edge when an element is not using them
      // anymore. The vertex, edge or the whole Node can auto-destruct.
      // The hash-table pointer needs to be known then to remove the node.
      void UnrefVertex(HashTable<Node>& nodes);
      void UnrefEdge(HashTable<Node>& nodes);

      ~Node();
   };

   ///
   struct Face : public RefCount, public Hashed4
   {
      int attribute; ///< boundary element attribute, -1 if internal face
      int index; ///<  internal numbering, set by IndexFaces to 0..NF-1

      Face(int id) : Hashed4(id), attribute(-1) {}

      // overloaded Unref without auto-destruction
      int Unref() { return --ref_count; }
   };

   Array<Element*> root_elements;
   HashTable<Node> nodes;
   HashTable<Face> faces;

   int num_leaf_elements;

   int FaceSplitType(Node* v1, Node* v2, Node* v3, Node* v4,
                     Node* mid[5] = NULL /* optional output of middle nodes*/);

   Node* GetMidVertex(Node* n1, Node* n2);

   Element* NewElement(Node* n0, Node* n1, Node* n2, Node* n3,
                       Node* n4, Node* n5, Node* n6, Node* n7,
                       int attr,
                       int fattr0, int fattr1, int fattr2,
                       int fattr3, int fattr4, int fattr5);

   void RefVertices(Element* elem);
   void UnrefVertices(Element* elem);

   void RefEdgesFaces(Element *elem);
   void UnrefEdgesFaces(Element *elem);

   int IndexVertices();
   int IndexEdges();
   int IndexFaces();

   void GetLeafElements(Element* e, Array< ::Element*>& elements,
                                    Array< ::Element*>& boundary);

   void DeleteHierarchy(Element* elem);


   // interpolation

   struct Dependency
   {
      int dof;
      double coef;

      Dependency(int dof, double coef) : dof(dof), coef(coef) {}
   };

   typedef Array<Dependency> DepList;

   /// Holds temporary data for each vertex during the interpolation algorithm.
   struct VertexData
   {
      int dof;      ///< original nonconforming FESpace-assigned DOF number
      int true_dof; ///< conforming true DOF number, -1 if slave DOF
      DepList dep_list;

      bool Independent() const { return true_dof >= 0; }
      bool Dependent() const { return true_dof == -1; }
      bool Processed() const { return true_dof > -2; }
   };

   VertexData* v_data; ///< vertex temporary data
/*   InterpolationData* e_data; ///< edge temporary data
   InterpolationData* f_data; ///< face temporary data*/

   struct MasterFace
   {
      const Node* v[4];
      const Node* e[4];
      const Face* face;
      const FiniteElement *face_fe;
   };

   void ConstrainFace(Node* v0, Node* v1, Node* v2, Node* v3,
                      IsoparametricTransformation& face_T,
                      MasterFace* master, int level);

   void VisitFaces(Element* elem, const FiniteElementCollection *fec);

   int next_true_dof;


};


#endif
