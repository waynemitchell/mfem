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
      int index;

      Vertex() {}
      Vertex(double x, double y, double z)
         { pos[0] = x, pos[1] = y, pos[2] = z; }
   };

   ///
   struct Edge : public RefCount
   {
      int index;
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
      int attribute;

      Face(int id) : Hashed4(id), attribute(-1) {}

      // overloaded Unref without auto-destruction
      int Unref() { return --ref_count; }
   };

   Array<Element*> root_elements;
   HashTable<Node> nodes;
   HashTable<Face> faces;

   int num_leaf_elements;

//   int FaceSplitType(int v1, int v2, int v3, int v4);

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

   void GetLeafElements(Element* e, Array< ::Element*>& elements,
                                    Array< ::Element*>& boundary);

   void DeleteHierarchy(Element* elem);

   struct Dependency
   {
      int dof;
      double coef;
   };

   typedef Array<Dependency> DepList;
/*   DepList* v_dep;
   DepList* e_dep;
   DepList* f_dep;*/

};


#endif
