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

#include "mesh.hpp"
#include "../fem/fespace.hpp"
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
      int ref_type; // bit mask of X,Y,Z refinements (bits 0,1,2, respectively)
      union {
         Node* node[8]; // element corners (ref_type == 0, i.e, not refined)
         Element* child[8]; // children (ref_type != 0)
      };

      Element() : ref_type(0) {}

      Element(Node* n0, Node* n1, Node* n2, Node* n3,
              Node* n4, Node* n5, Node* n6, Node* n7) : ref_type(0)
      {
         node[0] = n0, node[1] = n1, node[2] = n2, node[3] = n3;
         node[4] = n4, node[5] = n5, node[6] = n6, node[7] = n7;
      }

      ~Element();
   };

   /** Returns one of the elements that are based on the elements of the
       original mesh. These are the roots of the refinement trees. */
   Element* GetRootElement(int index) { return root_elements[index]; }
   int NRootElements() const { return root_elements.Size(); }

   void Refine(Element* elem, int ref_type /*= 7*/);
   void Derefine(Element* elem);

   SparseMatrix *GetInterpolation(Mesh *f_mesh, FiniteElementSpace *f_fes);

   ~NCMeshHex();

protected:

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
      bool dependent;

      Vertex() {}
      Vertex(double x, double y, double z)
         { pos[0] = x, pos[1] = y, pos[2] = z; }
   };

   ///
   struct Edge : public RefCount
   {
      int depth;
      bool dependent;
   };

   ///
   struct Node : public Hashed2
   {
      Vertex* vertex;
      Edge* edge;

      Node(int id) : Hashed2(id), vertex(NULL), edge(NULL) {}

      // Bump ref count on a vertex or an edge or create them. Used when an
      // element starts using a vertex or an edge.
      void RefVertex();
      void RefEdge();

      // Decrement ref on vertex or edge when an element is not using them
      // anymore. The vertex, edge or the whole Node can auto-destruct.
      // The hash-table pointer needs to be known then to remove the node.
      void UnrefVertex();
      void UnrefEdge();
   };

   ///
   struct Face : public Hashed4, public RefCount
   {
      bool dependent;

      Face(int id) : Hashed4(id) {}
   };


   Array<Element*> root_elements;
   HashTable<Node> nodes;
   HashTable<Face> faces;

   int FaceSplitType(int v1, int v2, int v3, int v4);

   Node* GetMidVertex(Node* n1, Node* n2);
   void RefElementNodes(Element* elem);
   void UnrefElementNodes(Element* elem);

   struct Dependency
   {
      int dof;
      double coef;
   };

   typedef Array<Dependency> DepList;
   DepList* v_dep;
   DepList* e_dep;
   DepList* f_dep;

};


#endif
