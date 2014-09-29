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

class SparseMatrix;
class FiniteElementCollection;
class IsoparametricTransformation;


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
      int attribute;
      int ref_type; // bit mask of X,Y,Z refinements (bits 0,1,2, respectively)
      union
      {
         Node* node[8]; // element corners (if ref_type == 0)
         Element* child[8]; // 2-8 children (if ref_type != 0)
      };

      Element(int attr);
   };

   /** Returns one of the elements that are based on the elements of the
       original mesh. These are the roots of the refinement trees. */
   Element* GetRootElement(int index) { return root_elements[index]; }

   /** Returns a leaf (unrefined) elements. NOTE: leaf elements are enumerated
       when converting from NCMeshHex to Mesh. Only use this function after. */
   Element* GetLeafElement(int index) { return leaf_elements[index]; }

   int NRootElements() const { return root_elements.Size(); }
   int NLeafElements() const { return leaf_elements.Size(); }

   bool Refine(Element* elem, int ref_type);
   void Derefine(Element* elem);

   bool Refine(int index, int ref_type)
      { return Refine(GetLeafElement(index), ref_type); }
   void Derefine(int index)
      { Derefine(GetLeafElement(index)); }

   SparseMatrix *GetInterpolation(Mesh *f_mesh, FiniteElementSpace *fes);

   ~NCMeshHex();

protected: // interface for Mesh to be able to construct itself from us

   void GetVertices(Array< ::Vertex>& vertices);
   void GetElements(Array< ::Element*>& elements,
                    Array< ::Element*>& boundary);

   Array<Element*> leaf_elements; // initialized by GetElements

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
      Vertex(double x, double y, double z) : index(-1)
         { pos[0] = x, pos[1] = y, pos[2] = z; }
   };

   ///
   struct Edge : public RefCount
   {
      int index; ///< internal numbering, set by IndexEdges to 0..NE-1
   };

   ///
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
      // anymore. The vertex, edge or the whole Node can auto-destruct.
      // The hash-table pointer needs to be known then to remove the node.
      void UnrefVertex(HashTable<Node>& nodes);
      void UnrefEdge(HashTable<Node>& nodes);

      ~Node();
   };

   ///
   struct Face : public RefCount, public Hashed4<Face>
   {
      int attribute; ///< boundary element attribute, -1 if internal face
      int index; ///<  internal numbering, set by IndexFaces to 0..NF-1

      Face(int id) : Hashed4<Face>(id), attribute(-1) {}

      bool Boundary() const { return attribute >= 0; }

      // overloaded Unref without auto-destruction
      int Unref() { return --ref_count; }
   };

   Array<Element*> root_elements;
   HashTable<Node> nodes;
   HashTable<Face> faces;

   Vertex* NewVertex(Node* v1, Node* v2);
   Node* GetMidEdgeVertex(Node* v1, Node* v2);
   Node* GetMidFaceVertex(Node* e1, Node* e2, Node* e3, Node* e4);

   Element* NewElement(Node* n0, Node* n1, Node* n2, Node* n3,
                       Node* n4, Node* n5, Node* n6, Node* n7,
                       int attr,
                       int fattr0, int fattr1, int fattr2,
                       int fattr3, int fattr4, int fattr5);

   int FaceSplitType(Node* v1, Node* v2, Node* v3, Node* v4,
                     Node* mid[5] = NULL /* optional output of middle nodes*/);

   bool LegalAnisoSplit(Node* v1, Node* v2, Node* v3, Node* v4, int level = 0);
   void CheckAnisoFace(Node* v1, Node* v2, Node* v3, Node* v4,
                       Node* mid12, Node* mid34);

   void RefElementNodes(Element *elem);
   void UnrefElementNodes(Element *elem);

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

      Dependency(int dof, double coef)
         : dof(dof), coef(coef) {}
   };

   typedef Array<Dependency> DepList;

   /// Holds temporary data for each vertex during the interpolation algorithm.
   struct VertexData
   {
      int dof;      ///< original nonconforming FESpace-assigned DOF number
      int true_dof; ///< conforming true DOF number, -1 if slave
      bool finalized; ///< true if cP matrix row is known for this DOF

      DepList dep_list; ///< what other DOFs does this vertex depend on?
      bool Independent() const { return !dep_list.Size(); }

      VertexData() : true_dof(-1), finalized(false) {}
   };

   VertexData* v_data; ///< vertex temporary data

   bool VertexFinalizable(VertexData& vd);

   struct MasterFace
   {
      Node* v[4];
      Node* e[4];
      Face* face;
      const FiniteElement *face_fe;
   };

   void ConstrainFace(Node* v0, Node* v1, Node* v2, Node* v3,
                      IsoparametricTransformation& face_T,
                      MasterFace* master, int level);

   void ProcessMasterFace(Node* node[4], Face* face,
                          const FiniteElementCollection *fec);

};


#endif
