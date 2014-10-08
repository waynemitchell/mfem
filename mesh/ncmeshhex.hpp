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
//class FiniteElementCollection;
class IsoparametricTransformation;
class FiniteElementSpace;


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

   ~NCMeshHex();

protected: // interface for Mesh to be able to construct itself from us

   void GetVerticesElementsBoundary(Array< ::Vertex>& vertices,
                                    Array< ::Element*>& elements,
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
      double pos[3]; ///< 3D position
      int index; ///< vertex number in the Mesh

      Vertex() {}
      Vertex(double x, double y, double z) : index(-1)
         { pos[0] = x, pos[1] = y, pos[2] = z; }
   };

   ///
   struct Edge : public RefCount
   {
      //int dof; ///< first edge DOF number (nonconforming)
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
      Element* elem[2]; ///< up to 2 elements sharing the face
      int index; ///< face number in the Mesh

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
   Array<Element*> leaf_elements; // updated by GetLeafElements

   HashTable<Node> nodes;
   HashTable<Face> faces;

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

   Element* NewElement(Node* n0, Node* n1, Node* n2, Node* n3,
                       Node* n4, Node* n5, Node* n6, Node* n7,
                       int attr,
                       int fattr0, int fattr1, int fattr2,
                       int fattr3, int fattr4, int fattr5);

   Vertex* NewVertex(Node* v1, Node* v2);
   Node* GetMidEdgeVertex(Node* v1, Node* v2);
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
      int true_dof; ///< assigned conforming true DOF number, -1 if slave
      bool finalized; ///< true if cP matrix row is known for this DOF

      DepList dep_list; ///< list of other DOFs this DOF depends on
      bool Independent() const { return !dep_list.Size(); }

      DofData() : true_dof(-1), finalized(false) {}
   };

   DofData* dof_data; ///< vertex temporary data

   FiniteElementSpace* space;

   void ConstrainFace(Node* v0, Node* v1, Node* v2, Node* v3,
                      IsoparametricTransformation& face_T,
                      Array<int>& master_dofs, int level);

   void ProcessMasterFace(Node* node[4], Face* face);

   bool DofFinalizable(DofData& vd);

   // utility

   void FaceSplitLevel(Node* v1, Node* v2, Node* v3, Node* v4,
                       int& h_level, int& v_level);

   void CountSplits(Element* elem, int splits[3]);


};


#endif
