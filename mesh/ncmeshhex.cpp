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

#include "mesh_headers.hpp"
#include "../fem/fem.hpp"

#include <cmath>

#include "ncmeshhex.hpp"

/** This holds in one place the constants about the geometries we support
    (triangles, quads, cubes) */
struct GeomInfo
{
   int nv, ne, nf, nfv; // number of: vertices, edge, faces, face vertices
   int edges[12][2];    // edge vertices (up to 12 edges)
   int faces[6][4];     // face vertices (up to 6 faces)

   bool initialized;
   GeomInfo() : initialized(false) {}
   void Initialize(const Element* elem);
};

static GeomInfo GI[Geometry::NumGeom];

static GeomInfo& gi_hex  = GI[Geometry::CUBE];
static GeomInfo& gi_quad = GI[Geometry::SQUARE];
static GeomInfo& gi_tri  = GI[Geometry::TRIANGLE];

void GeomInfo::Initialize(const Element* elem)
{
   if (initialized) return;

   nv = elem->GetNVertices();
   ne = elem->GetNEdges();
   nf = elem->GetNFaces(nfv);

   for (int i = 0; i < ne; i++)
      for (int j = 0; j < 2; j++)
         edges[i][j] = elem->GetEdgeVertices(i)[j];

   for (int i = 0; i < nf; i++)
      for (int j = 0; j < nfv; j++)
         faces[i][j] = elem->GetFaceVertices(i)[j];

   initialized = true;
}


NCMeshHex::NCMeshHex(const Mesh *mesh)
{
   Dim = mesh->Dimension();

   // create the NCMeshHex::Element struct for each Mesh element
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      const ::Element *elem = mesh->GetElement(i);
      const int *v = elem->GetVertices();

      int geom = elem->GetGeometryType();
      if (geom != Geometry::TRIANGLE &&
          geom != Geometry::SQUARE &&
          geom != Geometry::CUBE)
      {
         mfem_error("NCMesh: only triangles, quads and hexes are supported.");
      }

      // initialize edge/face tables for this type of element
      GI[geom].Initialize(elem);

      // create our Element struct for this element
      Element* nc_elem = new Element(geom, elem->GetAttribute());
      root_elements.Append(nc_elem);

      for (int j = 0; j < GI[geom].nv; j++)
      {
         // root nodes are special: they have p1 == p2 == orig. mesh vertex id
         Node* node = nodes.Get(v[j], v[j]);

         if (!node->vertex)
         {
            // create a vertex in the node and initialize its position
            const double* pos = mesh->GetVertex(v[j]);
            node->vertex = new Vertex(pos[0], pos[1], pos[2]);
         }

         nc_elem->node[j] = node;
      }

      // increase reference count of all nodes the element is using
      // (NOTE: this will also create and reference all edge and face nodes)
      RefElementNodes(nc_elem);

      // make links from faces back to the element
      RegisterFaces(nc_elem);
   }

   // store boundary element attributes
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      const ::Element *be = mesh->GetBdrElement(i);
      const int *v = be->GetVertices();

      Node* node[4];
      for (int i = 0; i < be->GetNVertices(); i++)
      {
         node[i] = nodes.Peek(v[i], v[i]);
         if (!node[i])
            mfem_error("NCMesh: boundary elements inconsistent.");
      }

      if (be->GetType() == ::Element::QUADRILATERAL)
      {
         Face* face = faces.Peek(node[0], node[1], node[2], node[3]);
         if (!face)
            mfem_error("NCMesh: boundary face not found.");

         face->attribute = be->GetAttribute();
      }
      else if (be->GetType() == ::Element::SEGMENT)
      {
         Edge* edge = nodes.Peek(node[0], node[1])->edge;
         if (!edge)
            mfem_error("NCMesh: boundary edge not found.");

         edge->attribute = be->GetAttribute();
      }
      else
         mfem_error("NCMesh: only segment and quadrilateral boundary "
                    "elements are supported.");
   }

   UpdateLeafElements();
}

NCMeshHex::~NCMeshHex()
{
   for (int i = 0; i < root_elements.Size(); i++)
      DeleteHierarchy(root_elements[i]);
}

void NCMeshHex::DeleteHierarchy(Element* elem)
{
   if (elem->ref_type)
   {
      for (int i = 0; i < 8; i++)
         if (elem->child[i])
            DeleteHierarchy(elem->child[i]);
   }
   else
   {
      UnrefElementNodes(elem);
   }
   delete elem;
}


//// Node and Face Memory Management ///////////////////////////////////////////

void NCMeshHex::Node::RefVertex()
{
   MFEM_ASSERT(vertex, "NCMeshHex::Node::RefVertex: can't create vertex here.");
   vertex->Ref();
}

void NCMeshHex::Node::RefEdge()
{
   if (!edge) edge = new Edge;
   edge->Ref();
}

void NCMeshHex::Node::UnrefVertex(HashTable<Node> &nodes)
{
   MFEM_ASSERT(vertex, "Cannot unref a nonexistent vertex.");
   if (!vertex->Unref()) vertex = NULL;
   if (!vertex && !edge) nodes.Delete(this);
}

void NCMeshHex::Node::UnrefEdge(HashTable<Node> &nodes)
{
   MFEM_ASSERT(this, "Node not found.");
   MFEM_ASSERT(edge, "Cannot unref a nonexistent edge.");
   if (!edge->Unref()) edge = NULL;
   if (!vertex && !edge) nodes.Delete(this);
}

NCMeshHex::Node::~Node()
{
   MFEM_ASSERT(!vertex && !edge, "Node was not unreffed properly.");
   if (vertex) delete vertex;
   if (edge) delete edge;
}

void NCMeshHex::RefElementNodes(Element *elem)
{
   Node** node = elem->node;
   GeomInfo& gi = GI[elem->geom];

   // ref all vertices
   for (int i = 0; i < gi.nv; i++)
      node[i]->RefVertex();

   // ref all edges (possibly creating them)
   for (int i = 0; i < gi.ne; i++)
   {
      const int* ev = gi.edges[i];
      nodes.Get(node[ev[0]], node[ev[1]])->RefEdge();
   }

   // ref all faces (possibly creating them)
   for (int i = 0; i < gi.nf; i++)
   {
      const int* fv = gi.faces[i];
      faces.Get(node[fv[0]], node[fv[1]], node[fv[2]], node[fv[3]])->Ref();
      // NOTE: face->RegisterElement called elsewhere to avoid having
      //       to store 3 element pointers temporarily in the face when refining
   }
}

void NCMeshHex::UnrefElementNodes(Element *elem)
{
   Node** node = elem->node;
   GeomInfo& gi = GI[elem->geom];

   // unref all faces (possibly destroying them)
   for (int i = 0; i < gi.nf; i++)
   {
      const int* fv = gi.faces[i];
      Face* face = faces.Peek(node[fv[0]], node[fv[1]], node[fv[2]], node[fv[3]]);
      face->ForgetElement(elem);
      if (!face->Unref()) faces.Delete(face);
   }

   // unref all edges (possibly destroying them)
   for (int i = 0; i < gi.ne; i++)
   {
      const int* ev = gi.edges[i];
      //nodes.Peek(node[ev[0]], node[ev[1]])->UnrefEdge(nodes); -- pre-aniso
      PeekAltParents(node[ev[0]], node[ev[1]])->UnrefEdge(nodes);
   }

   // unref all vertices (possibly destroying them)
   for (int i = 0; i < gi.nv; i++)
      elem->node[i]->UnrefVertex(nodes);
}

void NCMeshHex::Face::RegisterElement(Element* e)
{
   if (elem[0] == NULL)
      elem[0] = e;
   else if (elem[1] == NULL)
      elem[1] = e;
   else
      MFEM_ASSERT(0, "Can't have 3 elements in Face::elem[2].");
}

void NCMeshHex::Face::ForgetElement(Element* e)
{
   if (elem[0] == e)
      elem[0] = NULL;
   else if (elem[1] == e)
      elem[1] = NULL;
   else
      MFEM_ASSERT(0, "Element not found in Face::elem[2].");
}

void NCMeshHex::RegisterFaces(Element* elem)
{
   Node** node = elem->node;
   GeomInfo& gi = GI[elem->geom];

   for (int i = 0; i < gi.nf; i++)
   {
      const int* fv = gi.faces[i];
      Face* face = faces.Peek(node[fv[0]], node[fv[1]], node[fv[2]], node[fv[3]]);
      face->RegisterElement(elem);
   }
}

NCMeshHex::Element* NCMeshHex::Face::GetSingleElement() const
{
   if (elem[0])
   {
      MFEM_ASSERT(!elem[1], "Not a single element face.");
      return elem[0];
   }
   else
   {
      MFEM_ASSERT(elem[1], "No elements in face.");
      return elem[1];
   }
}

NCMeshHex::Node* NCMeshHex::PeekAltParents(Node* v1, Node* v2)
{
   Node* mid = nodes.Peek(v1, v2);
   if (!mid)
   {
      /* In rare cases, a mid-face node exists under alternate parents w1, w2
         (see picture) instead of the requested parents v1, v2. This is an
         inconsistent situation that may exist temporarily as a result of
         "nodes.Reparent" while doing anisotropic splits, before forced
         refinements are all processed. This function attempts to retrieve such
         a node. An extra twist is that w1 and w2 may themselves need to be
         obtained using this very function.

                        v1->p1      v1       v1->p2
                              *------*------*
                              |      |      |
                              |      |mid   |
                           w1 *------*------* w2
                              |      |      |
                              |      |      |
                              *------*------*
                        v2->p1      v2       v2->p2

         NOTE: this function would not be needed if the elements remembered
         pointers to their edge nodes. We have however opted to save memory
         at the cost of this computation, which is only necessary when forced
         refinements are being done. */

      if ((v1->p1 != v1->p2) && (v2->p1 != v2->p2)) // non-top-level nodes?
      {
         Node *v1p1 = nodes.Peek(v1->p1), *v1p2 = nodes.Peek(v1->p2);
         Node *v2p1 = nodes.Peek(v2->p1), *v2p2 = nodes.Peek(v2->p2);

         Node* w1 = PeekAltParents(v1p1, v2p1);
         Node* w2 = w1 ? PeekAltParents(v1p2, v2p2) : NULL /* optimization */;

         if (!w1 || !w2) // one more try may be needed as p1, p2 are unordered
            w1 = PeekAltParents(v1p1, v2p2),
            w2 = w1 ? PeekAltParents(v1p2, v2p1) : NULL /* optimization */;

         if (w1 && w2) // got both alternate parents?
            mid = nodes.Peek(w1, w2);
      }
   }
   return mid;
}


//// Refinement & Derefinement /////////////////////////////////////////////////

NCMeshHex::Element::Element(int geom, int attr)
   : geom(geom), attribute(attr), ref_type(0), index(-1)
{
   memset(node, 0, sizeof(node));

   // NOTE: in 2D the 8-element node/child arrays are not optimal, however,
   // testing shows we would only save 17% of the total NCMeshHex memory if
   // 4-element arrays were used (e.g. through templates); we thus prefer to
   // keep the code as simple as possible.
}

NCMeshHex::Element*
   NCMeshHex::NewHexahedron(Node* n0, Node* n1, Node* n2, Node* n3,
                            Node* n4, Node* n5, Node* n6, Node* n7,
                            int attr,
                            int fattr0, int fattr1, int fattr2,
                            int fattr3, int fattr4, int fattr5)
{
   // create new unrefined element, initialize nodes
   Element* e = new Element(Geometry::CUBE, attr);
   e->node[0] = n0, e->node[1] = n1, e->node[2] = n2, e->node[3] = n3;
   e->node[4] = n4, e->node[5] = n5, e->node[6] = n6, e->node[7] = n7;

   // get face nodes and assign face attributes
   Face* f[6];
   for (int i = 0; i < gi_hex.nf; i++)
   {
      const int* fv = gi_hex.faces[i];
      f[i] = faces.Get(e->node[fv[0]], e->node[fv[1]],
                       e->node[fv[2]], e->node[fv[3]]);
   }

   f[0]->attribute = fattr0,  f[1]->attribute = fattr1;
   f[2]->attribute = fattr2,  f[3]->attribute = fattr3;
   f[4]->attribute = fattr4,  f[5]->attribute = fattr5;

   return e;
}

NCMeshHex::Element*
   NCMeshHex::NewQuadrilateral(Node* n0, Node* n1, Node* n2, Node* n3,
                               int attr,
                               int eattr0, int eattr1, int eattr2, int eattr3)
{
   // create new unrefined element, initialize nodes
   Element* e = new Element(Geometry::SQUARE, attr);
   e->node[0] = n0, e->node[1] = n1, e->node[2] = n2, e->node[3] = n3;

   // get edge nodes and assign edge attributes
   Edge* edge[4];
   for (int i = 0; i < gi_quad.ne; i++)
   {
      const int* ev = gi_quad.edges[i];
      Node* node = nodes.Get(e->node[ev[0]], e->node[ev[1]]);
      if (!node->edge) node->edge = new Edge;
      edge[i] = node->edge;
   }

   edge[0]->attribute = eattr0;
   edge[1]->attribute = eattr1;
   edge[2]->attribute = eattr2;
   edge[3]->attribute = eattr3;

   return e;
}

NCMeshHex::Element*
   NCMeshHex::NewTriangle(Node* n0, Node* n1, Node* n2,
                          int attr, int eattr0, int eattr1, int eattr2)
{
   // create new unrefined element, initialize nodes
   Element* e = new Element(Geometry::TRIANGLE, attr);
   e->node[0] = n0, e->node[1] = n1, e->node[2] = n2;

   // get edge nodes and assign edge attributes
   Edge* edge[3];
   for (int i = 0; i < gi_tri.ne; i++)
   {
      const int* ev = gi_tri.edges[i];
      Node* node = nodes.Get(e->node[ev[0]], e->node[ev[1]]);
      if (!node->edge) node->edge = new Edge;
      edge[i] = node->edge;
   }

   edge[0]->attribute = eattr0;
   edge[1]->attribute = eattr1;
   edge[2]->attribute = eattr2;

   return e;
}

NCMeshHex::Vertex* NCMeshHex::NewVertex(Node* v1, Node* v2)
{
   MFEM_ASSERT(v1->vertex && v2->vertex,
               "NCMeshHex::NewVertex: missing parent vertices.");

   // get the midpoint between v1 and v2
   Vertex* v = new Vertex;
   for (int i = 0; i < 3; i++)
      v->pos[i] = (v1->vertex->pos[i] + v2->vertex->pos[i]) * 0.5;

   return v;
}

NCMeshHex::Node* NCMeshHex::GetMidEdgeVertex(Node* v1, Node* v2)
{
   // in 3D we must be careful about getting the mid-edge node
   Node* mid = PeekAltParents(v1, v2);
   if (!mid) mid = nodes.Get(v1, v2);
   if (!mid->vertex) mid->vertex = NewVertex(v1, v2);
   return mid;
}

NCMeshHex::Node* NCMeshHex::GetMidEdgeVertexSimple(Node* v1, Node* v2)
{
   // simple version for 2D cases
   Node* mid = nodes.Get(v1, v2);
   if (!mid->vertex) mid->vertex = NewVertex(v1, v2);
   return mid;
}

NCMeshHex::Node*
   NCMeshHex::GetMidFaceVertex(Node* e1, Node* e2, Node* e3, Node* e4)
{
   // mid-face node can be created either from (e1, e3) or from (e2, e4)
   Node* midf = nodes.Peek(e1, e3);
   if (midf)
   {
      if (!midf->vertex) midf->vertex = NewVertex(e1, e3);
      return midf;
   }
   else
   {
      midf = nodes.Get(e2, e4);
      if (!midf->vertex) midf->vertex = NewVertex(e2, e4);
      return midf;
   }
}

//
inline bool NCMeshHex::NodeSetX1(Node* node, Node** n)
   { return node == n[0] || node == n[3] || node == n[4] || node == n[7]; }

inline bool NCMeshHex::NodeSetX2(Node* node, Node** n)
   { return node == n[1] || node == n[2] || node == n[5] || node == n[6]; }

inline bool NCMeshHex::NodeSetY1(Node* node, Node** n)
   { return node == n[0] || node == n[1] || node == n[4] || node == n[5]; }

inline bool NCMeshHex::NodeSetY2(Node* node, Node** n)
   { return node == n[2] || node == n[3] || node == n[6] || node == n[7]; }

inline bool NCMeshHex::NodeSetZ1(Node* node, Node** n)
   { return node == n[0] || node == n[1] || node == n[2] || node == n[3]; }

inline bool NCMeshHex::NodeSetZ2(Node* node, Node** n)
   { return node == n[4] || node == n[5] || node == n[6] || node == n[7]; }


void NCMeshHex::ForceRefinement(Node* v1, Node* v2, Node* v3, Node* v4)
{
   // get the element this face belongs to
   Face* face = faces.Peek(v1, v2, v3, v4);
   if (!face) return;

   Element* elem = face->GetSingleElement();
   MFEM_ASSERT(!elem->ref_type, "Element already refined.");

   Node** nodes = elem->node;

   // schedule the right split depending on face orientation
   if ((NodeSetX1(v1, nodes) && NodeSetX2(v2, nodes)) ||
       (NodeSetX1(v2, nodes) && NodeSetX2(v1, nodes)))
   {
      ref_stack.Append(RefStackItem(elem, 1)); // X split
   }
   else if ((NodeSetY1(v1, nodes) && NodeSetY2(v2, nodes)) ||
            (NodeSetY1(v2, nodes) && NodeSetY2(v1, nodes)))
   {
      ref_stack.Append(RefStackItem(elem, 2)); // Y split
   }
   else if ((NodeSetZ1(v1, nodes) && NodeSetZ2(v2, nodes)) ||
            (NodeSetZ1(v2, nodes) && NodeSetZ2(v1, nodes)))
   {
      ref_stack.Append(RefStackItem(elem, 4)); // Z split
   }
   else
      MFEM_ASSERT(0, "Inconsistent element/face structure.");
}


void NCMeshHex::CheckAnisoFace(Node* v1, Node* v2, Node* v3, Node* v4,
                               Node* mid12, Node* mid34, int level)
{
   /* When a face is getting split anisotropically (without loss of generality
      we assume a "vertical" split here, see picture), it is important to make
      sure that the mid-face vertex (midf) has mid34 and mid12 as parents.
      This is necessary for the face traversal algorithm and at places like
      Refine() that assume the mid-edge nodes to be accessible through the right
      parents. However, midf may already exist under the parents mid41 and
      mid23. In that case we need to "reparent" midf, i.e., reinsert it to the
      hash-table under the correct parents. This doesn't affect other nodes as
      all IDs stay the same, only the face refinement "tree" is affected.

                           v4      mid34      v3
                             *------*------*
                             |      |      |
                             |      |midf  |
                       mid41 *- - - *- - - * mid23
                             |      |      |
                             |      |      |
                             *------*------*
                          v1      mid12      v2

      This function is recusive, because the above applies to any node along the
      middle vertical edge. The function calls itself again for the bottom and
      upper half of the above picture. */

   Node* mid23 = nodes.Peek(v2, v3);
   Node* mid41 = nodes.Peek(v4, v1);
   if (mid23 && mid41)
   {
      Node* midf = nodes.Peek(mid23, mid41);
      if (midf)
      {
        nodes.Reparent(midf, mid12->id, mid34->id);

        CheckAnisoFace(v1, v2, mid23, mid41, mid12, midf, level+1);
        CheckAnisoFace(mid41, mid23, v3, v4, midf, mid34, level+1);
        return;
      }
   }

   /* Also, this is the place where forced refinements begin. In the picture,
      the edges mid12-midf and midf-mid34 should actually exist in the
      neighboring elements, otherwise the mesh is inconsistent and needs to be
      fixed. */

   if (level > 0)
      ForceRefinement(v1, v2, v3, v4);
}

void NCMeshHex::CheckIsoFace(Node* v1, Node* v2, Node* v3, Node* v4,
                             Node* e1, Node* e2, Node* e3, Node* e4, Node* midf)
{
   /* If anisotropic refinements are present in the mesh, we need to check
      isotropically split faces as well. The iso face can be thought to contain
      four anisotropic cases as in the function CheckAnisoFace, that still need
      to be checked for the correct parents. */

   CheckAnisoFace(v1, v2, e2, e4, e1, midf);
   CheckAnisoFace(e4, e2, v3, v4, midf, e3);
   CheckAnisoFace(v4, v1, e1, e3, e4, midf);
   CheckAnisoFace(e3, e1, v2, v3, midf, e2);
}


void NCMeshHex::Refine(Element* elem, int ref_type)
{
   if (!ref_type) return;

   // handle elements that may have been (force-) refined already
   if (elem->ref_type)
   {
      int remaining = ref_type & ~elem->ref_type;

      // do the remaining splits on the children
      for (int i = 0; i < 8; i++)
         if (elem->child[i])
            Refine(elem->child[i], remaining);

      return;
   }

   Node** no = elem->node;
   int attr = elem->attribute;

   Element* child[8];
   memset(child, 0, sizeof(child));

   // create child elements
   if (elem->geom == Geometry::CUBE)
   {
      // get parent's face attributes
      int fa[6];
      for (int i = 0; i < gi_hex.nf; i++)
      {
         const int* fv = gi_hex.faces[i];
         Face* face = faces.Peek(no[fv[0]], no[fv[1]], no[fv[2]], no[fv[3]]);
         fa[i] = face->attribute;
      }

      /* Vertex numbering is assumed to be as follows:

               7              6
                +------------+                Faces: 0 bottom
               /|           /|                       1 front
            4 / |        5 / |                       2 right
             +------------+  |                       3 back
             |  |         |  |                       4 left
             |  +---------|--+                       5 top
             | / 3        | / 2       Z Y
             |/           |/          |/
             +------------+           *--X
            0              1                      */

      if (ref_type == 1) // split along X axis
      {
         Node* mid01 = GetMidEdgeVertex(no[0], no[1]);
         Node* mid23 = GetMidEdgeVertex(no[2], no[3]);
         Node* mid67 = GetMidEdgeVertex(no[6], no[7]);
         Node* mid45 = GetMidEdgeVertex(no[4], no[5]);

         child[0] = NewHexahedron(no[0], mid01, mid23, no[3],
                                  no[4], mid45, mid67, no[7], attr,
                                  fa[0], fa[1], -1, fa[3], fa[4], fa[5]);

         child[1] = NewHexahedron(mid01, no[1], no[2], mid23,
                                  mid45, no[5], no[6], mid67, attr,
                                  fa[0], fa[1], fa[2], fa[3], -1, fa[5]);

         CheckAnisoFace(no[0], no[1], no[5], no[4], mid01, mid45);
         CheckAnisoFace(no[2], no[3], no[7], no[6], mid23, mid67);
         CheckAnisoFace(no[4], no[5], no[6], no[7], mid45, mid67);
         CheckAnisoFace(no[3], no[2], no[1], no[0], mid23, mid01);
      }
      else if (ref_type == 2) // split along Y axis
      {
         Node* mid12 = GetMidEdgeVertex(no[1], no[2]);
         Node* mid30 = GetMidEdgeVertex(no[3], no[0]);
         Node* mid56 = GetMidEdgeVertex(no[5], no[6]);
         Node* mid74 = GetMidEdgeVertex(no[7], no[4]);

         child[0] = NewHexahedron(no[0], no[1], mid12, mid30,
                                  no[4], no[5], mid56, mid74, attr,
                                  fa[0], fa[1], fa[2], -1, fa[4], fa[5]);

         child[1] = NewHexahedron(mid30, mid12, no[2], no[3],
                                  mid74, mid56, no[6], no[7], attr,
                                  fa[0], -1, fa[2], fa[3], fa[4], fa[5]);

         CheckAnisoFace(no[1], no[2], no[6], no[5], mid12, mid56);
         CheckAnisoFace(no[3], no[0], no[4], no[7], mid30, mid74);
         CheckAnisoFace(no[5], no[6], no[7], no[4], mid56, mid74);
         CheckAnisoFace(no[0], no[3], no[2], no[1], mid30, mid12);
      }
      else if (ref_type == 4) // split along Z axis
      {
         Node* mid04 = GetMidEdgeVertex(no[0], no[4]);
         Node* mid15 = GetMidEdgeVertex(no[1], no[5]);
         Node* mid26 = GetMidEdgeVertex(no[2], no[6]);
         Node* mid37 = GetMidEdgeVertex(no[3], no[7]);

         child[0] = NewHexahedron(no[0], no[1], no[2], no[3],
                                  mid04, mid15, mid26, mid37, attr,
                                  fa[0], fa[1], fa[2], fa[3], fa[4], -1);

         child[1] = NewHexahedron(mid04, mid15, mid26, mid37,
                                  no[4], no[5], no[6], no[7], attr,
                                  -1, fa[1], fa[2], fa[3], fa[4], fa[5]);

         CheckAnisoFace(no[4], no[0], no[1], no[5], mid04, mid15);
         CheckAnisoFace(no[5], no[1], no[2], no[6], mid15, mid26);
         CheckAnisoFace(no[6], no[2], no[3], no[7], mid26, mid37);
         CheckAnisoFace(no[7], no[3], no[0], no[4], mid37, mid04);
      }
      else if (ref_type == 3) // XY split
      {
          Node* mid01 = GetMidEdgeVertex(no[0], no[1]);
          Node* mid12 = GetMidEdgeVertex(no[1], no[2]);
          Node* mid23 = GetMidEdgeVertex(no[2], no[3]);
          Node* mid30 = GetMidEdgeVertex(no[3], no[0]);

          Node* mid45 = GetMidEdgeVertex(no[4], no[5]);
          Node* mid56 = GetMidEdgeVertex(no[5], no[6]);
          Node* mid67 = GetMidEdgeVertex(no[6], no[7]);
          Node* mid74 = GetMidEdgeVertex(no[7], no[4]);

          Node* midf0 = GetMidFaceVertex(mid23, mid12, mid01, mid30);
          Node* midf5 = GetMidFaceVertex(mid45, mid56, mid67, mid74);

          child[0] = NewHexahedron(no[0], mid01, midf0, mid30,
                                   no[4], mid45, midf5, mid74, attr,
                                   fa[0], fa[1], -1, -1, fa[4], fa[5]);

          child[1] = NewHexahedron(mid01, no[1], mid12, midf0,
                                   mid45, no[5], mid56, midf5, attr,
                                   fa[0], fa[1], fa[2], -1, -1, fa[5]);

          child[2] = NewHexahedron(midf0, mid12, no[2], mid23,
                                   midf5, mid56, no[6], mid67, attr,
                                   fa[0], -1, fa[2], fa[3], -1, fa[5]);

          child[3] = NewHexahedron(mid30, midf0, mid23, no[3],
                                   mid74, midf5, mid67, no[7], attr,
                                   fa[0], -1, -1, fa[3], fa[4], fa[5]);

          CheckAnisoFace(no[0], no[1], no[5], no[4], mid01, mid45);
          CheckAnisoFace(no[1], no[2], no[6], no[5], mid12, mid56);
          CheckAnisoFace(no[2], no[3], no[7], no[6], mid23, mid67);
          CheckAnisoFace(no[3], no[0], no[4], no[7], mid30, mid74);

          CheckIsoFace(no[3], no[2], no[1], no[0], mid23, mid12, mid01, mid30, midf0);
          CheckIsoFace(no[4], no[5], no[6], no[7], mid45, mid56, mid67, mid74, midf5);
      }
      else if (ref_type == 5) // XZ split
      {
         Node* mid01 = GetMidEdgeVertex(no[0], no[1]);
         Node* mid23 = GetMidEdgeVertex(no[2], no[3]);
         Node* mid45 = GetMidEdgeVertex(no[4], no[5]);
         Node* mid67 = GetMidEdgeVertex(no[6], no[7]);

         Node* mid04 = GetMidEdgeVertex(no[0], no[4]);
         Node* mid15 = GetMidEdgeVertex(no[1], no[5]);
         Node* mid26 = GetMidEdgeVertex(no[2], no[6]);
         Node* mid37 = GetMidEdgeVertex(no[3], no[7]);

         Node* midf1 = GetMidFaceVertex(mid01, mid15, mid45, mid04);
         Node* midf3 = GetMidFaceVertex(mid23, mid37, mid67, mid26);

         child[0] = NewHexahedron(no[0], mid01, mid23, no[3],
                                  mid04, midf1, midf3, mid37, attr,
                                  fa[0], fa[1], -1, fa[3], fa[4], -1);

         child[1] = NewHexahedron(mid01, no[1], no[2], mid23,
                                  midf1, mid15, mid26, midf3, attr,
                                  fa[0], fa[1], fa[2], fa[3], -1, -1);

         child[2] = NewHexahedron(midf1, mid15, mid26, midf3,
                                  mid45, no[5], no[6], mid67, attr,
                                  -1, fa[1], fa[2], fa[3], -1, fa[5]);

         child[3] = NewHexahedron(mid04, midf1, midf3, mid37,
                                  no[4], mid45, mid67, no[7], attr,
                                  -1, fa[1], -1, fa[3], fa[4], fa[5]);

         CheckAnisoFace(no[3], no[2], no[1], no[0], mid23, mid01);
         CheckAnisoFace(no[2], no[6], no[5], no[1], mid26, mid15);
         CheckAnisoFace(no[6], no[7], no[4], no[5], mid67, mid45);
         CheckAnisoFace(no[7], no[3], no[0], no[4], mid37, mid04);

         CheckIsoFace(no[0], no[1], no[5], no[4], mid01, mid15, mid45, mid04, midf1);
         CheckIsoFace(no[2], no[3], no[7], no[6], mid23, mid37, mid67, mid26, midf3);
      }
      else if (ref_type == 6) // YZ split
      {
          Node* mid12 = GetMidEdgeVertex(no[1], no[2]);
          Node* mid30 = GetMidEdgeVertex(no[3], no[0]);
          Node* mid56 = GetMidEdgeVertex(no[5], no[6]);
          Node* mid74 = GetMidEdgeVertex(no[7], no[4]);

          Node* mid04 = GetMidEdgeVertex(no[0], no[4]);
          Node* mid15 = GetMidEdgeVertex(no[1], no[5]);
          Node* mid26 = GetMidEdgeVertex(no[2], no[6]);
          Node* mid37 = GetMidEdgeVertex(no[3], no[7]);

          Node* midf2 = GetMidFaceVertex(mid12, mid26, mid56, mid15);
          Node* midf4 = GetMidFaceVertex(mid30, mid04, mid74, mid37);

          child[0] = NewHexahedron(no[0], no[1], mid12, mid30,
                                   mid04, mid15, midf2, midf4, attr,
                                   fa[0], fa[1], fa[2], -1, fa[4], -1);

          child[1] = NewHexahedron(mid30, mid12, no[2], no[3],
                                   midf4, midf2, mid26, mid37, attr,
                                   fa[0], -1, fa[2], fa[3], fa[4], -1);

          child[2] = NewHexahedron(mid04, mid15, midf2, midf4,
                                   no[4], no[5], mid56, mid74, attr,
                                   -1, fa[1], fa[2], -1, fa[4], fa[5]);

          child[3] = NewHexahedron(midf4, midf2, mid26, mid37,
                                   mid74, mid56, no[6], no[7], attr,
                                   -1, -1, fa[2], fa[3], fa[4], fa[5]);

          CheckAnisoFace(no[4], no[0], no[1], no[5], mid04, mid15);
          CheckAnisoFace(no[0], no[3], no[2], no[1], mid30, mid12);
          CheckAnisoFace(no[3], no[7], no[6], no[2], mid37, mid26);
          CheckAnisoFace(no[7], no[4], no[5], no[6], mid74, mid56);

          CheckIsoFace(no[1], no[2], no[6], no[5], mid12, mid26, mid56, mid15, midf2);
          CheckIsoFace(no[3], no[0], no[4], no[7], mid30, mid04, mid74, mid37, midf4);
      }
      else if (ref_type == 7) // full isotropic refinement
      {
         Node* mid01 = GetMidEdgeVertex(no[0], no[1]);
         Node* mid12 = GetMidEdgeVertex(no[1], no[2]);
         Node* mid23 = GetMidEdgeVertex(no[2], no[3]);
         Node* mid30 = GetMidEdgeVertex(no[3], no[0]);

         Node* mid45 = GetMidEdgeVertex(no[4], no[5]);
         Node* mid56 = GetMidEdgeVertex(no[5], no[6]);
         Node* mid67 = GetMidEdgeVertex(no[6], no[7]);
         Node* mid74 = GetMidEdgeVertex(no[7], no[4]);

         Node* mid04 = GetMidEdgeVertex(no[0], no[4]);
         Node* mid15 = GetMidEdgeVertex(no[1], no[5]);
         Node* mid26 = GetMidEdgeVertex(no[2], no[6]);
         Node* mid37 = GetMidEdgeVertex(no[3], no[7]);

         Node* midf0 = GetMidFaceVertex(mid23, mid12, mid01, mid30);
         Node* midf1 = GetMidFaceVertex(mid01, mid15, mid45, mid04);
         Node* midf2 = GetMidFaceVertex(mid12, mid26, mid56, mid15);
         Node* midf3 = GetMidFaceVertex(mid23, mid37, mid67, mid26);
         Node* midf4 = GetMidFaceVertex(mid30, mid04, mid74, mid37);
         Node* midf5 = GetMidFaceVertex(mid45, mid56, mid67, mid74);

         Node* midel = GetMidEdgeVertex(midf1, midf3);

         child[0] = NewHexahedron(no[0], mid01, midf0, mid30,
                                  mid04, midf1, midel, midf4, attr,
                                  fa[0], fa[1], -1, -1, fa[4], -1);

         child[1] = NewHexahedron(mid01, no[1], mid12, midf0,
                                  midf1, mid15, midf2, midel, attr,
                                  fa[0], fa[1], fa[2], -1, -1, -1);

         child[2] = NewHexahedron(midf0, mid12, no[2], mid23,
                                  midel, midf2, mid26, midf3, attr,
                                  fa[0], -1, fa[2], fa[3], -1, -1);

         child[3] = NewHexahedron(mid30, midf0, mid23, no[3],
                                  midf4, midel, midf3, mid37, attr,
                                  fa[0], -1, -1, fa[3], fa[4], -1);

         child[4] = NewHexahedron(mid04, midf1, midel, midf4,
                                  no[4], mid45, midf5, mid74, attr,
                                  -1, fa[1], -1, -1, fa[4], fa[5]);

         child[5] = NewHexahedron(midf1, mid15, midf2, midel,
                                  mid45, no[5], mid56, midf5, attr,
                                  -1, fa[1], fa[2], -1, -1, fa[5]);

         child[6] = NewHexahedron(midel, midf2, mid26, midf3,
                                  midf5, mid56, no[6], mid67, attr,
                                  -1, -1, fa[2], fa[3], -1, fa[5]);

         child[7] = NewHexahedron(midf4, midel, midf3, mid37,
                                  mid74, midf5, mid67, no[7], attr,
                                  -1, -1, -1, fa[3], fa[4], fa[5]);

         CheckIsoFace(no[3], no[2], no[1], no[0], mid23, mid12, mid01, mid30, midf0);
         CheckIsoFace(no[0], no[1], no[5], no[4], mid01, mid15, mid45, mid04, midf1);
         CheckIsoFace(no[1], no[2], no[6], no[5], mid12, mid26, mid56, mid15, midf2);
         CheckIsoFace(no[2], no[3], no[7], no[6], mid23, mid37, mid67, mid26, midf3);
         CheckIsoFace(no[3], no[0], no[4], no[7], mid30, mid04, mid74, mid37, midf4);
         CheckIsoFace(no[4], no[5], no[6], no[7], mid45, mid56, mid67, mid74, midf5);
      }
      else
         mfem_error("NCMesh::Refine(): Invalid refinement type.");
   }
   else if (elem->geom == Geometry::SQUARE)
   {
      // get parent's edge attributes
      int ea0 = nodes.Peek(no[0], no[1])->edge->attribute;
      int ea1 = nodes.Peek(no[1], no[2])->edge->attribute;
      int ea2 = nodes.Peek(no[2], no[3])->edge->attribute;
      int ea3 = nodes.Peek(no[3], no[0])->edge->attribute;

      if (ref_type == 1) // X split
      {
         Node* mid01 = GetMidEdgeVertexSimple(no[0], no[1]);
         Node* mid23 = GetMidEdgeVertexSimple(no[2], no[3]);

         child[0] = NewQuadrilateral(no[0], mid01, mid23, no[3],
                                     attr, ea0, -1, ea2, ea3);

         child[1] = NewQuadrilateral(mid01, no[1], no[2], mid23,
                                     attr, ea0, ea1, ea2, -1);
      }
      else if (ref_type == 2) // Y split
      {
         Node* mid12 = GetMidEdgeVertexSimple(no[1], no[2]);
         Node* mid30 = GetMidEdgeVertexSimple(no[3], no[0]);

         child[0] = NewQuadrilateral(no[0], no[1], mid12, mid30,
                                     attr, ea0, ea1, -1, ea3);

         child[1] = NewQuadrilateral(mid30, mid12, no[2], no[3],
                                     attr, -1, ea1, ea2, ea3);
      }
      else if (ref_type == 3) // iso split
      {
         Node* mid01 = GetMidEdgeVertexSimple(no[0], no[1]);
         Node* mid12 = GetMidEdgeVertexSimple(no[1], no[2]);
         Node* mid23 = GetMidEdgeVertexSimple(no[2], no[3]);
         Node* mid30 = GetMidEdgeVertexSimple(no[3], no[0]);

         Node* midel = GetMidEdgeVertexSimple(mid01, mid23);

         child[0] = NewQuadrilateral(no[0], mid01, midel, mid30,
                                     attr, ea0, -1, -1, ea3);

         child[1] = NewQuadrilateral(mid01, no[1], mid12, midel,
                                     attr, ea0, ea1, -1, -1);

         child[2] = NewQuadrilateral(midel, mid12, no[2], mid23,
                                     attr, -1, ea1, ea2, -1);

         child[3] = NewQuadrilateral(mid30, midel, mid23, no[3],
                                     attr, -1, -1, ea2, ea3);
      }
      else
         mfem_error("NCMesh::Refine(): Invalid refinement type.");
   }
   else if (elem->geom == Geometry::TRIANGLE)
   {
      // get parent's edge attributes
      int ea0 = nodes.Peek(no[0], no[1])->edge->attribute;
      int ea1 = nodes.Peek(no[1], no[2])->edge->attribute;
      int ea2 = nodes.Peek(no[2], no[0])->edge->attribute;

      // isotropic split - the only ref_type available for triangles
      Node* mid01 = GetMidEdgeVertexSimple(no[0], no[1]);
      Node* mid12 = GetMidEdgeVertexSimple(no[1], no[2]);
      Node* mid20 = GetMidEdgeVertexSimple(no[2], no[0]);

      child[0] = NewTriangle(no[0], mid01, mid20, attr, ea0, -1, ea2);
      child[1] = NewTriangle(mid01, no[1], mid12, attr, ea0, ea1, -1);
      child[2] = NewTriangle(mid20, mid12, no[2], attr, -1, ea1, ea2);
      child[3] = NewTriangle(mid01, mid12, mid20, attr, -1, -1, -1);
   }
   else
      mfem_error("NCMesh::Refine(): Unsupported element geometry.");

   // start using the nodes of the children, create edges & faces
   for (int i = 0; i < 8; i++)
      if (child[i])
         RefElementNodes(child[i]);

   // sign off of all nodes of the parent, clean up unused nodes
   UnrefElementNodes(elem);

   // register the children in their faces once the parent is out of the way
   for (int i = 0; i < 8; i++)
      if (child[i])
         RegisterFaces(child[i]);

   // finish the refinement
   elem->ref_type = ref_type;
   memcpy(elem->child, child, sizeof(elem->child));
}


void NCMeshHex::Refine(const Array<NCRefinement>& refinements)
{
   // push all refinements on the stack in reverse order
   for (int i = refinements.Size()-1; i >= 0; i--)
   {
      const NCRefinement& ref = refinements[i];
      ref_stack.Append(RefStackItem(leaf_elements[ref.index], ref.ref_type));
   }

   // keep refining as long as the stack contains something
   int nforced = 0;
   while (ref_stack.Size())
   {
      RefStackItem ref = ref_stack.Last();
      ref_stack.DeleteLast();

      int size = ref_stack.Size();
      Refine(ref.elem, ref.ref_type);
      nforced += ref_stack.Size() - size;
   }

   /* TODO: the current algorithm of forced refinements is not optimal. As
      forced refinements spread through the mesh, some may not be necessary
      in the end, since the affected elements may still be scheduled for
      refinement that could stop the propagation. We should introduce the
      member Element::ref_pending that would show the intended refinement in
      the batch. A forced refinement would be combined with ref_pending to
      (possibly) stop the propagation earlier. */

   std::cout << "Refined " << refinements.Size() << " + " << nforced
             << " elements" << std::endl;

   UpdateLeafElements();
}


/*void NCMeshHex::Derefine(Element* elem)
{

}*/


//// Mesh Interface ////////////////////////////////////////////////////////////

void NCMeshHex::GetLeafElements(Element* e)
{
   if (!e->ref_type)
   {
      e->index = leaf_elements.Size();
      leaf_elements.Append(e);
   }
   else
   {
      e->index = -1;
      for (int i = 0; i < 8; i++)
         if (e->child[i])
            GetLeafElements(e->child[i]);
   }
}

void NCMeshHex::UpdateLeafElements()
{
   // collect leaf elements
   leaf_elements.SetSize(0);
   for (int i = 0; i < root_elements.Size(); i++)
      GetLeafElements(root_elements[i]);
}

void NCMeshHex::GetVerticesElementsBoundary(Array< ::Vertex>& vertices,
                                            Array< ::Element*>& elements,
                                            Array< ::Element*>& boundary)
{
   // count vertices and assign indices
   int num_vert = 0;
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
      if (it->vertex)
         it->vertex->index = num_vert++;

   // copy vertices
   vertices.SetSize(num_vert);
   int i = 0;
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
      if (it->vertex)
         vertices[i++].SetCoords(it->vertex->pos);

   UpdateLeafElements();

   elements.SetSize(leaf_elements.Size());
   boundary.SetSize(0);

   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      Element* nc_elem = leaf_elements[i];
      Node** node = nc_elem->node;
      GeomInfo& gi = GI[nc_elem->geom];

      // create an ::Element for each leaf Element
      ::Element* elem = NULL;
      switch (nc_elem->geom)
      {
      case Geometry::CUBE: elem = new Hexahedron; break;
      case Geometry::SQUARE: elem = new Quadrilateral; break;
      case Geometry::TRIANGLE: elem = new Triangle; break;
      }

      elements[i] = elem;
      elem->SetAttribute(nc_elem->attribute);
      for (int j = 0; j < gi.nv; j++)
         elem->GetVertices()[j] = node[j]->vertex->index;

      // create boundary elements
      if (nc_elem->geom == Geometry::CUBE)
      {
         for (int k = 0; k < gi.nf; k++)
         {
            const int* fv = gi.faces[k];
            Face* face = faces.Peek(node[fv[0]], node[fv[1]],
                                    node[fv[2]], node[fv[3]]);
            if (face->Boundary())
            {
               Quadrilateral* quad = new Quadrilateral;
               quad->SetAttribute(face->attribute);
               for (int j = 0; j < 4; j++)
                  quad->GetVertices()[j] = node[fv[j]]->vertex->index;

               boundary.Append(quad);
            }
         }
      }
      else // quad & triangle boundary elements
      {
         for (int k = 0; k < gi.ne; k++)
         {
            const int* ev = gi.edges[k];
            Edge* edge = nodes.Peek(node[ev[0]], node[ev[1]])->edge;
            if (edge->Boundary())
            {
               Segment* segment = new Segment;
               segment->SetAttribute(edge->attribute);
               for (int j = 0; j < 2; j++)
                  segment->GetVertices()[j] = node[ev[j]]->vertex->index;

               boundary.Append(segment);
            }
         }
      }
   }
}


//// Interpolation /////////////////////////////////////////////////////////////

int NCMeshHex::FaceSplitType(Node* v1, Node* v2, Node* v3, Node* v4,
                             Node* mid[4])
{
   // find edge nodes
   Node* e1 = nodes.Peek(v1, v2);
   Node* e2 = nodes.Peek(v2, v3);
   Node* e3 = e1 ? nodes.Peek(v3, v4) : NULL;
   Node* e4 = e2 ? nodes.Peek(v4, v1) : NULL;

   // optional: return the mid-edge nodes if requested
   if (mid) mid[0] = e1, mid[1] = e2, mid[2] = e3, mid[3] = e4;

   // try to get a mid-face node, either by (e1, e3) or by (e2, e4)
   Node *midf1 = NULL, *midf2 = NULL;
   if (e1 && e3) midf1 = nodes.Peek(e1, e3);
   if (e2 && e4) midf2 = nodes.Peek(e2, e4);

   // only one way to access the mid-face node must always exist
   MFEM_ASSERT(!(midf1 && midf2), "Incorrectly split face!");

   if (!midf1 && !midf2)
      return 0; // face not split

   if (midf1)
      return 1; // face split "vertically"
   else
      return 2; // face split "horizontally"
}

static void make_point_mat(double v0[], double v1[], double v2[], double v3[],
                           DenseMatrix& pm)
{
   for (int i = 0; i < 2; i++)
   {
      pm(i,0) = v0[i];
      pm(i,1) = v1[i];
      pm(i,2) = v2[i];
      pm(i,3) = v3[i];
   }
}

int NCMeshHex::find_node(Element* elem, Node* node)
{
   for (int i = 0; i < 8; i++)
      if (elem->node[i] == node)
         return i;

   mfem_error("Node not found.");
}

static int find_hex_face(int a, int b, int c)
{
   for (int i = 0; i < 6; i++)
   {
      const int* fv = gi_hex.faces[i];
      if ((a == fv[0] || a == fv[1] || a == fv[2] || a == fv[3]) &&
          (b == fv[0] || b == fv[1] || b == fv[2] || b == fv[3]) &&
          (c == fv[0] || c == fv[1] || c == fv[2] || c == fv[3]))
      {
         return i;
      }
   }
   mfem_error("Face not found.");
}

void NCMeshHex::ReorderFacePointMat(Node* v0, Node* v1, Node* v2, Node* v3,
                                    Element* elem, DenseMatrix& pm)
{
   int master[4] = {
      find_node(elem, v0), find_node(elem, v1),
      find_node(elem, v2), find_node(elem, v3)
   };

   int fi = find_hex_face(master[0], master[1], master[2]);
   const int* fv = gi_hex.faces[fi];

   DenseMatrix tmp(pm);
   for (int i = 0, j; i < 4; i++)
   {
      for (j = 0; j < 4; j++)
         if (fv[i] == master[j])
         {
            // pm.column(i) = tmp.column(j)
            for (int k = 0; k < pm.Height(); k++)
               pm(k,i) = tmp(k,j);
            break;
         }

      MFEM_ASSERT(j != 4, "Node not found.");
   }
}

inline int decode_dof(int dof, double& sign)
{
   if (dof >= 0)
      return (sign = 1.0, dof);
   else
      return (sign = -1.0, -1 - dof);
}

void NCMeshHex::AddDependencies(Array<int>& master_dofs,
                                Array<int>& slave_dofs,
                                DenseMatrix& I)
{
   // make each slave DOF dependent on all master DOFs
   for (int i = 0; i < slave_dofs.Size(); i++)
   {
      double ssign;
      int sdof = decode_dof(slave_dofs[i], ssign);
      DofData& sdata = dof_data[sdof];

      if (!sdata.dep_list.Size()) // not processed yet?
      {
         for (int j = 0; j < master_dofs.Size(); j++)
         {
            double coef = I(i, j);
            if (std::abs(coef) > 1e-12)
            {
               double msign;
               int mdof = decode_dof(master_dofs[j], msign);
               if (mdof != sdof)
                  sdata.dep_list.Append(Dependency(mdof, coef * ssign * msign));
            }
         }
      }
   }
}

void NCMeshHex::ConstrainEdge(Node* v0, Node* v1,
                              IsoparametricTransformation& edge_T,
                              Array<int>& master_dofs, int level)
{
   Node* mid = nodes.Peek(v0, v1);
   if (!mid) return;

   DenseMatrix& pm = edge_T.GetPointMat();
   double t0 = pm(0,0), t1 = pm(0,1);
   double tmid = (t0 + t1) / 2;

   if (mid->edge && level > 0)
   {
      // we need to make this edge constrained; get its DOFs
      Array<int> slave_dofs;
      space->GetEdgeDofs(mid->edge->index, slave_dofs);

      // handle slave edge orientation
      if (v0->vertex->index > v1->vertex->index)
         std::swap(pm(0,0), pm(0,1));

      // obtain the local interpolation matrix
      const FiniteElement* edge_fe =
         space->FEColl()->FiniteElementForGeometry(Geometry::SEGMENT);

      DenseMatrix I(edge_fe->GetDof());
      edge_fe->GetLocalInterpolation(edge_T, I);

      // make each slave DOF dependent on all master edge DOFs
      AddDependencies(master_dofs, slave_dofs, I);
   }

   // recurse deeper
   pm(0,0) = t0; pm(0,1) = tmid;
   ConstrainEdge(v0, mid, edge_T, master_dofs, level+1);

   pm(0,0) = tmid; pm(0,1) = t1;
   ConstrainEdge(mid, v1, edge_T, master_dofs, level+1);
}

void NCMeshHex::ConstrainFace(Node* v0, Node* v1, Node* v2, Node* v3,
                              IsoparametricTransformation& face_T,
                              Array<int>& master_dofs, int level)
{
   if (level > 0)
   {
      // check if we made it to a face that is not split further
      Face* face = faces.Peek(v0, v1, v2, v3);
      if (face)
      {
         // yes, we need to make this face constrained; get its DOFs
         Array<int> slave_dofs;
         space->GetFaceDofs(face->index, slave_dofs);

         // reorder face_T point matrix according to slave face orientation
         ReorderFacePointMat(v0, v1, v2, v3, face->GetSingleElement(),
                             face_T.GetPointMat());

         // obtain the local interpolation matrix
         const FiniteElement* face_fe =
            space->FEColl()->FiniteElementForGeometry(Geometry::SQUARE);

         DenseMatrix I(face_fe->GetDof());
         face_fe->GetLocalInterpolation(face_T, I);

         // make each slave DOF dependent on all master face DOFs
         AddDependencies(master_dofs, slave_dofs, I);
         return;
      }
   }

   // we need to recurse deeper, now determine how
   Node* mid[4];
   int split = FaceSplitType(v0, v1, v2, v3, mid);
   if (!split) return;

   // prepare also the middle points for the transformation
   DenseMatrix& pm = face_T.GetPointMat();
   double tmid[4][2] =
   {
      { (pm(0,0) + pm(0,1)) / 2,  (pm(1,0) + pm(1,1)) / 2 }, // bottom (0)
      { (pm(0,1) + pm(0,2)) / 2,  (pm(1,1) + pm(1,2)) / 2 }, // right (1)
      { (pm(0,2) + pm(0,3)) / 2,  (pm(1,2) + pm(1,3)) / 2 }, // top (2)
      { (pm(0,3) + pm(0,0)) / 2,  (pm(1,3) + pm(1,0)) / 2 }  // left (3)
   };
   double tv[4][2] = // backup of original points
   {
      { pm(0,0), pm(1,0) },
      { pm(0,1), pm(1,1) },
      { pm(0,2), pm(1,2) },
      { pm(0,3), pm(1,3) },
   };

   // TODO: use the PointMatrix class!

   if (split == 1) // "X" split face
   {
      make_point_mat(tv[0], tmid[0], tmid[2], tv[3], pm);
      ConstrainFace (  v0,   mid[0],  mid[2],   v3,
                     face_T, master_dofs, level+1);

      make_point_mat(tmid[0], tv[1], tv[2], tmid[2], pm);
      ConstrainFace ( mid[0],   v1,    v2,   mid[2],
                     face_T, master_dofs, level+1);
   }
   else if (split == 2) // "Y" split face
   {
      make_point_mat(tv[0], tv[1], tmid[1], tmid[3], pm);
      ConstrainFace (  v0,    v1,   mid[1],  mid[3],
                     face_T, master_dofs, level+1);

      make_point_mat(tmid[3], tmid[1], tv[2], tv[3], pm);
      ConstrainFace ( mid[3],  mid[1],   v2,    v3,
                     face_T, master_dofs, level+1);
   }
}

void NCMeshHex::ProcessMasterEdge(Node* node[2], Node* edge)
{
   // set up a face transformation that will keep track of our position
   // within the master edge
   IsoparametricTransformation edge_T;
   edge_T.SetFE(&SegmentFE);

   // initial transformation is identity (interval 0..1)
   DenseMatrix& pm = edge_T.GetPointMat();
   pm.SetSize(1, 2);
   pm(0,0) = 0;
   pm(0,1) = 1;

   if (node[0]->vertex->index > node[1]->vertex->index)
      std::swap(pm(0,0), pm(0,1));

   // get a list of DOFs on the master edge
   Array<int> master_dofs;
   space->GetEdgeDofs(edge->edge->index, master_dofs);

   ConstrainEdge(node[0], node[1], edge_T, master_dofs, 0);
}

void NCMeshHex::ProcessMasterFace(Node* node[4], Face* face)
{
   // set up a face transformation that will keep track of our position
   // within the master face
   IsoparametricTransformation face_T;
   face_T.SetFE(&QuadrilateralFE);

   // initial transformation is identity (vertices of the unit square)
   DenseMatrix& pm = face_T.GetPointMat();
   pm.SetSize(2, 4);
   pm(0,0) = 0;  pm(0,1) = 1;  pm(0,2) = 1;  pm(0,3) = 0;
   pm(1,0) = 0;  pm(1,1) = 0;  pm(1,2) = 1;  pm(1,3) = 1;

   // get a list of DOFs on the master face
   Array<int> master_dofs;
   space->GetFaceDofs(face->index, master_dofs);

   ConstrainFace(node[0], node[1], node[2], node[3], face_T, master_dofs, 0);
}

bool NCMeshHex::DofFinalizable(DofData& dd)
{
   // are all constraining DOFs finalized?
   for (int i = 0; i < dd.dep_list.Size(); i++)
      if (!dof_data[dd.dep_list[i].dof].finalized)
         return false;

   return true;
}

SparseMatrix*
   NCMeshHex::GetInterpolation(Mesh* mesh, FiniteElementSpace *space)
{
   // get a list of our Vertices
   Node** vertex_nodes = new Node*[mesh->GetNV()];
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
   {
      if (it->vertex)
      {
         MFEM_ASSERT(it->vertex->index != -1, "Vertices not indexed.");
         vertex_nodes[it->vertex->index] = it;
      }
   }

   // pull edge numbering from the Mesh
   {
      Array<int> ev;
      for (int i = 0; i < mesh->GetNEdges(); i++)
      {
         mesh->GetEdgeVertices(i, ev);
         Node* node = nodes.Peek(vertex_nodes[ev[0]], vertex_nodes[ev[1]]);

         MFEM_ASSERT(node && node->edge, "Edge not found.");
         node->edge->index = i;
      }
   }

   // pull face numbering from the Mesh
   for (int i = 0; i < mesh->GetNFaces(); i++)
   {
      const int* fv = mesh->GetFace(i)->GetVertices();
      Face* face = faces.Peek(vertex_nodes[fv[0]], vertex_nodes[fv[1]],
                              vertex_nodes[fv[2]], vertex_nodes[fv[3]]);

      MFEM_ASSERT(face, "Face not found.");
      face->index = i;
   }

   delete [] vertex_nodes;

   // allocate temporary data for each DOF
   int n_dofs = space->GetNDofs();
   dof_data = new DofData[n_dofs];

   this->space = space;

   // visit edges and faces of leaf elements
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      Element* elem = leaf_elements[i];
      MFEM_ASSERT(!elem->ref_type, "Not a leaf element.");
      GeomInfo& gi = GI[elem->geom];

      // visit edges of 'elem'
      for (int j = 0; j < gi.ne; j++)
      {
         const int* ev = gi.edges[j];
         Node* node[2] = { elem->node[ev[0]], elem->node[ev[1]] };

         Node* edge = nodes.Peek(node[0], node[1]);
         MFEM_ASSERT(edge && edge->edge, "Edge not found!");

         // this edge could contain slave edges that need constraining; traverse
         // them recursively and make them dependent on this master edge
         ProcessMasterEdge(node, edge);
      }

      // visit faces of 'elem'
      for (int j = 0; j < gi.nf; j++)
      {
         Node* node[4];
         const int* fv = gi.faces[j];
         for (int k = 0; k < 4; k++)
            node[k] = elem->node[fv[k]];

         Face* face = faces.Peek(node[0], node[1], node[2], node[3]);
         MFEM_ASSERT(face, "Face not found!");

         if (face->ref_count == 1 && !face->Boundary())
         {
            // this is a potential master face that could be constraining
            // smaller faces adjacent to it; traverse them recursively and
            // make them dependent on this master face
            ProcessMasterFace(node, face);
         }
      }
   }

   // DOFs that stayed independent are true DOFs
   int n_true_dofs = 0;
   for (int i = 0; i < n_dofs; i++)
   {
      DofData& dd = dof_data[i];
      if (dd.Independent())
         n_true_dofs++;
   }

   // create the conforming prolongation matrix
   SparseMatrix* cP = new SparseMatrix(n_dofs, n_true_dofs);

   // put identity in the matrix for true DOFs
   for (int i = 0, true_dof = 0; i < n_dofs; i++)
   {
      DofData& dd = dof_data[i];
      if (dd.Independent())
      {
         cP->Add(i, true_dof++, 1.0);
         dd.finalized = true;
      }
   }

   // resolve dependencies of slave DOFs
   bool finished;
   int n_finalized = n_true_dofs;
   do
   {
      finished = true;
      for (int i = 0; i < n_dofs; i++)
      {
         DofData& dd = dof_data[i];
         if (!dd.finalized && DofFinalizable(dd))
         {
            for (int j = 0; j < dd.dep_list.Size(); j++)
            {
               Dependency& dep = dd.dep_list[j];

               Array<int> cols;
               Vector srow;
               cP->GetRow(dep.dof, cols, srow);

               for (int k = 0; k < cols.Size(); k++)
                  cP->Add(i, cols[k], dep.coef * srow[k]);
            }

            dd.finalized = true;
            n_finalized++;
            finished = false;
         }
      }
   }
   while (!finished);

   // if everything is consistent (mesh, face orientations, etc.), we should
   // be able to finalize all slave DOFs, otherwise it's a serious error
   if (n_finalized != n_dofs)
      mfem_error("Error creating cP matrix.");

   delete [] dof_data;

   cP->Finalize();
   return cP;
}


//// Coarse to fine transformations ////////////////////////////////////////////

void NCMeshHex::PointMatrix::GetMatrix(DenseMatrix& point_matrix) const
{
   point_matrix.SetSize(points[0].dim, np);
   for (int i = 0; i < np; i++)
      for (int j = 0; j < points[0].dim; j++)
         point_matrix(j, i) = points[i].coord[j];
}

void NCMeshHex::GetFineTransforms(Element* elem, int coarse_index,
                                  FineTransform* transforms,
                                  const PointMatrix& pm)
{
   if (!elem->ref_type)
   {
      // we got to a leaf, store the fine element transformation
      FineTransform& ft = transforms[elem->index];
      ft.coarse_index = coarse_index;
      pm.GetMatrix(ft.point_matrix);
      return;
   }

   // recurse into the finer children, adjusting the point matrix
   if (elem->geom == Geometry::CUBE)
   {
       if (elem->ref_type == 1) // split along X axis
       {
          Point mid01(pm(0), pm(1)), mid23(pm(2), pm(3));
          Point mid67(pm(6), pm(7)), mid45(pm(4), pm(5));

          GetFineTransforms(elem->child[0], coarse_index, transforms,
                            PointMatrix(pm(0), mid01, mid23, pm(3),
                                        pm(4), mid45, mid67, pm(7)));

          GetFineTransforms(elem->child[1], coarse_index, transforms,
                            PointMatrix(mid01, pm(1), pm(2), mid23,
                                        mid45, pm(5), pm(6), mid67));
       }
       else if (elem->ref_type == 2) // split along Y axis
       {
          Point mid12(pm(1), pm(2)), mid30(pm(3), pm(0));
          Point mid56(pm(5), pm(6)), mid74(pm(7), pm(4));

          GetFineTransforms(elem->child[0], coarse_index, transforms,
                            PointMatrix(pm(0), pm(1), mid12, mid30,
                                        pm(4), pm(5), mid56, mid74));

          GetFineTransforms(elem->child[1], coarse_index, transforms,
                            PointMatrix(mid30, mid12, pm(2), pm(3),
                                        mid74, mid56, pm(6), pm(7)));
       }
       else if (elem->ref_type == 4) // split along Z axis
       {
          Point mid04(pm(0), pm(4)), mid15(pm(1), pm(5));
          Point mid26(pm(2), pm(6)), mid37(pm(3), pm(7));

          GetFineTransforms(elem->child[0], coarse_index, transforms,
                            PointMatrix(pm(0), pm(1), pm(2), pm(3),
                                        mid04, mid15, mid26, mid37));

          GetFineTransforms(elem->child[1], coarse_index, transforms,
                            PointMatrix(mid04, mid15, mid26, mid37,
                                        pm(4), pm(5), pm(6), pm(7)));
       }
       else if (elem->ref_type == 3) // XY split
       {
           Point mid01(pm(0), pm(1)), mid12(pm(1), pm(2));
           Point mid23(pm(2), pm(3)), mid30(pm(3), pm(0));
           Point mid45(pm(4), pm(5)), mid56(pm(5), pm(6));
           Point mid67(pm(6), pm(7)), mid74(pm(7), pm(4));

           Point midf0(mid23, mid12, mid01, mid30);
           Point midf5(mid45, mid56, mid67, mid74);

           GetFineTransforms(elem->child[0], coarse_index, transforms,
                             PointMatrix(pm(0), mid01, midf0, mid30,
                                         pm(4), mid45, midf5, mid74));

           GetFineTransforms(elem->child[1], coarse_index, transforms,
                             PointMatrix(mid01, pm(1), mid12, midf0,
                                         mid45, pm(5), mid56, midf5));

           GetFineTransforms(elem->child[2], coarse_index, transforms,
                             PointMatrix(midf0, mid12, pm(2), mid23,
                                         midf5, mid56, pm(6), mid67));

           GetFineTransforms(elem->child[3], coarse_index, transforms,
                             PointMatrix(mid30, midf0, mid23, pm(3),
                                         mid74, midf5, mid67, pm(7)));
       }
       else if (elem->ref_type == 5) // XZ split
       {
          Point mid01(pm(0), pm(1)), mid23(pm(2), pm(3));
          Point mid45(pm(4), pm(5)), mid67(pm(6), pm(7));
          Point mid04(pm(0), pm(4)), mid15(pm(1), pm(5));
          Point mid26(pm(2), pm(6)), mid37(pm(3), pm(7));

          Point midf1(mid01, mid15, mid45, mid04);
          Point midf3(mid23, mid37, mid67, mid26);

          GetFineTransforms(elem->child[0], coarse_index, transforms,
                            PointMatrix(pm(0), mid01, mid23, pm(3),
                                        mid04, midf1, midf3, mid37));

          GetFineTransforms(elem->child[1], coarse_index, transforms,
                            PointMatrix(mid01, pm(1), pm(2), mid23,
                                        midf1, mid15, mid26, midf3));

          GetFineTransforms(elem->child[2], coarse_index, transforms,
                            PointMatrix(midf1, mid15, mid26, midf3,
                                        mid45, pm(5), pm(6), mid67));

          GetFineTransforms(elem->child[3], coarse_index, transforms,
                            PointMatrix(mid04, midf1, midf3, mid37,
                                        pm(4), mid45, mid67, pm(7)));
       }
       else if (elem->ref_type == 6) // YZ split
       {
           Point mid12(pm(1), pm(2)), mid30(pm(3), pm(0));
           Point mid56(pm(5), pm(6)), mid74(pm(7), pm(4));
           Point mid04(pm(0), pm(4)), mid15(pm(1), pm(5));
           Point mid26(pm(2), pm(6)), mid37(pm(3), pm(7));

           Point midf2(mid12, mid26, mid56, mid15);
           Point midf4(mid30, mid04, mid74, mid37);

           GetFineTransforms(elem->child[0], coarse_index, transforms,
                             PointMatrix(pm(0), pm(1), mid12, mid30,
                                         mid04, mid15, midf2, midf4));

           GetFineTransforms(elem->child[1], coarse_index, transforms,
                             PointMatrix(mid30, mid12, pm(2), pm(3),
                                         midf4, midf2, mid26, mid37));

           GetFineTransforms(elem->child[2], coarse_index, transforms,
                             PointMatrix(mid04, mid15, midf2, midf4,
                                         pm(4), pm(5), mid56, mid74));

           GetFineTransforms(elem->child[3], coarse_index, transforms,
                             PointMatrix(midf4, midf2, mid26, mid37,
                                         mid74, mid56, pm(6), pm(7)));
       }
       else if (elem->ref_type == 7) // full isotropic refinement
       {
          Point mid01(pm(0), pm(1)), mid12(pm(1), pm(2));
          Point mid23(pm(2), pm(3)), mid30(pm(3), pm(0));
          Point mid45(pm(4), pm(5)), mid56(pm(5), pm(6));
          Point mid67(pm(6), pm(7)), mid74(pm(7), pm(4));
          Point mid04(pm(0), pm(4)), mid15(pm(1), pm(5));
          Point mid26(pm(2), pm(6)), mid37(pm(3), pm(7));

          Point midf0(mid23, mid12, mid01, mid30);
          Point midf1(mid01, mid15, mid45, mid04);
          Point midf2(mid12, mid26, mid56, mid15);
          Point midf3(mid23, mid37, mid67, mid26);
          Point midf4(mid30, mid04, mid74, mid37);
          Point midf5(mid45, mid56, mid67, mid74);

          Point midel(midf1, midf3);

          GetFineTransforms(elem->child[0], coarse_index, transforms,
                            PointMatrix(pm(0), mid01, midf0, mid30,
                                        mid04, midf1, midel, midf4));

          GetFineTransforms(elem->child[1], coarse_index, transforms,
                            PointMatrix(mid01, pm(1), mid12, midf0,
                                        midf1, mid15, midf2, midel));

          GetFineTransforms(elem->child[2], coarse_index, transforms,
                            PointMatrix(midf0, mid12, pm(2), mid23,
                                        midel, midf2, mid26, midf3));

          GetFineTransforms(elem->child[3], coarse_index, transforms,
                            PointMatrix(mid30, midf0, mid23, pm(3),
                                        midf4, midel, midf3, mid37));

          GetFineTransforms(elem->child[4], coarse_index, transforms,
                            PointMatrix(mid04, midf1, midel, midf4,
                                        pm(4), mid45, midf5, mid74));

          GetFineTransforms(elem->child[5], coarse_index, transforms,
                            PointMatrix(midf1, mid15, midf2, midel,
                                        mid45, pm(5), mid56, midf5));

          GetFineTransforms(elem->child[6], coarse_index, transforms,
                            PointMatrix(midel, midf2, mid26, midf3,
                                        midf5, mid56, pm(6), mid67));

          GetFineTransforms(elem->child[7], coarse_index, transforms,
                            PointMatrix(midf4, midel, midf3, mid37,
                                        mid74, midf5, mid67, pm(7)));
       }
   }
   else if (elem->geom == Geometry::SQUARE)
   {
      if (elem->ref_type == 1) // X split
      {
         Point mid01(pm(0), pm(1)), mid23(pm(2), pm(3));

         GetFineTransforms(elem->child[0], coarse_index, transforms,
                           PointMatrix(pm(0), mid01, mid23, pm(3)));

         GetFineTransforms(elem->child[1], coarse_index, transforms,
                           PointMatrix(mid01, pm(1), pm(2), mid23));
      }
      else if (elem->ref_type == 2) // Y split
      {
         Point mid12(pm(1), pm(2)), mid30(pm(3), pm(0));

         GetFineTransforms(elem->child[0], coarse_index, transforms,
                           PointMatrix(pm(0), pm(1), mid12, mid30));

         GetFineTransforms(elem->child[1], coarse_index, transforms,
                           PointMatrix(mid30, mid12, pm(2), pm(3)));
      }
      else if (elem->ref_type == 3) // iso split
      {
         Point mid01(pm(0), pm(1)), mid12(pm(1), pm(2));
         Point mid23(pm(2), pm(3)), mid30(pm(3), pm(0));
         Point midel(mid01, mid23);

         GetFineTransforms(elem->child[0], coarse_index, transforms,
                           PointMatrix(pm(0), mid01, midel, mid30));

         GetFineTransforms(elem->child[1], coarse_index, transforms,
                           PointMatrix(mid01, pm(1), mid12, midel));

         GetFineTransforms(elem->child[2], coarse_index, transforms,
                           PointMatrix(midel, mid12, pm(2), mid23));

         GetFineTransforms(elem->child[3], coarse_index, transforms,
                           PointMatrix(mid30, midel, mid23, pm(3)));
      }
   }
   else if (elem->geom == Geometry::TRIANGLE)
   {
      Point mid01(pm(0), pm(1)), mid12(pm(1), pm(2)), mid20(pm(2), pm(0));

      GetFineTransforms(elem->child[0], coarse_index, transforms,
                        PointMatrix(pm(0), mid01, mid20));

      GetFineTransforms(elem->child[1], coarse_index, transforms,
                        PointMatrix(mid01, pm(1), mid12));

      GetFineTransforms(elem->child[2], coarse_index, transforms,
                        PointMatrix(mid20, mid12, pm(2)));

      GetFineTransforms(elem->child[3], coarse_index, transforms,
                        PointMatrix(mid01, mid12, mid20));
   }
}

NCMeshHex::FineTransform* NCMeshHex::GetFineTransforms()
{
   if (!coarse_elements.Size())
      mfem_error("You need to call MarkCoarseLevel before calling Refine and "
                 "GetFineTransformations.");

   FineTransform* transforms = new FineTransform[leaf_elements.Size()];

   // get transformations for fine elements, starting from coarse elements
   for (int i = 0; i < coarse_elements.Size(); i++)
   {
      Element* c_elem = coarse_elements[i];

      if (c_elem->geom == Geometry::CUBE)
      {
         PointMatrix pm(Point(0,0,0), Point(1,0,0), Point(1,1,0), Point(0,1,0),
                        Point(0,0,1), Point(1,0,1), Point(1,1,1), Point(0,1,1));

         GetFineTransforms(c_elem, i, transforms, pm);
      }
      else if (c_elem->geom == Geometry::SQUARE)
      {
         PointMatrix pm(Point(0,0), Point(1,0), Point(1,1), Point(0,1));
         GetFineTransforms(c_elem, i, transforms, pm);
      }
      else if (c_elem->geom == Geometry::TRIANGLE)
      {
         PointMatrix pm(Point(0,0), Point(1,0), Point(0,1));
         GetFineTransforms(c_elem, i, transforms, pm);
      }
      else
         mfem_error("NCMeshHex::GetFineTransforms: Bad geometry.");

      // TODO: detect non-refined elements and return empty matrices as identities
   }

   // get rid of the coarse level array to save memory
   coarse_elements.DeleteAll();

   return transforms;
}


//// Utility ///////////////////////////////////////////////////////////////////

void NCMeshHex::FaceSplitLevel(Node* v1, Node* v2, Node* v3, Node* v4,
                               int& h_level, int& v_level)
{
   int hl1, hl2, vl1, vl2;
   Node* mid[4];

   switch (FaceSplitType(v1, v2, v3, v4, mid))
   {
   case 0: // not split
      h_level = v_level = 0;
      break;

   case 1: // vertical
      FaceSplitLevel(v1, mid[0], mid[2], v4, hl1, vl1);
      FaceSplitLevel(mid[0], v2, v3, mid[2], hl2, vl2);
      h_level = std::max(hl1, hl2);
      v_level = std::max(vl1, vl2) + 1;
      break;

   default: // horizontal
      FaceSplitLevel(v1, v2, mid[1], mid[3], hl1, vl1);
      FaceSplitLevel(mid[3], mid[1], v3, v4, hl2, vl2);
      h_level = std::max(hl1, hl2) + 1;
      v_level = std::max(vl1, vl2);
   }
}

static int max4(int a, int b, int c, int d)
{
   return std::max(std::max(a, b), std::max(c, d));
}

void NCMeshHex::CountSplits(Element* elem, int splits[3])
{
   Node** node = elem->node;
   GeomInfo& gi = GI[elem->geom];

   MFEM_ASSERT(elem->geom == Geometry::CUBE, "TODO");

   int level[6][2];
   for (int i = 0; i < gi.nf; i++)
   {
      const int* fv = gi.faces[i];
      FaceSplitLevel(node[fv[0]], node[fv[1]], node[fv[2]], node[fv[3]],
                     level[i][1], level[i][0]);
   }

   splits[0] = max4(level[0][0], level[1][0], level[3][0], level[5][0]);
   splits[1] = max4(level[0][1], level[2][0], level[4][0], level[5][1]);
   splits[2] = max4(level[1][1], level[2][1], level[3][1], level[4][1]);
}

void NCMeshHex::LimitNCLevel(int max_level)
{
   if (max_level < 1)
      mfem_error("NCMeshHex::LimitNCLevel: max_level must be 1 or greater.");

   while (1)
   {
      UpdateLeafElements();

      Array<NCRefinement> refinements;
      for (int i = 0; i < leaf_elements.Size(); i++)
      {
         int splits[3];
         CountSplits(leaf_elements[i], splits);

         int ref_type = 0;
         for (int k = 0; k < 3; k++)
            if (splits[k] > max_level)
               ref_type |= (1 << k);

         if (ref_type)
            refinements.Append(NCRefinement(i, ref_type));
      }

      if (!refinements.Size()) break;

      Refine(refinements);
   }
}

int NCMeshHex::CountElements(Element* elem)
{
   int n = 1;
   if (elem->ref_type)
   {
      for (int i = 0; i < 8; i++)
         if (elem->child[i])
            n += CountElements(elem->child[i]);
   }
   return n;
}

long NCMeshHex::MemoryUsage()
{
   int num_elem = 0;
   for (int i = 0; i < root_elements.Size(); i++)
      num_elem += CountElements(root_elements[i]);

   int num_vert = 0, num_edges = 0;
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
   {
      if (it->vertex) num_vert++;
      if (it->edge) num_edges++;
   }

   return num_elem * sizeof(Element) +
          num_vert * sizeof(Vertex) +
          num_edges * sizeof(Edge) +
          nodes.MemoryUsage() +
          faces.MemoryUsage() +
          root_elements.Size() * sizeof(Element*) +
          leaf_elements.Size() * sizeof(Element*) +
          sizeof(*this);
}
