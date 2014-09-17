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

#include "ncmeshhex.hpp"


NCMeshHex::NCMeshHex(const Mesh *mesh)
{
   // create our Element struct for each mesh element
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      const ::Element *elem = mesh->GetElement(i);
      const int *v = elem->GetVertices();

      if (elem->GetType() != ::Element::HEXAHEDRON)
         mfem_error("NCMeshHex: only hexahedra supported.");

      Element* nc_elem = new Element(elem->GetAttribute());
      root_elements.Append(nc_elem);

      for (int j = 0; j < 8; j++)
      {
         // root nodes are special: they have p1 == p2 == orig. mesh vertex id
         Node* node = nodes.Get(v[j], v[j]);

         // create a vertex in the node and initialize its position
         const double* pos = mesh->GetVertex(v[j]);
         node->vertex = new Vertex(pos[0], pos[1], pos[2]);

         nc_elem->node[j] = node;
      }

      // increase reference count of all nodes the element is using
      // (note: this will also create and reference all edge nodes)
      RefElementNodes(nc_elem);
   }

   num_leaf_elements = root_elements.Size();

   // store boundary element attributes
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      const ::Element *be = mesh->GetBdrElement(i);
      const int *v = be->GetVertices();

      if (be->GetType() != ::Element::QUADRILATERAL)
         mfem_error("NCMeshHex: only quadrilateral boundary "
                    "elements supported.");

      Node* node[4];
      for (int i = 0; i < 4; i++)
      {
         node[i] = nodes.Peek(v[i], v[i]);
         if (!node[i])
            mfem_error("NCMeshHex: boundary elements inconsistent.");
      }

      Face* face = faces.Peek(node[0], node[1], node[2], node[3]);
      if (!face)
         mfem_error("NCMeshHex: face not found.");

      face->attribute = be->GetAttribute();
   }
}

NCMeshHex::~NCMeshHex()
{
   for (int i = 0; i < root_elements.Size(); i++)
      delete root_elements[i];
}

NCMeshHex::Element::~Element() // FIXME!!!
{
   if (ref_type)
   {
      for (int i = 0; i < 8; i++)
         if (child[i])
            delete child[i];
   }
   else
   {
      for (int i = 0; i < 8; i++)
         node[i]->UnrefVertex();
   }
}

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

void NCMeshHex::Node::UnrefVertex()
{
   MFEM_ASSERT(vertex, "Cannot unref a nonexistent vertex.");
   if (!vertex->Unref()) vertex = NULL;
   if (!vertex && !edge) delete this;
}

void NCMeshHex::Node::UnrefEdge()
{
   MFEM_ASSERT(edge, "Cannot unref a nonexistent edge.");
   if (!edge->Unref()) edge = NULL;
   if (!vertex && !edge) delete this;
}

NCMeshHex::Node::~Node()
{
   if (vertex) delete vertex;
   if (edge) delete edge;
}

static Hexahedron hexahedron; // used for a list of edges

const int hex_faces[6][4] = // TODO: this should be shared somehow
{{3, 2, 1, 0}, {0, 1, 5, 4},
 {1, 2, 6, 5}, {2, 3, 7, 6},
 {3, 0, 4, 7}, {4, 5, 6, 7}};


void NCMeshHex::RefElementNodes(Element *elem)
{
   Node** node = elem->node;

   // ref all vertices
   for (int i = 0; i < 8; i++)
      node[i]->RefVertex();

   // ref all edges (possibly creating them)
   for (int i = 0; i < hexahedron.GetNEdges(); i++)
   {
      const int* ev = hexahedron.GetEdgeVertices(i);
      nodes.Get(node[ev[0]], node[ev[1]])->RefEdge();
   }

   // ref all faces (possibly creating them)
   for (int i = 0; i < 6; i++)
   {
      const int* fv = hex_faces[i];
      faces.Get(node[fv[0]], node[fv[1]], node[fv[2]], node[fv[3]])->Ref();
   }
}

void NCMeshHex::UnrefElementNodes(Element *elem)
{
   Node** node = elem->node;

   // unref all faces (possibly destroying them)
   for (int i = 0; i < 6; i++)
   {
      const int* fv = hex_faces[i];
      Face* face = faces.Peek(node[fv[0]], node[fv[1]], node[fv[2]], node[fv[3]]);
      if (!face->Unref()) faces.Delete(face);
   }

   // unref all edges (possibly destroying them)
   for (int i = 0; i < hexahedron.GetNEdges(); i++)
   {
      const int* ev = hexahedron.GetEdgeVertices(i);
      nodes.Peek(node[ev[0]], node[ev[1]])->UnrefEdge();
   }

   // unref all vertices (possibly destroying them)
   for (int i = 0; i < 8; i++)
      node[i]->UnrefVertex();
}


//// Refinement & Derefinement /////////////////////////////////////////////////

NCMeshHex::Node* NCMeshHex::GetMidVertex(Node* n1, Node* n2)
{
   Node* mid = nodes.Get(n1, n2);
   if (!mid->vertex)
   {
      MFEM_ASSERT(n1->vertex && n2->vertex,
                  "NCMeshHex::CreateMidVertex: missing parent vertices");

      mid->vertex = new Vertex;
      for (int i = 0; i < 3; i++)
         mid->vertex->pos[i] = (n1->vertex->pos[i] + n2->vertex->pos[i]) * 0.5;
   }
   return mid;
}

NCMeshHex::Element*
   NCMeshHex::NewElement(Node* n0, Node* n1, Node* n2, Node* n3,
                         Node* n4, Node* n5, Node* n6, Node* n7,
                         int attr,
                         int fattr0, int fattr1, int fattr2,
                         int fattr3, int fattr4, int fattr5)
{
   // create new unrefined element, initialize nodes
   Element* e = new Element(attr);
   e->node[0] = n0, e->node[1] = n1, e->node[2] = n2, e->node[3] = n3;
   e->node[4] = n4, e->node[5] = n5, e->node[6] = n6, e->node[7] = n7;

   // get face nodes and assign face attributes
   Face* f[6];
   for (int i = 0; i < 6; i++)
   {
      const int* fv = hex_faces[i];
      f[i] = faces.Get(e->node[fv[0]], e->node[fv[1]],
                       e->node[fv[2]], e->node[fv[3]]);
   }

   f[0]->attribute = fattr0,  f[1]->attribute = fattr1;
   f[2]->attribute = fattr2,  f[3]->attribute = fattr3;
   f[4]->attribute = fattr4,  f[5]->attribute = fattr5;

   return e;
}


void NCMeshHex::Refine(Element* elem, int ref_type)
{
   if (elem->ref_type)
      mfem_error("NCMeshHex::Refine: element already refined.");

   // TODO: do combined splits at once
   // TODO: check for incompatible refinements between neighbors!!!

   Node** n = elem->node;
   int attr = elem->attribute;

   // get element's face attributes
   int fa[6];
   for (int i = 0; i < 6; i++)
   {
      const int* fv = hex_faces[i];
      fa[i] = faces.Peek(elem->node[fv[0]], elem->node[fv[1]],
                         elem->node[fv[2]], elem->node[fv[3]])->attribute;

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

   Element *child0, *child1;

   if (ref_type == 1) // split along X axis
   {
      Node* mid01 = GetMidVertex(n[0], n[1]);
      Node* mid23 = GetMidVertex(n[2], n[3]);
      Node* mid67 = GetMidVertex(n[6], n[7]);
      Node* mid45 = GetMidVertex(n[4], n[5]);

      child0 = NewElement(n[0], mid01, mid23, n[3],
                          n[4], mid45, mid67, n[7], attr,
                          fa[0], fa[1], -1, fa[3], fa[4], fa[5]);

      child1 = NewElement(mid01, n[1], n[2], mid23,
                          mid45, n[5], n[6], mid67, attr,
                          fa[0], fa[1], fa[2], fa[3], -1, fa[5]);
   }
   else if (ref_type == 2) // split along Y axis
   {
      Node* mid12 = GetMidVertex(n[1], n[2]);
      Node* mid30 = GetMidVertex(n[3], n[0]);
      Node* mid56 = GetMidVertex(n[5], n[6]);
      Node* mid74 = GetMidVertex(n[7], n[4]);

      child0 = NewElement(n[0], n[1], mid12, mid30,
                          n[4], n[5], mid56, mid74, attr,
                          fa[0], fa[1], fa[2], -1, fa[4], fa[5]);

      child1 = NewElement(mid30, mid12, n[2], n[3],
                          mid74, mid56, n[6], n[7], attr,
                          fa[0], -1, fa[2], fa[3], fa[4], fa[5]);
   }
   else if (ref_type == 4) // split along Z axis
   {
      Node* mid04 = GetMidVertex(n[0], n[4]);
      Node* mid15 = GetMidVertex(n[1], n[5]);
      Node* mid26 = GetMidVertex(n[2], n[6]);
      Node* mid37 = GetMidVertex(n[3], n[7]);

      child0 = NewElement(n[0], n[1], n[2], n[3],
                          mid04, mid15, mid26, mid37, attr,
                          fa[0], fa[1], fa[2], fa[3], fa[4], -1);

      child1 = NewElement(mid04, mid15, mid26, mid37,
                          n[4], n[5], n[6], n[7], attr,
                          -1, fa[1], fa[2], fa[3], fa[4], fa[5]);
   }

   // start using the nodes of the children (plus create edge nodes)
   RefElementNodes(child0);
   RefElementNodes(child1);

   // sign off of the nodes of the parent (some may get destroyed)
   UnrefElementNodes(elem);

   // mark the original element as refined
   elem->ref_type = ref_type;
   memset(elem->child, 0, sizeof(elem->child));
   elem->child[0] = child0;
   elem->child[1] = child1;

   // keep track of the number of leaf elements
   num_leaf_elements += 1;
}

void NCMeshHex::Derefine(Element* elem)
{
   // TODO
}


//// Mesh interface ////////////////////////////////////////////////////////////

int NCMeshHex::IndexVertices()
{
   int num_vert = 0;
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
      if (it->vertex)
         it->vertex->index = num_vert++;

   return num_vert;
}

int NCMeshHex::IndexEdges()
{
   int num_edges = 0;
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
      if (it->edge)
         it->edge->index = num_edges++;

   return num_edges;
}

void NCMeshHex::GetVertices(Array< ::Vertex>& vertices)
{
   int num_vert = IndexVertices();
   vertices.SetSize(num_vert);

   // copy vertices
   int i = 0;
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
      if (it->vertex)
         vertices[i++].SetCoords(it->vertex->pos);
}

void NCMeshHex::GetLeafElements(Element* e,
                                Array< ::Element*>& elements,
                                Array< ::Element*>& boundary)
{
   if (!e->ref_type)
   {
      Hexahedron* hex = new Hexahedron;
      hex->SetAttribute(e->attribute);
      for (int i = 0; i < 8; i++)
         hex->GetVertices()[i] = e->node[i]->vertex->index;

      elements.Append(hex);

      // also return boundary elements
      for (int i = 0; i < 6; i++)
      {
         const int* fv = hex_faces[i];
         NCMeshHex::Face* face = faces.Peek(e->node[fv[0]], e->node[fv[1]],
                                            e->node[fv[2]], e->node[fv[3]]);
         if (face->attribute >= 0)
         {
            Quadrilateral* quad = new Quadrilateral;
            quad->SetAttribute(face->attribute);
            for (int i = 0; i < 4; i++)
               quad->GetVertices()[i] = e->node[fv[i]]->vertex->index;

            boundary.Append(quad);
         }
      }
   }
   else
   {
      for (int i = 0; i < 8; i++)
         if (e->child[i])
            GetLeafElements(e->child[i], elements, boundary);
   }
}

void NCMeshHex::GetElements(Array< ::Element*>& elements,
                            Array< ::Element*>& boundary)
{
   // NOTE: this assumes GetVertices has already been called
   // so their 'index' member is valid

   elements.SetSize(num_leaf_elements);
   elements.SetSize(0);

   for (int i = 0; i < root_elements.Size(); i++)
      GetLeafElements(root_elements[i], elements, boundary);
}

/*void NCMeshHex::GetBdrElements(Array< ::Element*>& boundary)
{
   for (HashTable<Face>::Iterator it(faces); it; ++it)
   {
      if (it->attribute >= 0)
      {
         Quadrilateral* quad = new Quadrilateral;
         quad->SetAttribute(it->attribute);

         for (int i = 0; i < 4; i++)
         {
            Node* node = nodes.Peek(it->p[i]); // get one of the parent nodes
            quad->GetVertices()[i] = node->vertex->index;
         }

         boundary.Append(quad);
      }
   }
}*/


//// Interpolation /////////////////////////////////////////////////////////////




/* THINGS MISSING:
 *  - proper destruction of nodes (remove from hash table)
 *  - Refine: incompatible + multiple
 */
