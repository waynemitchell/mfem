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
         mfem_error("NCMeshHex::NCMeshHex: only hexahedrons supported.");

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

/*NCMeshHex::Node::~Node()
{
   if (vertex) delete vertex;
   if (edge) delete edge;
}*/

static Hexahedron hexahedron;

void NCMeshHex::RefElementNodes(Element *elem)
{
   // ref all vertices
   for (int i = 0; i < 8; i++)
      elem->node[i]->RefVertex();

   // ref all edges, possibly creating them
   for (int i = 0; i < hexahedron.GetNEdges(); i++)
   {
      const int* ev = hexahedron.GetEdgeVertices(i);
      Node* node = nodes.Get(elem->node[ev[0]], elem->node[ev[1]]);
      node->RefEdge();
   }
}

void NCMeshHex::UnrefElementNodes(Element *elem)
{
   // unref all edges
   for (int i = 0; i < hexahedron.GetNEdges(); i++)
   {
      const int* ev = hexahedron.GetEdgeVertices(i);
      Node* node = nodes.Get(elem->node[ev[0]], elem->node[ev[1]]);
      node->UnrefEdge();
   }

   // unref all vertices
   for (int i = 0; i < 8; i++)
      elem->node[i]->UnrefVertex();
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


void NCMeshHex::Refine(Element* elem, int ref_type)
{
   if (elem->ref_type)
      mfem_error("NCMeshHex::Refine: element already refined.");

   // TODO: do combined splits at once
   // TODO: check for incompatible refinements between neighbors!!!

   Node** n = elem->node;
   int attr = elem->attribute;

   /* Vertex numbering is assumed to be as follows:

            7              6
             +------------+
            /|           /|
         4 / |        5 / |
          +------------+  |
          |  |         |  |
          |  +---------|--+
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

      child0 = new Element(n[0], mid01, mid23, n[3],
                           n[4], mid45, mid67, n[7], attr);

      child1 = new Element(mid01, n[1], n[2], mid23,
                           mid45, n[5], n[6], mid67, attr);
   }
   else if (ref_type == 2) // split along Y axis
   {
      Node* mid12 = GetMidVertex(n[1], n[2]);
      Node* mid30 = GetMidVertex(n[3], n[0]);
      Node* mid56 = GetMidVertex(n[5], n[6]);
      Node* mid74 = GetMidVertex(n[7], n[4]);

      child0 = new Element(n[0], n[1], mid12, mid30,
                           n[4], n[5], mid56, mid74, attr);

      child1 = new Element(mid12, n[2], n[3], mid30,
                           mid56, n[6], n[7], mid74, attr);
   }
   else if (ref_type == 4) // split along Z axis
   {
      Node* mid04 = GetMidVertex(n[0], n[4]);
      Node* mid15 = GetMidVertex(n[1], n[5]);
      Node* mid26 = GetMidVertex(n[2], n[6]);
      Node* mid37 = GetMidVertex(n[3], n[7]);

      child0 = new Element(n[0], n[1], n[2], n[3],
                           mid04, mid15, mid26, mid37, attr);

      child1 = new Element(mid04, mid15, mid26, mid37,
                           n[4], n[5], n[6], n[7], attr);
   }

   // start using the nodes of the children (plus create edge nodes)
   RefElementNodes(child0);
   RefElementNodes(child1);

   // sign off of the nodes of the parent (some may get destroyed)
   UnrefElementNodes(elem);

   // finish the refinement
   memset(elem->child, 0, sizeof(elem->child));
   elem->child[0] = child0;
   elem->child[1] = child1;
   elem->ref_type = ref_type;

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

static void GetLeafElements(NCMeshHex::Element* e, Array<Element*>& elements)
{
   if (!e->ref_type)
   {
      Hexahedron* hex = new Hexahedron;
      hex->SetAttribute(e->attribute);
      for (int i = 0; i < 8; i++)
         hex->GetVertices()[i] = e->node[i]->vertex->index;

      elements.Append(hex);
   }
   else
   {
      for (int i = 0; i < 8; i++)
         if (e->child[i])
            GetLeafElements(e->child[i], elements);
   }
}

void NCMeshHex::GetElements(Array< ::Element*>& elements)
{
   // NOTE: this assumes GetVertices has already been called
   // so their 'index' member is valid

   elements.SetSize(num_leaf_elements);
   elements.SetSize(0);

   for (int i = 0; i < root_elements.Size(); i++)
      GetLeafElements(root_elements[i], elements);
}

void NCMeshHex::GetBdrElements(Array< ::Element*>& boundary)
{

}


//// Interpolation /////////////////////////////////////////////////////////////




/* THINGS MISSING:
 *  - proper destruction of nodes (remove from hash table)
 *  - Refine: incompatible + multiple
 */
