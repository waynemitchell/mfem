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

// TODO: this should be somewhere else
const int hex_edges[12][2] =
{{0, 1}, {1, 2}, {3, 2}, {0, 3},
 {4, 5}, {5, 6}, {7, 6}, {4, 7},
 {0, 4}, {1, 5}, {2, 6}, {3, 7}};

const int hex_faces[6][4] =
{{3, 2, 1, 0}, {0, 1, 5, 4},
 {1, 2, 6, 5}, {2, 3, 7, 6},
 {3, 0, 4, 7}, {4, 5, 6, 7}};


NCMeshHex::NCMeshHex(const Mesh *mesh)
{
   // create the NCMeshHex::Element struct for each mesh element
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

         if (!node->vertex)
         {
            // create a vertex in the node and initialize its position
            const double* pos = mesh->GetVertex(v[j]);
            node->vertex = new Vertex(pos[0], pos[1], pos[2]);
         }

         nc_elem->node[j] = node;
      }

      // increase reference count of the element's vertices
      RefVertices(nc_elem);
      // create edges and faces
      RefEdgesFaces(nc_elem);
   }

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
      UnrefEdgesFaces(elem);
   }
   UnrefVertices(elem);
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

void NCMeshHex::RefVertices(Element* elem)
{
   // ref all vertices
   for (int i = 0; i < 8; i++)
      elem->node[i]->RefVertex();
}

void NCMeshHex::UnrefVertices(Element* elem)
{
   // unref all vertices (possibly destroying them)
   for (int i = 0; i < 8; i++)
      elem->node[i]->UnrefVertex(nodes);
}

void NCMeshHex::RefEdgesFaces(Element *elem)
{
   // NOTE: vertices must exist for this to work!
   Node** node = elem->node;

   // ref all edges (possibly creating them)
   for (int i = 0; i < 12; i++)
   {
      const int* ev = hex_edges[i];
      nodes.Get(node[ev[0]], node[ev[1]])->RefEdge();
   }

   // ref all faces (possibly creating them)
   for (int i = 0; i < 6; i++)
   {
      const int* fv = hex_faces[i];
      faces.Get(node[fv[0]], node[fv[1]], node[fv[2]], node[fv[3]])->Ref();
   }
}

void NCMeshHex::UnrefEdgesFaces(Element *elem)
{
   // NOTE: vertices must exist for this to work!
   Node** node = elem->node;

   // unref all faces (possibly destroying them)
   for (int i = 0; i < 6; i++)
   {
      const int* fv = hex_faces[i];
      Face* face = faces.Peek(node[fv[0]], node[fv[1]], node[fv[2]], node[fv[3]]);
      if (!face->Unref()) faces.Delete(face);
   }

   // unref all edges (possibly destroying them)
   for (int i = 0; i < 12; i++)
   {
      const int* ev = hex_edges[i];
      nodes.Peek(node[ev[0]], node[ev[1]])->UnrefEdge(nodes);
   }
}


//// Refinement & Derefinement /////////////////////////////////////////////////

NCMeshHex::Element::Element(int attr)
   : ref_type(0), attribute(attr)
{
   memset(node, 0, sizeof(node));
   memset(child, 0, sizeof(child));
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


NCMeshHex::Vertex* NCMeshHex::NewVertex(Node* v1, Node* v2)
{
   MFEM_ASSERT(v1->vertex && v2->vertex,
               "NCMeshHex::NewVertex: missing parent vertices.");

   Vertex* v = new Vertex;
   for (int i = 0; i < 3; i++)
      v->pos[i] = (v1->vertex->pos[i] + v2->vertex->pos[i]) * 0.5;

   return v;
}

NCMeshHex::Node* NCMeshHex::GetMidEdgeVertex(Node* v1, Node* v2)
{
   Node* mid = nodes.Get(v1, v2);
   if (!mid->vertex) mid->vertex = NewVertex(v1, v2);
   return mid;
}

NCMeshHex::Node*
   NCMeshHex::GetMidFaceVertex(Node* e1, Node* e2, Node* e3, Node* e4)
{
   // mid-face node must be created either from (e1, e3) or from (e2, e4)
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


int NCMeshHex::FaceSplitType(Node* v1, Node* v2, Node* v3, Node* v4,
                             Node* mid[5])
{
   // find edge nodes
   Node* e1 = nodes.Peek(v1, v2);
   Node* e2 = nodes.Peek(v2, v3);
   Node* e3 = nodes.Peek(v3, v4);
   Node* e4 = nodes.Peek(v4, v1);

   // optional: return the mid-edge nodes if requested
   if (mid) mid[0] = e1, mid[1] = e2, mid[2] = e3, mid[3] = e4, mid[4] = NULL;

   // try to get a mid-face node, either by (e1, e3) or by (e2, e4)
   Node *midf1, *midf2;
   if (e1 && e3) midf1 = nodes.Peek(e1, e3);
   if (e2 && e4) midf2 = nodes.Peek(e2, e4);

   // only one way to to access the mid-face node must always exist
   MFEM_ASSERT(!(midf1 && midf2), "Incorrectly split face!");

   if (!midf1 && !midf2) return 0; // face not split

   Node* midf = midf1 ? midf1 : midf2;
   if (mid) mid[4] = midf;

   if (midf->vertex) return 3; // face split both ways

   MFEM_ASSERT(midf->edge, "Nodes inconsistent!");

   return midf1 ? 1 : 2; // face split "vertically" or "horizontally"
}


void NCMeshHex::Refine(Element* elem, int ref_type)
{
   if (elem->ref_type)
      mfem_error("NCMeshHex::Refine: element already refined.");

   // TODO: do combined splits at once
   // TODO: check for incompatible refinements between neighbors!!!

   Node** node = elem->node;
   Element** child = elem->child;
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
      Node* mid01 = GetMidEdgeVertex(node[0], node[1]);
      Node* mid23 = GetMidEdgeVertex(node[2], node[3]);
      Node* mid67 = GetMidEdgeVertex(node[6], node[7]);
      Node* mid45 = GetMidEdgeVertex(node[4], node[5]);

      child[0] = NewElement(node[0], mid01, mid23, node[3],
                            node[4], mid45, mid67, node[7], attr,
                            fa[0], fa[1], -1, fa[3], fa[4], fa[5]);

      child[1] = NewElement(mid01, node[1], node[2], mid23,
                            mid45, node[5], node[6], mid67, attr,
                            fa[0], fa[1], fa[2], fa[3], -1, fa[5]);
   }
   else if (ref_type == 2) // split along Y axis
   {
      Node* mid12 = GetMidEdgeVertex(node[1], node[2]);
      Node* mid30 = GetMidEdgeVertex(node[3], node[0]);
      Node* mid56 = GetMidEdgeVertex(node[5], node[6]);
      Node* mid74 = GetMidEdgeVertex(node[7], node[4]);

      child[0] = NewElement(node[0], node[1], mid12, mid30,
                            node[4], node[5], mid56, mid74, attr,
                            fa[0], fa[1], fa[2], -1, fa[4], fa[5]);

      child[1] = NewElement(mid30, mid12, node[2], node[3],
                            mid74, mid56, node[6], node[7], attr,
                            fa[0], -1, fa[2], fa[3], fa[4], fa[5]);
   }
   else if (ref_type == 4) // split along Z axis
   {
      Node* mid04 = GetMidEdgeVertex(node[0], node[4]);
      Node* mid15 = GetMidEdgeVertex(node[1], node[5]);
      Node* mid26 = GetMidEdgeVertex(node[2], node[6]);
      Node* mid37 = GetMidEdgeVertex(node[3], node[7]);

      child[0] = NewElement(node[0], node[1], node[2], node[3],
                            mid04, mid15, mid26, mid37, attr,
                            fa[0], fa[1], fa[2], fa[3], fa[4], -1);

      child[1] = NewElement(mid04, mid15, mid26, mid37,
                            node[4], node[5], node[6], node[7], attr,
                            -1, fa[1], fa[2], fa[3], fa[4], fa[5]);
   }
   else if (ref_type == 3) // XY split
   {
      // TODO
   }
   else if (ref_type == 5) // XZ split
   {
      Node* mid01 = GetMidEdgeVertex(node[0], node[1]);
      Node* mid23 = GetMidEdgeVertex(node[2], node[3]);
      Node* mid45 = GetMidEdgeVertex(node[4], node[5]);
      Node* mid67 = GetMidEdgeVertex(node[6], node[7]);

      Node* mid04 = GetMidEdgeVertex(node[0], node[4]);
      Node* mid15 = GetMidEdgeVertex(node[1], node[5]);
      Node* mid26 = GetMidEdgeVertex(node[2], node[6]);
      Node* mid37 = GetMidEdgeVertex(node[3], node[7]);

      Node* midf1 = GetMidFaceVertex(mid01, mid15, mid45, mid04);
      Node* midf3 = GetMidFaceVertex(mid23, mid37, mid67, mid26);

      child[0] = NewElement(node[0], mid01, mid23, node[3],
                            mid04, midf1, midf3, mid37, attr,
                            fa[0], fa[1], -1, fa[3], fa[4], -1);

      child[1] = NewElement(mid01, node[1], node[2], mid23,
                            midf1, mid15, mid26, midf3, attr,
                            fa[0], fa[1], fa[2], fa[3], -1, -1);

      child[2] = NewElement(midf1, mid15, mid26, midf3,
                            mid45, node[5], node[6], mid67, attr,
                            -1, fa[1], fa[2], fa[3], -1, fa[5]);

      child[3] = NewElement(mid04, midf1, midf3, mid37,
                            node[4], mid45, mid67, node[7], attr,
                            -1, fa[1], -1, fa[3], fa[4], fa[5]);
   }
   else if (ref_type == 6) // YZ split
   {
      // TODO
   }
   else if (ref_type == 7) // full isotropic refinement
   {
      Node* mid01 = GetMidEdgeVertex(node[0], node[1]);
      Node* mid12 = GetMidEdgeVertex(node[1], node[2]);
      Node* mid23 = GetMidEdgeVertex(node[2], node[3]);
      Node* mid30 = GetMidEdgeVertex(node[3], node[0]);

      Node* mid45 = GetMidEdgeVertex(node[4], node[5]);
      Node* mid56 = GetMidEdgeVertex(node[5], node[6]);
      Node* mid67 = GetMidEdgeVertex(node[6], node[7]);
      Node* mid74 = GetMidEdgeVertex(node[7], node[4]);

      Node* mid04 = GetMidEdgeVertex(node[0], node[4]);
      Node* mid15 = GetMidEdgeVertex(node[1], node[5]);
      Node* mid26 = GetMidEdgeVertex(node[2], node[6]);
      Node* mid37 = GetMidEdgeVertex(node[3], node[7]);

      Node* midf0 = GetMidFaceVertex(mid23, mid12, mid01, mid30);
      Node* midf1 = GetMidFaceVertex(mid01, mid15, mid45, mid04);
      Node* midf2 = GetMidFaceVertex(mid12, mid26, mid56, mid15);
      Node* midf3 = GetMidFaceVertex(mid23, mid37, mid67, mid26);
      Node* midf4 = GetMidFaceVertex(mid30, mid04, mid74, mid37);
      Node* midf5 = GetMidFaceVertex(mid45, mid56, mid67, mid74);

      Node* midel = GetMidEdgeVertex(midf1, midf3);

      child[0] = NewElement(node[0], mid01, midf0, mid30,
                            mid04, midf1, midel, midf4, attr,
                            fa[0], fa[1], -1, -1, fa[4], -1);

      child[1] = NewElement(mid01, node[1], mid12, midf0,
                            midf1, mid15, midf2, midel, attr,
                            fa[0], fa[1], fa[2], -1, -1, -1);

      child[2] = NewElement(midf0, mid12, node[2], mid23,
                            midel, midf2, mid26, midf3, attr,
                            fa[0], -1, fa[2], fa[3], -1, -1);

      child[3] = NewElement(mid30, midf0, mid23, node[3],
                            midf4, midel, midf3, mid37, attr,
                            fa[0], -1, -1, fa[3], fa[4], -1);

      child[4] = NewElement(mid04, midf1, midel, midf4,
                            node[4], mid45, midf5, mid74, attr,
                            -1, fa[1], -1, -1, fa[4], fa[5]);

      child[5] = NewElement(midf1, mid15, midf2, midel,
                            mid45, node[5], mid56, midf5, attr,
                            -1, fa[1], fa[2], -1, -1, fa[5]);

      child[6] = NewElement(midel, midf2, mid26, midf3,
                            midf5, mid56, node[6], mid67, attr,
                            -1, -1, fa[2], fa[3], -1, fa[5]);

      child[7] = NewElement(midf4, midel, midf3, mid37,
                            mid74, midf5, mid67, node[7], attr,
                            -1, -1, -1, fa[3], fa[4], fa[5]);
   }

   // start using the nodes of the children, create edges & faces
   for (int i = 0; i < 8; i++)
      if (child[i])
      {
         RefVertices(child[i]);
         RefEdgesFaces(child[i]);
      }

   // sign off of the edges & faces of the parent, but retain the corners
   UnrefEdgesFaces(elem);

   elem->ref_type = ref_type;
}


void NCMeshHex::Derefine(Element* elem)
{
/*   // check that all children are leafs   TODO: maybe derefine recursively?
   for (int i = 0; i < 8; i++)
      if (elem->child[i] && elem->child[i]->ref_type)
         mfem_error("NCMeshHex::Derefine: can't derefine element whose child "
                    "is refined.");

   for (int i = 0; i < 8; i++)
   {
      if (elem->child[i])
      {
         UnrefEdgesFaces(elem->child[i]);
         UnrefVertices(elem->child[i]);
         delete elem->child[i];
         elem->child[i] = NULL;
      }
   }

   RefEdgesFaces(elem);

   elem->ref_type = 0;

   // FIXME: restore boundary attributes!*/
}


//// Mesh Interface ////////////////////////////////////////////////////////////

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
      leaf_elements.Append(e);

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

   elements.SetSize(0);
   boundary.SetSize(0);
   leaf_elements.SetSize(0);

   for (int i = 0; i < root_elements.Size(); i++)
      GetLeafElements(root_elements[i], elements, boundary);
}


//// Interpolation /////////////////////////////////////////////////////////////

static void make_point_mat(double v0[], double v1[], double v2[], double v3[],
                           DenseMatrix &pm)
{
   for (int i = 0; i < 2; i++)
   {
      pm(i,0) = v0[i];
      pm(i,1) = v1[i];
      pm(i,2) = v2[i];
      pm(i,3) = v3[i];
   }
}

void NCMeshHex::ConstrainFace(Node* v0, Node* v1, Node* v2, Node* v3,
                              IsoparametricTransformation& face_T,
                              MasterFace* master, int level)
{
   if (level > 0)
   {
      // check if we made it to a face that is not split further
      Face* face = faces.Peek(v0, v1, v2, v3);
      if (face)
      {
         // yes, we need to make this face constrained
         DenseMatrix I(master->face_fe->GetDof());
         master->face_fe->GetLocalInterpolation(face_T, I);

         Node* v[4] = { v0, v1, v2, v3 };
         for (int i = 0; i < 4; i++)
         {
            // make v[i] dependent on all master face DOFs
            VertexData& vd = v_data[v[i]->vertex->index];
            if (vd.Independent())
            {
               for (int j = 0; j < 4; j++)
               {
                  double coef = I(i, j);
                  if (fabs(coef) > 1e-12)
                  {
                     int dof = v_data[master->v[j]->vertex->index].dof;
                     if (dof != vd.dof)
                        vd.dep_list.Append(Dependency(dof, coef));
                  }
               }
            }
         }

         return;
      }
   }

   // we need to recurse deeper, now determine how
   Node* mid[5];
   int split = FaceSplitType(v0, v1, v2, v3, mid);

   // prepare also the middle points for the transformation
   DenseMatrix& pm = face_T.GetPointMat();
   double tmid[5][2] =
   {
      { (pm(0,0) + pm(0,1)) / 2,  (pm(1,0) + pm(1,1)) / 2 }, // bottom (0)
      { (pm(0,1) + pm(0,2)) / 2,  (pm(1,1) + pm(1,2)) / 2 }, // right (1)
      { (pm(0,2) + pm(0,3)) / 2,  (pm(1,2) + pm(1,3)) / 2 }, // top (2)
      { (pm(0,3) + pm(0,0)) / 2,  (pm(1,3) + pm(1,0)) / 2 }, // left (3)
      { (pm(0,0) + pm(0,1) + pm(0,2) + pm(0,3)) / 4,
        (pm(1,0) + pm(1,1) + pm(1,2) + pm(1,3)) / 4 }  // middle (4)
   };
   double tv[4][2] = // backup of original points
   {
      { pm(0,0), pm(1,0) },
      { pm(0,1), pm(1,1) },
      { pm(0,2), pm(1,2) },
      { pm(0,3), pm(1,3) },
   };

   if (split == 1) // "X" split face
   {
      make_point_mat(tv[0], tmid[0], tmid[2], tv[3], pm);
      ConstrainFace (  v0,   mid[0],  mid[2],   v3,  face_T, master, level+1);

      make_point_mat(tmid[0], tv[1], tv[2], tmid[2], pm);
      ConstrainFace ( mid[0],   v1,    v2,   mid[2], face_T, master, level+1);
   }
   else if (split == 2) // "Y" split face
   {
      make_point_mat(tv[0], tv[1], tmid[1], tmid[3], pm);
      ConstrainFace (  v0,    v1,   mid[1],  mid[3], face_T, master, level+1);

      make_point_mat(tmid[3], tmid[1], tv[2], tv[3], pm);
      ConstrainFace ( mid[3],  mid[1],   v2,    v3,  face_T, master, level+1);
   }
   else if (split == 3) // 4-way split face
   {
      make_point_mat(tv[0], tmid[0], tmid[4], tmid[3], pm);
      ConstrainFace (  v0,   mid[0],  mid[4],  mid[3], face_T, master, level+1);

      make_point_mat(tmid[0], tv[1], tmid[1], tmid[4], pm);
      ConstrainFace ( mid[0],   v1,   mid[1],  mid[4], face_T, master, level+1);

      make_point_mat(tmid[4], tmid[1], tv[2], tmid[2], pm);
      ConstrainFace ( mid[4],  mid[1],   v2,   mid[2], face_T, master, level+1);

      make_point_mat(tmid[3], tmid[4], tmid[2], tv[3], pm);
      ConstrainFace ( mid[3],  mid[4],  mid[2],   v3,  face_T, master, level+1);
   }
}


void NCMeshHex::ProcessMasterFace(Node* node[4], Face* face,
                                  const FiniteElementCollection *fec)
{
   // set up a face transformation that will keep track of our position
   // within the master face
   IsoparametricTransformation face_T;
   face_T.SetFE(&QuadrilateralFE);

   // initial transformation is identity (vertices of the unit square)
   DenseMatrix& pm = face_T.GetPointMat();
   pm.SetSize(2, 4);
   pm(0, 0) = 0;  pm(0, 1) = 1;  pm(0, 2) = 1;  pm(0, 3) = 0;
   pm(1, 0) = 0;  pm(1, 1) = 0;  pm(1, 2) = 1;  pm(1, 3) = 1;

   // package all constraining nodes in one struct
   MasterFace master;
   for (int i = 0; i < 4; i++)
   {
      master.v[i] = node[i];
      //master.e[j] = ...
   }
   master.face = face;
   master.face_fe = fec->FiniteElementForGeometry(Geometry::SQUARE);

   ConstrainFace(node[0], node[1], node[2], node[3], face_T, &master, 0);
}


bool NCMeshHex::VertexFinalizable(VertexData& vd)
{
   // are all constraining DOFs finalized?
   for (int i = 0; i < vd.dep_list.Size(); i++)
      if (!v_data[vd.dep_list[i].dof /*FIXME*/].finalized)
         return false;

   return true;
}


SparseMatrix*
   NCMeshHex::GetInterpolation(Mesh*, FiniteElementSpace *fes)
{
   int num_vert = IndexVertices();
   //int num_edges = IndexEdges();
   //int num_faces = IndexFaces();

   v_data = new VertexData[num_vert];
   //e_data = new InterpolationData[num_edges];
   //f_data = new InterpolationData[num_faces];

   // get nonconforming DOF numbers
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
   {
      if (it->vertex)
      {
         int index = it->vertex->index;
         v_data[index].dof = index;
      }
   }

   // visit faces of leaf elements
   for (int i = 0; i < leaf_elements.Size(); i++)
   {
      Element* elem = leaf_elements[i];
      MFEM_ASSERT(!elem->ref_type, "Not a leaf element.");

      for (int i = 0; i < 6; i++)
      {
         Node* node[4];
         const int* fv = hex_faces[i];
         for (int j = 0; j < 4; j++)
            node[j] = elem->node[fv[j]];

         Face* face = faces.Peek(node[0], node[1], node[2], node[3]);
         MFEM_ASSERT(face, "Face not found!");

         if (face->ref_count == 1 && !face->Boundary())
         {
            // we found a face that is complete on one side and refined on the
            // other; we need to start a recursive process that will make all
            // DOFs lying on the other side dependent on this master face
            ProcessMasterFace(node, face, fes->FEColl());
         }
      }
   }

   // assign true DOFs to vertices that stayed independent
   int next_true_dof = 0;
   for (int i = 0; i < num_vert; i++)
   {
      VertexData& vd = v_data[i];
      if (vd.Independent())
         vd.true_dof = next_true_dof++;
   }

   // create the conforming prolongation matrix
   SparseMatrix* cP = new SparseMatrix(fes->GetNDofs(), next_true_dof);

   // put identity in the matrix for independent vertices
   for (int i = 0; i < num_vert; i++)
   {
      VertexData& vd = v_data[i];
      if (vd.Independent())
      {
         cP->Add(vd.dof, vd.true_dof, 1.0);
         vd.finalized = true;
      }
   }

   // resolve dependencies
   bool finished;
   do
   {
      finished = true;
      for (int i = 0; i < num_vert; i++)
      {
         VertexData& vd = v_data[i];
         if (!vd.finalized && VertexFinalizable(vd))
         {
            for (int j = 0; j < vd.dep_list.Size(); j++)
            {
               Dependency& dep = vd.dep_list[j];

               Array<int> cols;
               Vector srow;
               cP->GetRow(dep.dof, cols, srow);

               for (int k = 0; k < cols.Size(); k++)
                  cP->Add(vd.dof, cols[k], dep.coef * srow[k]);
            }
            vd.finalized = true;
            finished = false;
         }
      }
   }
   while (!finished);

   delete [] v_data;

   cP->Finalize();
   return cP;
}

