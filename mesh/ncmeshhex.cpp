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

   // start using the nodes of the children, create edges & faces
   RefVertices(child0); RefEdgesFaces(child0);
   RefVertices(child1); RefEdgesFaces(child1);

   // sign off of the edges & faces of the parent, but retain the corners
   UnrefEdgesFaces(elem);

   // mark the original element as refined
   elem->ref_type = ref_type;
   elem->child[0] = child0;
   elem->child[1] = child1;

   // keep track of the number of leaf elements
   num_leaf_elements += 1;
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
         num_leaf_elements--;
      }
   }

   RefEdgesFaces(elem);

   elem->ref_type = 0;
   num_leaf_elements++;

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
                              IsoparametricTransformation& face_T, int level)
{
   if (level > 0)
   {
      // check if we made it to a face that is not split further
      Face* face = faces.Peek(v0, v1, v2, v3);
      if (face)
      {
         // yes, we need to make everything on this face constrained


         return;
      }
   }

   // we need to recurse deeper, now determine how
   Node* mid[5];
   int split = FaceSplitType(v0, v1, v2, v3, mid);

   // prepare also the middle points for the transformation
   DenseMatrix& pm = face_T.GetPointMat();
   double t_mid[5][2] =
   {
      { pm(0,0) + pm(0,1),  pm(1,0) + pm(1,1) }, // bottom (0)
      { pm(0,1) + pm(0,2),  pm(1,1) + pm(1,2) }, // right (1)
      { pm(0,2) + pm(0,3),  pm(1,2) + pm(1,3) }, // top (2)
      { pm(0,3) + pm(0,0),  pm(1,3) + pm(1,0) }, // left (3)
      { pm(0,0) + pm(0,1) + pm(0,2) + pm(0,3),
        pm(1,0) + pm(1,1) + pm(1,2) + pm(1,3) }  // middle (4)
   };
   double t_v[4][2] = // backup of original points
   {
      { pm(0,0), pm(1,0) },
      { pm(0,1), pm(1,1) },
      { pm(0,2), pm(1,2) },
      { pm(0,3), pm(1,3) },
   };

   if (split == 1) // "X" split face
   {
      make_point_mat(t_v[0], t_mid[0], t_mid[2], t_v[3], pm);
      ConstrainFace (   v0,    mid[0],   mid[2],    v3,  face_T, level+1);

      make_point_mat(t_mid[0], t_v[1], t_v[2], t_mid[2], pm);
      ConstrainFace (  mid[0],    v1,     v2,    mid[2], face_T, level+1);
   }
   else if (split == 2) // "Y" split face
   {
      make_point_mat(t_v[0], t_v[1], t_mid[1], t_mid[3], pm);
      ConstrainFace (   v0,     v1,    mid[1],   mid[3], face_T, level+1);

      make_point_mat(t_mid[3], t_mid[1], t_v[2], t_v[3], pm);
      ConstrainFace (  mid[3],   mid[1],    v2,     v3,  face_T, level+1);
   }
   else if (split == 3) // 4-way split face
   {
      make_point_mat(t_v[0], t_mid[0], t_mid[4], t_mid[3], pm);
      ConstrainFace (   v0,    mid[0],   mid[4],   mid[3], face_T, level+1);

      make_point_mat(t_mid[0], t_v[1], t_mid[1], t_mid[4], pm);
      ConstrainFace (  mid[0],    v1,    mid[1],   mid[4], face_T, level+1);

      make_point_mat(t_mid[4], t_mid[1], t_v[2], t_mid[2], pm);
      ConstrainFace (  mid[4],   mid[1],    v2,    mid[2], face_T, level+1);

      make_point_mat(t_mid[3], t_mid[4], t_mid[2], t_v[3], pm);
      ConstrainFace (  mid[3],   mid[4],   mid[2],    v3,  face_T, level+1);
   }

   MFEM_ASSERT(0, "Should never get here.");
}

void NCMeshHex::VisitFaces(Element* elem)
{
   // we're done once we hit a leaf
   if (!elem->ref_type) return;

   // refined element: check its faces for constraints coming from the outside
   for (int i = 0; i < 6; i++)
   {
      const int* fv = hex_faces[i];
      Node** node = elem->node;
      Face* face = faces.Peek(node[fv[0]], node[fv[1]], node[fv[2]], node[fv[3]]);

      if (face)
      {
         // there is a complete face on the other side; we need to spawn
         // a new recursive process that will constrain everything lying on
         // this master face on our side

         // set up a face transformation that will keep track of our position
         // within the complete face
         IsoparametricTransformation face_T;
         face_T.SetFE(&QuadrilateralFE);

         // the initial transformation is identity (vertices of the unit square)
         DenseMatrix& pm = face_T.GetPointMat();
         pm.SetSize(2, 4);
         pm(0, 0) = 0;  pm(0, 1) = 1;  pm(0, 2) = 1;  pm(0, 3) = 0;
         pm(1, 0) = 0;  pm(1, 1) = 0;  pm(1, 2) = 1;  pm(1, 3) = 1;

         ConstrainFace(node[fv[0]], node[fv[1]], node[fv[2]], node[fv[3]],
                       face_T, 0);
      }
   }

   // go further down
   for (int i = 0; i < 8; i++)
      if (elem->child[i])
         VisitFaces(elem->child[i]);
}

SparseMatrix*
   NCMeshHex::GetInterpolation(Mesh *f_mesh, FiniteElementSpace *f_fes)
{
   int num_vert = IndexVertices();
   //int num_edges = IndexEdges();
   //int num_faces = IndexFaces();

   v_data = new InterpolationData[num_vert];
   //e_data = new InterpolationData[num_edges];
   //f_data = new InterpolationData[num_faces];

   // assign true DOF numbers to vertices
   int true_vert = 0;
   for (HashTable<Node>::Iterator it(nodes); it; ++it)
   {
      if (!it->vertex) continue;

      int index = it->vertex->index;
      InterpolationData& vd = v_data[index];

      vd.dof = index; // nonconforming: same numbering as in mesh

      if (!it->edge) // independent vertex  FIXME!!!!! faces
         vd.true_dof = true_vert++;
      else  // dependent vertex (in the middle of some edge)
         vd.true_dof = -1;
   }

   // traverse hierarych top-down, find constraining faces
   for (int i = 0; i < root_elements.Size(); i++)
      VisitFaces(root_elements[i]);
}

