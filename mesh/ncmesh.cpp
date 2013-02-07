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

NCMesh::Edge::Edge(Edge *my_parent)
   : parent(my_parent)
{
   id = -1;
   child[0] = child[1] = NULL;
   face[0] = face[1] = NULL;
   vertex[0] = vertex[1] = NULL;
}

void NCMesh::Edge::Refine(Face *newly_refined_face, Face *child0, Face *child1,
                          Data_manager *man)
{
   if (!this->isRefined())
   {
      // allocate children
      Vertex *new_vertex;
      if (!man)
      {
         child[0] = new Edge(this);
         child[1] = new Edge(this);
         new_vertex = new Vertex;
      }
      else
      {
         child[0] = man->NewEdge(this);
         child[1] = man->NewEdge(this);
         new_vertex = man->NewEdgeVertex(this);
      }
      child[0]->vertex[0] = vertex[0];
      child[0]->vertex[1] = new_vertex;
      child[1]->vertex[0] = new_vertex;
      child[1]->vertex[1] = vertex[1];
      if (man)
      {
         // check if the edge is a boundary edge:
         // *** check if all ancestors are truly-refined, this could be
         // improved if 'truly_refined' flag is added to the Edge class.
         {
            bool truly_refined = isTrulyRefined(); // is the other face NULL?
            for (Edge *ancestor = parent; truly_refined && ancestor;
                 ancestor = ancestor->parent)
            {
               truly_refined = truly_refined && ancestor->isTrulyRefined();
            }
            if (truly_refined)
               man->TrulyRefineEdge(this, NULL);
         }
      }
   }
   else
   {
      if (man)
         man->TrulyRefineEdge(this, newly_refined_face);
   }

   // set the faces (corresponding to the 'newly_refined_face') in the children
   if (newly_refined_face == face[0])
   {
      child[0]->face[0] = child0;
      child[1]->face[0] = child1;
   }
   else
   {
#ifdef MFEM_DEBUG
      if (newly_refined_face != face[1])
         mfem_error("NCMesh::Edge::Refine");
#endif
      child[0]->face[1] = child1;
      child[1]->face[1] = child0;
   }
}

void NCMesh::Edge::Derefine(Face *derefined_face, Data_manager *man)
{
   int side = (derefined_face == face[0]) ? 0 : 1;
#ifdef MFEM_DEBUG
   if (derefined_face != face[side])
      mfem_error("NCMesh::Edge::Derefine");
#endif
   if (face[1-side] == NULL)
      FreeHierarchy(man, false);
   else
   {
      SimpleEdge_iterator edge(this);
      for (++edge; edge; ++edge)
      {
         if (edge->face[1-side] == NULL)
         {
            --edge;
            edge->FreeHierarchy(man, false);
         }
         else
            edge->face[side] = NULL;
      }
   }
   if (man)
      man->DerefineEdge(this, derefined_face);
}

void NCMesh::Edge::SetInteriorVertices()
{
   // recursive implementation
   if (this->isRefined())
   {
      child[0]->vertex[1]->Set(vertex[0], vertex[1]);
      child[0]->SetInteriorVertices();
      child[1]->SetInteriorVertices();
   }
}

int NCMesh::Edge::Check() const
{
   // check parent
   if (parent &&
       !(this == parent->child[0] || this == parent->child[1]))
      return 1;

   // check children
   if (this->isRefined())
   {
      if (!(child[0] && child[1] &&
            this == child[0]->parent && this == child[1]->parent))
         return 2;

      // check chilren's vertices
      if (!(vertex[0] == child[0]->vertex[0] &&
            vertex[1] == child[1]->vertex[1] &&
            child[0]->vertex[1] == child[1]->vertex[0]))
         return 3;
   }

   // check face[0]
   if (face[0] &&
       !(this == face[0]->edge[0] || this == face[0]->edge[1] ||
         this == face[0]->edge[2] || this == face[0]->edge[3]))
      return 4;

   // check face[1]
   if (face[1] &&
       !(this == face[1]->edge[0] || this == face[1]->edge[1] ||
         this == face[1]->edge[2] || this == face[1]->edge[3]))
      return 5;

   // check that at least one of face[0] and face[1] is defined
   if (!(face[0] || face[1]))
      return 6;

   return 0;
}

void NCMesh::Edge::FreeHierarchy(Data_manager *man, bool free_this)
{
   Edge *stop = free_this ? NULL : this;
   if (!man)
   {
      for (DependentVertex_iterator vi(this); vi; ++vi)
         delete vi;
      for (SimpleEdge_iterator ei(this, false); ei != stop; )
         delete ei--;
   }
   else
   {
      for (DependentVertex_iterator vi(this); vi; ++vi)
         man->FreeVertex(vi);
      for (SimpleEdge_iterator ei(this, false); ei != stop; )
         man->FreeEdge(ei--);
   }
   if (!free_this)
      child[0] = NULL;
}

NCMesh::Face::Face(Face *my_parent)
   : parent(my_parent)
{
   id = -1;
   child[0] = child[1] = child[2] = child[3] = NULL;
   edge[0] = edge[1] = edge[2] = edge[3] = NULL;
}

void NCMesh::Face::Refine(Data_manager *man)
{
   if (this->isRefined())
      return;

   // allocate children
   if (!man)
   {
      child[0] = new Face(this);
      child[1] = new Face(this);
      child[2] = new Face(this);
      child[3] = new Face(this);
   }
   else
   {
      child[0] = man->NewFace(this);
      child[1] = man->NewFace(this);
      child[2] = man->NewFace(this);
      child[3] = man->NewFace(this);
   }

   // refine the edges (if not refined already) and set their
   // childrens's faces corresponding to 'this' face
   if (!this->isQuad())
   {
      edge[0]->Refine(this, child[0], child[1], man);
      edge[1]->Refine(this, child[1], child[2], man);
      edge[2]->Refine(this, child[2], child[0], man);
   }
   else
   {
      edge[0]->Refine(this, child[0], child[1], man);
      edge[1]->Refine(this, child[1], child[2], man);
      edge[2]->Refine(this, child[2], child[3], man);
      edge[3]->Refine(this, child[3], child[0], man);
   }

   // allocate the new edges
   Edge *new_edge[4];
   if (!man)
   {
      new_edge[0] = new Edge(NULL);
      new_edge[1] = new Edge(NULL);
      new_edge[2] = new Edge(NULL);
   }
   else
   {
      new_edge[0] = man->NewEdge(NULL);
      new_edge[1] = man->NewEdge(NULL);
      new_edge[2] = man->NewEdge(NULL);
   }

   // set children edges
   if (!this->isQuad())
   {
      if (this == edge[0]->face[0])
      {
         child[0]->edge[1] = edge[0]->child[0];
         child[1]->edge[0] = edge[0]->child[1];
      }
      else
      {
         child[0]->edge[1] = edge[0]->child[1];
         child[1]->edge[0] = edge[0]->child[0];
      }

      if (this == edge[1]->face[0])
      {
         child[1]->edge[1] = edge[1]->child[0];
         child[2]->edge[0] = edge[1]->child[1];
      }
      else
      {
         child[1]->edge[1] = edge[1]->child[1];
         child[2]->edge[0] = edge[1]->child[0];
      }

      if (this == edge[2]->face[0])
      {
         child[0]->edge[0] = edge[2]->child[1];
         child[2]->edge[1] = edge[2]->child[0];
      }
      else
      {
         child[0]->edge[0] = edge[2]->child[0];
         child[2]->edge[1] = edge[2]->child[1];
      }

      child[0]->edge[2] = new_edge[1];
      child[1]->edge[2] = new_edge[2];
      child[2]->edge[2] = new_edge[0];

      child[3]->edge[0] = new_edge[0];
      child[3]->edge[1] = new_edge[1];
      child[3]->edge[2] = new_edge[2];

      // set the adjacent faces of the new edges
      new_edge[0]->face[0] = child[3];
      new_edge[0]->face[1] = child[2];

      new_edge[1]->face[0] = child[3];
      new_edge[1]->face[1] = child[0];

      new_edge[2]->face[0] = child[3];
      new_edge[2]->face[1] = child[1];

      // set the vertices of the new edges
      new_edge[0]->vertex[0] = edge[1]->child[0]->vertex[1];
      new_edge[0]->vertex[1] = edge[2]->child[0]->vertex[1];

      new_edge[1]->vertex[0] = edge[2]->child[0]->vertex[1];
      new_edge[1]->vertex[1] = edge[0]->child[0]->vertex[1];

      new_edge[2]->vertex[0] = edge[0]->child[0]->vertex[1];
      new_edge[2]->vertex[1] = edge[1]->child[0]->vertex[1];
   }
   else // the face is a quad
   {
      Vertex *new_vertex;
      if (!man)
      {
         new_edge[3] = new Edge(NULL);
         new_vertex = new Vertex;
      }
      else
      {
         new_edge[3] = man->NewEdge(NULL);
         new_vertex = man->NewQuadVertex(this);
      }

      if (this == edge[0]->face[0])
      {
         child[0]->edge[0] = edge[0]->child[0];
         child[1]->edge[0] = edge[0]->child[1];
      }
      else
      {
         child[0]->edge[0] = edge[0]->child[1];
         child[1]->edge[0] = edge[0]->child[0];
      }
      if (this == edge[3]->face[0])
      {
         child[0]->edge[3] = edge[3]->child[1];
         child[3]->edge[3] = edge[3]->child[0];
      }
      else
      {
         child[0]->edge[3] = edge[3]->child[0];
         child[3]->edge[3] = edge[3]->child[1];
      }
      if (this == edge[1]->face[0])
      {
         child[1]->edge[1] = edge[1]->child[0];
         child[2]->edge[1] = edge[1]->child[1];
      }
      else
      {
         child[1]->edge[1] = edge[1]->child[1];
         child[2]->edge[1] = edge[1]->child[0];
      }
      if (this == edge[2]->face[0])
      {
         child[2]->edge[2] = edge[2]->child[0];
         child[3]->edge[2] = edge[2]->child[1];
      }
      else
      {
         child[2]->edge[2] = edge[2]->child[1];
         child[3]->edge[2] = edge[2]->child[0];
      }

      child[0]->edge[1] = new_edge[0];
      child[0]->edge[2] = new_edge[3];

      child[1]->edge[2] = new_edge[1];
      child[1]->edge[3] = new_edge[0];

      child[2]->edge[0] = new_edge[1];
      child[2]->edge[3] = new_edge[2];

      child[3]->edge[0] = new_edge[3];
      child[3]->edge[1] = new_edge[2];

      // set the adjacent faces of the new edges
      new_edge[0]->face[0] = child[0];
      new_edge[0]->face[1] = child[1];

      new_edge[1]->face[0] = child[2];
      new_edge[1]->face[1] = child[1];

      new_edge[2]->face[0] = child[3];
      new_edge[2]->face[1] = child[2];

      new_edge[3]->face[0] = child[3];
      new_edge[3]->face[1] = child[0];

      // set the vertices of the new edges
      new_edge[0]->vertex[0] = edge[0]->child[0]->vertex[1];
      new_edge[0]->vertex[1] = new_vertex;

      new_edge[1]->vertex[0] = new_vertex;
      new_edge[1]->vertex[1] = edge[1]->child[0]->vertex[1];

      new_edge[2]->vertex[0] = new_vertex;
      new_edge[2]->vertex[1] = edge[2]->child[0]->vertex[1];

      new_edge[3]->vertex[0] = edge[3]->child[0]->vertex[1];
      new_edge[3]->vertex[1] = new_vertex;
   }

   if (man)
      man->RefineFace(this);
}

void NCMesh::Face::Derefine(Data_manager *man)
{
   if (!this->isRefined())
      return;

   FreeHierarchy(man, false);
   if (this->isQuad())
      edge[3]->Derefine(this, man);
   edge[2]->Derefine(this, man);
   edge[1]->Derefine(this, man);
   edge[0]->Derefine(this, man);
   if (man)
      man->DerefineFace(this);
}

int NCMesh::Face::GetVertices(Vertex **face_vert)
{
   if (this == edge[0]->face[0])
   {
      face_vert[0] = edge[0]->vertex[0];
      face_vert[1] = edge[0]->vertex[1];
   }
   else
   {
      face_vert[0] = edge[0]->vertex[1];
      face_vert[1] = edge[0]->vertex[0];
   }
   if (!this->isQuad())
   {
      if (this == edge[1]->face[0])
         face_vert[2] = edge[1]->vertex[1];
      else
         face_vert[2] = edge[1]->vertex[0];
      return 3;
   }
   else
   {
      if (this == edge[2]->face[0])
      {
         face_vert[2] = edge[2]->vertex[0];
         face_vert[3] = edge[2]->vertex[1];
      }
      else
      {
         face_vert[2] = edge[2]->vertex[1];
         face_vert[3] = edge[2]->vertex[0];
      }
      return 4;
   }
}

void NCMesh::Face::SetReferenceVertices()
{
   Vertex *fv[4];

   GetVertices(fv);
   fv[0]->Set(0., 0.);
   fv[1]->Set(1., 0.);
   if (!this->isQuad())
   {
      fv[2]->Set(0., 1.);
      edge[0]->SetInteriorVertices();
      edge[1]->SetInteriorVertices();
      edge[2]->SetInteriorVertices();
   }
   else
   {
      fv[2]->Set(1., 1.);
      fv[3]->Set(0., 1.);
      edge[0]->SetInteriorVertices();
      edge[1]->SetInteriorVertices();
      edge[2]->SetInteriorVertices();
      edge[3]->SetInteriorVertices();
   }
}

void NCMesh::Face::SetInteriorVertices()
{
   // recursive implementation
   if (this->isRefined())
   {
      if (!this->isQuad())
      {
         child[3]->edge[0]->SetInteriorVertices();
         child[3]->edge[1]->SetInteriorVertices();
         child[3]->edge[2]->SetInteriorVertices();
      }
      else
      {
         child[0]->edge[1]->vertex[1]->Set(
            edge[0]->child[0]->vertex[1],
            edge[2]->child[0]->vertex[1]);
         child[0]->edge[1]->SetInteriorVertices();
         child[1]->edge[2]->SetInteriorVertices();
         child[2]->edge[3]->SetInteriorVertices();
         child[3]->edge[0]->SetInteriorVertices();
      }
      child[0]->SetInteriorVertices();
      child[1]->SetInteriorVertices();
      child[2]->SetInteriorVertices();
      child[3]->SetInteriorVertices();
   }
}

int NCMesh::Face::Check() const
{
   // check parent
   if (parent &&
       !(this == parent->child[0] || this == parent->child[1] ||
         this == parent->child[2] || this == parent->child[3]))
      return 1;

   // check children
   if (this->isRefined() &&
       !(child[0] && child[1] && child[2] && child[3] &&
         this == child[0]->parent && this == child[1]->parent &&
         this == child[2]->parent && this == child[3]->parent))
      return 2;

   // check edge[0]
   if (!edge[0] ||
       !(this == edge[0]->face[0] || this == edge[0]->face[1]))
      return 3;

   // check edge[1]
   if (!edge[1] ||
       !(this == edge[1]->face[0] || this == edge[1]->face[1]))
      return 4;

   // check edge[2]
   if (!edge[2] ||
       !(this == edge[2]->face[0] || this == edge[2]->face[1]))
      return 5;

   // check edge[3]
   if (edge[3] &&
       !(this == edge[3]->face[0] || this == edge[3]->face[1]))
      return 6;

   // check edges' vertices
   Vertex *p[8];
   for (int i = 0, n = this->isQuad() ? 4 : 3; i < n; i++)
      if (this == edge[i]->face[0])
      {
         p[2*i+0] = edge[i]->vertex[0];
         p[2*i+1] = edge[i]->vertex[1];
      }
      else
      {
         p[2*i+0] = edge[i]->vertex[1];
         p[2*i+1] = edge[i]->vertex[0];
      }
   if (!this->isQuad())
   {
      if (!(p[1] == p[2] && p[3] == p[4] && p[5] == p[0]))
         return 7;
   }
   else
   {
      if (!(p[1] == p[2] && p[3] == p[4] && p[5] == p[6] && p[7] == p[0]))
         return 7;
   }

   return 0;
}

void NCMesh::Face::FreeHierarchy(Data_manager *man, bool free_this)
{
   Face *stop = free_this ? NULL : this;
   SimpleFace_iterator fi(this);
   for ( ; fi; ++fi)
      if (fi->isRefined())
      {
         // delete the interior edges
         if (!fi->isQuad())
         {
            fi->child[3]->edge[2]->FreeHierarchy(man);
            fi->child[3]->edge[1]->FreeHierarchy(man);
            fi->child[3]->edge[0]->FreeHierarchy(man);
         }
         else
         {
            // delete the interior vertex
            if (!man)
               delete fi->child[0]->edge[1]->vertex[1];
            else
               man->FreeVertex(fi->child[0]->edge[1]->vertex[1]);

            fi->child[3]->edge[0]->FreeHierarchy(man);
            fi->child[2]->edge[3]->FreeHierarchy(man);
            fi->child[1]->edge[2]->FreeHierarchy(man);
            fi->child[0]->edge[1]->FreeHierarchy(man);
         }
      }
   if (!man)
   {
      for (fi.end(); fi != stop; )
         delete fi--;
   }
   else
   {
      for (fi.end(); fi != stop; )
         man->FreeFace(fi--);
   }
   if (!free_this)
      child[0] = NULL;
}

void NCMesh::SimpleFace_iterator::next()
{
   if (face->isRefined())
   {
      face = face->child[0];
      return;
   }
   for (Face *parent; face != root; face = parent)
   {
      parent = face->parent;
      if (face == parent->child[0])
         face = parent->child[1];
      else if (face == parent->child[1])
         face = parent->child[2];
      else if (face == parent->child[2])
         face = parent->child[3];
      else
      {
#ifdef MFEM_DEBUG
         if (face != parent->child[3])
            mfem_error("NCMesh::SimpleFace_iterator::next()");
#endif
         continue;
      }
      return;
   }
   face = NULL;
}

void NCMesh::SimpleFace_iterator::prev()
{
   if (face != root)
   {
      Face *parent = face->parent;
      if (face == parent->child[3])
         face = parent->child[2];
      else if (face == parent->child[2])
         face = parent->child[1];
      else if (face == parent->child[1])
         face = parent->child[0];
      else
      {
#ifdef MFEM_DEBUG
         if (face != parent->child[0])
            mfem_error("NCMesh::SimpleFace_revrse_iterator::prev()");
#endif
         face = parent;
         return;
      }
      find_leaf();
      return;
   }
   face = NULL;
}

NCMesh::AllFace_iterator::AllFace_iterator(const NonconformingMesh &ncmesh,
                                           int c_face_start)
   : c_faces(ncmesh.c_faces)
{
   c_face_idx = c_face_start;
   if (c_face_idx < c_faces.Size())
      face = c_faces[c_face_idx];
   else
      face = NULL;
}

void NCMesh::AllFace_iterator::next_up()
{
   for (Face *parent; (parent = face->parent); face = parent)
   {
      if (face == parent->child[0])
         face = parent->child[1];
      else if (face == parent->child[1])
         face = parent->child[2];
      else if (face == parent->child[2])
         face = parent->child[3];
      else
      {
#ifdef MFEM_DEBUG
         if (face != parent->child[3])
            mfem_error("NCMesh::AllFace_iterator::next()");
#endif
         continue;
      }
      return;
   }
   next_coarse();
}

NCMesh::RefinedFace_iterator::RefinedFace_iterator(
   const NonconformingMesh &ncmesh) : AllFace_iterator(ncmesh)
{
   for ( ; face; next_coarse())
      if (face->isRefined())
         return;
}

void NCMesh::RefinedFace_iterator::next()
{
#if 0
   for (AllFace_iterator::next(); face; AllFace_iterator::next())
      if (face->isRefined())
         return;
#else
   // 'face' is a refined face, try its children
   for (int j = 0; j < 4; j++)
      if (face->child[j]->isRefined())
      {
         face = face->child[j];
         return;
      }

   // no refined children; try going up the tree
   for (Face *parent; (parent = face->parent); face = parent)
   {
      // try next sibling
      int j = 0;
      while (face != parent->child[j])
         j++;
      for (j++; j < 4; j++)
      {
         face = parent->child[j];
         if (face->isRefined())
            return;
      }
   }

   // reached the root of the current coarse face tree;
   // find the next coarse face that is refined
   for (c_face_idx++; c_face_idx < c_faces.Size(); c_face_idx++)
   {
      face = c_faces[c_face_idx];
      if (face->isRefined())
         return;
   }
#endif

   // no other refined faces
   face = NULL;
}

void NCMesh::SimpleEdge_iterator::next_up()
{
   for (Edge *parent; edge != root; edge = parent)
   {
      parent = edge->parent;
      if (edge == parent->child[0])
      {
         edge = parent->child[1];
         return;
      }
   }
   edge = NULL;
}

void NCMesh::SimpleEdge_iterator::next()
{
   if (edge->isRefined())
      edge = edge->child[0];
   else
      next_up();
}

void NCMesh::SimpleEdge_iterator::prev()
{
   if (edge != root)
   {
      Edge *parent = edge->parent;
      if (edge == parent->child[1])
      {
         edge = parent->child[0];
         find_leaf();
      }
      else
      {
#ifdef MFEM_DEBUG
         if (edge != parent->child[0])
            mfem_error("NCMesh::SimpleEdge_iterator::prev");
#endif
         edge = parent;
      }
   }
   else
      edge = NULL;
}

NCMesh::AllEdge_iterator::AllEdge_iterator(const NonconformingMesh &ncmesh,
                                           int c_edge_start)
   : face(ncmesh), c_edges(ncmesh.c_edges)
{
   c_edge_idx = c_edge_start;
   if (c_edge_idx < c_edges.Size())
      edge = c_edges[c_edge_idx];
   else
      edge = NULL; // no edges -> no faces either
}

NCMesh::Vertex *NCMesh::AllEdge_iterator::next_up()
{
   for (Edge *parent; (parent = edge->parent); edge = parent)
   {
      if (edge == parent->child[0])
      {
         edge = parent->child[1];
         return NULL;
      }
#ifdef MFEM_DEBUG
      if (edge != parent->child[1])
         mfem_error("NCMesh::AllEdge_iterator::next_up()");
#endif
   }

   // reached an edge without parent
   if (c_edge_idx >= 0) // still processing coarse edges
   {
      // try next coarse edge
      c_edge_idx++;
      if (c_edge_idx < c_edges.Size())
         edge = c_edges[c_edge_idx];
      else
      {
         // no more coarse edges: start with the first refined face
         c_edge_idx = -1;
         if (face)
            goto found_next_face;
         edge = NULL;
      }
      return NULL;
   }

   // 'edge' must be one of the new edges of 'face'
   if (!face->isQuad())
   {
      if (edge == face->child[3]->edge[0])
         edge = face->child[3]->edge[1];
      else if (edge == face->child[3]->edge[1])
         edge = face->child[3]->edge[2];
      else
      {
#ifdef MFEM_DEBUG
         if (edge != face->child[3]->edge[2])
            mfem_error("NCMesh::AllEdge_iterator::next_up()");
#endif
         goto find_next_face;
      }
   }
   else
   {
      if (edge == face->child[0]->edge[1])
         edge = face->child[1]->edge[2];
      else if (edge == face->child[1]->edge[2])
         edge = face->child[2]->edge[3];
      else if (edge == face->child[2]->edge[3])
         edge = face->child[3]->edge[0];
      else
      {
#ifdef MFEM_DEBUG
         if (edge != face->child[3]->edge[0])
            mfem_error("NCMesh::AllEdge_iterator::next_up()");
#endif
         goto find_next_face;
      }
   }
   return NULL;

find_next_face:
   // find the next refined face of all the faces;
   ++face;
   if (!face)
   {
      edge = NULL;
      return NULL;
   }

found_next_face:
   // take its first new_edge
   if (!face->isQuad())
      edge = face->child[3]->edge[0];
   else
   {
      edge = face->child[0]->edge[1];
      return edge->vertex[1];
   }
   return NULL;
}

NCMesh::TrulyRefinedEdge_iterator::TrulyRefinedEdge_iterator(
   const NonconformingMesh &ncmesh) : AllEdge_iterator(ncmesh)
{
   for (c_edge_idx = 0; c_edge_idx < c_edges.Size(); c_edge_idx++)
   {
      edge = c_edges[c_edge_idx];
      if (edge->isTrulyRefined())
         return;
   }
   edge = NULL;
}

NCMesh::Vertex *NCMesh::TrulyRefinedEdge_iterator::next_or_vertex()
{
   if (edge->isTrulyRefined())
   {
      if (edge->child[0]->isTrulyRefined())
      {
         edge = edge->child[0];
         return NULL;
      }
      else if (edge->child[1]->isTrulyRefined())
      {
         edge = edge->child[1];
         return NULL;
      }
   }
   while (true)
   {
      Vertex *v = next_up();
      if (v)
         return v;
      if (!edge || edge->isTrulyRefined())
         return NULL;
   }
}

void NCMesh::TrulyRefinedEdge_iterator::next()
{
   if (edge->child[0]->isTrulyRefined())
      edge = edge->child[0];
   else if (edge->child[1]->isTrulyRefined())
      edge = edge->child[1];
   else
      for (next_up(); edge; next_up())
         if (edge->isTrulyRefined())
            return;
}

void NCMesh::AllVertex_iterator::next()
{
   if (c_vert_idx >= 0)
   {
      c_vert_idx++;
      if (c_vert_idx < c_verts.Size())
      {
         vert = c_verts[c_vert_idx];
         return;
      }
      c_vert_idx = -1;
      for ( ; edge; ++edge)
         if (edge->isRefined())
         {
            vert = edge->child[0]->vertex[1];
            return;
         }
      vert = NULL;
      return;
   }
   if (c_vert_idx < -1)
   {
      c_vert_idx = -1;
      if (edge->isRefined())
      {
         vert = edge->child[0]->vertex[1];
         return;
      }
   }
   while (true)
   {
      if ((vert = edge.next_or_vertex()))
      {
         c_vert_idx = -2;
         return;
      }
      if (!edge)
         return; // vert is NULL
      if (edge->isRefined())
      {
         vert = edge->child[0]->vertex[1];
         return;
      }
   }
}

void NCMesh::AllVertex_iterator::parent_vertices(Vertex **pv)
{
   if (c_vert_idx >= 0) // coarse vertex
   {
      pv[0] = vert;
      pv[1] = NULL;
   }
   else if (c_vert_idx == -1) // middle of an edge
   {
      pv[0] = edge->vertex[0];
      pv[1] = edge->vertex[1];
   }
   else // middle of a quad
   {
      Face *quad = edge.face;
      pv[0] = quad->edge[0]->child[0]->vertex[1];
      pv[1] = quad->edge[2]->child[0]->vertex[1];
   }
}

void NCMesh::Vertex_iterator::next()
{
   if (c_vert_idx >= 0)
   {
      c_vert_idx++;
      if (c_vert_idx < c_verts.Size())
      {
         vert = c_verts[c_vert_idx];
         return;
      }
      c_vert_idx = -1;
      if (edge)
         vert = edge->child[0]->vertex[1];
      else
         vert = NULL;
      return;
   }
   if (c_vert_idx < -1)
   {
      c_vert_idx = -1;
      if (edge->isTrulyRefined())
      {
         vert = edge->child[0]->vertex[1];
         return;
      }
   }
   if ((vert = edge.next_or_vertex()))
   {
      c_vert_idx = -2;
      return;
   }
   // vert is NULL
   if (edge)
      vert = edge->child[0]->vertex[1];
}

int NCMesh::Data_manager::GetEdgeCoords(double a, int c_edge_idx, double *x)
{
   for (int i = 0; i < c_face_nv; i++)
      if (c_edge_idx == c_face_edges[i])
      {
         const IntegrationPoint &ip0 = ref_verts->IntPoint(i);
         const IntegrationPoint &ip1 = ref_verts->IntPoint((i+1)%c_face_nv);
         if (c_face != c_face->edge[i]->face[0])
         {
#ifdef MFEM_DEBUG
            if (c_face != c_face->edge[i]->face[1])
               return 2;
#endif
            a = 1.0 - a;
         }
         x[0] = (1.0 - a)*ip0.x + a*ip1.x;
         x[1] = (1.0 - a)*ip0.y + a*ip1.y;
         return 0;
      }
   return 1;
}

void NCMesh::Data_manager::Avg(const Vertex *v0, const Vertex *v1, Vertex *v)
{
   int type0, i0, type1, i1;
   double x0[2], x1[2];

   if (v0->x[0] == 0.0)  // v0 is a coarse vertex
      i0 = -((int)v0->x[1]), type0 = 0;
   else if (v0->x[1] <= 0.0) // v0 is interior to a coarse edge
      i0 = -((int)v0->x[1]), type0 = 1;
   else // v0 is interior to a coarse face
      type0 = 2;

   if (v1->x[0] == 0.0)  // v1 is a coarse vertex
      i1 = -((int)v1->x[1]), type1 = 0;
   else if (v1->x[1] <= 0.0) // v1 is interior to a coarse edge
      i1 = -((int)v1->x[1]), type1 = 1;
   else // v1 is interior to a coarse face
      type1 = 2;

   switch (3*type0 + type1)
   {
   case 0: // two vertices
      v->x[0] = 0.5;
      for (int i = 0; i < c_face_nv; i++)
      {
         Edge *edge = c_face->edge[i];
         if ((v0 == edge->vertex[0] && v1 == edge->vertex[1]) ||
             (v1 == edge->vertex[0] && v0 == edge->vertex[1]))
         {
            v->x[1] = -c_face_edges[i];
            return;
         }
      }
      break;

   case 1: // v0 is vertex, v1 is edge
      if (v0 == ncmesh->c_edges[i1]->vertex[0])
         v->x[0] = v1->x[0]/2;
      else
      {
#ifdef MFEM_DEBUG
         if (v0 != ncmesh->c_edges[i1]->vertex[1])
            break;
#endif
         v->x[0] = (v1->x[0] + 1.0)/2;
      }
      v->x[1] = v1->x[1];
      return;

   case 2: // v0 is vertex, v1 is interior
      // not possible
      break;

   case 3: // v0 is edge, v1 is vertex
      if (v1 == ncmesh->c_edges[i0]->vertex[0])
         v->x[0] = v0->x[0]/2;
      else
      {
#ifdef MFEM_DEBUG
         if (v1 != ncmesh->c_edges[i0]->vertex[1])
            break;
#endif
         v->x[0] = (v0->x[0] + 1.0)/2;
      }
      v->x[1] = v0->x[1];
      return;

   case 4: // v0 is edge, v1 is edge
      if (i0 == i1)
      {
         v->x[0] = (v0->x[0] + v1->x[0])/2;
         v->x[1] = v0->x[1];
      }
      else
      {
         if (GetEdgeCoords(v0->x[0], i0, x0) ||
             GetEdgeCoords(v1->x[0], i1, x1))
            break;
         v->x[0] = (x0[0] + x1[0])/2;
         v->x[1] = (x0[1] + x1[1])/2;
      }
      return;

   case 5: // v0 is edge, v1 is interior
      if (GetEdgeCoords(v0->x[0], i0, x0))
         break;
      v->x[0] = (x0[0] + v1->x[0])/2;
      v->x[1] = (x0[1] + v1->x[1])/2;
      return;

   case 6: // v0 is interior, v1 is vertex
      // not possible
      break;

   case 7: // v0 is interior, v1 is edge
      if (GetEdgeCoords(v1->x[0], i1, x1))
         break;
      v->x[0] = (x1[0] + v0->x[0])/2;
      v->x[1] = (x1[1] + v0->x[1])/2;
      return;

   case 8: // both v0 and v1 are interior
      v->x[0] = (v0->x[0] + v1->x[0])/2;
      v->x[1] = (v0->x[1] + v1->x[1])/2;
      return;
   }

   mfem_error("NCMesh::Data_manager::Avg");
}

const IntegrationPoint &NCMesh::Data_manager::GetCoarseVertexCoords(
   const Vertex *v)
{
   for (int j = 0; j < c_face_nv; j++)
      if (v == c_face_verts[j])
         return ref_verts->IntPoint(j);
   mfem_error("NCMesh::Data_manager::GetCoarseVertexCoords");
   return ref_verts->IntPoint(0);
}

void NCMesh::Data_manager::GetVertexCoords(const Vertex *v, double *y)
{
   if (v->x[0] == 0.0)  // v is a coarse vertex
   {
      const IntegrationPoint &ip = GetCoarseVertexCoords(v);
      y[0] = ip.x;
      y[1] = ip.y;
      return;
   }
   else if (v->x[1] <= 0.0) // v is interior to a coarse edge
   {
      if (GetEdgeCoords(v->x[0], -((int)v->x[1]), y) == 0)
         return;
   }
   else // v is interior to the current coarse face
   {
      y[0] = v->x[0];
      y[1] = v->x[1];
      return;
   }
   mfem_error("NCMesh::Data_manager::GetVertexCoords");
}

void NCMesh::Data_manager::SetCoarseFace(int _c_face_idx)
{
   if (c_face_idx == _c_face_idx)
      return;

   c_face_idx = _c_face_idx;
   c_face = ncmesh->c_faces[c_face_idx];
   c_face_nv = c_face->GetVertices(c_face_verts);
   c_face_edges = &ncmesh->c_face_edge[4*c_face_idx];
   if (!c_face->isQuad())
      ref_verts = Geometries.GetVertices(Geometry::TRIANGLE);
   else
      ref_verts = Geometries.GetVertices(Geometry::SQUARE);
}

void NCMesh::Data_manager::GetFaceToCoarseFacePointMatrix(
   Face *face, DenseMatrix &pm)
{
   Vertex *fv[4];
   int nv = face->GetVertices(fv);
   pm.SetSize(2, nv);
   for (int i = 0; i < nv; i++)
      GetVertexCoords(fv[i], &pm(0,i));
}

void NCMesh::Data_manager::GetFaceToCoarseFaceIPT(
   Face *face, IntegrationPointTransformation &ipT)
{
   GetFaceToCoarseFacePointMatrix(face, ipT.Transf.GetPointMat());
   if (!face->isQuad())
      ipT.Transf.SetFE(&TriangleFE);
   else
      ipT.Transf.SetFE(&QuadrilateralFE);
}

void NCMesh::Data_manager::AdjustEdgeData(Edge *edge, bool v0, bool v1)
{
   // recursive implentation
   if (v0)
   {
      if (v1)
      {
         if (edge->isTrulyRefined())
         {
            AdjustEdgeData(edge->child[0], true, false);
            AdjustEdgeData(edge->child[1], false, true);
         }
         else
            AdjustRealEdgeData(edge);
      }
      else
      {
         while (edge->isTrulyRefined())
            edge = edge->child[0];
         AdjustRealEdgeData(edge);
      }
   }
   else
   {
      if (v1)
      {
         while (edge->isTrulyRefined())
            edge = edge->child[1];
         AdjustRealEdgeData(edge);
      }
   }
}

void NCMesh::Data_manager::AdjustTriangleData(
   Face *face, bool e0, bool e1, bool e2)
{
   //recursive implementation
   if (!(e0 || e1 || e2))
      return;

   AdjustInteriorFaceData(face);

   if (!face->isRefined())
      return;

   Edge **new_edge = face->child[3]->edge;
   // adjust the new edges
   AdjustEdgeData(new_edge[0], e1, e2);
   AdjustEdgeData(new_edge[1], e2, e0);
   AdjustEdgeData(new_edge[2], e0, e1);

   // adjust the children
   AdjustTriangleData(face->child[0], e2, e0, e2 || e0);
   AdjustTriangleData(face->child[1], e0, e1, e0 || e1);
   AdjustTriangleData(face->child[2], e1, e2, e1 || e2);
   AdjustTriangleData(face->child[3], e1 || e2, e2 || e0, e0 || e1);
}

void NCMesh::Data_manager::AdjustQuadData(
   Face *face, bool e0, bool e1, bool e2, bool e3)
{
   // recursive implementation
   if (!(e0 || e1 || e2 || e3))
      return;

   AdjustInteriorFaceData(face);

   if (!face->isRefined())
      return;

   // adjust the new edges
   AdjustEdgeData(face->child[0]->edge[1], e0, false); // new_edge[0]
   AdjustEdgeData(face->child[1]->edge[2], false, e1); // new_edge[1]
   AdjustEdgeData(face->child[2]->edge[3], false, e2); // new_edge[2]
   AdjustEdgeData(face->child[3]->edge[0], e3, false); // new_edge[3]

   // adjust the children
   AdjustQuadData(face->child[0], e0, e0, e3, e3);
   AdjustQuadData(face->child[1], e0, e1, e1, e0);
   AdjustQuadData(face->child[2], e1, e1, e2, e2);
   AdjustQuadData(face->child[3], e3, e2, e2, e3);
}

void NCMesh::Data_manager::AdjustFaceData(Face *face, Edge *edge)
{
   if (!face->isQuad())
   {
      if (edge == face->edge[0])
         AdjustTriangleData(face, true, false, false);
      else if (edge == face->edge[1])
         AdjustTriangleData(face, false, true, false);
      else
         AdjustTriangleData(face, false, false, true);
   }
   else
   {
      if (edge == face->edge[0])
         AdjustQuadData(face, true, false, false, false);
      else if (edge == face->edge[1])
         AdjustQuadData(face, false, true, false, false);
      else if (edge == face->edge[2])
         AdjustQuadData(face, false, false, true, false);
      else
         AdjustQuadData(face, false, false, false, true);
   }
}

NCMesh::Vertex *NCMesh::Data_manager::NewCoarseVertex(int c_vert_idx)
{
   return new Vertex(c_vert_idx);
}

NCMesh::Vertex *NCMesh::Data_manager::NewEdgeVertex(Edge *edge)
{
   Vertex *vtx = new Vertex;
   Avg(edge, vtx);
   return vtx;
}

NCMesh::Vertex *NCMesh::Data_manager::NewQuadVertex(Face *quad)
{
   Vertex *vtx = new Vertex;
   Avg(quad, vtx);
   return vtx;
}

NCMesh::NonconformingMesh(int geom)
{
   bool tri = (geom == Geometry::TRIANGLE);

   data_manager = NULL;

   c_verts.SetSize(tri ? 3 : 4);
   c_edges.SetSize(tri ? 3 : 4);
   c_faces.SetSize(1);

   c_verts[0] = new Vertex;
   c_verts[1] = new Vertex;
   c_verts[2] = new Vertex;
   if (!tri)
      c_verts[3] = new Vertex;
   for (int i = 0; i < c_edges.Size(); i++)
   {
      c_edges[i] = new Edge(NULL);
      c_edges[i]->vertex[0] = c_verts[i];
      c_edges[i]->vertex[1] = c_verts[(i + 1)%c_verts.Size()];
   }
   c_faces[0] = new Face(NULL);

   Face *f = c_faces[0];
   f->edge[0] = c_edges[0];
   f->edge[1] = c_edges[1];
   f->edge[2] = c_edges[2];
   f->edge[0]->face[0] = f;
   f->edge[1]->face[0] = f;
   f->edge[2]->face[0] = f;
   if (!tri)
   {
      f->edge[3] = c_edges[3];
      f->edge[3]->face[0] = f;
   }
}

NCMesh::NonconformingMesh(const Mesh *mesh, Data_manager *man)
{
   if (mesh->Dimension() != 2)
      mfem_error("NonconformingMesh::NonconformingMesh : not a 2D mesh!");

   data_manager = man;
   if (man)
      data_manager->ncmesh = this;

   c_verts.SetSize(mesh->GetNV());
   c_edges.SetSize(mesh->GetNEdges());
   c_faces.SetSize(mesh->GetNE());

   if (man)
      c_face_edge.SetSize(4*c_faces.Size());

   if (man)
   {
      for (int i = 0; i < c_verts.Size(); i++)
         c_verts[i] = man->NewCoarseVertex(i);
      for (int i = 0; i < c_edges.Size(); i++)
         c_edges[i] = man->NewEdge(NULL);
      for (int i = 0; i < c_faces.Size(); i++)
         c_faces[i] = man->NewFace(NULL);
   }
   else
   {
      for (int i = 0; i < c_verts.Size(); i++)
         c_verts[i] = new Vertex;
      for (int i = 0; i < c_edges.Size(); i++)
         c_edges[i] = new Edge(NULL);
      for (int i = 0; i < c_faces.Size(); i++)
         c_faces[i] = new Face(NULL);
   }

   Array<int> el_edges, el_edge_or;

   for (int i = 0; i < c_faces.Size(); i++)
   {
      Face *face = c_faces[i];
      const Element *elem = mesh->GetElement(i);
      const int *v = elem->GetVertices();

      mesh->GetElementEdges(i, el_edges, el_edge_or);

      for (int j = 0; j < el_edges.Size(); j++)
      {
         int c_edge_idx = el_edges[j];
         Edge *edge = c_edges[c_edge_idx];
         const int *ev = elem->GetEdgeVertices(j);

         if (!edge->face[0])
         {
            edge->face[0] = face;
            edge->vertex[0] = c_verts[v[ev[0]]];
            edge->vertex[1] = c_verts[v[ev[1]]];
         }
         else if (!edge->face[1])
            edge->face[1] = face;
         else
            cout << "NonconformingMesh::NonconformingMesh : Coarse edge "
                 << el_edges[j]
                 << " already has two adjacent edges" << endl;

         face->edge[j] = edge;

         if (man)
            c_face_edge[4*i+j] = c_edge_idx;
      }
   }
}

Mesh *NCMesh::GetRefinedMesh(Mesh *c_mesh, bool remove_curv, int bdr_type)
   const
{
   Mesh *f_mesh;
   int num_vert = 0, num_elem = 0, num_bdr_elem = 0;

   if (c_mesh->NURBSext && !remove_curv)
   {
      cout << "NCMesh::GetRefinedMesh :"
         " NURBS meshes are not supported" << endl;
      return NULL;
   }

   for (AllVertex_iterator vi(*this); vi; ++vi)
      vi->id = num_vert++;

   for (Face_iterator fi(*this); fi; ++fi)
      num_elem++;

   if (bdr_type == 0)
   {
      for (int i = 0; i < c_mesh->GetNBE(); i++)
      {
         int c_edge = c_mesh->GetBdrElementEdgeIndex(i);
         for (Edge_iterator ei(*this, c_edge);
              ei && ei.CoarseEdge() == c_edge; ++ei)
         {
            num_bdr_elem++;
         }
      }
   }
   else if (bdr_type == 2)
   {
      for (int i = 0; i < c_edges.Size(); i++)
      {
         for (Edge_iterator ei(*this, i);
              ei && ei.CoarseEdge() == i; ++ei)
         {
            num_bdr_elem++;
         }
      }
   }

   f_mesh = new Mesh(2, num_vert, num_elem, num_bdr_elem);

   double x[2] = { 0., 0. };
   for (AllVertex_iterator vi(*this); vi; ++vi)
      f_mesh->AddVertex(x);

   Vertex *fv[4];
   int verts[4], mesh_type = 0;

   for (Face_iterator fi(*this); fi; ++fi)
   {
      int attr = c_mesh->GetAttribute(fi.CoarseFace());
      fi->GetVertices(fv);
      for (int i = 0, n = fi->isQuad() ? 4 : 3; i < n; i++)
         verts[i] = fv[i]->id;
      if (!fi->isQuad())
      {
         mesh_type |= 1;
         f_mesh->AddTriangle(verts, attr);
      }
      else
      {
         mesh_type |= 2;
         f_mesh->AddQuad(verts, attr);
      }
   }

   if (bdr_type == 0)
   {
      for (int i = 0; i < c_mesh->GetNBE(); i++)
      {
         int c_edge = c_mesh->GetBdrElementEdgeIndex(i);
         int attr = c_mesh->GetBdrAttribute(i);
         for (Edge_iterator ei(*this, c_edge);
              ei && ei.CoarseEdge() == c_edge; ++ei)
         {
            verts[0] = ei->vertex[0]->id;
            verts[1] = ei->vertex[1]->id;
            f_mesh->AddBdrSegment(verts, attr);
         }
      }
   }
   else if (bdr_type == 2)
   {
      for (int i = 0; i < c_edges.Size(); i++)
      {
         int attr = 1;
         for (Edge_iterator ei(*this, i);
              ei && ei.CoarseEdge() == i; ++ei)
         {
            verts[0] = ei->vertex[0]->id;
            verts[1] = ei->vertex[1]->id;
            f_mesh->AddBdrSegment(verts, attr);
         }
      }
   }

   if (mesh_type == 1)
      f_mesh->FinalizeTriMesh(1, 0);
   else if (mesh_type == 2)
      f_mesh->FinalizeQuadMesh(1, 0);
   else
      mfem_error("NCMesh::GetRefinedMesh");

   if (bdr_type == 1)
      f_mesh->GenerateBoundaryElements();

   GridFunction *c_nodes = c_mesh->GetNodes();
   if (!c_nodes || remove_curv)
   {
      int cur_c_face = -1;
      ElementTransformation *c_tr;
      IntegrationPoint ip;
      Vector f_vert;
      DenseMatrix pointmat;

      for (Face_iterator fi(*this); fi; ++fi)
      {
         if (fi.CoarseFace() != cur_c_face)
         {
            cur_c_face = fi.CoarseFace();
            if (!data_manager)
            {
               c_faces[cur_c_face]->SetReferenceVertices();
               c_faces[cur_c_face]->SetInteriorVertices();
            }
            else
            {
               data_manager->SetCoarseFace(cur_c_face);
            }
            c_tr = c_mesh->GetElementTransformation(cur_c_face);
         }
         fi->GetVertices(fv);
         if (!data_manager)
         {
            for (int i = 0, n = fi->isQuad() ? 4 : 3; i < n; i++)
            {
               ip.Set2(fv[i]->x);
               c_tr->Transform(ip, f_vert);
               double *f_v = f_mesh->GetVertex(fv[i]->id);
               f_v[0] = f_vert(0);
               f_v[1] = f_vert(1);
            }
         }
         else
         {
            data_manager->GetFaceToCoarseFacePointMatrix(fi, pointmat);
            for (int i = 0; i < pointmat.Width(); i++)
            {
               ip.Set2(&pointmat(0,i));
               c_tr->Transform(ip, f_vert);
               double *f_v = f_mesh->GetVertex(fv[i]->id);
               f_v[0] = f_vert(0);
               f_v[1] = f_vert(1);
            }
         }
      }

      // adjust the dependent vertices
      for (Edge_iterator ei(*this); ei; ++ei)
      {
         for (DependentVertex_iterator vi(ei); vi; ++vi)
         {
            const Edge *parent = vi.parent();
            double *ev0 = f_mesh->GetVertex(parent->vertex[0]->id);
            double *ev1 = f_mesh->GetVertex(parent->vertex[1]->id);
            double *mid = f_mesh->GetVertex(vi->id);
            mid[0] = (ev0[0] + ev1[0])/2;
            mid[1] = (ev0[1] + ev1[1])/2;
         }
      }
   }
   else
   {
      GridFunction *f_nodes =
         GetRefinedGridFunction(c_mesh, c_nodes, f_mesh, false);

      f_mesh->NewNodes(*f_nodes, true);
   }

   return f_mesh;
}

void NCMesh::GetRefinedGridFunction(Mesh *c_mesh, GridFunction *c_sol,
                                    Mesh *f_mesh, GridFunction *f_sol) const
{
   int n, cur_c_face = -1, f_face = 0;
   Vertex *fv[4];
   IntegrationPointTransformation ipT;
   IsoparametricTransformation &T = ipT.Transf;
   DenseMatrix I;
   Array<int> c_vdofs, f_vdofs;
   Vector c_vals, f_vals, c_shape;
   IntegrationPoint ip;
   const FiniteElement *c_fe, *f_fe;
   const NodalFiniteElement *nodal_f_fe;
   int vdim = c_sol->FESpace()->GetVDim();
   bool same_fec = !strcmp(c_sol->FESpace()->FEColl()->Name(),
                           f_sol->FESpace()->FEColl()->Name());

   for (Face_iterator fi(*this); fi; ++fi, f_face++)
   {
      if (fi.CoarseFace() != cur_c_face)
      {
         cur_c_face = fi.CoarseFace();
         if (!data_manager)
         {
            c_faces[cur_c_face]->SetReferenceVertices();
            c_faces[cur_c_face]->SetInteriorVertices();
         }
         else
            data_manager->SetCoarseFace(cur_c_face);
         c_sol->FESpace()->GetElementVDofs(cur_c_face, c_vdofs);
         c_sol->GetSubVector(c_vdofs, c_vals);
         c_fe = c_sol->FESpace()->GetFE(cur_c_face);
      }
      if (!data_manager)
      {
         fi->GetVertices(fv);
         if (!fi->isQuad())
         {
            n = 3;
            T.SetFE(&TriangleFE);
         }
         else
         {
            n = 4;
            T.SetFE(&QuadrilateralFE);
         }
         T.GetPointMat().SetSize(2, n);
         for (int i = 0; i < n; i++)
         {
            T.GetPointMat()(0,i) = fv[i]->x[0];
            T.GetPointMat()(1,i) = fv[i]->x[1];
         }
      }
      else
      {
         data_manager->GetFaceToCoarseFaceIPT(fi, ipT);
      }
      f_sol->FESpace()->GetElementVDofs(f_face, f_vdofs);
      f_fe = f_sol->FESpace()->GetFE(f_face);
      nodal_f_fe = dynamic_cast<const NodalFiniteElement *>(f_fe);
      if (nodal_f_fe)
      {
         f_vals.SetSize(vdim*f_fe->GetDof());
         c_shape.SetSize(c_fe->GetDof());
         for (int i = 0; i < f_fe->GetDof(); i++)
         {
            ipT.Transform(f_fe->GetNodes().IntPoint(i), ip);
            c_fe->CalcShape(ip, c_shape);
            for (int d = 0; d < vdim; d++)
               f_vals(d*f_fe->GetDof()+i) =
                  c_shape * (c_vals.GetData() + d*c_shape.Size());
         }
         f_sol->SetSubVector(f_vdofs, f_vals);
      }
      else if (same_fec)
      {
         I.SetSize(c_fe->GetDof());
         c_fe->GetLocalInterpolation(T, I);
         f_vals.SetSize(f_vdofs.Size());
         for (int d = 0; d < vdim; d++)
            I.Mult(c_vals.GetData() + d*I.Size(),
                   f_vals.GetData() + d*I.Size());
         f_sol->SetSubVector(f_vdofs, f_vals);
      }
      else
      {
         mfem_error("NCMesh::GetRefinedGridFunction :\n"
                    "   the fine and coarse spaces differ and\n"
                    "   the fine space is not nodal.");
      }
#ifdef MFEM_DEBUG
      T.SetIntPoint(&Geometries.GetCenter(c_fe->GetGeomType()));
      if (T.Weight() < 0.)
         cout << "\nNegative Jacobian\n" << endl;
#endif
   }
}

GridFunction *NCMesh::GetRefinedGridFunction(Mesh *c_mesh, GridFunction *c_sol,
                                             Mesh *f_mesh, bool linearize)
   const
{
   FiniteElementSpace *c_fes = c_sol->FESpace();
   FiniteElementCollection *f_fec;
   if (linearize)
      f_fec = new LinearFECollection;
   else
      f_fec = FiniteElementCollection::New(c_fes->FEColl()->Name());
   FiniteElementSpace *f_fes = new FiniteElementSpace(
      f_mesh, f_fec, c_fes->GetVDim(), c_fes->GetOrdering());
   GridFunction *f_sol = new GridFunction(f_fes);
   f_sol->MakeOwner(f_fec);

   GetRefinedGridFunction(c_mesh, c_sol, f_mesh, f_sol);

   if (linearize) // adjust the dependent vertices
   {
      for (Edge_iterator ei(*this); ei; ++ei)
         for (DependentVertex_iterator vi(ei); vi; ++vi)
         {
            const Edge *parent = vi.parent();
            (*f_sol)(vi->id) = ((*f_sol)(parent->vertex[0]->id) +
                                (*f_sol)(parent->vertex[1]->id))/2;
         }
   }

   return f_sol;
}

SparseMatrix *NCMesh::GetInterpolation(Mesh *f_mesh, FiniteElementSpace *f_fes)
{
   SparseMatrix *P;
   const FiniteElementCollection *fec = f_fes->FEColl();
   Array<int> f_dofs;

   int t_nv = 0, t_ne = 0, t_nf = 0;

   for (Vertex_iterator vi(*this); vi; ++vi)
      t_nv++;
   for (Edge_iterator ei(*this); ei; ++ei)
      t_ne++;
   for (Face_iterator fi(*this); fi; ++fi)
      fi->id = t_nf++;

   int nvdofs, te_off, nedofs, tf_off, nfdofs, t_ndofs;
   int f_geom = f_mesh->GetElementBaseGeometry(0);
   nvdofs = fec->DofForGeometry(Geometry::POINT);
   nedofs = fec->DofForGeometry(Geometry::SEGMENT);
   nfdofs = fec->DofForGeometry(f_geom);
   te_off = t_nv * nvdofs;
   tf_off = te_off + t_ne * nedofs;
   t_ndofs = tf_off + t_nf * nfdofs;

   P = new SparseMatrix(f_fes->GetNDofs(), t_ndofs);

   if (nvdofs > 0)
   {
      t_nv = 0;
      for (Vertex_iterator vi(*this); vi; ++vi, t_nv++)
         for (int j = 0; j < nvdofs; j++)
            P->Set(nvdofs*vi->id + j, nvdofs*t_nv + j, 1.0);
   }

   if (nvdofs + nedofs > 0)
   {
      IsoparametricTransformation edge_T;
      edge_T.SetFE(&SegmentFE);
      DenseMatrix &pm = edge_T.GetPointMat();
      pm.SetSize(1, 2);
      const FiniteElement *e_fe =
         fec->FiniteElementForGeometry(Geometry::SEGMENT);
      const int *edge_dof_ord[2];
      edge_dof_ord[0] = fec->DofOrderForOrientation(Geometry::SEGMENT, +1);
      edge_dof_ord[1] = fec->DofOrderForOrientation(Geometry::SEGMENT, -1);
      DenseMatrix edge_P(e_fe->GetDof());
      Vector full_edge_P_row, P_row;
      Array<int> t_dofs(nedofs); // only interior true edge dofs
      Array<int> tv_dofs, all_t_dofs, tv_off(2*nvdofs+1), cols;
      tv_off[0] = 0;
      Array<double> tv_rows;
      const Table &f_el_edge = f_mesh->ElementToEdgeTable();
      Vertex *t_vtx[2];
      int fe_off = nvdofs*f_mesh->GetNV();
      t_ne = 0;
      for (Edge_iterator ei(*this); ei; ++ei, t_ne++)
      {
         int te_or = (ei->vertex[0]->id < ei->vertex[1]->id) ? +1 : -1;
         int i_or = (te_or > 0) ? 0 : 1;
         // determine t_dofs for this edge, excluding the vertex dofs
         {
            int i = 0;
            const int *ind = edge_dof_ord[i_or];
            for (int j = 0; j < nedofs; j++)
            {
               int k = ind[j];
               if (k >= 0)
                  t_dofs[i++] = te_off + nedofs*t_ne + k;
               else
                  t_dofs[i++] = -1 - (te_off + nedofs*t_ne + (-1 - k));
            }
         }
         int f_edge_idx;
         // determine f_edge_idx for this real edge as an edge in f_mesh
         {
            Face *face = ei->face[0];
            if (!face || face->isRefined())
               face = ei->face[1];
            int i, n = f_el_edge.RowSize(face->id);
            for (i = 0; ei != face->edge[i]; i++)
               if (i >= n)
                  mfem_error("NCMesh::GetInterpolation");
            f_edge_idx = f_el_edge.GetRow(face->id)[i];
         }
         f_fes->GetEdgeDofs(f_edge_idx, f_dofs); // vertex + edge dofs

         // do not set vertex dofs
         for (int i = 0; i < t_dofs.Size(); i++)
            P->Set(f_dofs[2*nvdofs+i], t_dofs[i], 1.0);

         if (!ei->isRefined())
            continue;

         t_vtx[0] = ei->vertex[i_or];
         t_vtx[1] = ei->vertex[1-i_or];
         // extract and merge the rows of P corresponding to the vertices of
         // the current real edge, ei
         {
            tv_dofs.SetSize(0);
            tv_rows.SetSize(0);
            if (nvdofs > 0)
            {
               for (int k = 0; k < 2; k++)
                  for (int j = 0; j < nvdofs; j++)
                  {
                     P->GetRow(nvdofs*t_vtx[k]->id + j, cols, P_row);
#ifdef MFEM_DEBUG
                     if (cols.Size() == 0)
                        mfem_error("NCMesh::GetInterpolation");
#endif
                     tv_dofs.Append(cols);
                     {
                        Array<double> P_row_array(P_row, P_row.Size());
                        tv_rows.Append(P_row_array);
                     }
                     tv_off[nvdofs*k+j+1] = tv_off[nvdofs*k+j] + cols.Size();
                  }
            }
            tv_dofs.Copy(all_t_dofs);
            all_t_dofs.Append(t_dofs);
         }

         t_vtx[0]->x[0] = 0.0;
         t_vtx[1]->x[0] = 1.0;
         for (SimpleEdge_iterator edge(ei); edge; ++edge)
         {
            if (edge->isRefined())
            {
               edge->child[0]->vertex[1]->x[0] =
                  (edge->vertex[0]->x[0] + edge->vertex[1]->x[0])/2;
            }
            else
            {
               int V[2] = { edge->vertex[0]->id, edge->vertex[1]->id };
               int fe_or = (V[0] < V[1]) ? +1 : -1;
               pm(0,0) = edge->vertex[i_or]->x[0];
               pm(0,1) = edge->vertex[1-i_or]->x[0];
               fe_or *= te_or;
// #ifdef MFEM_DEBUG
               // make sure edge_T has positive Jacobian
               if (pm(0,0) >= pm(0,1))
                  mfem_error("NCMesh::GetInterpolation");
// #endif
               e_fe->GetLocalInterpolation(edge_T, edge_P);
               // determine f_edge_idx for this dependent edge as an edge
               // in f_mesh
               {
                  Face *face = edge->face[0];
                  if (!face)
                     face = edge->face[1];
                  int i, n = f_el_edge.RowSize(face->id);
                  for (i = 0; edge != face->edge[i]; i++)
                     if (i >= n)
                        mfem_error("NCMesh::GetInterpolation");
                  f_edge_idx = f_el_edge.GetRow(face->id)[i];
               }
               // define f_dofs for this dependent edge, taking into account
               // the orientation in fe_or
               {
                  // f_fes->GetEdgeDofs(f_edge_idx, f_dofs);
                  int i = 0;
                  if (nvdofs > 0)
                  {
                     if (i_or)
                        Swap<int>(V[0], V[1]);
                     for (int k = 0; k < 2; k++)
                        for (int j = 0; j < nvdofs; j++)
                           f_dofs[i++] = nvdofs*V[k] + j;
                  }
                  const int *ind =
                     (fe_or > 0) ? edge_dof_ord[0] : edge_dof_ord[1];
                  int m = fe_off + nedofs*f_edge_idx;
                  for (int j = 0; j < nedofs; j++)
                  {
                     int k = ind[j];
                     if (k >= 0)
                        f_dofs[i++] = m + k;
                     else
                        f_dofs[i++] = -1 - (m + (-1 - k));
                  }
               }
               for (int i = 0; i < f_dofs.Size(); i++)
               {
                  if (!P->RowIsEmpty(f_dofs[i]))
                     continue;

                  full_edge_P_row.SetSize(all_t_dofs.Size());

                  for (int j = 0; j < 2*nvdofs; j++)
                  {
                     double a = edge_P(i,j);
                     for (int k = tv_off[j]; k < tv_off[j+1]; k++)
                        full_edge_P_row(k) = a*tv_rows[k];
                  }
                  for (int j = 0; j < t_dofs.Size(); j++)
                  {
                     full_edge_P_row(tv_dofs.Size()+j) = edge_P(i,2*nvdofs+j);
                  }

                  // all_t_dofs may have the same index twice !
                  P->AddRow(f_dofs[i], all_t_dofs, full_edge_P_row);
                  // P->SetRow(f_dofs[i], all_t_dofs, full_edge_P_row);
               }
            }
         }
      }
   }

   if (nfdofs > 0)
   {
      t_nf = 0;
      for (Face_iterator fi(*this); fi; ++fi, t_nf++)
      {
         f_fes->GetElementInteriorDofs(t_nf, f_dofs);
         for (int i = 0; i < f_dofs.Size(); i++)
            P->Set(f_dofs[i], tf_off + t_nf*nfdofs + i, 1.0);
      }
   }

   P->Finalize();

   return P;
}

NonconformingMesh::~NonconformingMesh()
{
   for (int i = 0; i < c_faces.Size(); i++)
      c_faces[i]->FreeHierarchy(data_manager);

   for (int i = 0; i < c_edges.Size(); i++)
      c_edges[i]->FreeHierarchy(data_manager);

   if (!data_manager)
      for (int i = 0; i < c_verts.Size(); i++)
         delete c_verts[i];
   else
      for (int i = 0; i < c_verts.Size(); i++)
         data_manager->FreeVertex(c_verts[i]);
}
