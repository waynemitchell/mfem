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

#ifdef MFEM_USE_MPI

#include "mesh_headers.hpp"

// for osockstream ... temporary
#include "../general/osockstream.hpp"
#include "../general/sets.hpp"

ParMesh::ParMesh(MPI_Comm comm, Mesh &mesh, int *partitioning_,
                 int part_method)
{
   int i, j;
   int *partitioning;

   MyComm = comm;
   MPI_Comm_size(MyComm, &NRanks);
   MPI_Comm_rank(MyComm, &MyRank);

   if (partitioning_)
      partitioning = partitioning_;
   else
      partitioning = mesh.GeneratePartitioning(NRanks, part_method);

   Array<int> vert;
   Array<int> vert_global_local(mesh.GetNV());
   int vert_counter, element_counter, bdrelem_counter;

   // make sure that the same edges are marked in the elements,
   // boundary elements and the faces of the serial mesh
   mesh.MarkEdges();

   // build vert_global_local
   for (i = 0; i < vert_global_local.Size(); i++)
      vert_global_local[i] = -1;

   element_counter = 0;
   vert_counter = 0;
   for (i = 0; i < mesh.GetNE(); i++)
      if (partitioning[i] == MyRank)
      {
         mesh.GetElementVertices(i, vert);
         element_counter++;
         for (j = 0; j < vert.Size(); j++)
            if (vert_global_local[vert[j]] < 0)
               vert_global_local[vert[j]] = vert_counter++;
      }

   NumOfVertices = vert_counter;
   NumOfElements = element_counter;
   vertices.SetSize(NumOfVertices);

   // preserve ordering when running in serial
   if (NRanks == 1)
      for (i = 0; i < vert_global_local.Size(); i++)
         vert_global_local[i] = i;

   // determine vertices
   for (i = 0; i < vert_global_local.Size(); i++)
      if (vert_global_local[i] >= 0)
         vertices[vert_global_local[i]].SetCoords(mesh.GetVertex(i));

   // determine elements
   element_counter = 0;
   elements.SetSize (NumOfElements);
   for (i = 0; i < mesh.GetNE(); i++)
      if (partitioning[i] == MyRank)
      {
         elements[element_counter] = mesh.GetElement (i) -> Duplicate();
         int *v = elements[element_counter] -> GetVertices();
         int nv = elements[element_counter] -> GetNVertices();
         for (j = 0; j < nv; j++)
            v[j] = vert_global_local[v[j]];
         element_counter++;
      }

   Table *edge_element = NULL;

   // build boundary elements
   if (Dim == 3)
   {
      NumOfBdrElements = 0;
      for (i = 0; i < mesh.GetNBE(); i++)
      {
         int face = mesh.GetBdrElementEdgeIndex (i);
         int el1, el2;
         mesh.GetFaceElements (face, &el1, &el2);
         if (partitioning[el1] == MyRank ||
             (el2 >= 0 && partitioning[el2] == MyRank))
            NumOfBdrElements++;
      }

      bdrelem_counter = 0;
      boundary.SetSize (NumOfBdrElements);
      for (i = 0; i < mesh.GetNBE(); i++)
      {
         int face = mesh.GetBdrElementEdgeIndex (i);
         int el1, el2;
         mesh.GetFaceElements (face, &el1, &el2);
         if (partitioning[el1] == MyRank ||
             (el2 >= 0 && partitioning[el2] == MyRank))
         {
            boundary[bdrelem_counter] = mesh.GetBdrElement (i) -> Duplicate();
            int *v = boundary[bdrelem_counter] -> GetVertices();
            int nv = boundary[bdrelem_counter] -> GetNVertices();
            for (j = 0; j < nv; j++)
               v[j] = vert_global_local[v[j]];
            bdrelem_counter++;
         }
      }

   }
   else if (Dim == 2)
   {
      edge_element = new Table;
      Transpose(mesh.ElementToEdgeTable(), *edge_element, mesh.GetNEdges());

      NumOfBdrElements = 0;
      for (i = 0; i < mesh.GetNBE(); i++)
      {
         int edge = mesh.GetBdrElementEdgeIndex (i);
         int el1, el2 = -1;
         el1 = edge_element -> GetRow(edge)[0];
         if (edge_element -> RowSize(edge) == 2)
            el2 = edge_element -> GetRow(edge)[1];
         if (partitioning[el1] == MyRank ||
             (el2 >= 0 && partitioning[el2] == MyRank))
            NumOfBdrElements++;
      }

      bdrelem_counter = 0;
      boundary.SetSize (NumOfBdrElements);
      for (i = 0; i < mesh.GetNBE(); i++)
      {
         int edge = mesh.GetBdrElementEdgeIndex (i);
         int el1, el2 = -1;
         el1 = edge_element -> GetRow(edge)[0];
         if (edge_element -> RowSize(edge) == 2)
            el2 = edge_element -> GetRow(edge)[1];
         if (partitioning[el1] == MyRank ||
             (el2 >= 0 && partitioning[el2] == MyRank))
         {
            boundary[bdrelem_counter] = mesh.GetBdrElement (i) -> Duplicate();
            int *v = boundary[bdrelem_counter] -> GetVertices();
            int nv = boundary[bdrelem_counter] -> GetNVertices();
            for (j = 0; j < nv; j++)
               v[j] = vert_global_local[v[j]];
            bdrelem_counter++;
         }
      }
   }

   meshgen = mesh.MeshGenerator();

   attributes.SetSize (mesh.attributes.Size());
   for (i = 0; i < attributes.Size(); i++)
      attributes[i] = mesh.attributes[i];
   bdr_attributes.SetSize (mesh.bdr_attributes.Size());
   for (i = 0; i < bdr_attributes.Size(); i++)
      bdr_attributes[i] = mesh.bdr_attributes[i];

   InitTables();

   el_to_edge = new Table;
   NumOfEdges = Mesh::GetElementToEdgeTable(*el_to_edge, be_to_edge);

   STable3D *faces_tbl = NULL;
   if (Dim == 3)
   {
      faces_tbl = GetElementToFaceTable (1);
      GenerateFaces();
   }

   ListOfIntegerSets  groups;
   IntegerSet         group;

   // the first group is the local one
   group.Recreate (1, &MyRank);
   groups.Insert(group);

   // determine shared faces
   int sface_counter = 0;
   Array<int> face_group (mesh.GetNFaces());
   for (i = 0; i < face_group.Size(); i++)
   {
      int el[2];
      face_group[i] = -1;
      mesh.GetFaceElements (i, &el[0], &el[1]);
      if (el[1] >= 0)
      {
         el[0] = partitioning[el[0]];
         el[1] = partitioning[el[1]];
         if ((el[0] == MyRank && el[1] != MyRank) ||
             (el[0] != MyRank && el[1] == MyRank))
         {
            group.Recreate (2, el);
            face_group[i] = groups.Insert (group) - 1;
            sface_counter++;
         }
      }
   }

   // determine shared edges
   int sedge_counter = 0;
   if (!edge_element)
   {
      edge_element = new Table;
      Transpose(mesh.ElementToEdgeTable(), *edge_element, mesh.GetNEdges());
   }
   for (i = 0; i < edge_element->Size(); i++)
   {
      int me = 0, others = 0;
      for (j = edge_element->GetI()[i]; j < edge_element->GetI()[i+1]; j++)
      {
         edge_element->GetJ()[j] = partitioning[edge_element->GetJ()[j]];
         if (edge_element->GetJ()[j] == MyRank)
            me = 1;
         else
            others = 1;
      }

      if (me && others)
      {
         sedge_counter++;
         group.Recreate (edge_element->RowSize(i), edge_element->GetRow(i));
         edge_element->GetRow(i)[0] = groups.Insert (group) - 1;
      }
      else
         edge_element->GetRow(i)[0] = -1;
   }

   // determine shared vertices
   int svert_counter = 0;
   Table *vert_element = mesh.GetVertexToElementTable(); // we must delete this

   for (i = 0; i < vert_element -> Size(); i++)
   {
      int me = 0, others = 0;
      for (j = vert_element->GetI()[i]; j < vert_element->GetI()[i+1]; j++)
      {
         vert_element->GetJ()[j] = partitioning[vert_element->GetJ()[j]];
         if (vert_element->GetJ()[j] == MyRank)
            me = 1;
         else
            others = 1;
      }

      if (me && others)
      {
         svert_counter++;
         group.Recreate (vert_element->RowSize(i), vert_element->GetRow(i));
         vert_element->GetRow(i)[0] = groups.Insert (group) - 1;
      }
      else
         vert_element->GetRow(i)[0] = -1;
   }

   delete [] partitioning;

   // build group_sface
   group_sface.MakeI(groups.Size()-1);

   for (i = 0; i < face_group.Size(); i++)
      if (face_group[i] >= 0)
         group_sface.AddAColumnInRow(face_group[i]);

   group_sface.MakeJ();

   sface_counter = 0;
   for (i = 0; i < face_group.Size(); i++)
      if (face_group[i] >= 0)
         group_sface.AddConnection(face_group[i],sface_counter++);

   group_sface.ShiftUpI();

   // build group_sedge
   group_sedge.MakeI(groups.Size()-1);

   for (i = 0; i < edge_element->Size(); i++)
      if (edge_element->GetRow(i)[0] >= 0)
         group_sedge.AddAColumnInRow(edge_element->GetRow(i)[0]);


   group_sedge.MakeJ();

   sedge_counter = 0;
   for (i = 0; i < edge_element->Size(); i++)
      if (edge_element->GetRow(i)[0] >= 0)
         group_sedge.AddConnection(edge_element->GetRow(i)[0],
                                   sedge_counter++);

   group_sedge.ShiftUpI();

   // build group_svert
   group_svert.MakeI(groups.Size()-1);

   for (i = 0; i < vert_element -> Size(); i++)
      if (vert_element -> GetRow(i)[0] >= 0)
         group_svert.AddAColumnInRow(vert_element -> GetRow(i)[0]);

   group_svert.MakeJ();

   svert_counter = 0;
   for (i = 0; i < vert_element -> Size(); i++)
      if (vert_element -> GetRow(i)[0] >= 0)
         group_svert.AddConnection(vert_element -> GetRow(i)[0],
                                   svert_counter++);

   group_svert.ShiftUpI();

   // build shared_faces and sface_lface
   shared_faces.SetSize (sface_counter);
   sface_lface. SetSize (sface_counter);

   if (Dim == 3)
   {
      sface_counter = 0;
      for (i = 0; i < face_group.Size(); i++)
         if (face_group[i] >= 0)
         {
            shared_faces[sface_counter] = mesh.GetFace(i) -> Duplicate();
            int *v = shared_faces[sface_counter] -> GetVertices();
            int nv = shared_faces[sface_counter] -> GetNVertices();
            for (j = 0; j < nv; j++)
               v[j] = vert_global_local[v[j]];
            switch (shared_faces[sface_counter] -> GetType())
            {
            case Element::TRIANGLE:
               sface_lface[sface_counter] = (*faces_tbl)(v[0], v[1], v[2]);
               break;
            case Element::QUADRILATERAL:
               sface_lface[sface_counter] = (*faces_tbl)(v[0], v[1], v[2], v[3]);
               break;
            }
            sface_counter++;
         }

      delete faces_tbl;
   }

   // build shared_edges and sedge_ledge
   shared_edges.SetSize (sedge_counter);
   sedge_ledge. SetSize (sedge_counter);

   {
      DSTable v_to_v(NumOfVertices);
      GetVertexToVertexTable(v_to_v);

      sedge_counter = 0;
      for (i = 0; i < edge_element->Size(); i++)
         if (edge_element->GetRow(i)[0] >= 0)
         {
            mesh.GetEdgeVertices(i, vert);

            shared_edges[sedge_counter] =
               new Segment(vert_global_local[vert[0]],
                           vert_global_local[vert[1]], 1);

            if ((sedge_ledge[sedge_counter] =
                 v_to_v(vert_global_local[vert[0]],
                        vert_global_local[vert[1]])) < 0)
            {
               cerr << "\n\n\n" << MyRank << ": ParMesh::ParMesh: "
                    << "ERROR in v_to_v\n\n" << endl;
               mfem_error();
            }

            sedge_counter++;
         }
   }

   delete edge_element;

   // build svert_lvert
   svert_lvert.SetSize(svert_counter);

   svert_counter = 0;
   for (i = 0; i < vert_element -> Size(); i++)
      if (vert_element -> GetRow(i)[0] >= 0)
         svert_lvert[svert_counter++] = vert_global_local[i];

   delete vert_element;

   // build group_lproc, group_mgroupandproc and lproc_proc
   groups.AsTable(group_lproc); // group_lproc = group_proc

   Table group_mgroupandproc;
   group_mgroupandproc.SetDims (group_lproc.Size(),
                                group_lproc.Size_of_connections() +
                                group_lproc.Size());
   for (i = 0; i < group_mgroupandproc.Size(); i++)
   {
      j = group_mgroupandproc.GetI()[i];
      group_mgroupandproc.GetI()[i+1] = j + group_lproc.RowSize(i) + 1;
      group_mgroupandproc.GetJ()[j] = i;
      j++;
      for (int k = group_lproc.GetI()[i];
           j < group_mgroupandproc.GetI()[i+1]; j++, k++)
         group_mgroupandproc.GetJ()[j] = group_lproc.GetJ()[k];
   }

   Array<int> proc_lproc (NRanks);
   for (i = 0; i < NRanks; i++)
      proc_lproc[i] = -1;

   int lproc_counter = 0;
   for (i = 0; i < group_lproc.Size_of_connections(); i++)
      if (proc_lproc[group_lproc.GetJ()[i]] < 0)
         proc_lproc[group_lproc.GetJ()[i]] = lproc_counter++;

   lproc_proc.SetSize(lproc_counter);
   for (i = 0; i < NRanks; i++)
      if (proc_lproc[i] >= 0)
         lproc_proc[proc_lproc[i]] = i;

   for (i = 0; i < group_lproc.Size_of_connections(); i++)
      group_lproc.GetJ()[i] = proc_lproc[group_lproc.GetJ()[i]];

   // build groupmaster_lproc
   groupmaster_lproc.SetSize(groups.Size());

   /* simplest choice of the group owner
      for (i = 0; i < groups.Size(); i++)
      groupmaster_lproc[i] = proc_lproc[groups.PickElementInSet(i)];
   */

   // load-balanced choice of the group owner
   for (i = 0; i < groups.Size(); i++)
      groupmaster_lproc[i] = proc_lproc[groups.PickRandomElementInSet(i)];

   // build group_mgroup
   group_mgroup.SetSize (groups.Size());

   int send_counter = 0;
   int recv_counter = 0;
   for (i = 1; i < groups.Size(); i++)
      if (groupmaster_lproc[i] != 0) // we are not the master
         recv_counter++;
      else
         send_counter += group_lproc.RowSize (i)-1;

   MPI_Request *requests = new MPI_Request[send_counter];
   MPI_Status  *statuses = new MPI_Status[send_counter];

   int max_recv_size = 0;
   send_counter = 0;
   for (i = 1; i < groups.Size(); i++)
   {
      if (groupmaster_lproc[i] == 0) // we are the master
      {
         group_mgroup[i] = i;

         for (j = group_lproc.GetI()[i];
              j < group_lproc.GetI()[i+1]; j++)
         {
            if (group_lproc.GetJ()[j] != 0)
            {
               MPI_Isend (group_mgroupandproc.GetRow (i),
                          group_mgroupandproc.RowSize (i),
                          MPI_INT,
                          lproc_proc[group_lproc.GetJ()[j]],
                          822,
                          MyComm,
                          &requests[send_counter]);
               send_counter++;
//               cout << "ParMesh::ParMesh: " << MyRank
//                    << " ---> " << lproc_proc[group_lproc.GetJ()[j]] << endl;
            }
         }
      }
      else // we are not the master
         if (max_recv_size < group_lproc.RowSize(i))
            max_recv_size = group_lproc.RowSize(i);
   }
   max_recv_size++;

   if (recv_counter > 0)
   {
      int count;
      MPI_Status status;
      int *recv_buf = new int[max_recv_size];
      for ( ; recv_counter > 0; recv_counter--)
      {
         MPI_Recv (recv_buf, max_recv_size, MPI_INT,
                   MPI_ANY_SOURCE, 822, MyComm, &status);

         MPI_Get_count (&status, MPI_INT, &count);

//         cout << "ParMesh::ParMesh: " << MyRank
//              << " <--- " << status.MPI_SOURCE << endl;

         group.Recreate (count-1, recv_buf+1);
//         group_mgroup[groups.Lookup (group)] = recv_buf[0];
         group_mgroup[i=groups.Lookup (group)] = recv_buf[0];

         if (lproc_proc[groupmaster_lproc[i]] != status.MPI_SOURCE)
         {
            cerr << "\n\n\nParMesh::ParMesh: " << MyRank
                 << ": ERROR\n\n\n" << endl;
            mfem_error();
         }
      }
      delete [] recv_buf;
   }

   MPI_Waitall (send_counter, requests, statuses);

   delete [] statuses;
   delete [] requests;
}

void ParMesh::GroupEdge(int group, int i, int &edge, int &o)
{
   int sedge = group_sedge.GetJ()[group_sedge.GetI()[group-1]+i];
   edge = sedge_ledge[sedge];
   int *v = shared_edges[sedge] -> GetVertices();
   o = (v[0] < v[1]) ? (+1) : (-1);
}

void ParMesh::GroupFace(int group, int i, int &face, int &o)
{
   int sface = group_sface.GetJ()[group_sface.GetI()[group-1]+i];
   face = sface_lface[sface];
   // face gives the base orientation
   if (faces[face] -> GetType() == Element::TRIANGLE)
      o = GetTriOrientation (faces[face] -> GetVertices(),
                             shared_faces[sface] -> GetVertices());
   if (faces[face] -> GetType() == Element::QUADRILATERAL)
      o = GetQuadOrientation (faces[face] -> GetVertices(),
                              shared_faces[sface] -> GetVertices());
}


// For a line segment with vertices v[0] and v[1], return a number with
// the following meaning:
// 0 - the edge was not refined
// 1 - the edge e was refined once by splitting v[0],v[1]
int ParMesh::GetEdgeSplittings(Element *edge, const DSTable &v_to_v,
                               int *middle)
{
   int m, *v = edge->GetVertices();

   if ((m = v_to_v(v[0], v[1])) != -1 && middle[m] != -1)
      return 1;
   else
      return 0;
}

// For a triangular face with (correctly ordered) vertices v[0], v[1], v[2]
// return a number with the following meaning:
// 0 - the face was not refined
// 1 - the face was refined once by splitting v[0],v[1]
// 2 - the face was refined twice by splitting v[0],v[1] and then v[1],v[2]
// 3 - the face was refined twice by splitting v[0],v[1] and then v[0],v[2]
// 4 - the face was refined three times (as in 2+3)
int ParMesh::GetFaceSplittings(Element *face, const DSTable &v_to_v,
                               int *middle)
{
   int m, right = 0;
   int number_of_splittings = 0;
   int *v = face->GetVertices();

   if ((m = v_to_v(v[0], v[1])) != -1 && middle[m] != -1)
   {
      number_of_splittings++;
      if ((m = v_to_v(v[1], v[2])) != -1 && middle[m] != -1)
      {
         right = 1;
         number_of_splittings++;
      }
      if ((m = v_to_v(v[2], v[0])) != -1 && middle[m] != -1)
         number_of_splittings++;

      switch (number_of_splittings)
      {
      case 2:
         if (right == 0)
            number_of_splittings++;
         break;
      case 3:
         number_of_splittings++;
         break;
      }
   }

   return number_of_splittings;
}

void ParMesh::LocalRefinement(const Array<int> &marked_el, int type)
{
   int i;

   if (Dim == 3)
   {
      int nedges;

      if (WantTwoLevelState)
      {
         c_NumOfVertices    = NumOfVertices;
         c_NumOfElements    = NumOfElements;
         c_NumOfBdrElements = NumOfBdrElements;
      }

      int uniform_refinement = 0;
      if (type < 0)
      {
         type = -type;
         uniform_refinement = 1;
      }

      // 1. Get table of vertex to vertex connections.
      DSTable v_to_v(NumOfVertices);
      GetVertexToVertexTable(v_to_v);

      // 2. Get edge to element connections in arrays edge1 and edge2
      nedges = v_to_v.NumberOfEntries();
      int *middle = new int[nedges];

      for(i=0; i<nedges; i++)
         middle[i] = -1;

      // 3. Do the red refinement.
      int ii;
      switch (type)
      {
      case 1:
         for(i=0; i<marked_el.Size(); i++)
            Bisection ( marked_el[i], v_to_v, NULL, NULL, middle);
         break;
      case 2:
         for(i=0; i<marked_el.Size(); i++) {
            Bisection ( marked_el[i], v_to_v, NULL, NULL, middle);

            Bisection ( NumOfElements - 1, v_to_v, NULL, NULL, middle);
            Bisection ( marked_el[i], v_to_v, NULL, NULL, middle);
         }
         break;
      case 3:
         for(i=0; i<marked_el.Size(); i++) {
            Bisection ( marked_el[i], v_to_v, NULL, NULL, middle);

            ii = NumOfElements - 1;
            Bisection ( ii, v_to_v, NULL, NULL, middle);
            Bisection ( NumOfElements - 1, v_to_v, NULL, NULL, middle);
            Bisection ( ii, v_to_v, NULL, NULL, middle);

            Bisection ( marked_el[i], v_to_v, NULL, NULL, middle);
            Bisection ( NumOfElements-1, v_to_v, NULL, NULL, middle);
            Bisection ( marked_el[i], v_to_v, NULL, NULL, middle);
         }
         break;
      }

      if (WantTwoLevelState)
      {
         RefinedElement::State = RefinedElement::FINE;
         State = Mesh::TWO_LEVEL_FINE;
      }

      // 4. Do the green refinement (to get conforming mesh).
      int need_refinement;
      int refined_edge[5][3] = {{0, 0, 0},
                                {1, 0, 0},
                                {1, 1, 0},
                                {1, 0, 1},
                                {1, 1, 1}};
      int faces_in_group, max_faces_in_group = 0;
      // face_splittings identify how the shared faces have been split
      int **face_splittings = new int*[GetNGroups()-1];
      for (i = 0; i < GetNGroups()-1; i++)
      {
         faces_in_group = GroupNFaces(i+1);
         face_splittings[i] = new int[faces_in_group];
         if (faces_in_group > max_faces_in_group)
            max_faces_in_group = faces_in_group;
      }
      int neighbor, *iBuf = new int[max_faces_in_group];

      do {
         need_refinement = 0;
         for (i = 0; i < NumOfElements; i++)
         {
            if (elements[i]->NeedRefinement(v_to_v, middle))
            {
               need_refinement = 1;
               Bisection(i, v_to_v, NULL, NULL, middle);
            }
         }

         if (uniform_refinement)
            continue;

         Array<int> group_faces;
         int *v, j, k, c;
         double coord[3];

         MPI_Request request;
         MPI_Status  status;

         // if the mesh is locally conforming start making it globally conforming
         if (need_refinement == 0)
         {
            MPI_Barrier(MyComm);

            //==== (a) send the type of interface splitting ========
            for (i = 0; i < GetNGroups()-1; i++)
            {
               group_sface.GetRow(i, group_faces);
               faces_in_group = group_faces.Size();
               // it is enough to communicate through the faces
               if (faces_in_group != 0)
               {
                  for (j = 0; j < faces_in_group; j++)
                     face_splittings[i][j] =
                        GetFaceSplittings(shared_faces[group_faces[j]], v_to_v, middle);
                  j = group_lproc.GetI()[i+1];
                  if (group_lproc.GetJ()[j] == 0)
                     neighbor = lproc_proc[group_lproc.GetJ()[j+1]];
                  else
                     neighbor = lproc_proc[group_lproc.GetJ()[j]];
                  MPI_Isend(face_splittings[i], faces_in_group, MPI_INT,
                            neighbor, 0, MyComm, &request);
               }
            }

            //==== (b) receive the type of interface splitting =====
            for (i = 0; i < GetNGroups()-1; i++)
            {
               group_sface.GetRow(i, group_faces);
               faces_in_group = group_faces.Size();
               if (faces_in_group != 0)
               {
                  j = group_lproc.GetI()[i+1];
                  if (group_lproc.GetJ()[j] == 0)
                     neighbor = lproc_proc[group_lproc.GetJ()[j+1]];
                  else
                     neighbor = lproc_proc[group_lproc.GetJ()[j]];
                  MPI_Recv(iBuf, faces_in_group, MPI_INT, neighbor,
                           MPI_ANY_TAG, MyComm, &status);

                  for (j = 0; j < faces_in_group; j++)
                     if (iBuf[j] != face_splittings[i][j])
                     {
                        v = shared_faces[group_faces[j]] -> GetVertices();
                        for (k = 0; k < 3; k++)
                           if (refined_edge[iBuf[j]][k] == 1 &&
                               refined_edge[face_splittings[i][j]][k] == 0)
                           {
                              ii = v_to_v(v[k], v[(k+1)%3]);
                              if (middle[ii] == -1)
                              {
                                 need_refinement = 1;
                                 middle[ii] = NumOfVertices++;
                                 for (c = 0; c < 3; c++)
                                    coord[c] = 0.5 * (vertices[v[k]](c) +
                                                      vertices[v[(k+1)%3]](c));
                                 Vertex V(coord, Dim);
                                 vertices.Append(V);
                              }
                           }
                     }
               }
            }

            ii = need_refinement;
            MPI_Allreduce(&ii, &need_refinement, 1, MPI_INT, MPI_LOR, MyComm);
         }
      }
      while (need_refinement == 1);

      for (i = 0; i < GetNGroups()-1; i++)
         delete [] face_splittings[i];
      delete [] face_splittings;

      delete [] iBuf;

      // 5. Update the boundary elements.
      do {
         need_refinement = 0;
         for (i = 0; i < NumOfBdrElements; i++)
            if (boundary[i]->NeedRefinement(v_to_v, middle))
            {
               need_refinement = 1;
               Bisection(i, v_to_v, middle);
            }
      }
      while (need_refinement == 1);

      // 5a. Update the groups after refinement.
      RefineGroups(v_to_v, middle);

      // 6. Un-mark the Pf elements.
      int refinement_edges[2], type, flag;
      for (i = 0; i < NumOfElements; i++)
      {
         Element *El = elements[i];
         while (El->GetType() == Element::BISECTED)
            El = ((BisectedElement *) El)->FirstChild;
         ((Tetrahedron *) El)->ParseRefinementFlag (refinement_edges,
                                                    type, flag);
         if (type == Tetrahedron::TYPE_PF)
            ((Tetrahedron *) El) ->
               CreateRefinementFlag (refinement_edges, Tetrahedron::TYPE_PU, flag);
      }

      NumOfBdrElements = boundary.Size();

      // 7. Free the allocated memory.
      delete [] middle;

#ifdef MFEM_DEBUG
      CheckElementOrientation();
#endif

      if (el_to_edge != NULL)
      {
         if (WantTwoLevelState)
         {
            c_el_to_edge = el_to_edge;
            f_el_to_edge = new Table (NumOfElements, 12); // 12 for hexs
            c_bel_to_edge = bel_to_edge;
            bel_to_edge = NULL;
            NumOfEdges = GetElementToEdgeTable(*f_el_to_edge, be_to_edge);
            el_to_edge = f_el_to_edge;
            f_bel_to_edge = bel_to_edge;
         }
         else
            NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      }

      if (el_to_face != NULL)
      {
         GetElementToFaceTable();
         GenerateFaces();
      }

      if (WantTwoLevelState) {
         f_NumOfVertices    = NumOfVertices;
         f_NumOfElements    = NumOfElements;
         f_NumOfBdrElements = NumOfBdrElements;
      }
   } //  'if (Dim == 3)'


   if (Dim == 2)
   {
      int i, j, ind, nedges;
      Array<int> v;

      if (WantTwoLevelState) {
         c_NumOfVertices    = NumOfVertices;
         c_NumOfElements    = NumOfElements;
         c_NumOfBdrElements = NumOfBdrElements;
      }

      int uniform_refinement = 0;
      if (type < 0) {
         type = -type;
         uniform_refinement = 1;
      }

      // 1. Get table of vertex to vertex connections.
      DSTable v_to_v(NumOfVertices);
      GetVertexToVertexTable(v_to_v);

      // 2. Get edge to element connections in arrays edge1 and edge2
      nedges = v_to_v.NumberOfEntries();
      int *edge1  = new int[nedges];
      int *edge2  = new int[nedges];
      int *middle = new int[nedges];

      for(i=0; i<nedges; i++)
         edge1[i] = edge2[i] = middle[i] = -1;

      for(i=0; i<NumOfElements; i++){
         elements[i]->GetVertices( v);
         for(j=1; j<v.Size(); j++){
            ind = v_to_v(v[j-1], v[j]);
            (edge1[ind] == -1) ? (edge1[ind] = i) : (edge2[ind] = i);
         }
         ind = v_to_v(v[0], v[v.Size()-1]);
         (edge1[ind] == -1) ? (edge1[ind] = i) : (edge2[ind] = i);
      }

      // 3. Do the red refinement.
      for(i=0; i<marked_el.Size(); i++)
         RedRefinement(marked_el[i], v_to_v, edge1, edge2, middle);

      if (WantTwoLevelState) {
         RefinedElement::State = RefinedElement::FINE;
         State = Mesh::TWO_LEVEL_FINE;
      }

      // 4. Do the green refinement (to get conforming mesh).
      int need_refinement;
      int refined_edge[5][3] = {{0, 0, 0},
                                {1, 0, 0},
                                {1, 1, 0},
                                {1, 0, 1},
                                {1, 1, 1}};
      int edges_in_group, max_edges_in_group = 0;
      // edge_splittings identify how the shared edges have been split
      int **edge_splittings = new int*[GetNGroups()-1];
      for (i = 0; i < GetNGroups()-1; i++) {
         edges_in_group = GroupNEdges(i+1);
         edge_splittings[i] = new int[edges_in_group];
         if (edges_in_group > max_edges_in_group)
            max_edges_in_group = edges_in_group;
      }
      int neighbor, *iBuf = new int[max_edges_in_group];

      do {
         need_refinement = 0;
         for(i=0; i<nedges; i++)
            if (middle[i] != -1 && edge1[i] != -1) {
               need_refinement = 1;
               GreenRefinement(edge1[i], v_to_v, edge1, edge2, middle);
            }

         if (uniform_refinement)
            continue;

         Array<int> group_edges;
         int *v, ii,j, k, c;
         double coord[3];

         MPI_Request request;
         MPI_Status  status;

         // if the mesh is locally conforming start making it globally conforming
         if (need_refinement == 0)
         {
            MPI_Barrier(MyComm);

            //==== (a) send the type of interface splitting ========
            for (i = 0; i < GetNGroups()-1; i++) {
               group_sedge.GetRow(i, group_edges);
               edges_in_group = group_edges.Size();
               // it is enough to communicate through the edges
               if (edges_in_group != 0) {
                  for (j = 0; j < edges_in_group; j++)
                     edge_splittings[i][j] = GetEdgeSplittings(shared_edges[group_edges[j]], v_to_v, middle);
                  j = group_lproc.GetI()[i+1];
                  if (group_lproc.GetJ()[j] == 0)
                     neighbor = lproc_proc[group_lproc.GetJ()[j+1]];
                  else
                     neighbor = lproc_proc[group_lproc.GetJ()[j]];
                  MPI_Isend (edge_splittings[i], edges_in_group, MPI_INT,
                             neighbor, 0, MyComm, &request);
               }
            }

            //==== (b) receive the type of interface splitting =====
            for (i = 0; i < GetNGroups()-1; i++)
            {
               group_sedge.GetRow(i, group_edges);
               edges_in_group = group_edges.Size();
               if (edges_in_group != 0)
               {
                  j = group_lproc.GetI()[i+1];
                  if (group_lproc.GetJ()[j] == 0)
                     neighbor = lproc_proc[group_lproc.GetJ()[j+1]];
                  else
                     neighbor = lproc_proc[group_lproc.GetJ()[j]];
                  MPI_Recv(iBuf, edges_in_group, MPI_INT, neighbor,
                           MPI_ANY_TAG, MyComm, &status);

                  for (j = 0; j < edges_in_group; j++)
                     if (iBuf[j] != edge_splittings[i][j])
                     {
                        v = shared_edges[group_edges[j]] -> GetVertices();
                        for (k = 0; k < 3; k++)
                           if (refined_edge[iBuf[j]][k] == 1 &&
                               refined_edge[edge_splittings[i][j]][k] == 0)
                           {
                              ii = v_to_v(v[k], v[(k+1)%3]);
                              if (middle[ii] == -1)
                              {
                                 need_refinement = 1;
                                 middle[ii] = NumOfVertices++;
                                 for (c = 0; c < 3; c++)
                                    coord[c] = 0.5 * (vertices[v[k]](c) +
                                                      vertices[v[(k+1)%3]](c));
                                 Vertex V(coord, Dim);
                                 // TODO: (Dim == 2) here!?
                                 vertices.Append(V);
                              }
                           }
                     }
               }
            }

            ii = need_refinement;
            MPI_Allreduce (&ii, &need_refinement, 1, MPI_INT, MPI_LOR, MyComm);
         }
      } while (need_refinement == 1);

      for (i = 0; i < GetNGroups()-1; i++)
         delete [] edge_splittings[i];
      delete [] edge_splittings;

      delete [] iBuf;

      // 5. Update the boundary elements.
      int v1[2], v2[2], bisect, temp;
      temp = NumOfBdrElements;
      for(i=0; i<temp; i++){
         boundary[i]->GetVertices(v);
         bisect = v_to_v(v[0],v[1]);
         if (middle[bisect] != -1){  // the element was refined (needs updating)
            if (boundary[i]->GetType() == Element::SEGMENT){
               v1[0] =           v[0]; v1[1] = middle[bisect];
               v2[0] = middle[bisect]; v2[1] =           v[1];

               if (WantTwoLevelState)
               {
                  boundary.Append( new Segment(v2, boundary[i]->GetAttribute()) );
                  BisectedElement *aux = BEMemory.Alloc();
                  aux -> SetCoarseElem (boundary[i]);
                  aux->FirstChild = new Segment(v1, boundary[i]->GetAttribute());
                  aux->SecondChild = NumOfBdrElements;
                  boundary[i] = aux;
                  NumOfBdrElements ++;
               }
               else
               {
                  boundary[i]->SetVertices( v1 );
                  boundary.Append( new Segment(v2, boundary[i]->GetAttribute()) );
               }
            }
            else
               cerr << "Only bisection of segment is implemented for bdr elem.\n";
         }
      }
      NumOfBdrElements = boundary.Size();

      // 5a. Update the groups after refinement.
      RefineGroups(v_to_v, middle);

      // 6. Free the allocated memory.
      delete [] edge1;
      delete [] edge2;
      delete [] middle;

#ifdef MFEM_DEBUG
      CheckElementOrientation();
#endif

      if (el_to_edge != NULL)
      {
         if (WantTwoLevelState)
         {
            c_el_to_edge = el_to_edge;
            f_el_to_edge = new Table (NumOfElements, 3); // triangles
            // need to save  'be_to_el' -- NOT saved for now
            NumOfEdges = GetElementToEdgeTable(*f_el_to_edge, be_to_edge);
            el_to_edge = f_el_to_edge;
         }
         else
            NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      }

   } //  'if (Dim == 2)'
}

void ParMesh::RefineGroups(const DSTable &v_to_v, int *middle)
{
   int i, attr, newv[3], ind, f_ind;
   Array<int> v;

   int group;
   Array<int> group_verts, group_edges, group_faces;

   STable3D *faces_tbl = GetElementToFaceTable(1);

   // To update the groups after a refinement, we observe that:
   // - every (new and old) vertex, edge and face belongs to exactly one group
   // - the refinement does not create new groups
   // - a new vertex appears only as the middle of a refined edge
   // - a face can be refined 2, 3 or 4 times producing new edges and faces

   int *I_group_svert, *J_group_svert;
   int *I_group_sedge, *J_group_sedge;
   int *I_group_sface, *J_group_sface;

   I_group_svert = new int[GetNGroups()+1];
   I_group_sedge = new int[GetNGroups()+1];
   I_group_sface = new int[GetNGroups()+1];

   I_group_svert[0] = I_group_svert[1] = 0;
   I_group_sedge[0] = I_group_sedge[1] = 0;
   I_group_sface[0] = I_group_sface[1] = 0;

   // overestimate the size of the J arrays
   J_group_svert = new int[group_svert.Size_of_connections()
                           + group_sedge.Size_of_connections()];
   J_group_sedge = new int[2*group_sedge.Size_of_connections()
                           + 3*group_sface.Size_of_connections()];
   J_group_sface = new int[4*group_sface.Size_of_connections()];

   for (group = 0; group < GetNGroups()-1; group++)
   {
      // Get the group shared objects
      group_svert.GetRow(group, group_verts);
      group_sedge.GetRow(group, group_edges);
      group_sface.GetRow(group, group_faces);

      // Check which edges have been refined
      for(i = 0; i < group_sedge.RowSize(group); i++)
      {
         shared_edges[group_edges[i]] -> GetVertices(v);
         ind = middle[v_to_v(v[0], v[1])];
         if (ind != -1)
         {
            // add a vertex
            group_verts.Append(svert_lvert.Append(ind)-1);
            // update the edges
            attr = shared_edges[group_edges[i]]->GetAttribute();
            shared_edges.Append(new Segment(v[1],ind,attr));
            group_edges.Append(sedge_ledge.Append(-1)-1);
            newv[0] = v[0]; newv[1] = ind;
            shared_edges[group_edges[i]] -> SetVertices(newv);
         }
      }

      // Check which faces have been refined
      for (i = 0; i < group_sface.RowSize(group); i++)
      {
         shared_faces[group_faces[i]] -> GetVertices(v);
         ind = middle[v_to_v(v[0], v[1])];
         if (ind != -1)
         {
            attr = shared_faces[group_faces[i]]->GetAttribute();
            // add the refinement edge
            shared_edges.Append(new Segment(v[2],ind,attr));
            group_edges.Append(sedge_ledge.Append(-1)-1);
            // add a face
            newv[0] = v[2]; newv[1] = v[0]; newv[2] = ind;
            shared_faces[group_faces[i]] -> SetVertices(newv);
            f_ind = group_faces.Size();
            shared_faces.Append(new Triangle(v[1],v[2],ind,attr));
            group_faces.Append(sface_lface.Append(-1)-1);

            // check if the left face has also been refined
            shared_faces[group_faces[i]] -> GetVertices(v);
            ind = middle[v_to_v(v[0], v[1])];
            if (ind != -1)
            {
               // add the refinement edge
               shared_edges.Append(new Segment(v[2],ind,attr));
               group_edges.Append(sedge_ledge.Append(-1)-1);
               // add a face
               newv[0] = v[2]; newv[1] = v[0]; newv[2] = ind;
               shared_faces[group_faces[i]] -> SetVertices(newv);
               shared_faces.Append(new Triangle(v[1],v[2],ind,attr));
               group_faces.Append(sface_lface.Append(-1)-1);
            }

            // check if the right face has also been refined
            shared_faces[group_faces[f_ind]] -> GetVertices(v);
            ind = middle[v_to_v(v[0], v[1])];
            if (ind != -1)
            {
               // add the refinement edge
               shared_edges.Append(new Segment(v[2],ind,attr));
               group_edges.Append(sedge_ledge.Append(-1)-1);
               // add a face
               newv[0] = v[2]; newv[1] = v[0]; newv[2] = ind;
               shared_faces[group_faces[f_ind]] -> SetVertices(newv);
               shared_faces.Append(new Triangle(v[1],v[2],ind,attr));
               group_faces.Append(sface_lface.Append(-1)-1);
            }
         }
      }

      I_group_svert[group+1] = I_group_svert[group] + group_verts.Size();
      I_group_sedge[group+1] = I_group_sedge[group] + group_edges.Size();
      I_group_sface[group+1] = I_group_sface[group] + group_faces.Size();

      int *J;
      J = J_group_svert+I_group_svert[group];
      for (i = 0; i < group_verts.Size(); i++)
         J[i] = group_verts[i];
      J = J_group_sedge+I_group_sedge[group];
      for (i = 0; i < group_edges.Size(); i++)
         J[i] = group_edges[i];
      J = J_group_sface+I_group_sface[group];
      for (i = 0; i < group_faces.Size(); i++)
         J[i] = group_faces[i];
   }

   // Fix the local numbers of shared edges and faces
   DSTable new_v_to_v(NumOfVertices);
   GetVertexToVertexTable(new_v_to_v);
   for (i = 0; i < shared_edges.Size(); i++)
   {
      shared_edges[i]->GetVertices(v);
      sedge_ledge[i] = new_v_to_v(v[0], v[1]);
   }
   for (i = 0; i < shared_faces.Size(); i++)
   {
      shared_faces[i]->GetVertices(v);
      sface_lface[i] = (*faces_tbl)(v[0], v[1], v[2]);
   }

   group_svert.Recreate(I_group_svert, J_group_svert);
   group_sedge.Recreate(I_group_sedge, J_group_sedge);
   group_sface.Recreate(I_group_sface, J_group_sface);

   delete faces_tbl;
}

void ParMesh::HexUniformRefinement()
{
   int oedge = NumOfVertices;
   int oface = oedge + NumOfEdges;
   int oelem = oface + NumOfFaces;

   int i;
   int *v, *e, *f;
   int vv[4];

   DSTable v_to_v(NumOfVertices);
   GetVertexToVertexTable(v_to_v);
   STable3D *faces_tbl = GetElementToFaceTable(1);

   vertices.SetSize(oelem + NumOfElements);
   for (i = 0; i < NumOfElements; i++)
   {
      v = elements[i]->GetVertices();

      AverageVertices(v, 8, oelem+i);

      f = el_to_face->GetRow(i);

      vv[0] = v[3], vv[1] = v[2], vv[2] = v[1], vv[3] = v[0];
      AverageVertices (vv, 4, oface+f[0]);
      vv[0] = v[0], vv[1] = v[1], vv[2] = v[5], vv[3] = v[4];
      AverageVertices (vv, 4, oface+f[1]);
      vv[0] = v[1], vv[1] = v[2], vv[2] = v[6], vv[3] = v[5];
      AverageVertices (vv, 4, oface+f[2]);
      vv[0] = v[2], vv[1] = v[3], vv[2] = v[7], vv[3] = v[6];
      AverageVertices (vv, 4, oface+f[3]);
      vv[0] = v[3], vv[1] = v[0], vv[2] = v[4], vv[3] = v[7];
      AverageVertices (vv, 4, oface+f[4]);
      vv[0] = v[4], vv[1] = v[5], vv[2] = v[6], vv[3] = v[7];
      AverageVertices (vv, 4, oface+f[5]);

      e = el_to_edge -> GetRow(i);

      vv[0] = v[0], vv[1] = v[1]; AverageVertices (vv, 2, oedge+e[0]);
      vv[0] = v[1], vv[1] = v[2]; AverageVertices (vv, 2, oedge+e[1]);
      vv[0] = v[2], vv[1] = v[3]; AverageVertices (vv, 2, oedge+e[2]);
      vv[0] = v[3], vv[1] = v[0]; AverageVertices (vv, 2, oedge+e[3]);
      vv[0] = v[4], vv[1] = v[5]; AverageVertices (vv, 2, oedge+e[4]);
      vv[0] = v[5], vv[1] = v[6]; AverageVertices (vv, 2, oedge+e[5]);
      vv[0] = v[6], vv[1] = v[7]; AverageVertices (vv, 2, oedge+e[6]);
      vv[0] = v[7], vv[1] = v[4]; AverageVertices (vv, 2, oedge+e[7]);
      vv[0] = v[0], vv[1] = v[4]; AverageVertices (vv, 2, oedge+e[8]);
      vv[0] = v[1], vv[1] = v[5]; AverageVertices (vv, 2, oedge+e[9]);
      vv[0] = v[2], vv[1] = v[6]; AverageVertices (vv, 2, oedge+e[10]);
      vv[0] = v[3], vv[1] = v[7]; AverageVertices (vv, 2, oedge+e[11]);
   }

   int attr;
   elements.SetSize (8 * NumOfElements);
   for (i = NumOfElements - 1; i >= 0 ; i--) {
      attr = elements[i] -> GetAttribute();
      v = elements[i] -> GetVertices();
      e = el_to_edge -> GetRow(i);
      f = el_to_face -> GetRow(i);

      elements[8*i+1] = new Hexahedron (oedge+e[0], v[1], oedge+e[1], oface+f[0],
                                        oface+f[1], oedge+e[9], oface+f[2], oelem+i,
                                        attr);
      elements[8*i+2] = new Hexahedron (oface+f[0], oedge+e[1], v[2], oedge+e[2],
                                        oelem+i, oface+f[2], oedge+e[10], oface+f[3],
                                        attr);
      elements[8*i+3] = new Hexahedron (oedge+e[3], oface+f[0], oedge+e[2], v[3],
                                        oface+f[4], oelem+i, oface+f[3], oedge+e[11],
                                        attr);
      elements[8*i+4] = new Hexahedron (oedge+e[8], oface+f[1], oelem+i, oface+f[4],
                                        v[4], oedge+e[4], oface+f[5], oedge+e[7],
                                        attr);
      elements[8*i+5] = new Hexahedron (oface+f[1], oedge+e[9], oface+f[2], oelem+i,
                                        oedge+e[4], v[5], oedge+e[5], oface+f[5],
                                        attr);
      elements[8*i+6] = new Hexahedron (oelem+i, oface+f[2], oedge+e[10], oface+f[3],
                                        oface+f[5], oedge+e[5], v[6], oedge+e[6],
                                        attr);
      elements[8*i+7] = new Hexahedron (oface+f[4], oelem+i, oface+f[3], oedge+e[11],
                                        oedge+e[7], oface+f[5], oedge+e[6], v[7],
                                        attr);

      elements[8*i] = elements[i];
      v[1] = oedge+e[0];
      v[2] = oface+f[0];
      v[3] = oedge+e[3];
      v[4] = oedge+e[8];
      v[5] = oface+f[1];
      v[6] = oelem+i;
      v[7] = oface+f[4];
   }

   boundary.SetSize (4 * NumOfBdrElements);
   for (i = NumOfBdrElements - 1; i >= 0 ; i--)
   {
      attr = boundary[i] -> GetAttribute();
      v = boundary[i] -> GetVertices();
      e = bel_to_edge -> GetRow(i);
      f = & be_to_face[i];

      boundary[4*i+1] = new Quadrilateral (oedge+e[0], v[1], oedge+e[1],
                                           oface+f[0], attr);
      boundary[4*i+2] = new Quadrilateral (oface+f[0], oedge+e[1], v[2],
                                           oedge+e[2], attr);
      boundary[4*i+3] = new Quadrilateral (oedge+e[3], oface+f[0], oedge+e[2],
                                           v[3], attr);

      boundary[4*i] = boundary[i];
      v[1] = oedge+e[0];
      v[2] = oface+f[0];
      v[3] = oedge+e[3];
   }

   NumOfVertices    = oelem + NumOfElements;
   NumOfElements    = 8 * NumOfElements;
   NumOfBdrElements = 4 * NumOfBdrElements;

   // update the groups
   {
      int i, attr, newv[4], ind, m[5];
      Array<int> v;

      int group;
      Array<int> group_verts, group_edges, group_faces;

      int *I_group_svert, *J_group_svert;
      int *I_group_sedge, *J_group_sedge;
      int *I_group_sface, *J_group_sface;

      I_group_svert = new int[GetNGroups()+1];
      I_group_sedge = new int[GetNGroups()+1];
      I_group_sface = new int[GetNGroups()+1];

      I_group_svert[0] = I_group_svert[1] = 0;
      I_group_sedge[0] = I_group_sedge[1] = 0;
      I_group_sface[0] = I_group_sface[1] = 0;

      // compute the size of the J arrays
      J_group_svert = new int[group_svert.Size_of_connections()
                              + group_sedge.Size_of_connections()
                              + group_sface.Size_of_connections()];
      J_group_sedge = new int[2*group_sedge.Size_of_connections()
                              + 4*group_sface.Size_of_connections()];
      J_group_sface = new int[4*group_sface.Size_of_connections()];

      for (group = 0; group < GetNGroups()-1; group++)
      {
         // Get the group shared objects
         group_svert.GetRow(group, group_verts);
         group_sedge.GetRow(group, group_edges);
         group_sface.GetRow(group, group_faces);

         // Process the edges that have been refined
         for (i = 0; i < group_sedge.RowSize(group); i++)
         {
            shared_edges[group_edges[i]] -> GetVertices(v);
            ind = oedge+v_to_v(v[0],v[1]);
            // add a vertex
            group_verts.Append(svert_lvert.Append(ind)-1);
            // update the edges
            attr = shared_edges[group_edges[i]]->GetAttribute();
            shared_edges.Append(new Segment(v[1],ind,attr));
            group_edges.Append(sedge_ledge.Append(-1)-1);
            newv[0] = v[0]; newv[1] = ind;
            shared_edges[group_edges[i]] -> SetVertices(newv);
         }

         // Process the faces that have been refined
         for (i = 0; i < group_sface.RowSize(group); i++)
         {
            shared_faces[group_faces[i]]->GetVertices(v);
            attr = shared_faces[group_faces[i]]->GetAttribute();
            // add the refinement edges
            m[0] = oface+(*faces_tbl)(v[0], v[1], v[2], v[3]);
            m[1] = oedge+v_to_v(v[0],v[1]);
            m[2] = oedge+v_to_v(v[1],v[2]);
            m[3] = oedge+v_to_v(v[2],v[3]);
            m[4] = oedge+v_to_v(v[3],v[0]);
            shared_edges.Append(new Segment(m[1],m[0],attr));
            group_edges.Append(sedge_ledge.Append(-1)-1);
            shared_edges.Append(new Segment(m[2],m[0],attr));
            group_edges.Append(sedge_ledge.Append(-1)-1);
            shared_edges.Append(new Segment(m[3],m[0],attr));
            group_edges.Append(sedge_ledge.Append(-1)-1);
            shared_edges.Append(new Segment(m[4],m[0],attr));
            group_edges.Append(sedge_ledge.Append(-1)-1);
            // update faces
            newv[0] = v[0]; newv[1] = m[1]; newv[2] = m[0]; newv[3] = m[4];
            shared_faces[group_faces[i]] -> SetVertices(newv);
            shared_faces.Append(new Quadrilateral(m[1],v[1],m[2],m[0],attr));
            group_faces.Append(sface_lface.Append(-1)-1);
            shared_faces.Append(new Quadrilateral(m[0],m[2],v[2],m[3],attr));
            group_faces.Append(sface_lface.Append(-1)-1);
            shared_faces.Append(new Quadrilateral(m[4],m[0],m[3],v[3],attr));
            group_faces.Append(sface_lface.Append(-1)-1);
         }

         I_group_svert[group+1] = I_group_svert[group] + group_verts.Size();
         I_group_sedge[group+1] = I_group_sedge[group] + group_edges.Size();
         I_group_sface[group+1] = I_group_sface[group] + group_faces.Size();

         int *J;
         J = J_group_svert+I_group_svert[group];
         for (i = 0; i < group_verts.Size(); i++)
            J[i] = group_verts[i];
         J = J_group_sedge+I_group_sedge[group];
         for (i = 0; i < group_edges.Size(); i++)
            J[i] = group_edges[i];
         J = J_group_sface+I_group_sface[group];
         for (i = 0; i < group_faces.Size(); i++)
            J[i] = group_faces[i];
      }

      // Fix the local numbers of shared edges and faces
      DSTable new_v_to_v(NumOfVertices);
      GetVertexToVertexTable(new_v_to_v);
      for (i = 0; i < shared_edges.Size(); i++)
      {
         shared_edges[i]->GetVertices(v);
         sedge_ledge[i] = new_v_to_v(v[0], v[1]);
      }

      delete faces_tbl;
      faces_tbl = GetElementToFaceTable(1);
      for (i = 0; i < shared_faces.Size(); i++)
      {
         shared_faces[i]->GetVertices(v);
         sface_lface[i] = (*faces_tbl)(v[0], v[1], v[2], v[3]);
      }
      delete faces_tbl;

      group_svert.Recreate(I_group_svert,J_group_svert);
      group_sedge.Recreate(I_group_sedge,J_group_sedge);
      group_sface.Recreate(I_group_sface,J_group_sface);
   }

   //  GetElementToFaceTable();
   GenerateFaces();
   CheckBdrElementOrientation();

   NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
}

void ParMesh::Print (ostream &out) const
{
   if (Dim == 3 && meshgen == 1)
   {
      int i, j, nv;
      const int *ind;

      out << "NETGEN_Neutral_Format\n";
      // print the vertices
      out << NumOfVertices << '\n';
      for(i=0; i<NumOfVertices; i++)
      {
         for(j=0; j<Dim; j++)
            out << " " << vertices[i](j);
         out << '\n';
      }

      // print the elements
      out << NumOfElements << '\n';
      for(i=0; i<NumOfElements; i++)
      {
         nv = elements[i] -> GetNVertices ();
         ind = elements[i] -> GetVertices ();
         out << elements[i] -> GetAttribute();
         for(j=0; j<nv; j++)
            out << " " << ind[j]+1;
         out << '\n';
      }

      // print the boundary + shared faces information
      out << NumOfBdrElements + shared_faces.Size() << '\n';
      // boundary
      for(i=0; i< NumOfBdrElements; i++)
      {
         nv = boundary[i] -> GetNVertices ();
         ind = boundary[i] -> GetVertices ();
         out << boundary[i] -> GetAttribute();
         for(j=0; j<nv; j++)
            out << " " << ind[j]+1;
         out << '\n';
      }
      // shared faces
      for(i=0; i<shared_faces.Size() ; i++)
      {
         nv = shared_faces[i] -> GetNVertices ();
         ind = shared_faces[i] -> GetVertices ();
         out << shared_faces[i] -> GetAttribute();
         for(j=0; j<nv; j++)
            out << " " << ind[j]+1;
         out << '\n';
      }
   }

   if (Dim == 3 && meshgen == 2)
   {
      int i, j, nv;
      const int *ind;

      out << "TrueGrid" << endl
          << "1 " << NumOfVertices << " " << NumOfElements << " 0 0 0 0 0 0 0" << endl
          << "0 0 0 1 0 0 0 0 0 0 0" << endl
          << "0 0 " << NumOfBdrElements+shared_faces.Size()
          << " 0 0 0 0 0 0 0 0 0 0 0 0 0" << endl
          << "0.0 0.0 0.0 0 0 0.0 0.0 0 0.0" << endl
          << "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0" << endl;

      // print the vertices
      for (i = 0; i < NumOfVertices; i++)
         out << i+1 << " 0.0 " << vertices[i](0) << " " << vertices[i](1)
             << " " << vertices[i](2) << " 0.0 " << endl;

      // print the elements
      for (i = 0; i < NumOfElements; i++) {
         nv = elements[i] -> GetNVertices ();
         ind = elements[i] -> GetVertices ();
         out << i+1 << " " << elements[i] -> GetAttribute();
         for(j = 0; j < nv; j++)
            out << " " << ind[j]+1;
         out << endl;
      }

      // print the boundary information
      for(i=0; i< NumOfBdrElements; i++) {
         nv = boundary[i] -> GetNVertices ();
         ind = boundary[i] -> GetVertices ();
         out << boundary[i] -> GetAttribute();
         for(j=0; j<nv; j++)
            out << " " << ind[j]+1;
         out << " 1.0 1.0 1.0 1.0" << endl;
      }

      // print the shared faces information
      for(i=0; i<shared_faces.Size() ; i++)
      {
         nv = shared_faces[i] -> GetNVertices ();
         ind = shared_faces[i] -> GetVertices ();
         out << shared_faces[i] -> GetAttribute();
         for(j=0; j<nv; j++)
            out << " " << ind[j]+1;
         out << " 1.0 1.0 1.0 1.0" << endl;
      }
   }

   if (Dim == 2)
   {
      int i, j, attr;
      Array<int> v;

      out << "areamesh2" << endl << endl;

      // print the boundary + shared edges information
      out << NumOfBdrElements + shared_edges.Size() << endl;
      // boundary
      for (i = 0; i < NumOfBdrElements; i++)
      {
         attr = boundary[i]->GetAttribute();
         boundary[i]->GetVertices(v);
         out << attr << "     ";
         for (j = 0; j < v.Size(); j++)
            out << v[j] + 1 << "   ";
         out << endl;
      }
      // shared edges
      for (i = 0; i < shared_edges.Size(); i++)
      {
         attr = shared_edges[i]->GetAttribute();
         shared_edges[i]->GetVertices(v);
         out << attr << "     ";
         for (j = 0; j < v.Size(); j++)
            out << v[j] + 1 << "   ";
         out << endl;
      }

      // print the elements
      out << NumOfElements << endl;
      for (i = 0; i < NumOfElements; i++)
      {
         attr = elements[i]->GetAttribute();
         elements[i]->GetVertices(v);

         out << attr << "   ";
         if ((j = GetElementType (i)) == Element::TRIANGLE)
            out << 3 << "   ";
         else
            if (j == Element::QUADRILATERAL)
               out << 4 << "   ";
            else
               if (j == Element::SEGMENT)
                  out << 2 << "   ";
         for (j = 0; j < v.Size(); j++)
            out << v[j] + 1 << "  ";
         out << endl;
      }

      // print the vertices
      out << NumOfVertices << endl;
      for (i = 0; i < NumOfVertices; i++)
      {
         for (j = 0; j < Dim; j++)
            out << vertices[i](j) << " ";
         out << endl;
      }
   }
}

void ParMesh::PrintAsOne (ostream &out)
{
   if (Dim == 3 && meshgen == 1)
   {
      int i, j, k, nv, ne, p;
      const int *ind, *v;
      MPI_Status status;
      Array<double> vert;
      Array<int> ints;

      if (MyRank == 0)
      {
         out << "NETGEN_Neutral_Format\n";
         // print the vertices
         ne = NumOfVertices;
         MPI_Reduce (&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         out << nv << '\n';
         for(i=0; i<NumOfVertices; i++)
         {
            for(j=0; j<Dim; j++)
               out << " " << vertices[i](j);
            out << '\n';
         }
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv (&nv, 1, MPI_INT, p, 444, MyComm, &status);
            vert.SetSize (Dim*nv);
            MPI_Recv (&vert[0], Dim*nv, MPI_DOUBLE, p, 445, MyComm, &status);
            for(i = 0; i < nv; i++)
            {
               for(j=0; j<Dim; j++)
                  out << " " << vert[Dim*i+j];
               out << '\n';
            }
         }

         // print the elements
         nv = NumOfElements;
         MPI_Reduce (&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         out << ne << '\n';
         for(i=0; i<NumOfElements; i++)
         {
            nv = elements[i] -> GetNVertices ();
            ind = elements[i] -> GetVertices ();
            out << 1;
            for(j=0; j<nv; j++)
               out << " " << ind[j]+1;
            out << '\n';
         }
         k = NumOfVertices;
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv (&nv, 1, MPI_INT, p, 444, MyComm, &status);
            MPI_Recv (&ne, 1, MPI_INT, p, 446, MyComm, &status);
            ints.SetSize (4*ne);
            MPI_Recv (&ints[0], 4*ne, MPI_INT, p, 447, MyComm, &status);
            for(i = 0; i < ne; i++)
            {
               out << p+1;
               for(j=0; j<4; j++)
                  out << " " << k+ints[i*4+j]+1;
               out << '\n';
            }
            k += nv;
         }
         // print the boundary + shared faces information
         nv = NumOfBdrElements + shared_faces.Size();
         MPI_Reduce (&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         out << ne << '\n';
         // boundary
         for(i=0; i< NumOfBdrElements; i++)
         {
            nv = boundary[i] -> GetNVertices ();
            ind = boundary[i] -> GetVertices ();
            out << 1;
            for(j=0; j<nv; j++)
               out << " " << ind[j]+1;
            out << '\n';
         }
         // shared faces
         for(i=0; i< shared_faces.Size(); i++)
         {
            nv = shared_faces[i] -> GetNVertices ();
            ind = shared_faces[i] -> GetVertices ();
            out << 1;
            for(j=0; j<nv; j++)
               out << " " << ind[j]+1;
            out << '\n';
         }
         k = NumOfVertices;
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv (&nv, 1, MPI_INT, p, 444, MyComm, &status);
            MPI_Recv (&ne, 1, MPI_INT, p, 446, MyComm, &status);
            ints.SetSize (3*ne);
            MPI_Recv (&ints[0], 3*ne, MPI_INT, p, 447, MyComm, &status);
            for(i = 0; i < ne; i++)
            {
               out << p+1;
               for(j=0; j<3; j++)
                  out << " " << k+ints[i*3+j]+1;
               out << '\n';
            }
            k += nv;
         }
      }
      else
      {
         ne = NumOfVertices;
         MPI_Reduce (&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send (&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         vert.SetSize (Dim*NumOfVertices);
         for (i = 0; i < NumOfVertices; i++)
            for (j = 0; j < Dim; j++)
               vert[Dim*i+j] = vertices[i](j);
         MPI_Send (&vert[0], Dim*NumOfVertices, MPI_DOUBLE,
                   0, 445, MyComm);
         // elements
         ne = NumOfElements;
         MPI_Reduce (&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send (&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         MPI_Send (&NumOfElements, 1, MPI_INT, 0, 446, MyComm);
         ints.SetSize(NumOfElements*4);
         for (i = 0; i < NumOfElements; i++)
         {
            v = elements[i] -> GetVertices();
            for (j = 0; j < 4; j++)
               ints[4*i+j] = v[j];
         }
         MPI_Send (&ints[0], 4*NumOfElements, MPI_INT, 0, 447, MyComm);
         // boundary + shared faces
         nv = NumOfBdrElements + shared_faces.Size();
         MPI_Reduce (&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send (&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         ne = NumOfBdrElements + shared_faces.Size();
         MPI_Send (&ne, 1, MPI_INT, 0, 446, MyComm);
         ints.SetSize (3*ne);
         for (i = 0; i < NumOfBdrElements; i++)
         {
            v = boundary[i] -> GetVertices();
            for (j = 0; j < 3; j++)
               ints[3*i+j] = v[j];
         }
         for ( ; i < ne; i++)
         {
            v = shared_faces[i-NumOfBdrElements] -> GetVertices();
            for (j = 0; j < 3; j++)
               ints[3*i+j] = v[j];
         }
         MPI_Send (&ints[0], 3*ne, MPI_INT, 0, 447, MyComm);
      }
   }

   if (Dim == 3 && meshgen == 2)
   {
      int i, j, k, nv, ne, p;
      const int *ind, *v;
      MPI_Status status;
      Array<double> vert;
      Array<int> ints;

      int TG_nv, TG_ne, TG_nbe;

      if (MyRank == 0)
      {
         MPI_Reduce (&NumOfVertices, &TG_nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Reduce (&NumOfElements, &TG_ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         nv = NumOfBdrElements + shared_faces.Size();
         MPI_Reduce (&nv, &TG_nbe, 1, MPI_INT, MPI_SUM, 0, MyComm);

         out << "TrueGrid" << endl
             << "1 " << TG_nv << " " << TG_ne << " 0 0 0 0 0 0 0" << endl
             << "0 0 0 1 0 0 0 0 0 0 0" << endl
             << "0 0 " << TG_nbe << " 0 0 0 0 0 0 0 0 0 0 0 0 0" << endl
             << "0.0 0.0 0.0 0 0 0.0 0.0 0 0.0" << endl
             << "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0" << endl;

         // print the vertices
         nv = TG_nv;
         for(i=0; i<NumOfVertices; i++)
            out << i+1 << " 0.0 " << vertices[i](0) << " " << vertices[i](1)
                << " " << vertices[i](2) << " 0.0 " << endl;
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv (&nv, 1, MPI_INT, p, 444, MyComm, &status);
            vert.SetSize (Dim*nv);
            MPI_Recv (&vert[0], Dim*nv, MPI_DOUBLE, p, 445, MyComm, &status);
            for(i = 0; i < nv; i++)
               out << i+1 << " 0.0 " << vert[Dim*i] << " " << vert[Dim*i+1]
                   << " " << vert[Dim*i+2] << " 0.0 " << endl;
         }

         // print the elements
         ne = TG_ne;
         for(i=0; i<NumOfElements; i++)
         {
            nv = elements[i] -> GetNVertices ();
            ind = elements[i] -> GetVertices ();
            out << i+1 << " " << 1;
            for(j=0; j<nv; j++)
               out << " " << ind[j]+1;
            out << '\n';
         }
         k = NumOfVertices;
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv (&nv, 1, MPI_INT, p, 444, MyComm, &status);
            MPI_Recv (&ne, 1, MPI_INT, p, 446, MyComm, &status);
            ints.SetSize (8*ne);
            MPI_Recv (&ints[0], 8*ne, MPI_INT, p, 447, MyComm, &status);
            for(i = 0; i < ne; i++)
            {
               out << i+1 << " " << p+1;
               for(j=0; j<8; j++)
                  out << " " << k+ints[i*8+j]+1;
               out << '\n';
            }
            k += nv;
         }

         // print the boundary + shared faces information
         ne = TG_nbe;
         // boundary
         for(i=0; i< NumOfBdrElements; i++)
         {
            nv = boundary[i] -> GetNVertices ();
            ind = boundary[i] -> GetVertices ();
            out << 1;
            for(j=0; j<nv; j++)
               out << " " << ind[j]+1;
            out << " 1.0 1.0 1.0 1.0" << endl;
         }
         // shared faces
         for(i=0; i< shared_faces.Size(); i++)
         {
            nv = shared_faces[i] -> GetNVertices ();
            ind = shared_faces[i] -> GetVertices ();
            out << 1;
            for(j=0; j<nv; j++)
               out << " " << ind[j]+1;
            out << " 1.0 1.0 1.0 1.0" << endl;
         }
         k = NumOfVertices;
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv (&nv, 1, MPI_INT, p, 444, MyComm, &status);
            MPI_Recv (&ne, 1, MPI_INT, p, 446, MyComm, &status);
            ints.SetSize (4*ne);
            MPI_Recv (&ints[0], 4*ne, MPI_INT, p, 447, MyComm, &status);
            for(i = 0; i < ne; i++)
            {
               out << p+1;
               for(j=0; j<4; j++)
                  out << " " << k+ints[i*4+j]+1;
               out << " 1.0 1.0 1.0 1.0" << endl;
            }
            k += nv;
         }
      }
      else
      {
         ne = NumOfVertices;
         MPI_Reduce (&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send (&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         vert.SetSize (Dim*NumOfVertices);
         for (i = 0; i < NumOfVertices; i++)
            for (j = 0; j < Dim; j++)
               vert[Dim*i+j] = vertices[i](j);
         MPI_Send (&vert[0], Dim*NumOfVertices, MPI_DOUBLE,
                   0, 445, MyComm);
         // elements
         ne = NumOfElements;
         MPI_Reduce (&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send (&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         MPI_Send (&NumOfElements, 1, MPI_INT, 0, 446, MyComm);
         ints.SetSize(NumOfElements*8);
         for (i = 0; i < NumOfElements; i++)
         {
            v = elements[i] -> GetVertices();
            for (j = 0; j < 8; j++)
               ints[8*i+j] = v[j];
         }
         MPI_Send (&ints[0], 8*NumOfElements, MPI_INT, 0, 447, MyComm);
         // boundary + shared faces
         nv = NumOfBdrElements + shared_faces.Size();
         MPI_Reduce (&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send (&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         ne = NumOfBdrElements + shared_faces.Size();
         MPI_Send (&ne, 1, MPI_INT, 0, 446, MyComm);
         ints.SetSize (4*ne);
         for (i = 0; i < NumOfBdrElements; i++)
         {
            v = boundary[i] -> GetVertices();
            for (j = 0; j < 4; j++)
               ints[4*i+j] = v[j];
         }
         for ( ; i < ne; i++)
         {
            v = shared_faces[i-NumOfBdrElements] -> GetVertices();
            for (j = 0; j < 4; j++)
               ints[4*i+j] = v[j];
         }
         MPI_Send (&ints[0], 4*ne, MPI_INT, 0, 447, MyComm);
      }
   }

   if (Dim == 2)
   {
      int i, j, k, attr, nv, ne, p;
      Array<int> v;
      MPI_Status status;
      Array<double> vert;
      Array<int> ints;


      if (MyRank == 0)
      {
         out << "areamesh2" << endl << endl;

         // print the boundary + shared edges information
         nv = NumOfBdrElements + shared_edges.Size();
         MPI_Reduce (&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         out << ne << endl;
         // boundary
         for (i = 0; i < NumOfBdrElements; i++)
         {
            attr = boundary[i]->GetAttribute();
            boundary[i]->GetVertices(v);
            out << attr << "     ";
            for (j = 0; j < v.Size(); j++)
               out << v[j] + 1 << "   ";
            out << endl;
         }
         // shared edges
         for (i = 0; i < shared_edges.Size(); i++)
         {
            attr = shared_edges[i]->GetAttribute();
            shared_edges[i]->GetVertices(v);
            out << attr << "     ";
            for (j = 0; j < v.Size(); j++)
               out << v[j] + 1 << "   ";
            out << endl;
         }
         k = NumOfVertices;
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv (&nv, 1, MPI_INT, p, 444, MyComm, &status);
            MPI_Recv (&ne, 1, MPI_INT, p, 446, MyComm, &status);
            ints.SetSize (2*ne);
            MPI_Recv (&ints[0], 2*ne, MPI_INT, p, 447, MyComm, &status);
            for (i = 0; i < ne; i++) {
               out << p+1;
               for (j = 0; j < 2; j++)
                  out << " " << k+ints[i*2+j]+1;
               out << '\n';
            }
            k += nv;
         }

         // print the elements
         nv = NumOfElements;
         MPI_Reduce (&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         out << ne << '\n';
         for (i = 0; i < NumOfElements; i++)
         {
            attr = elements[i]->GetAttribute();
            elements[i]->GetVertices(v);
            out << 1 << "   " << 3 << "   ";
            for (j = 0; j < v.Size(); j++)
               out << v[j] + 1 << "  ";
            out << endl;
         }
         k = NumOfVertices;
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv (&nv, 1, MPI_INT, p, 444, MyComm, &status);
            MPI_Recv (&ne, 1, MPI_INT, p, 446, MyComm, &status);
            ints.SetSize (3*ne);
            MPI_Recv (&ints[0], 3*ne, MPI_INT, p, 447, MyComm, &status);
            for(i = 0; i < ne; i++) {
               out << p+1 << " " << 3;
               for (j = 0; j < 3; j++)
                  out << " " << k+ints[i*3+j]+1;
               out << '\n';
            }
            k += nv;
         }

         // print the vertices
         ne = NumOfVertices;
         MPI_Reduce (&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         out << nv << '\n';
         for (i = 0; i < NumOfVertices; i++) {
            for (j = 0; j < Dim; j++)
               out << vertices[i](j) << " ";
            out << endl;
         }
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv (&nv, 1, MPI_INT, p, 444, MyComm, &status);
            vert.SetSize (Dim*nv);
            MPI_Recv (&vert[0], Dim*nv, MPI_DOUBLE, p, 445, MyComm, &status);
            for(i = 0; i < nv; i++)
            {
               for(j=0; j<Dim; j++)
                  out << " " << vert[Dim*i+j];
               out << '\n';
            }
         }
      } else {
         // boundary + shared faces
         nv = NumOfBdrElements + shared_edges.Size();
         MPI_Reduce (&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send (&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         ne = NumOfBdrElements + shared_edges.Size();
         MPI_Send (&ne, 1, MPI_INT, 0, 446, MyComm);
         ints.SetSize (2*ne);
         for (i = 0; i < NumOfBdrElements; i++)
         {
            boundary[i] -> GetVertices(v);
            for (j = 0; j < 2; j++)
               ints[2*i+j] = v[j];
         }
         for ( ; i < ne; i++)
         {
            shared_edges[i-NumOfBdrElements] -> GetVertices(v);
            for (j = 0; j < 2; j++)
               ints[2*i+j] = v[j];
         }
         MPI_Send (&ints[0], 2*ne, MPI_INT, 0, 447, MyComm);
         // elements
         ne = NumOfElements;
         MPI_Reduce (&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send (&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         MPI_Send (&NumOfElements, 1, MPI_INT, 0, 446, MyComm);
         ints.SetSize(NumOfElements*3);
         for (i = 0; i < NumOfElements; i++)
         {
            elements[i] -> GetVertices(v);
            for (j = 0; j < 3; j++)
               ints[3*i+j] = v[j];
         }
         MPI_Send (&ints[0], 3*NumOfElements, MPI_INT, 0, 447, MyComm);
         // vertices
         ne = NumOfVertices;
         MPI_Reduce (&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send (&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         vert.SetSize (Dim*NumOfVertices);
         for (i = 0; i < NumOfVertices; i++)
            for (j = 0; j < Dim; j++)
               vert[Dim*i+j] = vertices[i](j);
         MPI_Send (&vert[0], Dim*NumOfVertices, MPI_DOUBLE,
                   0, 445, MyComm);
      }
   }
}

void ParMesh::PrintAsOneAndScaled (ostream &out, double sf)
{
   if (Dim == 3)
   {
      int i, j, k, nv, ne, p;
      const int *ind, *v;
      MPI_Status status;
      Array<double> vert;
      Array<int> ints;

      double cg[Dim]; // center of gravity

      if (MyRank == 0)
      {
         out << "NETGEN_Neutral_Format\n";
         // print the vertices
         ne = NumOfVertices;
         MPI_Reduce (&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         out << nv << '\n';

         for(j=0; j<Dim; j++)
            cg[j] = 0.0;
         for(i=0; i<NumOfVertices; i++)
            for(j=0; j<Dim; j++)
               cg[j] +=  vertices[i](j);
         for(j=0; j<Dim; j++)
            cg[j] /= NumOfVertices;

         for(i=0; i<NumOfVertices; i++) {
            for(j=0; j<Dim; j++)
               out << " " << sf*vertices[i](j) + (1-sf)*cg[j];
            out << '\n';
         }

         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv (&nv, 1, MPI_INT, p, 444, MyComm, &status);
            vert.SetSize (Dim*nv);
            MPI_Recv (&vert[0], Dim*nv, MPI_DOUBLE, p, 445, MyComm, &status);

            for(j=0; j<Dim; j++)
               cg[j] = 0.0;
            for(i=0; i<nv; i++)
               for(j=0; j<Dim; j++) {
                  cg[j] +=  vert[Dim*i+j];
               }
            for(j=0; j<Dim; j++)
               cg[j] /= nv;

            for(i=0; i<nv; i++) {
               for(j=0; j<Dim; j++)
                  out << " " << sf*vert[Dim*i+j] + (1-sf)*cg[j];
               out << '\n';
            }
         }

         // print the elements
         nv = NumOfElements;
         MPI_Reduce (&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         out << ne << '\n';
         for(i=0; i<NumOfElements; i++)
         {
            nv = elements[i] -> GetNVertices ();
            ind = elements[i] -> GetVertices ();
            out << 1;
            for(j=0; j<nv; j++)
               out << " " << ind[j]+1;
            out << '\n';
         }
         k = NumOfVertices;
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv (&nv, 1, MPI_INT, p, 444, MyComm, &status);
            MPI_Recv (&ne, 1, MPI_INT, p, 446, MyComm, &status);
            ints.SetSize (4*ne);
            MPI_Recv (&ints[0], 4*ne, MPI_INT, p, 447, MyComm, &status);
            for(i = 0; i < ne; i++)
            {
               out << p+1;
               for(j=0; j<4; j++)
                  out << " " << k+ints[i*4+j]+1;
               out << '\n';
            }
            k += nv;
         }
         // print the boundary + shared faces information
         nv = NumOfBdrElements + shared_faces.Size();
         MPI_Reduce (&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         out << ne << '\n';
         // boundary
         for(i=0; i< NumOfBdrElements; i++)
         {
            nv = boundary[i] -> GetNVertices ();
            ind = boundary[i] -> GetVertices ();
            out << 1;
            for(j=0; j<nv; j++)
               out << " " << ind[j]+1;
            out << '\n';
         }
         // shared faces
         for(i=0; i< shared_faces.Size(); i++)
         {
            nv = shared_faces[i] -> GetNVertices ();
            ind = shared_faces[i] -> GetVertices ();
            out << 1;
            for(j=0; j<nv; j++)
               out << " " << ind[j]+1;
            out << '\n';
         }
         k = NumOfVertices;
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv (&nv, 1, MPI_INT, p, 444, MyComm, &status);
            MPI_Recv (&ne, 1, MPI_INT, p, 446, MyComm, &status);
            ints.SetSize (3*ne);
            MPI_Recv (&ints[0], 3*ne, MPI_INT, p, 447, MyComm, &status);
            for(i = 0; i < ne; i++)
            {
               out << p+1;
               for(j=0; j<3; j++)
                  out << " " << k+ints[i*3+j]+1;
               out << '\n';
            }
            k += nv;
         }
      }
      else
      {
         ne = NumOfVertices;
         MPI_Reduce (&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send (&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         vert.SetSize (Dim*NumOfVertices);
         for (i = 0; i < NumOfVertices; i++)
            for (j = 0; j < Dim; j++)
               vert[Dim*i+j] = vertices[i](j);
         MPI_Send (&vert[0], Dim*NumOfVertices, MPI_DOUBLE,
                   0, 445, MyComm);
         // elements
         ne = NumOfElements;
         MPI_Reduce (&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send (&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         MPI_Send (&NumOfElements, 1, MPI_INT, 0, 446, MyComm);
         ints.SetSize(NumOfElements*4);
         for (i = 0; i < NumOfElements; i++)
         {
            v = elements[i] -> GetVertices();
            for (j = 0; j < 4; j++)
               ints[4*i+j] = v[j];
         }
         MPI_Send (&ints[0], 4*NumOfElements, MPI_INT, 0, 447, MyComm);
         // boundary + shared faces
         nv = NumOfBdrElements + shared_faces.Size();
         MPI_Reduce (&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send (&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         ne = NumOfBdrElements + shared_faces.Size();
         MPI_Send (&ne, 1, MPI_INT, 0, 446, MyComm);
         ints.SetSize (3*ne);
         for (i = 0; i < NumOfBdrElements; i++)
         {
            v = boundary[i] -> GetVertices();
            for (j = 0; j < 3; j++)
               ints[3*i+j] = v[j];
         }
         for ( ; i < ne; i++)
         {
            v = shared_faces[i-NumOfBdrElements] -> GetVertices();
            for (j = 0; j < 3; j++)
               ints[3*i+j] = v[j];
         }
         MPI_Send (&ints[0], 3*ne, MPI_INT, 0, 447, MyComm);
      }
   }
}

ParMesh::~ParMesh()
{
   int i;

   for (i = 0; i < shared_faces.Size(); i++)
      FreeElement (shared_faces[i]);
   for (i = 0; i < shared_edges.Size(); i++)
      FreeElement (shared_edges[i]);

   ///  Is it necessary to call the destructor for Mesh? (No?)
   ///  The destructor for Mesh is called automatically. (checked)
}

#endif
