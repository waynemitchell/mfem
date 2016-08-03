// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.


#include "../config/config.hpp"

#ifdef MFEM_USE_SIDRE

#include "sidredatacollection.hpp"

#include "../fem/fem.hpp"

#include <string>
#include <cstdio>       // for snprintf()


#include "sidre/sidre.hpp"
#ifdef MFEM_USE_MPI
  #include "spio/IOManager.hpp"
#endif

namespace mfem
{

// class SidreDataCollection implementation
SidreDataCollection::SidreDataCollection(const std::string& collection_name, asctoolkit::sidre::DataGroup* dg)
  : mfem::DataCollection(collection_name.c_str()), parent_datagroup(dg)
{
   namespace sidre = asctoolkit::sidre;

   sidre_dc_group = dg->createGroup( collection_name );

   sidre_dc_group->createViewScalar("state/cycle", 0);
   sidre_dc_group->createViewScalar("state/time", 0.);
   sidre_dc_group->createViewScalar("state/domain", myid);
   sidre_dc_group->createViewScalar("state/time_step", 0.);

   sidre_dc_group->createGroup("array_data");
}

void SidreDataCollection::SetMesh(Mesh *new_mesh)
{
   DataCollection::SetMesh(new_mesh);

   // uses conduit's mesh blueprint

   namespace sidre = asctoolkit::sidre;
   sidre::DataGroup* grp = sidre_dc_group;

   int dim = new_mesh->Dimension();
   MFEM_ASSERT(dim >=1 && dim <= 3, "invalid mesh dimension");

   // Note: The coordinates in mfem always have three components (regardless of dim)
   //       but the mesh constructor can handle packed data
   const int NUM_COORDS = dim;

   // Retrieve some mesh attributes from mesh object
   int num_vertices = new_mesh->GetNV();
   int coordset_len = NUM_COORDS * num_vertices;

   int element_size = new_mesh->GetElement(0)->GetNVertices();
   int mesh_num_elements = new_mesh->GetNE();
   int mesh_num_indices = mesh_num_elements * element_size;

   int bnd_element_size = new_mesh->GetBdrElement(0)->GetNVertices();
   int bnd_num_elements = new_mesh->GetNBE();
   int bnd_num_indices = bnd_num_elements * bnd_element_size;

   bool has_bnd_elts = (bnd_num_indices > 0);
   bool isRestart = grp->hasGroup("topologies");
   if(!isRestart)
   {
      /// Setup the mesh blueprint

      // Add coordinate set
      grp->createViewString("coordsets/mesh/type", "explicit");
      grp->createView("coordsets/mesh/values/x");
      if(dim >= 2) { grp->createView("coordsets/mesh/values/y"); }
      if(dim >= 3) { grp->createView("coordsets/mesh/values/z"); }

      // Find the element shape
      // Note: Assumes homogeneous elements, so only check the first element
      std::string eltTypeStr = getElementName( static_cast<Element::Type>( new_mesh->GetElement(0)->GetType() ) );

      // Add mesh topology
      grp->createViewString("topologies/mesh/type", "unstructured");
      grp->createViewString("topologies/mesh/elements/shape",eltTypeStr);   // <-- Note: this comes form the mesh
      grp->createView("topologies/mesh/elements/connectivity");
      grp->createViewString("topologies/mesh/coordset", "mesh");
      grp->createViewString("topologies/mesh/mfem_grid_function", "nodes");

      if (has_bnd_elts)
      {
         grp->createViewString("topologies/mesh/boundary_topology", "boundary");
      }

      // Add mesh elements material attribute field
      grp->createViewString("fields/mesh_material_attribute/association", "Element");
      grp->createView("fields/mesh_material_attribute/values");
      grp->createViewString("fields/mesh_material_attribute/topology", "mesh");

      if (has_bnd_elts)
      {
         eltTypeStr = getElementName( static_cast<Element::Type>( new_mesh->GetBdrElement(0)->GetType() ) );

         // Add mesh boundary topology
         grp->createViewString("topologies/boundary/type", "unstructured");
         grp->createViewString("topologies/boundary/elements/shape",eltTypeStr);   // <-- Note: this comes form the mesh
         grp->createView("topologies/boundary/elements/connectivity");
         grp->createViewString("topologies/boundary/coordset", "mesh");

         // Add boundary elements material attribute field
         grp->createViewString("fields/boundary_material_attribute/association", "Element");
         grp->createView("fields/boundary_material_attribute/values");
         grp->createViewString("fields/boundary_material_attribute/topology", "boundary");
      }
   }

   // NOTE: This code can be consolidated to a single 'addElements' helper function.
   // It could be called twice - once for each topology to be added.
   sidre::DataView *mesh_elements_connectivity = grp->getView("topologies/mesh/elements/connectivity");
   sidre::DataView *mesh_material_attribute_values = grp->getView("fields/mesh_material_attribute/values");

   if (!isRestart)
   {
      mesh_elements_connectivity->allocate(sidre::INT_ID,mesh_num_indices);
      mesh_material_attribute_values->allocate(sidre::INT_ID,mesh_num_elements);

      sidre::DataBuffer* coordbuf = grp->getDataStore()
                                         ->createBuffer(sidre::DOUBLE_ID, coordset_len)
                                         ->allocate();

      grp->getView("coordsets/mesh/values/x")
         ->attachBuffer(coordbuf)
         ->apply(sidre::DOUBLE_ID, num_vertices, 0, NUM_COORDS);

      if(dim >= 2)
      {
         grp->getView("coordsets/mesh/values/y")
            ->attachBuffer(coordbuf)
            ->apply(sidre::DOUBLE_ID, num_vertices, 1, NUM_COORDS);
      }

      if(dim >= 3)
      {
         grp->getView("coordsets/mesh/values/z")
             ->attachBuffer(coordbuf)
             ->apply(sidre::DOUBLE_ID, num_vertices, 2, NUM_COORDS);
      }

   }

   // Change ownership of mesh data to sidre
   double *coord_values = grp->getView("coordsets/mesh/values/x")->getBuffer()->getData();

   new_mesh->ChangeElementDataOwnership(
             mesh_elements_connectivity->getArray(),
             element_size * mesh_num_elements,
             mesh_material_attribute_values->getArray(),
             mesh_num_elements,
             isRestart);

   new_mesh->ChangeVertexDataOwnership(
             coord_values,
             dim,
             coordset_len,
             isRestart);

   if (has_bnd_elts)
   {

      sidre::DataView *bnd_elements_connectivity = grp->getView("topologies/boundary/elements/connectivity");
      sidre::DataView *bnd_material_attribute_values = grp->getView("fields/boundary_material_attribute/values");

      if (!isRestart)
      {
         bnd_elements_connectivity->allocate(sidre::INT_ID,bnd_num_indices);
         bnd_material_attribute_values->allocate(sidre::INT_ID,bnd_num_elements);
      }

      new_mesh->ChangeBoundaryElementDataOwnership(
                bnd_elements_connectivity->getArray(),
                bnd_element_size * bnd_num_elements,
                bnd_material_attribute_values->getArray(),
                bnd_num_elements,
                isRestart);
   }

   // copy mesh nodes grid function

   //  When not restart, copy data from mesh to datastore
   //  In both cases, set the mesh version to point to this
   // Remove once we directly load the mesh into the datastore
   // Note: There is likely a much better way to do this

   const FiniteElementSpace* nFes = new_mesh->GetNodalFESpace();
   int sz = nFes->GetVSize();
   double* gfData = GetFieldData("nodes", sz);

   if(! isRestart)
   {
      double* meshNodeData = new_mesh->GetNodes()->GetData();
      std::memcpy(gfData, meshNodeData, sizeof(double) * sz);
   }

   new_mesh->GetNodes()->NewDataAndSize(gfData, sz);
   RegisterField( "nodes", new_mesh->GetNodes());
}

// Note - if this function is going to be permanent, we should consolidate code between this and 'SetMesh'.
void SidreDataCollection::CopyMesh(std::string name, Mesh *new_mesh)
{
   namespace sidre = asctoolkit::sidre;

   MFEM_ASSERT(!sidre_dc_group->hasGroup(name), "Data collection already has a snapshot of mesh named " << name);

   sidre::DataGroup* grp = sidre_dc_group->createGroup(name);

   int dim = new_mesh->Dimension();
   MFEM_ASSERT(dim >=1 && dim <= 3, "invalid mesh dimension");

   // Note: The coordinates in mfem always have three components (regardless of dim)
   //       but the mesh constructor can handle packed data.
   const int NUM_COORDS = dim;

   // Add coordinate set
   int num_vertices = new_mesh->GetNV();
   int coordset_len = NUM_COORDS * num_vertices;

   grp->createViewString("coordsets/mesh/type", "explicit");
   grp->createView("coordsets/mesh/values/x");
   if(dim >= 2)
   {
      grp->createView("coordsets/mesh/values/y");
   }
   if(dim >= 3)
   {
      grp->createView("coordsets/mesh/values/z");
   }

   sidre::DataBuffer* coordbuf = grp->getDataStore()
                                    ->createBuffer(sidre::DOUBLE_ID, coordset_len)
                                    ->allocate();

   grp->getView("coordsets/mesh/values/x")
      ->attachBuffer(coordbuf)
      ->apply(sidre::DOUBLE_ID, num_vertices, 0, NUM_COORDS);

   if(dim >= 2)
   {
       grp->getView("coordsets/mesh/values/y")
          ->attachBuffer(coordbuf)
          ->apply(sidre::DOUBLE_ID, num_vertices, 1, NUM_COORDS);
   }

   if(dim >= 3)
   {
       grp->getView("coordsets/mesh/values/z")
          ->attachBuffer(coordbuf)
          ->apply(sidre::DOUBLE_ID, num_vertices, 2, NUM_COORDS);
   }

   // Copy mesh coord data to sidre.
   double *coord_values = grp->getView("coordsets/mesh/values/x")->getBuffer()->getData();

   bool zeroCopy = false;
   bool copyOnly = true;

   new_mesh->ChangeVertexDataOwnership(
             coord_values,
             dim,
             coordset_len,
             zeroCopy,
		     copyOnly);


   // Add mesh topology
   int element_size = new_mesh->GetElement(0)->GetNVertices();
   int mesh_num_elements = new_mesh->GetNE();
   int mesh_num_indices = mesh_num_elements * element_size;

   // Find the element shape
   // Note: Assumes homogeneous elements, so only check the first element
   std::string eltTypeStr = getElementName( static_cast<Element::Type>( new_mesh->GetElement(0)->GetType() ) );

   grp->createViewString("topologies/mesh/type", "unstructured");
   grp->createViewString("topologies/mesh/elements/shape",eltTypeStr);
   grp->createView("topologies/mesh/elements/connectivity");
   grp->createViewString("topologies/mesh/coordset", "mesh");
   grp->createViewString("topologies/mesh/mfem_grid_function", "nodes");

   // Add mesh elements material attribute field
   grp->createViewString("fields/mesh_material_attribute/association", "Element");
   grp->createView("fields/mesh_material_attribute/values");
   grp->createViewString("fields/mesh_material_attribute/topology", "mesh");

   sidre::DataView *mesh_elements_connectivity = grp->getView("topologies/mesh/elements/connectivity");
   sidre::DataView *mesh_material_attribute_values = grp->getView("fields/mesh_material_attribute/values");

   mesh_elements_connectivity->allocate(sidre::INT_ID,mesh_num_indices);
   mesh_material_attribute_values->allocate(sidre::INT_ID,mesh_num_elements);

   new_mesh->ChangeElementDataOwnership(
             mesh_elements_connectivity->getArray(),
             element_size * mesh_num_elements,
             mesh_material_attribute_values->getArray(),
             mesh_num_elements,
             zeroCopy,
		     copyOnly);

   // Add mesh boundary topology ( if present )
   bool has_bnd_elts = (new_mesh->GetNBE() > 0);
   if (has_bnd_elts)
   {
      int bnd_element_size = new_mesh->GetBdrElement(0)->GetNVertices();
      int bnd_num_elements = new_mesh->GetNBE();
      int bnd_num_indices = bnd_num_elements * bnd_element_size;

      eltTypeStr = getElementName( static_cast<Element::Type>( new_mesh->GetBdrElement(0)->GetType() ) );

      grp->createViewString("topologies/mesh/boundary_topology", "boundary");

      // Add mesh boundary topology
      grp->createViewString("topologies/boundary/type", "unstructured");
      grp->createViewString("topologies/boundary/elements/shape",eltTypeStr);   // <-- Note: this comes form the mesh
      grp->createView("topologies/boundary/elements/connectivity");
      grp->createViewString("topologies/boundary/coordset", "mesh");

      // Add boundary elements material attribute field
      grp->createViewString("fields/boundary_material_attribute/association", "Element");
      grp->createView("fields/boundary_material_attribute/values");
      grp->createViewString("fields/boundary_material_attribute/topology", "boundary");

      sidre::DataView *bnd_elements_connectivity = grp->getView("topologies/boundary/elements/connectivity");
      sidre::DataView *bnd_material_attribute_values = grp->getView("fields/boundary_material_attribute/values");

      bnd_elements_connectivity->allocate(sidre::INT_ID,bnd_num_indices);
      bnd_material_attribute_values->allocate(sidre::INT_ID,bnd_num_elements);

      new_mesh->ChangeBoundaryElementDataOwnership(
                bnd_elements_connectivity->getArray(),
                bnd_element_size * bnd_num_elements,
                bnd_material_attribute_values->getArray(),
                bnd_num_elements,
                zeroCopy,
    			copyOnly);
   }

    // copy mesh nodes grid function
    // Redo this to not use GetFieldData ( just make a copy ).
/*

    const FiniteElementSpace* nFes = new_mesh->GetNodalFESpace();
    int sz = nFes->GetVSize();
    double* gfData = GetFieldData("nodes", sz);

    double* meshNodeData = new_mesh->GetNodes()->GetData();
    std::memcpy(gfData, meshNodeData, sizeof(double) * sz);

    RegisterField( "nodes", new_mesh->GetNodes());
*/
}

void SidreDataCollection::Load(const std::string& path, const std::string& protocol)
{
   bool useSerial = true;

   std::cout << "Loading Sidre checkpoint: " << path
             << " using protocol: " << protocol << std::endl;

   // write out in serial if non-mpi or for debug
#ifdef MFEM_USE_MPI

   useSerial = false;
   ParMesh *par_mesh = dynamic_cast<ParMesh*>(mesh);
   if (par_mesh)
   {
       asctoolkit::spio::IOManager reader(par_mesh->GetComm());
       reader.read(sidre_dc_group, path);
   }
   else
   {
       useSerial = true;
   }

#endif

   // write out in serial for debugging, or if MPI unavailable
   if(useSerial)
   {
       sidre_dc_group->getDataStore()->load(path, protocol); //, sidre_dc_group);
   }

   SetTime( sidre_dc_group->getView("state/time")->getData<double>() );
   SetCycle( sidre_dc_group->getView("state/cycle")->getData<int>() );
   SetTimeStep( sidre_dc_group->getView("state/time_step")->getData<double>() );
}

asctoolkit::sidre::DataGroup * SidreDataCollection::ConstructRootFileGroup()
{
/*

	// quads (unstructured)
	    Node &quads_idx = index_root["quads"];
	    // state
	    quads_idx["state/cycle"] = 42;
	    quads_idx["state/time"]  = 3.1415;
	    quads_idx["state/number_of_domains"]  = 1;
	    // coords
	    quads_idx["coordsets/coords/type"]         = "explicit";
	    quads_idx["coordsets/coords/coord_system"] = "xy";
	    quads_idx["coordsets/coords/path"]         = "quads/coords";
	    // topology
	    quads_idx["topologies/mesh/type"]     = "unstructured";
	    quads_idx["topologies/mesh/coordset"] = "coords";
	    quads_idx["topologies/mesh/path"]     = "quads/topology";
	    // fields
	        // pc
	        quads_idx["fields/braid_pc/association"] = "point";
	        quads_idx["fields/braid_pc/topology"]    = "mesh";
	        quads_idx["fields/braid_pc/number_of_components"] = 1;
	        quads_idx["fields/braid_pc/mesh/path"]   = "quads/fields/braid_pc";
	        // ec
	        quads_idx["fields/radial_ec/association"] = "element";
	        quads_idx["fields/radial_ec/topology"]    = "mesh";
	        quads_idx["fields/radial_ec/number_of_components"] = 1;
	        quads_idx["fields/radial_ec/mesh/path"]   = "quads/fields/radial_ec";
*/
}

void SidreDataCollection::Save()
{
   namespace sidre = asctoolkit::sidre;
   sidre::DataGroup* grp = sidre_dc_group->getGroup("state");

   grp->getView("cycle")->setScalar(cycle);
   grp->getView("time")->setScalar(time);
   grp->getView("time_step")->setScalar(time_step);

   std::string filename, protocol;

   std::stringstream fNameSstr;

   // Note: If non-empty, prefix_path has a separator ('/') at the end
   fNameSstr << prefix_path << name;

   if(cycle >= 0)
   {
       fNameSstr << "_" << cycle;
   }
   fNameSstr << "_" << num_procs ;

   // write out in serial if non-mpi or for debug
   bool useSerial = (myid == 0);

#ifdef MFEM_USE_MPI

   ParMesh *par_mesh = dynamic_cast<ParMesh*>(mesh);
   if (par_mesh)
   {
      asctoolkit::spio::IOManager writer(par_mesh->GetComm());
      writer.write(sidre_dc_group, num_procs, fNameSstr.str(), "sidre_hdf5");
   }
   else
   {
      useSerial = true;
   }

#endif

   // write out in serial for debugging, or if MPI unavailable
   if(useSerial)
   {
      protocol = "conduit_json";
      filename = fNameSstr.str() + "_ser.json";
      sidre_dc_group->getDataStore()->save(filename, protocol);//, sidre_dc_group);

      protocol = "sidre_hdf5";
      filename = fNameSstr.str() + "_ser.hdf5";
      sidre_dc_group->getDataStore()->save(filename, protocol);//, sidre_dc_group);
   }
}


bool SidreDataCollection::HasFieldData(const char *field_name)
{
   namespace sidre = asctoolkit::sidre;

   if( ! sidre_dc_group->getGroup("array_data")->hasView(field_name) )
   {
      return false;
   }

   sidre::DataView *view = sidre_dc_group->getGroup("array_data")
                                         ->getView(field_name);

   if( view == NULL)
   {
      return false;
   }

   if(! view->isApplied())
   {
      return false;
   }

   double* data = view->getArray();
   return (data != NULL);
}


double* SidreDataCollection::GetFieldData(const char *field_name, int sz)
{
   // NOTE: WE only handle scalar fields right now
   //       Need to add support for vector fields as well

   namespace sidre = asctoolkit::sidre;

   sidre::DataGroup* f = sidre_dc_group->getGroup("array_data");
   if( ! f->hasView( field_name ) )
   {
       f->createViewAndAllocate(field_name, sidre::DOUBLE_ID, sz);
   }
   else
   {
      // Need to handle a case where the user is requesting a larger field
      sidre::DataView* valsView = f->getView( field_name);
      int valSz = valsView->getNumElements();

      if(valSz < sz)
      {
         valsView->reallocate(sz);
      }
   }

    return f->getView(field_name)->getArray();
}

double* SidreDataCollection::GetFieldData(const char *field_name, int sz, const char *base_field, int offset, int stride)
{
   namespace sidre = asctoolkit::sidre;

   // Try to access /fields/<field_name>/values
   // If not present, try to create it as a different view into /fields/<base_field>/values
   //      with the given sz, stride and offset

   sidre::DataGroup* f = sidre_dc_group->getGroup("array_data");
   if( ! f->hasView( field_name ) )
   {
      if( f->hasView( base_field) && f->getView(base_field) )
      {
         sidre::DataBuffer* buff = f->getView(base_field)->getBuffer();
         f->createView(field_name, buff )->apply(sidre::DOUBLE_ID, sz, offset, stride);
      }
      else
      {
         return NULL;
      }
   }

   return f->getView(field_name)->getArray();
}

void SidreDataCollection::addScalarBasedGridFunction(const char* field_name, GridFunction *gf)
{
   // This function only makes sense when gf is not null
   MFEM_ASSERT( gf != NULL, "Attempted to register grid function with a null pointer");

   namespace sidre = asctoolkit::sidre;

   sidre::DataGroup* grp = sidre_dc_group->getGroup("fields")
                                         ->getGroup(field_name);

   const int numDofs = gf->FESpace()->GetVSize();

   /*
    *  Mesh blueprint for a scalar-based grid function is of the form
    *    /fields/field_name/basis
    *              -- string value is GridFunction's FEC::Name
    *    /fields/field_name/values
    *              -- array of size numDofs
    */


   // First check if we already have the data -- e.g. in restart mode
   if(grp->hasView("values") )
   {
      MFEM_ASSERT( grp->getView("values")->getArray() == gf->GetData(),
                   "Allocated array has different size than gridfunction");
      MFEM_ASSERT( grp->getView("values")->getNumElements() == numDofs,
                     "Allocated array has different size than gridfunction");
   }
   else
   {
      // Otherwise, we must add the view to the blueprint

      // If sidre allocated the data (via GetFieldData() ), use that
      if( HasFieldData(field_name))
      {
         sidre::DataView *vals = sidre_dc_group->getGroup("array_data")
                                               ->getView(field_name);

         const sidre::Schema& schema = vals->getSchema();
         int startOffset = schema.dtype().offset() / schema.dtype().element_bytes();

         sidre::DataBuffer* buff = vals->getBuffer();

         grp->createView("values",buff)
            ->apply(sidre::DOUBLE_ID, numDofs, startOffset);
      }
      else
      {
         // If we are not managing the grid function's data,
         // create a view with the external data
         grp->createView("values", gf->GetData())
            ->apply(sidre::DOUBLE_ID, numDofs);
      }
   }
}

void SidreDataCollection::addVectorBasedGridFunction(const char* field_name, GridFunction *gf)
{
   // This function only makes sense when gf is not null
   MFEM_ASSERT( gf != NULL, "Attempted to register grid function with a null pointer");

   namespace sidre = asctoolkit::sidre;

   sidre::DataGroup* grp = sidre_dc_group->getGroup("fields")
                                         ->getGroup(field_name);

   const int FLD_SZ = 20;
   char fidxName[FLD_SZ];

   int vdim = gf->FESpace()->GetVDim();
   int ndof = gf->FESpace()->GetNDofs();
   Ordering::Type ordering = gf->FESpace()->GetOrdering();

   /*
    *  Mesh blueprint for a vector-based grid function is of the form
    *    /fields/field_name/basis
    *              -- string value is GridFunction's FEC::Name
    *    /fields/field_name/values/x0
    *    /fields/field_name/values/x1
    *    ...
    *    /fields/field_name/values/xn
    *              -- each coordinate is an array of size ndof
    */


   // Check if the blueprint is already set up, and verify setup
   if(grp->hasGroup("values") )
   {
      sidre::DataGroup* fv = grp->getGroup("values");

      // Simple check that the first coord is pointing to the same data as the grid function
      MFEM_ASSERT( fv->hasView("x0")
                   && fv->getView("x0")->getArray() == gf->GetData()
                   , "DataCollection is pointing to different data than gridfunction");

      // Check that we have the right number of coords, each with the right size
      // Note: we are not testing the offsets and strides for each dimension
      for(int i=0; i<vdim; ++i)
      {
         std::snprintf(fidxName, FLD_SZ, "x%d", i);
         MFEM_ASSERT(fv->hasView(fidxName)
                     && fv->getView(fidxName)->getNumElements() == ndof
                    , "DataCollection organization does not match the blueprint"
                    );
      }
   }
   else
   {
      int offset =0;
      int stride =1;

      // Otherwise, we need to set up the blueprint
      // If we've already allocated the data, stride and offset the blueprint data appropriately
      if(HasFieldData(field_name))
      {
         sidre::DataView *vals = sidre_dc_group->getGroup("array_data")
                                               ->getView(field_name);

         sidre::DataBuffer* buff = vals->getBuffer();
         const sidre::Schema& schema = vals->getSchema();
         int startOffset = schema.dtype().offset() / schema.dtype().element_bytes();

         for(int i=0; i<vdim; ++i)
         {
            std::snprintf(fidxName, FLD_SZ, "values/x%d", i);

            switch(ordering)
            {
               case Ordering::byNODES:
                  offset = startOffset + i * ndof;
                  stride = 1;
                  break;
               case Ordering::byVDIM:
                  offset = startOffset + i;
                  stride = vdim;
                  break;
            }

            grp->createView(fidxName, buff)
               ->apply(sidre::DOUBLE_ID, ndof, offset, stride);
         }
      }
      else
      {
         // Else (we're not managing its data)
         // set the views up as external pointers

         for(int i=0; i<vdim; ++i)
         {
            std::snprintf(fidxName, FLD_SZ, "values/x%d", i);

            switch(ordering)
            {
               case Ordering::byNODES:
                  offset = i * ndof;
                  stride = 1;
                  break;
               case Ordering::byVDIM:
                  offset = i;
                  stride = vdim;
                  break;
            }

            grp->createView(fidxName, gf->GetData())
                ->apply(sidre::DOUBLE_ID, ndof, offset, stride);
         }
      }
   }
}

void SidreDataCollection::RegisterField(const char* field_name, GridFunction *gf)
{
   namespace sidre = asctoolkit::sidre;
   sidre::DataGroup* f = sidre_dc_group->getGroup("fields");

   if( gf != NULL )
   {
      // (Create on demand) and) access the group of the field
      if( !f->hasGroup( field_name ) )
      {
         f->createGroup( field_name );
      }

      sidre::DataGroup* grp = f->getGroup( field_name );


      // Set the basis string using the gf's finite element space, overwrite if necessary
      if(!grp->hasView("basis"))
      {
         grp->createViewString("basis", gf->FESpace()->FEColl()->Name());
      }
      else
      {  // overwrite the basis string
         grp->getView("basis")->setString(gf->FESpace()->FEColl()->Name() );
      }

      // Set the topology of the gridfunction.
      // This is always 'mesh' except for a special case with the boundary material attributes field.
      if(!grp->hasView("topology"))
      {
         grp->createViewString("topology", "mesh");
      }

      // Set the data views of the grid function -- either scalar-valued or vector-valued
      bool const isScalarValued = (gf->FESpace()->GetVDim() == 1);
      if(isScalarValued)
      {
         addScalarBasedGridFunction(field_name, gf);
      }
      else // vector valued
      {
         addVectorBasedGridFunction(field_name, gf);
      }
   }

   DataCollection::RegisterField(field_name, gf);
}

std::string SidreDataCollection::getElementName(Element::Type elementEnum)
{
   // Note -- the mapping from Element::Type to string is based on
   //   enum Element::Type { POINT, SEGMENT, TRIANGLE, QUADRILATERAL, TETRAHEDRON, HEXAHEDRON};
   // Note: -- the string names are from conduit's blueprint

   switch(elementEnum)
   {
      case Element::POINT:          return "points";
      case Element::SEGMENT:        return "lines";
      case Element::TRIANGLE:       return "tris";
      case Element::QUADRILATERAL:  return "quads";
      case Element::TETRAHEDRON:    return "tets";
      case Element::HEXAHEDRON:     return "hexs";
   }

   return "unknown";
}

} // end namespace mfem

#endif
