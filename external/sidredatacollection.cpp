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

  sidre_dc_group->createGroup("array_data");
}


void SidreDataCollection::SetupMeshBlueprint()
{
}

void SidreDataCollection::SetMesh(Mesh *new_mesh)
{
    DataCollection::SetMesh(new_mesh);

    // uses conduit's mesh blueprint

    namespace sidre = asctoolkit::sidre;
    sidre::DataGroup* grp = sidre_dc_group;

    int dim = new_mesh->Dimension();
    MFEM_ASSERT(dim >=1 && dim <= 3, "invalid mesh dimension");


    bool isRestart = grp->hasGroup("topology");
    if(!isRestart)
    {
        /// Setup the mesh blueprint

        // Find the element shape
        // Note: Assumes homogeneous elements, so only check the first element
        std::string eltTypeStr = getElementName( static_cast<Element::Type>( new_mesh->GetElement(0)->GetType() ) );
        
        // Add mesh topology
        grp->createViewString("topology/mesh/type", "unstructured");
        grp->createViewString("topology/mesh/elements/shape",eltTypeStr);
        grp->createView("topology/mesh/elements/connectivity");

        // Add mesh boundary topology
        grp->createViewString("topology/boundary/type", "unstructured");
        grp->createViewString("topology/boundary/elements/shape",eltTypeStr);
        grp->createView("topology/boundary/elements/connectivity");
 
        // Add coordinate set
        grp->createViewString("coordset/type", "explicit");
        grp->createView("coordset/values/x");
        if(dim >= 2) { grp->createView("coordset/values/y"); }
        if(dim >= 3) { grp->createView("coordset/values/z"); }

        // Add mesh elements material attribute field
        grp->createViewString("fields/mesh_material_attribute/association", "Element");
        grp->createView("fields/mesh_material_attribute/values");

        // Add boundary elements material attribute field
        grp->createViewString("fields/boundary_material_attribute/association", "Element");
        grp->createView("fields/boundary_material_attribute/values");
    }

    // NOTE: This code can be consolidated to a single 'addElements' helper function.
    // It could be called twice - once for each topology to be added.
    sidre::DataView *mesh_elements_connectivity;
    sidre::DataView *bnd_elements_connectivity;
    sidre::DataView *mesh_material_attribute_values;
    sidre::DataView *bnd_material_attribute_values;

    // Note: The coordinates in mfem always have three components
    const int NUM_COORDS = 3;

    int element_size = new_mesh->GetElement(0)->GetNVertices();
    int num_vertices = new_mesh->GetNV();
    int coordset_len = NUM_COORDS * num_vertices;

    int mesh_num_elements = new_mesh->GetNE();
    int bnd_num_elements = new_mesh->GetNBE();
    int mesh_num_indices = mesh_num_elements * element_size;
    int bnd_num_indices = bnd_num_elements * element_size;

    mesh_elements_connectivity = grp->getView("topology/mesh/elements/connectivity");
    mesh_material_attribute_values = grp->getView("fields/mesh_material_attribute/values");

    const bool has_bdry_elts = (bnd_num_elements > 0);

    bnd_elements_connectivity = grp->getView("topology/boundary/elements/connectivity");
    bnd_material_attribute_values = grp->getView("fields/boundary_material_attribute/values");

    if (!isRestart)
    {
       mesh_elements_connectivity->allocate(sidre::INT_ID,mesh_num_indices);
       mesh_material_attribute_values->allocate(sidre::INT_ID,mesh_num_elements);

       if(has_bdry_elts)
       {
         bnd_elements_connectivity->allocate(sidre::INT_ID,bnd_num_indices);
         bnd_material_attribute_values->allocate(sidre::INT_ID,bnd_num_elements);
       }

       sidre::DataBuffer* coordbuf = grp->getDataStore()
                                         ->createBuffer(sidre::DOUBLE_ID, coordset_len)
                                         ->allocate();

       grp->getView("coordset/values/x")
          ->attachBuffer(coordbuf)
          ->apply(sidre::DOUBLE_ID, num_vertices, 0, NUM_COORDS);

       if(dim >= 2) {
           grp->getView("coordset/values/y")
              ->attachBuffer(coordbuf)
              ->apply(sidre::DOUBLE_ID, num_vertices, 1, NUM_COORDS);
       }

       if(dim >= 3) {
           grp->getView("coordset/values/z")
              ->attachBuffer(coordbuf)
              ->apply(sidre::DOUBLE_ID, num_vertices, 2, NUM_COORDS);
       }

    }

    // Change ownership of mesh data to sidre
    double *coord_values = grp->getView("coordset/values/x")->getBuffer()->getData();

    new_mesh->ChangeElementDataOwnership(
            mesh_elements_connectivity->getArray(),
            element_size * mesh_num_elements,
            mesh_material_attribute_values->getArray(),
            mesh_num_elements,
            isRestart);


    if(has_bdry_elts) {
        new_mesh->ChangeBoundaryElementDataOwnership(
                bnd_elements_connectivity->getArray(),
                element_size * bnd_num_elements,
                bnd_material_attribute_values->getArray(),
                bnd_num_elements,
                isRestart);
    }

    new_mesh->ChangeVertexDataOwnership(
            coord_values,
            dim,
            coordset_len,
            isRestart);

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

void SidreDataCollection::Load(const std::string& path, const std::string& protocol)
{
	std::cout << "Loading Sidre checkpoint: " << path << " using protocol: " << protocol << std::endl;
//	sidre_dc_group->getDataStore()->load(path, protocol, sidre_dc_group);
	sidre_dc_group->getDataStore()->load(path, protocol);
   // we have to get this again because the group pointer may have changed
   sidre_dc_group = parent_datagroup->getGroup( name );
	SetTime( sidre_dc_group->getView("state/time")->getScalar() );
	SetCycle( sidre_dc_group->getView("state/cycle")->getScalar() );
}

void SidreDataCollection::Save()
{
    namespace sidre = asctoolkit::sidre;
    sidre::DataGroup* grp = sidre_dc_group->getGroup("state");

    grp->getView("cycle")->setScalar(cycle);
    grp->getView("time")->setScalar(time);

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
        sidre_dc_group->getDataStore()->save(filename, protocol);

        protocol = "sidre_hdf5";
        filename = fNameSstr.str() + "_ser.hdf5";
        sidre_dc_group->getDataStore()->save(filename, protocol);
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
        return false;

    if(! view->isApplied())
        return false;

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


void SidreDataCollection::RegisterField(const char* field_name, GridFunction *gf)
{
  namespace sidre = asctoolkit::sidre;
  sidre::DataGroup* f = sidre_dc_group->getGroup("fields");

  if( gf != NULL )
  {
      // (Create on demand) and) access the group of the field
      if( !f->hasGroup( field_name ) )
          f->createGroup( field_name );
      sidre::DataGroup* grp = f->getGroup( field_name );


      // Set the basis string using the gf's finite element space, overwrite if necessary
      if(!grp->hasView("basis"))
      {
          grp->createViewString("basis", gf->FESpace()->FEColl()->Name());
      }
      else
      {   // overwrite the basis string
          grp->getView("basis")->setString(gf->FESpace()->FEColl()->Name() );
      }


      // Set the data -- either scalar or vector-valued

      bool const isScalarValued = (gf->FESpace()->GetVDim() == 1);
      if(isScalarValued)
      {
          // First check if we have the data
          if(grp->hasView("values") )
          {
              MFEM_ASSERT( grp->getView("values")->getArray() == gf->GetData(),
                           "Allocated array has different size than gridfunction");
          }
          else
          {
              const int sz = gf->FESpace()->GetVSize();
              if( HasFieldData(field_name))
              {
                  sidre::DataView *vals = sidre_dc_group->getGroup("array_data")
                                                        ->getView(field_name);
                  sidre::DataBuffer* buff = vals->getBuffer();
                  grp->createView("values",buff)->apply(sidre::DOUBLE_ID, sz);
              }
              else
              {
                  // Handle case where we have gotten a grid function,
                  // but are not managing its data

                  grp->createView("values", sidre::DOUBLE_ID, sz)
                     ->setExternalDataPtr( gf->GetData());
              }
          }

      }
      else // vector valued
      {
          int vdim = gf->FESpace()->GetVDim();
          int ndof = gf->FESpace()->GetNDofs();
          Ordering::Type ordering = gf->FESpace()->GetOrdering();

          if(grp->hasGroup("values") )
          {
              // Check that the sizes are correct, need to modify below line for vector-valued data
              //MFEM_ASSERT( grp->getView("values")->getArray() == gf->GetData(),
              //             "Allocated array has different size than gridfunction");
          }
          else
          {
              if(HasFieldData(field_name))
              {
                  sidre::DataView *vals = sidre_dc_group->getGroup("array_data")
                                                        ->getView(field_name);
                  sidre::DataBuffer* buff = vals->getBuffer();

                  const int FLD_SZ = 20;
                  char fidxName[FLD_SZ];
                  int offset =0;
                  int stride = 0;

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

                      grp->createView(fidxName, buff)
                         ->apply(sidre::DOUBLE_ID, ndof, offset, stride);
                  }

              }
              else
              {
                  // Handle case where we have gotten a vector-valued grid function
                  // but we are not managing its data
              }
          }
      }
  }

  DataCollection::RegisterField(field_name, gf);
}

std::string SidreDataCollection::getElementName(Element::Type elementEnum)
{
   // Note -- the mapping from Element::Type to string is based on the enum Element::Type
   //   enum Types { POINT, SEGMENT, TRIANGLE, QUADRILATERAL, TETRAHEDRON, HEXAHEDRON};
   // Note: -- the string names are from conduit's blueprint

   switch(elementEnum)
   {
      case Element::POINT:
         return "point";
         break;

      case Element::SEGMENT:
         return "segment";
         break;

      case Element::TRIANGLE:
         return "tris";
         break;

      case Element::QUADRILATERAL:
         return "quads";
         break;

      case Element::TETRAHEDRON:
         return "tets";
         break;

      case Element::HEXAHEDRON:
         return "hexs";
         break;

      default:
         return "unknown";
   }

}

} // end namespace mfem

#endif
