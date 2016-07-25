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

        // Note -- the mapping from Element::Type to string is based on the enum Element::Type
        //   enum Types { POINT, SEGMENT, TRIANGLE, QUADRILATERAL, TETRAHEDRON, HEXAHEDRON};
        // Note: -- the string names are from conduit's blueprint

        std::string eltTypeStr;
        switch(new_mesh->GetElement(0)->GetType())
        {
        case Element::TRIANGLE:
            eltTypeStr = "tris";
            break;
        case Element::QUADRILATERAL:
            eltTypeStr = "quads";
            break;
        case Element::TETRAHEDRON:
            eltTypeStr = "tets";
            break;
        case Element::HEXAHEDRON:
            eltTypeStr = "hexs";
            break;
        case Element::POINT:
        case Element::SEGMENT:
        default:
            eltTypeStr = "<unsupported in blueprint>";
            break;
        }

        grp->createViewString("topology/type", "unstructured");
        grp->createViewString("topology/elements/shape",eltTypeStr);   // <-- Note: this comes form the mesh
        grp->createView("topology/elements/connectivity");

        grp->createViewString("coordset/type", "explicit");

        switch(dim) // Note-- intentional fall through on switch variable
        {
        case 3:     grp->createView("coordset/z");
        case 2:     grp->createView("coordset/y");
        case 1:     grp->createView("coordset/x");  break;
        }

        grp->createViewString("fields/material_attribute/association", "Element");
        grp->createView("fields/material_attribute/values");

    }

    sidre::DataView *elements_connectivity;
    sidre::DataView *material_attribute_values;

    int num_elements = new_mesh->GetNE();
    int element_size = new_mesh->GetElement(0)->GetNVertices();
    int num_indices = num_elements * element_size;
    int num_vertices = new_mesh->GetNV();
    int coordset_len = dim * num_vertices;

    elements_connectivity = grp->getView("topology/elements/connectivity");
    material_attribute_values = grp->getView("fields/material_attribute/values");

    if (!isRestart)
    {
       elements_connectivity->allocate(sidre::INT_ID,num_indices);
       material_attribute_values->allocate(sidre::INT_ID,num_elements);

       sidre::DataBuffer* coordbuf = grp->getDataStore()
                                         ->createBuffer(sidre::DOUBLE_ID, coordset_len)
                                         ->allocate();

       switch(dim) // Note-- intentional fall through on switch variable
       {
       case 3:
           grp->getView("coordset/z")->attachBuffer(coordbuf)
                                     ->apply(sidre::DOUBLE_ID, num_vertices, 2, dim);
       case 2:
           grp->getView("coordset/y")->attachBuffer(coordbuf)
                                     ->apply(sidre::DOUBLE_ID, num_vertices, 1, dim);
       case 1:
           grp->getView("coordset/x")->attachBuffer(coordbuf)
                                     ->apply(sidre::DOUBLE_ID, num_vertices, 0, dim);
       }
    }

    // Change ownership of mesh data to sidre
    double *coord_values = grp->getView("coordset/x")->getBuffer()->getData();

    new_mesh->ChangeElementDataOwnership(
            elements_connectivity->getArray(),
            element_size * num_elements,
            material_attribute_values->getArray(),
            num_elements,
            isRestart);
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


double* SidreDataCollection::GetFieldData(const char *field_name, int sz)
{
    // NOTE: WE only handle scalar fields right now
    //       Need to add support for vector fields as well

    namespace sidre = asctoolkit::sidre;

    sidre::DataGroup* f = sidre_dc_group->getGroup("fields");
    if( ! f->hasGroup( field_name ) )
    {
        sidre::DataGroup* grp = f->createGroup( field_name );
        grp->createViewAndAllocate("values", sidre::DOUBLE_ID, sz);
    }
    else
    {
        // Need to handle a case where the user is requesting a larger field
        sidre::DataView* valsView = f->getGroup( field_name)->getView("values");
        int valSz = valsView->getNumElements();

        if(valSz < sz)
        {
            valsView->reallocate(sz);
        }
    }

    return f->getGroup(field_name)->getView("values")->getArray();
}

double* SidreDataCollection::GetFieldData(const char *field_name, int sz, const char *base_field, int offset, int stride)
{
    namespace sidre = asctoolkit::sidre;

    // Try to access /fields/<field_name>/values
    // If not present, try to create it as a different view into /fields/<base_field>/values
    //      with the given sz, stride and offset

    sidre::DataGroup* f = sidre_dc_group->getGroup("fields");
    if( ! f->hasGroup( field_name ) )
    {
        if( f->hasGroup( base_field) && f->getGroup(base_field)->hasView( "values" ) )
        {
            sidre::DataView* baseV = f->getGroup(base_field)->getView("values");
            sidre::DataBuffer* buff = baseV->getBuffer();

            f->createGroup(field_name)->createView("values", buff )
                                      ->apply(sidre::DOUBLE_ID, sz, offset, stride);
        }
        else
        {
            return NULL;
        }
    }

    return f->getGroup(field_name)->getView("values")->getArray();
}


void SidreDataCollection::RegisterField(const char* field_name, GridFunction *gf)
{
  namespace sidre = asctoolkit::sidre;
  sidre::DataGroup* f = sidre_dc_group->getGroup("fields");

  if( gf != NULL )
  {
      if( !f->hasGroup( field_name ) )
          f->createGroup( field_name );

      sidre::DataGroup* grp = f->getGroup( field_name );

      // Set the gf data as external if it is not already in the datastore
      if(!grp->hasView("values"))
      {
          grp->createView("values", sidre::DOUBLE_ID, gf->Size())
             ->setExternalDataPtr( gf->GetData());
      }

      // Set the basis string using the gf's finite element space
      if(!grp->hasView("basis"))
      {
          grp->createViewString("basis", gf->FESpace()->FEColl()->Name());
      }
      else
      {   // overwrite the basis string
          grp->getView("basis")->setString(gf->FESpace()->FEColl()->Name() );
      }
  }

  DataCollection::RegisterField(field_name, gf);
}

} // end namespace mfem

#endif
