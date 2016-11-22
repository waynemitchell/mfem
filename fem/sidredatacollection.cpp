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

#include <cerrno>      // errno
#ifndef _WIN32
#include <sys/stat.h>  // mkdir
#else
#include <direct.h>    // _mkdir
#define mkdir(dir, mode) _mkdir(dir)
#endif

#include <string>
#include <iomanip>      // for setw, setfill
#include <cstdio>       // for snprintf()

#ifdef MFEM_USE_MPI
#include "spio/IOManager.hpp"
#endif

namespace mfem
{

// TODO - This code is duplicated from data collection.  It should obviously be changed to a class member of
// data collection so sidre dc can call it and this code doesn't have to be duplicated.
// Will change this later, but priority is to get ex9p working right now -- Aaron
static int create_directory(const std::string &dir_name, const Mesh *mesh,
                            int myid)
{
   // create directories recursively
   const char path_delim = '/';
   std::string::size_type pos = 0;
   int err;

   do
   {
      pos = dir_name.find(path_delim, pos+1);
      std::string subdir = dir_name.substr(0, pos);

#ifndef MFEM_USE_MPI
      err = mkdir(subdir.c_str(), 0777);
      err = (err && (errno != EEXIST)) ? 1 : 0;
#else
      const ParMesh *pmesh = dynamic_cast<const ParMesh*>(mesh);
      if (myid == 0 || pmesh == NULL)
      {
         err = mkdir(subdir.c_str(), 0777);
         err = (err && (errno != EEXIST)) ? 1 : 0;
         if (pmesh)
         {
            MPI_Bcast(&err, 1, MPI_INT, 0, pmesh->GetComm());
         }
      }
      else
      {
         // Wait for rank 0 to create the directory
         MPI_Bcast(&err, 1, MPI_INT, 0, pmesh->GetComm());
      }
#endif

   }
   while ( pos != std::string::npos );

   return err;
}

// Constructor that will automatically create the sidre data store and necessary data groups for domain and global data.
SidreDataCollection::SidreDataCollection(const std::string& collection_name,
                                         Mesh * the_mesh, bool own_mesh_data)
   : mfem::DataCollection(collection_name.c_str(), the_mesh),
     m_owns_datastore(true),
     m_owns_mesh_data(own_mesh_data),
     m_meshNodesGFName(""),
     m_loadCalled(false)
{
   namespace sidre = asctoolkit::sidre;
   m_datastore_ptr = new sidre::DataStore();

   sidre::DataGroup * global_grp = m_datastore_ptr->getRoot()->createGroup(
                                      collection_name + "_global");
   sidre::DataGroup * domain_grp = m_datastore_ptr->getRoot()->createGroup(
                                      collection_name);

   parent_datagroup= domain_grp->getParent();

   bp_grp = domain_grp->createGroup("blueprint");
   //Currently only rank 0 adds anything to bp_index.
   bp_index_grp = global_grp->createGroup("blueprint_index/" + name);
   simdata_grp = domain_grp->createGroup("sim");

   SetMesh(the_mesh);
}

// Second constructor that allows external code to specify data groups to place domain and global data in.

// TODO - Conduit will have the capability to generate a blueprint index group in the future.  When this
// is available, all the blueprint index code and the rootfile_dg can be removed from the data collection class.
SidreDataCollection::SidreDataCollection(const std::string& collection_name,
                                         asctoolkit::sidre::DataGroup* global_grp,
                                         asctoolkit::sidre::DataGroup* domain_grp,
                                         const std::string meshNodesGFName,
                                         bool own_mesh_data)
   : mfem::DataCollection(collection_name.c_str()),
     m_owns_datastore(false),
     m_owns_mesh_data(own_mesh_data),
     m_meshNodesGFName(meshNodesGFName),
     m_loadCalled(false),
     parent_datagroup( domain_grp->getParent() )
{
   bp_grp = domain_grp->createGroup("blueprint");

   //Currently only rank 0 adds anything to bp_index.
   bp_index_grp = global_grp->createGroup("blueprint_index/" + name);

   simdata_grp = domain_grp->createGroup("sim");
}

SidreDataCollection::~SidreDataCollection()
{
   if (m_owns_datastore)
   {
      delete m_datastore_ptr;
   }
}


void SidreDataCollection::createMeshBlueprintStubs(bool hasBP)
{
   if (!hasBP)
   {
      bp_grp->createGroup("state");
      bp_grp->createGroup("coordsets");
      bp_grp->createGroup("topologies");
      bp_grp->createGroup("fields");
   }

   // If rank is 0, set up blueprint index state group.
   if (myid == 0)
   {
      bp_index_grp->createGroup("state");
      bp_index_grp->createGroup("coordsets");
      bp_index_grp->createGroup("topologies");
      bp_index_grp->createGroup("fields");
   }

}

void SidreDataCollection::createMeshBlueprintState(bool hasBP)
{
   if (!hasBP)
   {
      // Set up blueprint state group.
      bp_grp->createViewScalar("state/cycle", 0);
      bp_grp->createViewScalar("state/time", 0.);
      bp_grp->createViewScalar("state/domain", myid);
      bp_grp->createViewScalar("state/time_step", 0.);


   }

   // If rank is 0, set up blueprint index state group.
   if (myid == 0)
   {
      bp_index_grp->createViewScalar("state/cycle", 0);
      bp_index_grp->createViewScalar("state/time", 0.);
      bp_index_grp->createViewScalar("state/number_of_domains", num_procs);
   }
}

void SidreDataCollection::createMeshBlueprintCoordset(bool hasBP)
{
   namespace sidre = asctoolkit::sidre;

   int dim = mesh->Dimension();
   MFEM_ASSERT(dim >=1 && dim <= 3, "invalid mesh dimension");

   // With the changes I (Tom) made Vertex is POD and we memcpy to and from the
   // mfem::Vertex mfem::Array requiring a full sized mfem::Vertex per vertex
   // aka double[3] -> dim always needs to be three
   // Note: The coordinates in mfem always have three components (regardless of dim)
   //       but the mesh constructor can handle packed data <- not anymore
   const int NUM_COORDS = 3;

   const int num_vertices = mesh->GetNV();
   const int coordset_len = NUM_COORDS * num_vertices;

   // Add blueprint if not present
   if ( !hasBP )
   {
      // Allocate buffer for coord values.
      sidre::DataBuffer* coordbuf = bp_grp->getDataStore()
                                    ->createBuffer(sidre::DOUBLE_ID, coordset_len)
                                    ->allocate();

      bp_grp->createViewString("coordsets/coords/type", "explicit");

      // Set up views for x, y, z values
      bp_grp->createView("coordsets/coords/values/x")
      ->attachBuffer(coordbuf)
      ->apply(sidre::DOUBLE_ID, num_vertices, 0, NUM_COORDS);

      if (dim >= 2)
      {
         bp_grp->createView("coordsets/coords/values/y")
         ->attachBuffer(coordbuf)
         ->apply(sidre::DOUBLE_ID, num_vertices, 1, NUM_COORDS);
      }
      if (dim >= 3)
      {
         bp_grp->createView("coordsets/coords/values/z")
         ->attachBuffer(coordbuf)
         ->apply(sidre::DOUBLE_ID, num_vertices, 2, NUM_COORDS);
      }
   }

   // If rank 0, set up blueprint index for coordinate set.
   if (myid == 0)
   {
      bp_index_grp->createViewString("coordsets/coords/path",
                                     bp_grp->getPathName() + "/coordsets/coords");
      bp_index_grp->getGroup("coordsets/coords")->copyView(
         bp_grp->getView("coordsets/coords/type") );

      std::string coord_system = "unknown";
      if ( bp_grp->getGroup("coordsets/coords/values")->hasView("x") &&
           bp_grp->getGroup("coordsets/coords/values")->hasView("y") )
      {
         if ( bp_grp->getGroup("coordsets/coords/values")->hasView("z") )
         {
            coord_system = "xyz";
         }
         else
         {
            coord_system = "xy";
         }
      }
      else
      {
         MFEM_ABORT("Unknown coordinate system.");
      }

      bp_index_grp->createViewString("coordsets/coords/coord_system", coord_system);
   }


   // Change ownership of mesh data to sidre
   double *coord_values =
      bp_grp->getView("coordsets/coords/values/x")->getBuffer()->getData();
   if (m_owns_mesh_data)
   {
      mesh->ChangeVertexDataOwnership(coord_values,coordset_len,hasBP);
   }
   else
   {
      mesh->CopyVertexData(coord_values,coordset_len);
   }
}

void SidreDataCollection::createMeshBlueprintTopologies(bool hasBP,
                                                        const std::string& mesh_name)
{
   namespace sidre = asctoolkit::sidre;

   const bool isBdry = (mesh_name == "boundary")? true : false;

   const int element_size = !isBdry
                            ? mesh->GetElement(0)->GetNVertices()
                            : mesh->GetBdrElement(0)->GetNVertices();

   const int num_elements = !isBdry
                            ? mesh->GetNE()
                            : mesh->GetNBE();

   const int num_indices = num_elements * element_size;

   // Find the element shape
   // Note: Assumes homogeneous elements, so only check the first element
   const std::string eltTypeStr = !isBdry
                                  ? getElementName( static_cast<Element::Type>( mesh->GetElement(0)->GetType() ) )
                                  : getElementName( static_cast<Element::Type>( mesh->GetBdrElement(
                                                                                   0)->GetType() ) );

   const std::string mesh_topo_str = "topologies/" + mesh_name;
   const std::string mesh_attr_str = "fields/"+ mesh_name +"_material_attribute";

   if ( !hasBP )
   {
      sidre::DataGroup* topology_grp = bp_grp->createGroup(mesh_topo_str);

      // Add mesh topology
      topology_grp->createViewString("type", "unstructured");
      topology_grp->createViewString("elements/shape",
                                     eltTypeStr);   // <-- Note: this comes form the mesh
      topology_grp->createViewAndAllocate("elements/connectivity",sidre::INT_ID,
                                          num_indices);
      topology_grp->createViewString("coordset", "coords");

      // If the nodes GF name is empty, then the user has not provided one AND one doesn't exist in the mfem mesh.
      if (!isBdry && !m_meshNodesGFName.empty() )
      {
         // NOTE: The view name will be changing
         //       from 'mfem_grid_function' to 'grid_function'
         //       in the next release.
         topology_grp->createViewString("mfem_grid_function",m_meshNodesGFName);
      }

      // Add material attribute field to blueprint
      sidre::DataGroup* attr_grp = bp_grp->createGroup(mesh_attr_str);
      attr_grp->createViewString("association", "element");
      attr_grp->createViewAndAllocate("values",sidre::INT_ID,num_elements);
      attr_grp->createViewString("topology", mesh_name );
   }

   // If rank 0, set up blueprint index for topologies group and material attribute field.
   if (myid == 0)
   {
      const std::string bp_grp_path = bp_grp->getPathName();

      // Create blueprint index for topologies.
      if (isBdry)
      {
         bp_index_grp->getGroup("topologies/mesh")
         ->copyView( bp_grp->getView("topologies/mesh/boundary_topology") );
      }

      sidre::DataGroup * bp_index_topo_grp = bp_index_grp->createGroup(mesh_topo_str);
      sidre::DataGroup* topology_grp = bp_grp->getGroup(mesh_topo_str);

      bp_index_topo_grp->createViewString("path", bp_grp_path + "/" + mesh_topo_str);
      bp_index_topo_grp->copyView( topology_grp->getView("type") );
      bp_index_topo_grp->copyView( topology_grp->getView("coordset") );

      // If the nodes GF name is empty, then the user has not provided one AND one doesn't exist in the mfem mesh.
      if (!isBdry && !m_meshNodesGFName.empty())
      {
         bp_index_topo_grp->copyView( topology_grp->getView("mfem_grid_function") );
      }

      // Create blueprint index for material attributes.
      sidre::DataGroup * bp_index_field_grp = bp_index_grp->createGroup(
                                                 mesh_attr_str);
      sidre::DataGroup * field_grp = bp_grp->getGroup(mesh_attr_str);

      bp_index_field_grp->createViewString( "path",
                                            bp_grp_path + "/" + mesh_attr_str );
      bp_index_field_grp->copyView( field_grp->getView("association") );
      bp_index_field_grp->copyView( field_grp->getView("topology") );

      int number_of_components = 1;
      if ( field_grp->hasGroup("values") )
      {
         number_of_components = field_grp->getGroup("values")->getNumViews();
      }

      bp_index_field_grp->createViewScalar("number_of_components",
                                           number_of_components);
   }

   // Finally, change ownership or copy the element arrays into Sidre
   sidre::DataView* conn_view = bp_grp->getGroup(
                                   mesh_topo_str)->getView("elements/connectivity");
   sidre::DataView* attr_view = bp_grp->getGroup(mesh_attr_str)->getView("values");
   if (m_owns_mesh_data)
   {
      if (!isBdry)
      {
         mesh->ChangeElementDataOwnership(conn_view->getArray(),num_indices,
                                          attr_view->getArray(),num_elements,hasBP);
      }
      else
      {
         mesh->ChangeBoundaryElementDataOwnership(conn_view->getArray(),num_indices,
                                                  attr_view->getArray(),num_elements,hasBP);
      }
   }
   else
   {
      if (!isBdry)
      {
         mesh->CopyElementData(conn_view->getArray(),num_indices,
                               attr_view->getArray(),num_elements);
      }
      else
      {
         mesh->CopyBoundaryElementData(conn_view->getArray(),num_indices,
                                       attr_view->getArray(),num_elements);
      }
   }
}

void SidreDataCollection::verifyMeshBlueprint()
{
   // Conduit will have a verify mesh blueprint capability in the future.
   // Add call to that when it's available to check actual contents in sidre.

   // If a nodes GF name was set, verify a field with that name was registered.
   if (!m_meshNodesGFName.empty())
   {
      MFEM_VERIFY( HasField( m_meshNodesGFName ),
                   "A nodes' position GF was not found with the name '" <<
                   m_meshNodesGFName
                   << "'.  Either the field was not registered, or the wrong mesh nodes GF name was provided in the Sidre DC constructor.");
   }
}

void SidreDataCollection::SetMesh(Mesh *new_mesh)
{
   namespace sidre = asctoolkit::sidre;

   DataCollection::SetMesh(new_mesh);

   if ( !simdata_grp->hasGroup("array_data") )
   {
      simdata_grp->createGroup("array_data");
   }

   bool hasBP = bp_grp->getNumViews() > 0 || bp_grp->getNumGroups() > 0;
   bool has_bnd_elts = (new_mesh->GetNBE() > 0);

   createMeshBlueprintStubs(hasBP);
   createMeshBlueprintState(hasBP);
   createMeshBlueprintCoordset(hasBP);

   createMeshBlueprintTopologies(hasBP, "mesh");

   if (has_bnd_elts)
   {
      bp_grp->createViewString("topologies/mesh/boundary_topology", "boundary");
      createMeshBlueprintTopologies(hasBP, "boundary");
   }


   if ( new_mesh->GetNodes() != NULL && m_meshNodesGFName == "")
   {
      m_meshNodesGFName = "_mesh_nodes_gf";
      /*
      if (m_owns_mesh_data)
      {
         const FiniteElementSpace* nFes = new_mesh->GetNodalFESpace();
         int sz = nFes->GetVSize();
         double* gfData = GetFieldData( m_nodePositionsFieldName.c_str(), sz);

         if(!hasBP)
         {
            double* meshNodeData = new_mesh->GetNodes()->GetData();
            std::memcpy(gfData, meshNodeData, sizeof(double) * sz);
         }

         new_mesh->GetNodes()->NewDataAndSize(gfData, sz);
      }
      */

      RegisterField( m_meshNodesGFName.c_str(), new_mesh->GetNodes());
   }
}

void SidreDataCollection::SetGroupPointers(asctoolkit::sidre::DataGroup *
                                           global_grp,
                                           asctoolkit::sidre::DataGroup * domain_grp)
{
   MFEM_VERIFY(domain_grp->hasGroup("blueprint"),
               "Domain group does not contain a blueprint group.");
   MFEM_VERIFY(global_grp->hasGroup("blueprint_index/" + name),
               "Global group does not contain a blueprint indexgroup.");
   MFEM_VERIFY(domain_grp->hasGroup("sim"),
               "Domain group does not contain a sim group.");

   bp_grp = domain_grp->getGroup("blueprint");
   bp_index_grp = global_grp->getGroup("blueprint_index/" + name);
   simdata_grp = domain_grp->getGroup("sim");
}


void SidreDataCollection::Load(const std::string& path,
                               const std::string& protocol)
{
   if ( m_loadCalled )
   {
      std::string
      error_msg( "Attempt to call SidreDataCollection::Load() more than once on collection: "
                 + name );
      MFEM_ABORT( error_msg);
   }
   else
   {
      simdata_grp->createViewScalar("loadCalled", 1);
      m_loadCalled = true;
   }

   namespace sidre = asctoolkit::sidre;

   sidre::DataStore * datastore = bp_grp->getDataStore();

   bool useSerial = true;

   // read in serial if non-mpi or for debug
#ifdef MFEM_USE_MPI

   useSerial = false;
   ParMesh *par_mesh = dynamic_cast<ParMesh*>(mesh);
   if (par_mesh)
   {
      //       sidre::DataGroup * domain_file_grp = bp_grp->getParent()->getParent();
      asctoolkit::spio::IOManager reader(par_mesh->GetComm());
      reader.read( datastore->getRoot(), path);
   }
   else
   {
      useSerial = true;
   }
#endif

   // read in serial for debugging, or if MPI unavailable
   if (useSerial)
   {
      datastore->load(path, protocol); //, sidre_dc_group);
   }

   // If the data collection created the datastore, it knows the layout of where the domain and global
   // groups are, and can restore them after the Load().
   //
   // If the data collection did not create the datastore, the host code must reset these pointers after
   // the load operation and also reset the state variables.
   if (m_owns_datastore)
   {
      SetGroupPointers(m_datastore_ptr->getRoot()->getGroup(name + "_global"),
                       m_datastore_ptr->getRoot()->getGroup(name));

      UpdateStateFromDS();
   }

}
void SidreDataCollection::UpdateStateFromDS()
{
   SetTime( bp_grp->getView("state/time")->getData<double>() );
   SetCycle( bp_grp->getView("state/cycle")->getData<int>() );
   SetTimeStep( bp_grp->getView("state/time_step")->getData<double>() );
}

void SidreDataCollection::UpdateStateToDS()
{
   bp_grp->getView("state/cycle")->setScalar(GetCycle());
   bp_grp->getView("state/time")->setScalar(GetTime());
   bp_grp->getView("state/time_step")->setScalar(GetTimeStep());

   if (myid == 0)
   {
      bp_index_grp->getView("state/cycle")->setScalar(GetCycle());
      bp_index_grp->getView("state/time")->setScalar(time);
   }
}

void SidreDataCollection::Save()
{
   std::string filename = name;
   std::string protocol = "sidre_hdf5";

   Save(filename, protocol);
}

void SidreDataCollection::Save(const std::string& filename,
                               const std::string& protocol)
{
   namespace sidre = asctoolkit::sidre;

   verifyMeshBlueprint();

   create_directory(prefix_path, mesh, myid);

   std::stringstream fNameSstr;

   // Note: If non-empty, prefix_path has a separator ('/') at the end
   fNameSstr << prefix_path << filename;

   if (GetCycle() >= 0)
   {
      fNameSstr << "_" << std::setfill('0') << std::setw(pad_digits) << GetCycle();
   }

   std::string file_path = fNameSstr.str();

   UpdateStateToDS();


#ifdef MFEM_USE_MPI
   const ParMesh *pmesh = dynamic_cast<const ParMesh*>(mesh);
   asctoolkit::spio::IOManager writer(pmesh->GetComm());
   sidre::DataStore * datastore = bp_grp->getDataStore();
   writer.write(datastore->getRoot(), num_procs, file_path, protocol);
#else
   // If serial, use sidre group writer.
   bp_grp->getDataStore()->save( file_path, protocol);//, sidre_dc_group);
#endif

   if (myid == 0)
   {
      sidre::DataGroup * blueprint_indicies_grp = bp_index_grp->getParent();
#ifdef MFEM_USE_MPI
      if (protocol == "sidre_hdf5")
      {
         writer.writeGroupToRootFile( blueprint_indicies_grp, file_path + ".root"  );
      }
      // Root file support only available in hdf5.
      else
      {
         writer.write(blueprint_indicies_grp, 1, file_path + ".root", protocol);
      }
#else
      // If serial, use sidre group writer.
      blueprint_indicies_grp->getDataStore()->save(file_path + ".root",
                                                   protocol);//, sidre_dc_group);
#endif

   }
#if 0
}
// If not hdf5, use sidre group writer for both parallel and serial.  SPIO only supports HDF5.
else
{
   if (myid == 0)
   {
      sidre::DataGroup * blueprint_indicies_grp = bp_index_grp->getParent();
      blueprint_indicies_grp->getDataStore()->save(file_path + ".root",
                                                   protocol);//, sidre_dc_group);
   }

   fNameSstr << "_" << myid;
   file_path = fNameSstr.str();
   bp_grp->getDataStore()->save(file_path, protocol);//, sidre_dc_group);

}
#endif

/*
   std::string _protocol;
   std::string _filename;

   _protocol = "conduit_json";
   _filename = fNameSstr.str() + ".conduit_json";
   bp_grp->getDataStore()->save(_filename, _protocol);//, sidre_dc_group);

   _protocol = "json";
   _filename = fNameSstr.str() + ".json";
   bp_grp->getDataStore()->save(_filename, _protocol);//, sidre_dc_group);

   _protocol = "sidre_hdf5";
   _filename = fNameSstr.str() + ".sidre_hdf5";
   bp_grp->getDataStore()->save(_filename, _protocol);//, sidre_dc_group);
*/
}

bool SidreDataCollection::HasFieldData(const std::string& field_name)
{
   namespace sidre = asctoolkit::sidre;

   if ( ! simdata_grp->getGroup("array_data")->hasView(field_name) )
   {
      return false;
   }

   sidre::DataView *view = simdata_grp->getGroup("array_data")
                           ->getView(field_name);

   if ( view == NULL)
   {
      return false;
   }

   if (! view->isApplied())
   {
      return false;
   }

   double* data = view->getArray();
   return (data != NULL);
}


double* SidreDataCollection::GetFieldData(const std::string& field_name, int sz)
{
   // NOTE: WE only handle scalar fields right now
   //       Need to add support for vector fields as well

   namespace sidre = asctoolkit::sidre;

   MFEM_VERIFY(simdata_grp->hasGroup("array_data") == true,
               "No group 'array_data' in data collection.  Verify that SetMesh was called to set the mesh in the data collection.");
   sidre::DataGroup* f = simdata_grp->getGroup("array_data");
   if ( ! f->hasView( field_name ) )
   {
      f->createViewAndAllocate(field_name, sidre::DOUBLE_ID, sz);
   }
   else
   {
      // Need to handle a case where the user is requesting a larger field
      sidre::DataView* valsView = f->getView( field_name);
      int valSz = valsView->getNumElements();

      if (valSz < sz)
      {
         valsView->reallocate(sz);
      }
   }

   return f->getView(field_name)->getArray();
}

double* SidreDataCollection::GetFieldData(const std::string& field_name, int sz,
                                          const std::string& base_field, int offset, int stride)
{
   namespace sidre = asctoolkit::sidre;

   // Try to access /fields/<field_name>/values
   // If not present, try to create it as a different view into /fields/<base_field>/values
   //      with the given sz, stride and offset

   sidre::DataGroup* f = simdata_grp->getGroup("array_data");
   if ( ! f->hasView( field_name ) )
   {
      if ( f->hasView( base_field) && f->getView(base_field) )
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

void SidreDataCollection::addScalarBasedGridFunction(const std::string&
                                                     field_name,
                                                     GridFunction *gf)
{
   // This function only makes sense when gf is not null
   MFEM_ASSERT( gf != NULL,
                "Attempted to register grid function with a null pointer");

   namespace sidre = asctoolkit::sidre;

   sidre::DataGroup* grp = bp_grp->getGroup( std::string("fields/") + field_name);

   const int numDofs = gf->FESpace()->GetVSize();

   /*
    *  Mesh blueprint for a scalar-based grid function is of the form
    *    /fields/field_name/basis
    *              -- string value is GridFunction's FEC::Name
    *    /fields/field_name/values
    *              -- array of size numDofs
    */


   // First check if we already have the data -- e.g. in restart mode
   if (grp->hasView("values") )
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
      if ( HasFieldData(field_name))
      {
         sidre::DataView *vals = simdata_grp->getGroup("array_data")
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

void SidreDataCollection::addVectorBasedGridFunction(const std::string&
                                                     field_name,
                                                     GridFunction *gf)
{
   // This function only makes sense when gf is not null
   MFEM_ASSERT( gf != NULL,
                "Attempted to register grid function with a null pointer");

   namespace sidre = asctoolkit::sidre;

   sidre::DataGroup* grp = bp_grp->getGroup( std::string("fields/") + field_name);

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
   if (grp->hasGroup("values") )
   {
      sidre::DataGroup* fv = grp->getGroup("values");

      // Simple check that the first coord is pointing to the same data as the grid function
      MFEM_ASSERT( fv->hasView("x0")
                   && fv->getView("x0")->getArray() == gf->GetData()
                   , "DataCollection is pointing to different data than gridfunction");

      // Check that we have the right number of coords, each with the right size
      // Note: we are not testing the offsets and strides for each dimension
      for (int i=0; i<vdim; ++i)
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
      if (HasFieldData(field_name))
      {
         sidre::DataView *vals = simdata_grp->getGroup("array_data")
                                 ->getView(field_name);

         sidre::DataBuffer* buff = vals->getBuffer();
         const sidre::Schema& schema = vals->getSchema();
         int startOffset = schema.dtype().offset() / schema.dtype().element_bytes();

         for (int i=0; i<vdim; ++i)
         {
            std::snprintf(fidxName, FLD_SZ, "values/x%d", i);

            switch (ordering)
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

         for (int i=0; i<vdim; ++i)
         {
            std::snprintf(fidxName, FLD_SZ, "values/x%d", i);

            switch (ordering)
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

// Should only be called on mpi rank 0 ( or if serial problem ).
void SidreDataCollection::DeregisterFieldInBPIndex(const std::string&
                                                   field_name)
{
   namespace sidre = asctoolkit::sidre;

   sidre::DataGroup * fields_grp = bp_index_grp->getGroup("fields");
   MFEM_VERIFY(fields_grp->hasGroup(field_name),
               "No field exists in blueprint index with name " << name);

   // Note: This will destroy all orphaned views or buffer classes under this group also.  If sidre owns this
   // field data, the memory will be deleted unless it's referenced somewhere else in sidre.
   fields_grp->destroyGroup(field_name);
}

// Should only be called on mpi rank 0 ( or if serial problem ).
void SidreDataCollection::RegisterFieldInBPIndex(asctoolkit::sidre::DataGroup *
                                                 bp_field_grp)
{
   namespace sidre = asctoolkit::sidre;
   const std::string& field_name = bp_field_grp->getName();
   sidre::DataGroup * bp_index_field_grp = bp_index_grp->createGroup("fields/"
                                                                     +field_name);

   bp_index_field_grp->createViewScalar( "path", bp_field_grp->getPathName() );
   bp_index_field_grp->copyView( bp_field_grp->getView("topology") );
   if (bp_field_grp->hasView("basis"))
   {
      bp_index_field_grp->copyView( bp_field_grp->getView("basis") );
   }
   else if (bp_field_grp->hasView("association"))
   {
      bp_index_field_grp->copyView( bp_field_grp->getView("association") );
   }
   else
   {
      std::string errorMessage = " Field " + bp_field_grp->getName() +
                                 " is missing association or basis entry in blueprint.";
      MFEM_ABORT( errorMessage );
   }


   // Note: The bp index requires GridFunction::VectorDim()
   //       since the GF might be scalar valued and have a vector basis
   //       (e.g. hdiv and hcurl spaces)
   const int number_of_components = GetField(field_name.c_str())->VectorDim();
   bp_index_field_grp->createViewScalar("number_of_components",
                                        number_of_components);
}

void SidreDataCollection::DeregisterField(const std::string& field_name)
{
   namespace sidre = asctoolkit::sidre;

   sidre::DataGroup * fields_grp = bp_grp->getGroup("fields");
   MFEM_VERIFY(fields_grp->hasGroup(field_name),
               "No field exists in blueprint with name " << field_name);

   // Note: This will destroy all orphaned views or buffer classes under this group also.  If sidre owns this
   // field data, the memory will be deleted unless it's referenced somewhere else in sidre.
   fields_grp->destroyGroup(field_name);

   if (!serial && myid == 0)
   {
      DeregisterFieldInBPIndex(field_name);
   }
}

void SidreDataCollection::RegisterField(const std::string& field_name,
                                        GridFunction *gf)
{
   namespace sidre = asctoolkit::sidre;
   sidre::DataGroup* f = bp_grp->getGroup("fields");

   if ( gf != NULL )
   {
      MFEM_VERIFY(!f->hasGroup( field_name ),
                  "Blueprint already has field registered with name '" << field_name <<
                  "', de-register the old field first.");

      sidre::DataGroup* grp = f->createGroup( field_name );

      // Set the basis string using the gf's finite element space, overwrite if necessary
      if (!grp->hasView("basis"))
      {
         grp->createViewString("basis", gf->FESpace()->FEColl()->Name());
      }
      else
      {
         // overwrite the basis string
         grp->getView("basis")->setString(gf->FESpace()->FEColl()->Name() );
      }

      // Set the topology of the gridfunction.
      // This is always 'mesh' except for a special case with the boundary material attributes field.
      if (!grp->hasView("topology"))
      {
         grp->createViewString("topology", "mesh");
      }

      // Set the data views of the grid function
      // e.g. the number of coefficients per DoF -- either scalar-valued or vector-valued
      bool const isScalarValued = (gf->FESpace()->GetVDim() == 1);
      if (isScalarValued)
      {
         addScalarBasedGridFunction(field_name, gf);
      }
      else // vector valued
      {
         addVectorBasedGridFunction(field_name, gf);
      }
   }

   DataCollection::RegisterField(field_name, gf);
   if (!serial && myid == 0)
   {
      RegisterFieldInBPIndex( f->getGroup(field_name) );
   }
}


std::string SidreDataCollection::getElementName(Element::Type elementEnum)
{
   // Note -- the mapping from Element::Type to string is based on
   //   enum Element::Type { POINT, SEGMENT, TRIANGLE, QUADRILATERAL, TETRAHEDRON, HEXAHEDRON};
   // Note: -- the string names are from conduit's blueprint

   switch (elementEnum)
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
