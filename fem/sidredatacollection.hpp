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

#ifndef MFEM_SIDREDATACOLLECTION
#define MFEM_SIDREDATACOLLECTION

#include "../config/config.hpp"

#ifdef MFEM_USE_SIDRE

#include "../fem/datacollection.hpp"
#include <sidre/sidre.hpp>

namespace mfem
{

/// Data collection with Sidre routines.
class SidreDataCollection : public DataCollection
{

public:

   // Adding some typedefs here for the variables types in MFEM that we will be
   // putting in Sidre. This is to avoid hard coding a bunch of SIDRE::ENUM
   // types in the sidre calls, in case MFEM ever wants to change some of it's
   // data types.
   // TODO - ask MFEM team if they have any interest in adding a few typedefs in
   // their classes, or if just hard-coding the type is better.  For now, I'll
   // just put the typedef's here.

   typedef int mfem_int_t;
   typedef double mfem_double_t;

   SidreDataCollection(const std::string& collection_name, Mesh *the_mesh,
                       bool owns_mesh_data=false);

   SidreDataCollection(const std::string& collection_name,
                       asctoolkit::sidre::DataGroup * global_grp,
                       asctoolkit::sidre::DataGroup * domain_grp,
                       const std::string meshNodesGFName = "mfem_default_mesh_nodes_gf",
                       bool owns_mesh_data=false);

   void DeregisterField(const std::string& field_name);

   void RegisterField(const std::string& field_name, GridFunction *gf);

   /// Delete all owned data.
   virtual ~SidreDataCollection();

   void SetMesh(Mesh *new_mesh);

   /// Reset the domain and global (root) datastore group pointers.
   /** These are set in the constructor, but if a host code changes the
       datastore contents ( such as wiping out the datastore and loading in new
       contents from a file, i.e. a restart ) these pointers will need to be
       reset to valid groups in the datastore.
    */
   void SetGroupPointers(asctoolkit::sidre::DataGroup * global_grp,
                         asctoolkit::sidre::DataGroup * domain_grp);

   asctoolkit::sidre::DataGroup * GetBPGroup() { return bp_grp; }
   asctoolkit::sidre::DataGroup * GetBPIndexGroup() { return bp_index_grp; }
   asctoolkit::sidre::DataGroup * GetSimGroup() { return simdata_grp; }

   void Save();

   void Save(const std::string& filename, const std::string& protocol);

   void Load(const std::string& path, const std::string& protocol);

   /** @brief Updates the DataCollection's state variables (e.g. cycle, time) to
       the values from the data store. */
   void UpdateStateFromDS();

   /** @brief Updates the data store's state variables (e.g. cycle, time) to the
       values from the DataCollection. */
   void UpdateStateToDS();

   /** @brief Gets a pointer to the associated field's view data (always an
       array of doubles).

       If the field does not exist, it will create a view of size @a sz. */
   double* GetFieldData(const std::string& field_name, int sz = 0);

   /// Gets a pointer to the data of @a field_name (always an array of doubles).
   /** Data is relative to the data associated with @a base_field.
       Returns null if @a base_field does not exist.
    */
   double* GetFieldData(const std::string& field_name, int sz,
                        const std::string& base_field, int offset = 0,
                        int stride = 1);

   bool HasFieldData(const std::string& field_name);

   /// Gets a pointer to the data (an array of template type T).
   /** If the array named by field_name does not exist, it will create a view of
       the appropriate size and allocate as appropriate.
       \note This function is not available in base DataCollection class.
    */
   template<typename T>
   T* GetArrayData(const std::string& field_name, int sz)
   {
      namespace sidre = asctoolkit::sidre;

      sidre::DataGroup* f = simdata_grp->getGroup("array_data");
      if ( ! f->hasView( field_name ) )
      {
         f->createViewAndAllocate(field_name, sidre::detail::SidreTT<T>::id, sz);
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

private:
   // Used if the sidre data collection is providing the datastore itself.
   const bool m_owns_datastore;

   // TODO - Need to evaluate if this bool member can be combined with own_data
   // in parent data collection class. m_owns_mesh_data indicates whether the
   // sidre dc owns the mesh element data and node positions gf. The DC base
   // class own_data indicates if the dc owns the mesh object pointer itself and
   // GF objects. Can we use one flag and just have DC own all objects vs none?
   const bool m_owns_mesh_data;

   std::string m_meshNodesGFName;
   bool m_loadCalled;

   // If the data collection owns the datastore, it will store a pointer to it.
   asctoolkit::sidre::DataStore * m_datastore_ptr;

protected:
   // why is this protected, but sidre dc is private?
   asctoolkit::sidre::DataGroup *parent_datagroup;

private:

   // If the data collection does not own the datastore, it will need pointers
   // to the blueprint and blueprint index group to use.
   asctoolkit::sidre::DataGroup * bp_grp;
   asctoolkit::sidre::DataGroup * bp_index_grp;

   // This is stored for convenience.
   asctoolkit::sidre::DataGroup * simdata_grp;

   // Private helper functions

   void DeregisterFieldInBPIndex(const std::string & field_name);
   void RegisterFieldInBPIndex(asctoolkit::sidre::DataGroup * bp_field_group);

   std::string getElementName( Element::Type elementEnum );

   /**
    * \brief A private helper function to set up the views associated with the
       data of a scalar valued grid function in the blueprint style.
    * \pre gf is not null
    * \note This function is expected to be called by RegisterField()
    * \note Handles cases where hierarchy is already set up,
    *      where the data was allocated by this data collection
    *      and where the gridfunction data is external to sidre
    */
   void addScalarBasedGridFunction(const std::string& field_name,
                                   GridFunction* gf);

   /**
    * \brief A private helper function to set up the views associated with the
       data of a vector valued grid function in the blueprint style.
    * \pre gf is not null
    * \note This function is expected to be called by RegisterField()
    * \note Handles cases where hierarchy is already set up,
    *      where the data was allocated by this data collection
    *      and where the gridfunction data is external to sidre
    */
   void addVectorBasedGridFunction(const std::string& field_name,
                                   GridFunction* gf);


   /// Sets up the four main mesh blueprint groups.
   /**
    * \param hasBP Indicates whether the blueprint has already been set up.
    */
   void createMeshBlueprintStubs(bool hasBP);

   /// Sets up the mesh blueprint 'state' group.
   /**
    * \param hasBP Indicates whether the blueprint has already been set up.
    */
   void createMeshBlueprintState(bool hasBP);

   /// Sets up the mesh blueprint 'coordsets' group.
   /**
    * \param hasBP Indicates whether the blueprint has already been set up.
    */
   void createMeshBlueprintCoordset(bool hasBP);

   /// Sets up the mesh blueprint 'topologies' group.
   /**
    * \param hasBP Indicates whether the blueprint has already been set up.
    * \param mesh_name The name of the topology.
    * \note Valid values for @a mesh_name are "mesh" and "boundary".
    */
   void createMeshBlueprintTopologies(bool hasBP, const std::string& mesh_name);

   /// Verifies that the contents of the mesh blueprint data is valid.
   void verifyMeshBlueprint();
};

} // end namespace mfem

#endif

#endif
