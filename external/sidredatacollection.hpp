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

namespace mfem
{

// Forward declare needed datastore class so we don't need to #include sidre headers in this header.
class DataGroup;

/// Data collection with Sidre routines
class SidreDataCollection : public DataCollection
{

protected:
   asctoolkit::sidre::DataGroup *parent_datagroup;

public:

  // Adding some typedefs here for the variables types in MFEM that we will be putting in Sidre
  // This is to avoid hard coding a bunch of SIDRE::ENUM types in the sidre calls, in case MFEM ever
  // wants to change some of it's data types.
  // TODO - ask MFEM team if they have any interest in adding a few typedefs in their classes, or
  // if just hard-coding the type is better.  For now, I'll just put the typedef's here.

  typedef int mfem_int_t;
  typedef double mfem_double_t;

  SidreDataCollection(const std::string& collection_name, asctoolkit::sidre::DataGroup * dg);

  void RegisterField(const char* field_name, GridFunction *gf);

  /// Verify we will delete the mesh and fields if we own them
  virtual ~SidreDataCollection() {}

  void SetMesh(Mesh *new_mesh);

  void setMeshStream(std::istream& input) {}

  void SetupMeshBlueprint();

  void Save();

  void Load(const std::string& path, const std::string& protocol = "sidre_hdf5");

  /**
   * Gets a pointer to the associated field's view data
   * If the field does not exist, it will create a view of the appropriate size
   */
  double* GetFieldData(const char *field_name, int sz = 0);

  /**
   * Gets a pointer to the data of field_name
   * Data is relative to the data associated with base_field
   * Returns null if base_field does not exist
   */
  double* GetFieldData(const char *field_name, int sz, const char *base_field, int offset = 0, int stride = 1);
private:
  // Private helper functions

  // why can't this be const? (the ElementAllocator)
  void addElements(asctoolkit::sidre::DataGroup * group, ElementAllocator*);

  void addField(asctoolkit::sidre::DataGroup * grp, mfem::GridFunction* gf);

  void addMesh( asctoolkit::sidre::DataGroup * grp);

  void addVertices(asctoolkit::sidre::DataGroup * grp);

  asctoolkit::sidre::DataGroup * sidre_dc_group;

};

} // end namespace mfem

#endif

#endif
