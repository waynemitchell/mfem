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

#ifndef MFEM_DATACOLLECTION
#define MFEM_DATACOLLECTION

#include "../config.hpp"
#include <string>
#include <map>

namespace mfem
{

class Mesh;
class GridFunction;
class RootData;


class DataCollection
{
protected:
   RootData *root_data;
   Mesh *mesh;

   int my_rank;
   bool own_data;
   std::map<std::string, GridFunction*> field_map;

   /// Interact with the Mesh files
   void SaveMesh();

   /// Interact with the Field files
   void SaveFields();
   void LoadFieldsFromRootData();

public:
   /// Constructors for initializing an empty collection so we can load a mesh and fields into it.
   /// If running in parallel, the mpi_rank is necessary 
   DataCollection(const char *collection_name, int mpi_rank = 0);

   /// Constructor for initializing a collection with a mesh
   DataCollection(const char *collection_name, Mesh *_mesh);

   /// Various Accessors
   const char* GetCollectionName();
   void SetOwnData(bool _own_data);
   void SetCycle(int c);
   int GetCycle();
   void SetTime(double t);
   double GetTime();
   void SetVisitParameters(int max_levels_of_detail);
   Mesh *GetMesh();

   /// Interact with the fields map
   void RegisterField(const char *field_name, GridFunction *gf);
   bool HasField(const char *name) {return field_map.count(name) == 1;};
   GridFunction *GetField(const char *field_name);

   /// Interact with MFEM data files
   void SaveData();
   void LoadMesh(int cycle = 0);
   void LoadField(const char *field_name);

   /// Interact with Visit data files
   void SaveVisitData();
   void LoadVisitData(int cycle = 0);

   /// We will delete the mesh and fields if we own them
   ~DataCollection();
};

}

#endif
