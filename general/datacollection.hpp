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

class FieldInfo
{
public:
   std::string association;
   int num_components;
   FieldInfo() {association = ""; num_components = 0;};
   FieldInfo(std::string _association, int _num_components) {association = _association; num_components = _num_components;};
};


class RootData
{
public:
   std::string base_name;
   int cycle;
   double time;
   int spatial_dim, topo_dim;
   int visit_max_levels_of_detail;
   int num_ranks;
   std::map<std::string, FieldInfo> field_info_map;

   RootData();

   /// Set FieldInfo
   void SetFieldInfo(std::string name, std::string association, int num_components) {field_info_map[name] = FieldInfo(association, num_components);};
   bool HasField(std::string name) {return field_info_map.count(name) == 1;};

   /// Interact with the json Visit root strings
   std::string GetVisitRootString();
   void ParseVisitRootString(std::string json_str);

   /// Interact with Visit Root files
   void SaveVisitRootFile();
   void LoadVisitRootFile(const char *fname);

   /// String conversions that will go away in C++11
   std::string to_padded_string(int i);
   std::string to_string(int i);
   int to_int(std::string str);
};


class DataCollection
{
protected:
   RootData root_data;
   Mesh* mesh;

   int my_rank;
   std::map<std::string, GridFunction*> field_map;

   /// Interact with the Mesh files
   void SaveMesh();
   void LoadMesh();

   /// Interact with the Field files
   void SaveFields();
   void LoadFields();

public:
   /// Construct passing in the necessary components
   DataCollection(const char* _colection_name, Mesh* _mesh);

   /// Accessors for the variables that can change
   void SetVisitParameters(int max_levels_of_detail) {root_data.visit_max_levels_of_detail = max_levels_of_detail;};

   /// Various Accessors
   void SetCycle(int c) {root_data.cycle = c;};
   int GetCycle() {return root_data.cycle;};
   void SetTime(double t) {root_data.time = t;};
   double GetTime() {return root_data.time;};
   Mesh *GetMesh() {return mesh;};
   GridFunction *GetField(const char *field_name);

   /// Interact with the fields map
   void RegisterField(const char *field_name, GridFunction *gf);
   bool HasField(const char *name) {return field_map.count(name) == 1;};

   /// Interact with Visit data files
   void SaveVisitData();
   void LoadVisitData(const char *_collection_name, int _cycle);

   /// For now we will leave the destructor empty and let the user delete the attached Mesh and GridFunctions
   ~DataCollection();
};

}

#endif
