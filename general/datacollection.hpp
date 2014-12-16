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

#include "picojson.hpp"

namespace mfem
{


class FieldInfo
{
public:
   FieldInfo(std::string a, int d) {assoc = a; depth = d;};
   FieldInfo() {assoc = ""; depth = 0;};
   std::string assoc;
   int depth;
};


class DataCollection
{
protected:
   std::string base_name;
   int num_domains;
   int spatial_dim;
   int topo_dim;
   int max_lods;
   int cycle;
   double time;
   std::map<std::string, FieldInfo> field_map;

   std::string to_string(int i);
   int to_int(std::string str);

public:
   /// Construct passing in the necessary components
   DataCollection(const char* bn, int nd, int sd, int td, int mlod);

   /// Accessors for the variables that can change
   void SetNumDomains(int nd) {num_domains = nd;};
   int GetNumDomains() {return num_domains;};
   void SetSpatialDim(int sd) {spatial_dim = sd;};
   int GetSpatialDim() {return spatial_dim;};
   void SetTopoDim(int td) {topo_dim = td;};
   int GetTopoDim() {return topo_dim;};  
   void SetMaxLods(int mlod) {max_lods = mlod;};
   int GetMaxLods() {return max_lods;};
   void SetCycle(int c) {cycle = c;};
   int GetCycle() {return cycle;};
   void SetTime(double t) {time = t;};
   double GetTime() {return time;};

   /// Interact with the fields map
   void SetField(const char *name, int depth);
   bool HasField(const char *name) {return field_map.count(name) == 1;};
   int GetFieldDepth(const char *name);

   /// Serialization to a Visit root file
   std::string GetVisitRootString();
   void SaveVisitRootFile();

   /// Parsing from Visit root file to Data collection object
   void ParseVisitRootString(std::string json_str);
   void LoadVisitRootFile(const char *fname);

   /// Empty destructor
   ~DataCollection() {};
};

}

#endif
