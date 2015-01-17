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

#include "picojson.hpp"
#include "../general/error.hpp"
#include "fem.hpp"
#include "../mesh/mesh.hpp"

#ifdef MFEM_USE_MPI
#include "../mesh/pmesh.hpp"
#endif

#include <fstream>
#include <sys/stat.h>

namespace mfem
{

using namespace std;

// Number of digits and format used for the cycle and MPI rank

const int file_ext_digits = 5;
const string file_ext_format = ".%05d";

// Helper string functions. Will go away in C++11

string to_string(int i)
{
   stringstream ss;
   ss << i;

   // trim leading spaces
   string out_str = ss.str();
   out_str = out_str.substr(out_str.find_first_not_of(" \t"));
   return out_str;
}

string to_padded_string(int i)
{
   ostringstream oss;
   oss << setw(file_ext_digits) << setfill('0') << i;
   return oss.str();
}

int to_int(string str)
{
   int i;
   stringstream(str) >> i;
   return i;
}

// class DataCollection implementation

DataCollection::DataCollection(const char *collection_name)
{
   name = collection_name;
   myid = 0;
   num_procs = 1;
   serial = 0;
   cycle = -1;
   time = 0.0;
   field_map.erase(field_map.begin(), field_map.end());
}

DataCollection::DataCollection(const char *collection_name, Mesh *_mesh)
{
   name = collection_name;
   mesh = _mesh;
   myid = 0;
   num_procs = 1;
   serial = 1;
#ifdef MFEM_USE_MPI
   ParMesh *par_mesh = dynamic_cast<ParMesh*>(mesh);
   if (par_mesh)
   {
      myid = par_mesh->GetMyRank();
      num_procs = par_mesh->GetNRanks();
      serial = 0;
   }
#endif
   own_data = false;
   cycle = -1;
   time = 0.0;
   field_map.erase(field_map.begin(), field_map.end());
}

void DataCollection::RegisterField(const char* name, GridFunction *gf)
{
   field_map[name] = gf;
}

GridFunction *DataCollection::GetField(const char *field_name)
{
   if (HasField(field_name))
      return field_map[field_name];
   else
      return NULL;
}

void DataCollection::Save()
{
   string dir_name = name;
   if (cycle == -1)
      dir_name = name;
   else
      dir_name = name + "_" + to_padded_string(cycle);
   mkdir(dir_name.c_str(), 0777);

   string mesh_name;
   if (serial)
      mesh_name = dir_name + "/mesh";
   else
      mesh_name = dir_name + "/mesh." + to_padded_string(myid);
   ofstream mesh_file(mesh_name.c_str());
   MFEM_ASSERT(mesh_file.is_open(),
               "Unable to open file for output: " << mesh_name);
   mesh->Print(mesh_file);
   mesh_file.close();

   for (map<string,GridFunction*>::iterator it = field_map.begin();
        it != field_map.end(); ++it)
   {
      string field_name;
      if (serial)
         field_name = dir_name + "/" + it->first;
      else
         field_name = dir_name + "/" + it->first + "." + to_padded_string(myid);
      ofstream field_file(field_name.c_str());
      MFEM_ASSERT(field_file.is_open(),
                  "Unable to open file for output: " << field_name);
      (it->second)->Save(field_file);
      field_file.close();
   }
}

DataCollection::~DataCollection()
{
   if (own_data)
   {
      delete mesh;
      for (map<string,GridFunction*>::iterator it = field_map.begin();
           it != field_map.end(); ++it)
         delete it->second;
   }
}


// class VisItDataCollection implementation

VisItDataCollection::VisItDataCollection(const char *collection_name)
   : DataCollection(collection_name)
{
   serial = 0; // just for the file extensions
   cycle  = 0;

   visit_max_levels_of_detail = 25;
   field_info_map.erase(field_info_map.begin(), field_info_map.end());
}

VisItDataCollection::VisItDataCollection(const char *collection_name, Mesh *mesh)
   : DataCollection(collection_name, mesh)
{
   // always assume a time-dependent parallel run
   serial = 0;
   cycle  = 0;

   spatial_dim = mesh->SpaceDimension();
   topo_dim = mesh->Dimension();
   visit_max_levels_of_detail = 25;
   field_info_map.erase(field_info_map.begin(), field_info_map.end());
}

void VisItDataCollection::RegisterField(const char* name, GridFunction *gf)
{
   DataCollection::RegisterField(name, gf);
   field_info_map[name] = VisItFieldInfo("nodes", gf->VectorDim());
}

void VisItDataCollection::SetVisItParameters(int max_levels_of_detail)
{
   visit_max_levels_of_detail = max_levels_of_detail;
}

void VisItDataCollection::Save()
{
   if (myid == 0)
   {
      string root_name = name + "_" + to_padded_string(cycle) + ".mfem_root";
      ofstream root_file(root_name.c_str());
      MFEM_ASSERT(root_file.is_open(),
                  "Unable to open VisIt Root file for output:  " << root_name);
      root_file << GetVisItRootString();
      root_file.close();
   }

   DataCollection::Save();
}

void VisItDataCollection::Load(int _cycle)
{
#ifdef MFEM_USE_MPI
   MFEM_ABORT("VisItDataCollection::Load() does not work in parallel");
#endif

   cycle = _cycle;
   string root_name = name + "_" + to_padded_string(cycle) + ".mfem_root";
   LoadVisItRootFile(root_name);
   LoadMesh();
   LoadFields();
   own_data = true;
}

void VisItDataCollection::LoadVisItRootFile(string root_name)
{
   ifstream root_file(root_name.c_str());
   MFEM_ASSERT(root_file.is_open(),
               "Unable to open VisIt Root file for input:  " << root_name);
   stringstream buffer;
   buffer << root_file.rdbuf();
   ParseVisItRootString(buffer.str());
   root_file.close();
}

void VisItDataCollection::LoadMesh()
{
   string mesh_fname = name + "_" + to_padded_string(cycle) + "/mesh."
      + to_padded_string(myid);
   ifstream file(mesh_fname.c_str());
   MFEM_ASSERT(file.is_open(), "Unable to open file for input:  " << mesh_fname);
   mesh = new Mesh(file); // todo in parallel
   file.close();
   spatial_dim = mesh->SpaceDimension();
   topo_dim = mesh->Dimension();
}

void VisItDataCollection::LoadFields()
{
   string path_left = name + "_" + to_padded_string(cycle) + "/";
   string path_right = "." + to_padded_string(myid);

   field_map.erase(field_map.begin(), field_map.end());
   for (map<string,VisItFieldInfo>::iterator it = field_info_map.begin();
        it != field_info_map.end(); ++it)
   {
      string fname = path_left + it->first + path_right;
      ifstream file(fname.c_str());
      MFEM_ASSERT(file.is_open(), "Unable to open file for input:  " << fname);
      field_map[it->first] = new GridFunction(mesh, file);
      file.close();
   }
}

string VisItDataCollection::GetVisItRootString()
{
   // Get the path string
   string path_str = name + "_" + to_padded_string(cycle) + "/";

   // We have to build the json tree inside out to get all the values in there
   picojson::object top, dsets, main, mesh, fields, field, mtags, ftags;

   // Build the mesh data
   mtags["spatial_dim"] = picojson::value(to_string(spatial_dim));
   mtags["topo_dim"] = picojson::value(to_string(topo_dim));
   mtags["max_lods"] = picojson::value(to_string(visit_max_levels_of_detail));
   mesh["path"] = picojson::value(path_str + "mesh" + file_ext_format);
   mesh["tags"] = picojson::value(mtags);

   // Build the fields data entries
   for (map<string,VisItFieldInfo>::iterator it = field_info_map.begin();
        it != field_info_map.end(); ++it)
   {
      ftags["assoc"] = picojson::value((it->second).association);
      ftags["comps"] = picojson::value(to_string((it->second).num_components));
      field["path"] = picojson::value(path_str + it->first + file_ext_format);
      field["tags"] = picojson::value(ftags);
      fields[it->first] = picojson::value(field);
   }

   main["cycle"] = picojson::value(double(cycle));
   main["time"] = picojson::value(time);
   main["domains"] = picojson::value(double(num_procs));
   main["mesh"] = picojson::value(mesh);
   if (!field_info_map.empty())
      main["fields"] = picojson::value(fields);

   dsets["main"] = picojson::value(main);
   top["dsets"] = picojson::value(dsets);

   return picojson::value(top).serialize(true);
}

void VisItDataCollection::ParseVisItRootString(string json)
{
   picojson::value top, dsets, main, mesh, fields;
   string parse_err = picojson::parse(top, json);
   MFEM_ASSERT(parse_err.empty(), "Unable to parse visit root data.");

   // Process "main"
   dsets = top.get("dsets");
   main = dsets.get("main");
   cycle = int(main.get("cycle").get<double>());
   time = main.get("time").get<double>();
   num_procs = int(main.get("domains").get<double>());
   mesh = main.get("mesh");
   fields = main.get("fields");

   // ... Process "mesh"
   string path = mesh.get("path").get<string>();
   size_t right_sep = path.find('_');
   MFEM_ASSERT(right_sep > 0, "Unable to parse visit root data.");
   name = path.substr(0, right_sep);

   spatial_dim = to_int(mesh.get("tags").get("spatial_dim").get<string>());
   topo_dim = to_int(mesh.get("tags").get("topo_dim").get<string>());
   visit_max_levels_of_detail = to_int(mesh.get("tags").get("max_lods").get<string>());

   // ... Process "fields"
   field_info_map.erase(field_info_map.begin(), field_info_map.end());
   if (fields.is<picojson::object>())
   {
      picojson::object fields_obj = fields.get<picojson::object>();
      for (picojson::object::iterator it = fields_obj.begin(); it != fields_obj.end(); ++it)
      {
         picojson::value tags = it->second.get("tags");
         field_info_map[it->first] = VisItFieldInfo(tags.get("assoc").get<string>(),
                                                    to_int(tags.get("comps").get<string>()));
      }
   }
}

}
