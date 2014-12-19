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


#include "datacollection.hpp"
#include "picojson.hpp"
#include "error.hpp"
#include "../fem/fem.hpp"
#include "../mesh/mesh.hpp"

#ifdef MFEM_USE_MPI
#include "../mesh/pmesh.hpp"
#endif

#include <fstream>

namespace mfem
{


///////////////////////////////// DataCollection Methods //////////////////////////////////////////

DataCollection::DataCollection(const char *_collection_name, Mesh *_mesh) 
{
   root_data.base_name = _collection_name;
   mesh = _mesh;
   my_rank = 0;
   root_data.num_ranks = 1;
#ifdef MFEM_USE_MPI
   ParMesh *par_mesh = dynamic_cast<ParMesh>(mesh);
   if (par_mesh) {
      my_rank = par_mesh->GetMyRank();
      num_ranks = par_mesh->GetNRanks();
   }
#endif

   root_data.cycle = 0;
   root_data.time = 0.0;
   root_data.spatial_dim = mesh->SpaceDimension();
   root_data.topo_dim = mesh->Dimension();
   root_data.visit_max_levels_of_detail = 25;

   field_map.erase(field_map.begin(), field_map.end());
}


void DataCollection::RegisterField(const char* name, GridFunction *gf)
{
   field_map[name] = gf;
   root_data.field_info_map[name] = FieldInfo("nodes", gf->VectorDim());
}


GridFunction *DataCollection::GetField(const char *field_name) 
{
   if (HasField(field_name))
      return field_map[field_name];
   else
      return NULL;
}



void DataCollection::SaveVisitData()
{
   root_data.SaveVisitRootFile();
   SaveMesh();
   SaveFields();
}


void DataCollection::LoadVisitData(const char *root_fname, int _cycle)
{
   root_data.LoadVisitRootFile(root_fname);
   LoadMesh();
   LoadFields();
}


void DataCollection::SaveMesh()
{
   std::string mesh_fname = root_data.base_name + "_" + root_data.to_padded_string(root_data.cycle) + "/"
                              + root_data.base_name + "_" + root_data.to_padded_string(root_data.cycle) + "."
                              + root_data.to_padded_string(my_rank) + ".mesh";

   std::ofstream file(mesh_fname.c_str());
   MFEM_ASSERT(file.is_open(), "Unable to open file for output:  " << mesh_fname);
   mesh->Print(file);
   file.close();
}


void DataCollection::LoadMesh()
{


}


void DataCollection::SaveFields()
{
   std::string path_left = root_data.base_name + "_" + root_data.to_padded_string(root_data.cycle) + "/"
                           + root_data.base_name + "_" + root_data.to_padded_string(root_data.cycle) + "_";
   std::string path_right = root_data.to_padded_string(my_rank) + ".gf";

   for (std::map<std::string,GridFunction*>::iterator it=field_map.begin(); it!=field_map.end(); ++it)
   {
      std::string fname = path_left + it->first + path_right;
      std::ofstream file(fname.c_str());
      MFEM_ASSERT(file.is_open(), "Unable to open file for output:  " << fname);
      (it->second)->Save(file);
      file.close();
   }
}


void DataCollection::LoadFields()
{


}



//////////////////////////////////////// RootData Methods /////////////////////////////////////////

RootData::RootData()
{
   base_name = "";
   cycle = 0;
   time = 0.0;
   spatial_dim = 0;
   topo_dim = 0;
   visit_max_levels_of_detail = 25;
   num_ranks = 0;
   field_info_map.erase(field_info_map.begin(), field_info_map.end());
}


void RootData::SaveVisitRootFile()
{
   std::string fname = base_name + "_" + to_padded_string(cycle) + ".mfem_root";

   std::ofstream file(fname.c_str());
   MFEM_ASSERT(file.is_open(), "Unable to open Visit Root file for output:  " << fname);
   file << GetVisitRootString();
   file.close();
}


void RootData::LoadVisitRootFile(const char *fname)
{
   std::ifstream file(fname);
   MFEM_ASSERT(file.is_open(), "Unable to open Visit Root file for input:  " << fname);
   std::stringstream buffer;
   buffer << file.rdbuf();
   ParseVisitRootString(buffer.str());
   file.close();
}


std::string RootData::GetVisitRootString()
{
   //Get the path string
   std::string path_str = base_name + "_" + to_padded_string(cycle) + "/" + base_name + "_" + to_padded_string(cycle);

   //We have to build the json tree inside out to get all the values in there
   picojson::object top, dsets, main, mesh, fields, field, mtags, ftags;

   //Build the mesh data
   mtags["spatial_dim"] = picojson::value(to_string(spatial_dim));
   mtags["topo_dim"] = picojson::value(to_string(topo_dim));
   mtags["max_lods"] = picojson::value(to_string(visit_max_levels_of_detail));
   mesh["path"] = picojson::value(path_str + ".%05d.mesh");
   mesh["tags"] = picojson::value(mtags);

   //Build the fields data entries
   for (std::map<std::string,FieldInfo>::iterator it=field_info_map.begin(); it!=field_info_map.end(); ++it)
   {
      ftags["assoc"] = picojson::value((it->second).association);
      ftags["comps"] = picojson::value(to_string((it->second).num_components));
      field["path"] = picojson::value(path_str + "_" + it->first + ".%05d.gf");
      field["tags"] = picojson::value(ftags);
      fields[it->first] = picojson::value(field);
   }

   main["cycle"] = picojson::value(double(cycle));
   main["time"] = picojson::value(time);
   main["domains"] = picojson::value(double(num_ranks));        //TODO:  Need a way to get the number of domains out of mesh?
   main["mesh"] = picojson::value(mesh);
   if (!field_info_map.empty())
      main["fields"] = picojson::value(fields);

   dsets["main"] = picojson::value(main);
   top["dsets"] = picojson::value(dsets);

   return picojson::value(top).serialize(true);
}


void RootData::ParseVisitRootString(std::string json)
{
   picojson::value top, dsets, main, mesh, fields;
   std::string parse_err = picojson::parse(top, json);
   MFEM_ASSERT(parse_err.empty(), "Unable to parse visit root data.");

   //Process "main"
   dsets = top.get("dsets");
   main = dsets.get("main");
   cycle = int(main.get("cycle").get<double>());
   time = main.get("time").get<double>();
   num_ranks = int(main.get("domains").get<double>()); 
   mesh = main.get("mesh");
   fields = main.get("fields");

   //....Process "mesh"
   std::string path = mesh.get("path").get<std::string>();
   std::size_t right_sep = path.find('_');
   MFEM_ASSERT(right_sep > 0, "Unable to parse visit root data.");
   base_name = path.substr(0, right_sep);

   spatial_dim = to_int(mesh.get("tags").get("spatial_dim").get<std::string>());
   topo_dim = to_int(mesh.get("tags").get("topo_dim").get<std::string>());
   visit_max_levels_of_detail = to_int(mesh.get("tags").get("max_lods").get<std::string>());

   //....Process "fields"
   field_info_map.erase(field_info_map.begin(), field_info_map.end());
   if (fields.is<picojson::object>())
   {
      picojson::object fields_obj = fields.get<picojson::object>();
      for (picojson::object::iterator it = fields_obj.begin(); it != fields_obj.end(); ++it)
      {
         picojson::value tags = it->second.get("tags");
         field_info_map[it->first] = FieldInfo(tags.get("assoc").get<std::string>(), 
                                                to_int(tags.get("comps").get<std::string>()));
      }
   }
}


//These little warts will go away in C++11
std::string RootData::to_string(int i)
{
   std::stringstream ss;
   ss << i;

   // trim leading spaces
   std::string out_str = ss.str();
   out_str = out_str.substr(out_str.find_first_not_of(" \t"));
   return out_str;
}

std::string RootData::to_padded_string(int i)
{
   std::ostringstream oss;
   oss << std::setw(5) << std::setfill('0') << i;
   return oss.str();
}

int RootData::to_int(std::string str)
{
   int i;
   std::stringstream(str) >> i;
   return i;
}


}
