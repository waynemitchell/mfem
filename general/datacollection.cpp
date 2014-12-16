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
#include "error.hpp"
#include <fstream>

namespace mfem
{

DataCollection::DataCollection(const char* bn, int nd, int sd, int td, int mlod) 
{
   base_name = bn;
   num_domains = nd; 
   spatial_dim = sd; 
   topo_dim = td; 
   max_lods = mlod;
   cycle = 0;
   time = 0.0;
   field_map.erase(field_map.begin(), field_map.end());
}


void DataCollection::SetField(const char* name, int depth)
{
   field_map[name] = FieldInfo("nodes", depth);
}


int DataCollection::GetFieldDepth(const char* name)
{
   if (HasField(name))
      return field_map[name].depth;
}


std::string DataCollection::GetVisitRootString()
{
   //Get the path string
   std::ostringstream oss;
   oss << std::setw(5) << std::setfill('0') << cycle;
   std::string path_str = base_name + "_" + oss.str() + "/" + base_name + "_" + oss.str();

   //We have to build the json tree inside out to get all the values in there
   picojson::object top, dsets, main, mesh, fields, field, mtags, ftags;

   //Build the mesh data
   mtags["spatial_dim"] = picojson::value(to_string(spatial_dim));
   mtags["topo_dim"] = picojson::value(to_string(topo_dim));
   mtags["max_lods"] = picojson::value(to_string(max_lods));
   mesh["path"] = picojson::value(path_str + ".%05d.mesh");
   mesh["tags"] = picojson::value(mtags);

   //Build the fields data entries
   for (std::map<std::string,FieldInfo>::iterator it=field_map.begin(); it!=field_map.end(); ++it)
   {
      ftags["assoc"] = picojson::value((it->second).assoc);
      ftags["comps"] = picojson::value(to_string((it->second).depth));
      field["path"] = picojson::value(path_str + "_" + it->first + ".%05d.gf");
      field["tags"] = picojson::value(ftags);
      fields[it->first] = picojson::value(field);
   }

   main["cycle"] = picojson::value(double(cycle));
   main["time"] = picojson::value(time);
   main["domains"] = picojson::value(double(num_domains));
   main["mesh"] = picojson::value(mesh);
   if (!field_map.empty())
      main["fields"] = picojson::value(fields);

   dsets["main"] = picojson::value(main);
   top["dsets"] = picojson::value(dsets);

   return picojson::value(top).serialize(true);
}

void DataCollection::SaveVisitRootFile()
{
   std::ostringstream oss;
   oss << std::setw(5) << std::setfill('0') << cycle;
   std::string fname = base_name + "_" + oss.str() + ".mfem_root";

   std::ofstream file(fname.c_str());
   MFEM_ASSERT(file.is_open(), "Unable to open Visit Root file:  " << fname);
   file << GetVisitRootString();
   file.close();
}


void DataCollection::ParseVisitRootString(std::string json)
{
   picojson::value top, dsets, main, mesh, fields;
   std::string parse_err = picojson::parse(top, json);
   MFEM_ASSERT(parse_err.empty(), "Unable to parse visit root data.");

   //Process "main"
   dsets = top.get("dsets");
   main = dsets.get("main");
   cycle = int(main.get("cycle").get<double>());
   time = main.get("time").get<double>();
   num_domains = int(main.get("domains").get<double>()); 
   mesh = main.get("mesh");
   fields = main.get("fields");

   //....Process "mesh"
   std::string path = mesh.get("path").get<std::string>();
   std::size_t right_sep = path.find('_');
   MFEM_ASSERT(right_sep > 0, "Unable to parse visit root data.");
   base_name = path.substr(0, right_sep);

   spatial_dim = to_int(mesh.get("tags").get("spatial_dim").get<std::string>());
   topo_dim = to_int(mesh.get("tags").get("topo_dim").get<std::string>());
   max_lods = to_int(mesh.get("tags").get("max_lods").get<std::string>());

   //....Process "fields"
   field_map.erase(field_map.begin(), field_map.end());
   if (fields.is<picojson::object>())
   {
      picojson::object fields_obj = fields.get<picojson::object>();
      for (picojson::object::iterator it = fields_obj.begin(); it != fields_obj.end(); ++it)
      {
         picojson::value tags = it->second.get("tags");
         field_map[it->first] = FieldInfo(tags.get("assoc").get<std::string>(), 
                                          to_int(tags.get("comps").get<std::string>()));
      }
   }
}


void DataCollection::LoadVisitRootFile(const char *fname)
{
   std::ifstream file(fname);
   MFEM_ASSERT(file.is_open(), "Unable to open Visit Root file for input:  " << fname);
   std::stringstream buffer;
   buffer << file.rdbuf();
   ParseVisitRootString(buffer.str());
   file.close();
}


//These little warts will go away in C++11
std::string DataCollection::to_string(int i)
{
   std::stringstream ss;
   ss << i;

   // trim leading spaces
   std::string out_str = ss.str();
   out_str = out_str.substr(out_str.find_first_not_of(" \t"));
   return out_str;
}

int DataCollection::to_int(std::string str)
{
   int i;
   std::stringstream(str) >> i;
   return i;
}

}
