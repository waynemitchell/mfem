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

#include "optparser.hpp"

namespace mfem
{

using namespace std;

void OptionsParser::Parse()
{
   for (int i = 1; i < argc; )
   {
      if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
      {
         // print help message
         error = -2;
         return;
      }

      for (int j = 0; true; j++)
      {
         if (j >= options.Size())
         {
            // unrecognized option
            error = i;
            return;
         }

         if (strcmp(argv[i], options[j].short_name) == 0 ||
             strcmp(argv[i], options[j].long_name) == 0)
         {
            OptionType type = options[j].type;

            i++;
            if (type != ENABLE && type != DISABLE && i >= argc)
            {
               // missing argument
               error = -1;
               return;
            }

            switch (options[j].type)
            {
            case INT:
               *(int *)(options[j].var_ptr) = atoi(argv[i++]);
               break;
            case DOUBLE:
               *(double *)(options[j].var_ptr) = atof(argv[i++]);
               break;
            case STRING:
               *(const char **)(options[j].var_ptr) = argv[i++];
               break;
            case ENABLE:
               *(bool *)(options[j].var_ptr) = true;
               break;
            case DISABLE:
               *(bool *)(options[j].var_ptr) = false;
               break;
            }

            break;
         }
      }
   }

   error = 0;
}

void OptionsParser::PrintOptions(ostream &out) const
{
   static const char *indent = "   ";

   out << "Options used:\n";
   for (int j = 0; j < options.Size(); j++)
   {
      OptionType type = options[j].type;

      out << indent;
      if (type == ENABLE)
      {
         if (*(bool *)(options[j].var_ptr) == true)
            out << options[j].long_name;
         else
            out << options[j+1].long_name;
         j++;
      }
      else
      {
         out << options[j].long_name << " ";
         switch (type)
         {
         case INT:
            out << *(int *)(options[j].var_ptr);
            break;
         case DOUBLE:
            out << *(double *)(options[j].var_ptr);
            break;
         case STRING:
            out << *(const char **)(options[j].var_ptr);
            break;
         }
      }
      out << '\n';
   }
}

void OptionsParser::PrintUsage(ostream &out) const
{
   static const char *indent = "   ";
   static const char *seprtr = ", ";
   static const char *descr_sep = "\n\t";
   static const char *line_sep = "";
   static const char *types[] = { " <int>", " <double>", " <string>", "", "" };

   out << line_sep;
   if (error == -1)
   {
      out << "Missing option after: " << argv[argc-1] << '\n' << line_sep;
   }
   else if (error > 0 && error < argc)
   {
      out << "Unrecongnized option: " << argv[error] << '\n' << line_sep;
   }

   out << "Usage: " << argv[0] << " [options] ...\n" << line_sep
       << "Options:\n" << line_sep;
   out << indent << "-h" << seprtr << "--help" << descr_sep
       << "Print this help message and exit.\n" << line_sep;
   for (int j = 0; j < options.Size(); j++)
   {
      OptionType type = options[j].type;

      out << indent << options[j].short_name << types[type]
          << seprtr << options[j].long_name << types[type]
          << seprtr;
      if (type == ENABLE)
      {
         j++;
         out << options[j].short_name << types[type] << seprtr
             << options[j].long_name << types[type] << seprtr
             << "current option: ";
         if (*(bool *)(options[j].var_ptr) == true)
            out << options[j-1].long_name;
         else
            out << options[j].long_name;
      }
      else
      {
         out << "current value: ";
         switch (type)
         {
         case INT:
            out << *(int *)(options[j].var_ptr);
            break;
         case DOUBLE:
            out << *(double *)(options[j].var_ptr);
            break;
         case STRING:
            out << *(const char **)(options[j].var_ptr);
            break;
         }
      }
      out << descr_sep;

      if (options[j].description)
         out << options[j].description << '\n';
      out << line_sep;
   }
}

}
