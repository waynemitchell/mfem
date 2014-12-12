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
#include "../linalg/vector.hpp"
#include <cctype>

namespace mfem
{

using namespace std;

int isValidAsInt(char * s)
{
	if ( s == NULL || *s == '\0' )
		return 0; //Empty string

	if ( *s == '+' || *s == '-' )
		++s;

	if ( *s == '\0')
		return 0;  //sign character only

	while(*s)
	{
		if( !isdigit(*s) )
			return 0;
		++s;
	}

	return 1;
}

int isValidAsDouble(char * s)
{
	//A valid floating point number for atof using the "C" locale is formed by
	// - an optional sign character (+ or -),
	// - followed by a sequence of digits, optionally containing a decimal-point character (.),
	// - optionally followed by an exponent part (an e or E character followed by an optional sign and a sequence of digits).

	if ( s == NULL || *s == '\0' )
		return 0; //Empty string

	if ( *s == '+' || *s == '-' )
		++s;

	if ( *s == '\0')
		return 0;  //sign character only

	while(*s)
	{
		if(!isdigit(*s))
			break;
		++s;
	}

	if(*s == '\0')
		return 1; //s = "123"

	if(*s == '.')
	{
		++s;
		while(*s)
		{
			if(!isdigit(*s))
				break;
			++s;
		}
		if(*s == '\0')
			return 1; //this is a fixed point double s = "123." or "123.45"
	}

	if(*s == 'e' || *s == 'E')
	{
		++s;
		return isValidAsInt(s);
	}
	else
		return 0; //we have encounter a wrong character


}

void parseArray(char * str, Array<int> & var)
{
	var.SetSize(0);
	std::stringstream input(str);
	int val;
	while( input >> val)
		var.Append(val);
}

void parseVector(char * str, Vector & var)
{
	int nentries = 0;
	   double val;
	{
	   std::stringstream input(str);
	   while( input >> val)
		 ++nentries;
	}

	var.SetSize(nentries);
	{
	   nentries = 0;
	   std::stringstream input(str);
       while( input >> val)
		 var(nentries++) = val;
	}
}

void OptionsParser::Parse()
{
	option_check.SetSize(options.Size());
	option_check = 0;
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

            if( option_check[j] )
            {
            	error = argc + j;
            	return;
            }
            option_check[j] = 1;

            i++;
            if (type != ENABLE && type != DISABLE && i >= argc)
            {
               // missing argument
               error = -1;
               return;
            }

            int isValid = 1;
            switch (options[j].type)
            {
            case INT:
               isValid = isValidAsInt(argv[i]);
               *(int *)(options[j].var_ptr) = atoi(argv[i++]);
               break;
            case DOUBLE:
               isValid = isValidAsDouble(argv[i]);
               *(double *)(options[j].var_ptr) = atof(argv[i++]);
               break;
            case STRING:
               *(const char **)(options[j].var_ptr) = argv[i++];
               break;
            case ENABLE:
               *(bool *)(options[j].var_ptr) = true;
               option_check[j+1] = 1;  //Do not allow to use the DISABLE Option
               break;
            case DISABLE:
               *(bool *)(options[j].var_ptr) = false;
               option_check[j-1] = 1;  //Do not allow to use the ENABLE Option
               break;
            case ARRAY:
            	parseArray(argv[i++], *(Array<int>*)(options[j].var_ptr) );
            	break;
            case VECTOR:
            	parseVector(argv[i++], *(Vector*)(options[j].var_ptr) );
            	break;
            }

            if(!isValid)
            {
            	error = 2*argc + i - 1;
            	return;
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
         case ARRAY:
        	 ((Array<int>*)(options[j].var_ptr))->Print(
        			 out, ((Array<int>*)(options[j].var_ptr))->Size() );
        	 break;
         case VECTOR:
        	 ((Vector*)(options[j].var_ptr))->Print(
        			 out, ((Vector*)(options[j].var_ptr))->Size() );
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
      out << "Unrecognized option: " << argv[error] << '\n' << line_sep;
   }
   else if ( error >= argc && error < 2*argc)
   {
	   if(options[error - argc].type == ENABLE )
		  out << "Option " << options[error - argc].long_name << " or "
		      << options[error - argc + 1].long_name << " provided multiple times\n" << line_sep;
	   else if(options[error - argc].type == DISABLE)
		  out << "Option " << options[error - argc - 1].long_name << " or "
		      << options[error - argc].long_name << " provided multiple times\n" << line_sep;
	   else
	      out << "Option " << options[error - argc].long_name << " provided multiple times\n" << line_sep;
   }
   else if (error > 2*argc)
   {
	   out << "Wrong option format " << argv[error - 2 * argc -1] << " " << argv[error - 2 * argc] << "\n";
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
         case ARRAY:
        	 ((Array<int>*)(options[j].var_ptr))->Print(
        			 out, ((Array<int>*)(options[j].var_ptr))->Size() );
        	 break;
         case VECTOR:
        	 ((Vector*)(options[j].var_ptr))->Print(
        			 out, ((Vector*)(options[j].var_ptr))->Size() );
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
