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

#ifndef MFEM_OPTPARSER
#define MFEM_OPTPARSER

#include "array.hpp"

namespace mfem
{

/** Class for parsing command-line options.

    The class is initialized with argc and argv, and new options are added with
    the AddOption method. Currently options of type bool, int, double and char*
    are supported.

    See the MFEM examples for sample use.
*/
class OptionsParser
{
public:
   enum OptionType { INT, DOUBLE, STRING, ENABLE, DISABLE };

private:
   struct Option
   {
      OptionType type;
      void *var_ptr;
      const char *short_name;
      const char *long_name;
      const char *description;

      Option(OptionType _type, void *_var_ptr, const char *_short_name,
             const char *_long_name, const char *_description)
         : type(_type), var_ptr(_var_ptr), short_name(_short_name),
           long_name(_long_name), description(_description) { }
   };

   int argc;
   char **argv;
   Array<Option> options;
   int error;

public:
   OptionsParser(int _argc, char *_argv[])
      : argc(_argc), argv(_argv)
   {
      error = 0;
   }
   void AddOption(bool *var, const char *enable_short_name,
                  const char *enable_long_name, const char *disable_short_name,
                  const char *disable_long_name, const char *description)
   {
      options.Append(Option(ENABLE, var, enable_short_name, enable_long_name,
                            description));
      options.Append(Option(DISABLE, var, disable_short_name, disable_long_name,
                            description));
   }
   void AddOption(int *var, const char *short_name, const char *long_name,
                  const char *description)
   {
      options.Append(Option(INT, var, short_name, long_name, description));
   }
   void AddOption(double *var, const char *short_name, const char *long_name,
                  const char *description)
   {
      options.Append(Option(DOUBLE, var, short_name, long_name, description));
   }
   void AddOption(const char **var, const char *short_name,
                  const char *long_name, const char *description)
   {
      options.Append(Option(STRING, var, short_name, long_name, description));
   }

   void Parse();
   bool Good() const { return (error == 0); }
   void PrintOptions(std::ostream &out) const;
   void PrintUsage(std::ostream &out) const;
};

}

#endif
