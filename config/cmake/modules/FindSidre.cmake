# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

# Defines the following variables:
#   - SIDRE_FOUND
#   - SIDRE_LIBRARIES
#   - SIDRE_INCLUDE_DIRS

include(MfemCmakeUtilities)
# Find in advance the other components of ATK needed by MFEM when SIDRE is
# enabled: SPIO, SLIC, and common. This way mfem_find_package() will not try to
# call find_package() for these "packages", so we do not need separate files for
# each of them, e.g. FindSPIO.cmake, FindSLIC.cmake, etc.
if (MFEM_USE_MPI)
  # SPIO is only needed in parallel.
  message(STATUS "Looking for Sidre dependencies: SPIO, SLIC, ATK_common ...")
  mfem_find_package(SPIO SPIO SIDRE_DIR "include" spio/IOManager.hpp
    "lib" spio "Paths to headers required by SPIO."
    "Libraries required by SPIO.")
  set(ATK_OTHER_LIBS SPIO)
else()
  message(STATUS "Looking for Sidre dependencies: SLIC, ATK_common ...")
endif()
mfem_find_package(SLIC SLIC SIDRE_DIR "include" slic/slic.hpp "lib" slic
  "Paths to headers required by SLIC." "Libraries required by SLIC.")
mfem_find_package(ATK_common ATK_common SIDRE_DIR "include" common/ATKMacros.hpp
  "lib" common "Paths to headers required by ATK_common."
  "Libraries required by ATK_common.")
set(ATK_OTHER_LIBS ${ATK_OTHER_LIBS} "SLIC" "ATK_common")
# Prepend ${ATK_OTHER_LIBS} to Sidre_REQUIRED_PACKAGES without forcing it into
# the cache.
set(Sidre_REQUIRED_PACKAGES ${ATK_OTHER_LIBS} ${Sidre_REQUIRED_PACKAGES})
mfem_find_package(Sidre SIDRE SIDRE_DIR "include" sidre/sidre.hpp "lib" sidre
  "Paths to headers required by Sidre." "Libraries required by Sidre.")
