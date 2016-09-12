# Find the libraries needed for backtrace in MFEM. This is a "meta-package",
# i.e. it simply combines all packages from MFEMBacktrace_REQUIRED_PACKAGES.
# Defines the following variables:
#   - MFEMBacktrace_FOUND
#   - MFEMBacktrace_LIBRARIES    (if needed)
#   - MFEMBacktrace_INCLUDE_DIRS (if needed)

include(MfemCmakeUtilities)
mfem_find_package(MFEMBacktrace MFEMBacktrace MFEMBacktrace_DIR "" "" "" ""
  "Paths to headers required by MFEM backtrace."
  "Libraries required by MFEM backtrace.")
