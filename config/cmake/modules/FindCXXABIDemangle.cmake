# Check for "abi::__cxa_demangle" in <cxxabi.h>.
# Defines the variables:
#    CXXABIDemangle_FOUND
#    CXXABIDemangle_LIBRARIES (if needed)

include(MfemCmakeUtilities)
mfem_find_library(CXXABIDemangle CXXABIDemangle ""
 "Library required for abi::__cxa_demangle." CXXABIDemangle_BUILD
  "
#include <cxxabi.h>
int main()
{
  int demangle_status;
  const char name[] = \"__ZN4mfem10mfem_errorEPKc\";
  char *name_demangle =
    abi::__cxa_demangle(name, NULL, NULL, &demangle_status);
  return 0;
}
")
