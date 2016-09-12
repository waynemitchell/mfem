# Find GnuTLS, searching ${GNUTLS_DIR} first. If not successful, tries the
# standard FindGnuTLS.cmake. Defines:
#    GNUTLS_FOUND
#    GNUTLS_INCLUDE_DIRS
#    GNUTLS_LIBRARIES

include(MfemCmakeUtilities)
set(_GnuTLS_REQUIRED_PACKAGES "ALT:" "GnuTLS")
mfem_find_package(_GnuTLS GNUTLS GNUTLS_DIR "include" gnutls/gnutls.h
  "lib" "gnutls;libgnutls"
  "Paths to headers required by GnuTLS." "Libraries required by GnuTLS.")
