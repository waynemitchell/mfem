# Find GnuTLS, searching ${GNUTLS_DIR} first.
# Defines (through FindGnuTLS.cmake):
#    GNUTLS_FOUND        (not cached)
#    GNUTLS_INCLUDE_DIRS (not cached)
#    GNUTLS_LIBRARIES    (not cached)
#    GNUTLS_INCLUDE_DIR  (cached, advanced)
#    GNUTLS_LIBRARY      (cached, advanced)
# Only the option REQUIRED is handled properly.

find_path(GNUTLS_INCLUDE_DIR gnutls/gnutls.h
   HINTS ${GNUTLS_DIR} $ENV{GNUTLS_DIR}
   PATH_SUFFIXES include
   NO_DEFAULT_PATH
   DOC "Directory with GnuTLS header.")

find_library(GNUTLS_LIBRARY NAMES gnutls libgnutls
   HINTS ${GNUTLS_DIR} $ENV{GNUTLS_DIR}
   PATH_SUFFIXES lib
   NO_DEFAULT_PATH
   DOC "The GnuTLS library.")

find_package(GnuTLS QUIET)

# Handle REQUIRED, message
set(GnuTLS_FIND_REQUIRED ${_GnuTLS_FIND_REQUIRED})
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GnuTLS
  " *** GnuTLS libarary not found. Please set GNUTLS_DIR."
  GNUTLS_LIBRARIES GNUTLS_INCLUDE_DIRS)
