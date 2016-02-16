// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_SOCKETSTREAM
#define MFEM_SOCKETSTREAM

#include "../config/config.hpp"
#include <iostream>

#ifdef MFEM_USE_GNUTLS
#include <gnutls/gnutls.h>
#include <gnutls/openpgp.h>
#endif

namespace mfem
{

class socketbuf : public std::streambuf
{
protected:
   int socket_descriptor;
   static const int buflen = 1024;
   char ibuf[buflen], obuf[buflen];

public:
   socketbuf()
   {
      socket_descriptor = -1;
   }

   explicit socketbuf(int sd)
   {
      socket_descriptor = sd;
      setp(obuf, obuf + buflen);
   }

   socketbuf(const char hostname[], int port)
   {
      socket_descriptor = -1;
      open(hostname, port);
   }

   /** Attach a new socket descriptor to the socketbuf.
       Returns the old socket descriptor which is NOT closed. */
   virtual int attach(int sd);

   int detach() { return attach(-1); }

   virtual int open(const char hostname[], int port);

   virtual int close();

   int getsocketdescriptor() { return socket_descriptor; }

   bool is_open() { return (socket_descriptor >= 0); }

   virtual ~socketbuf() { close(); }

protected:
   virtual int sync();

   virtual int_type underflow();

   virtual int_type overflow(int_type c = traits_type::eof());

   virtual std::streamsize xsgetn(char_type *__s, std::streamsize __n);

   virtual std::streamsize xsputn(const char_type *__s, std::streamsize __n);
};


class socketstream : public std::iostream
{
protected:
   socketbuf *buf__;

public:
   socketstream() : buf__(new socketbuf) { std::iostream::rdbuf(buf__); }

   socketstream(socketbuf *buf) : std::iostream(buf), buf__(buf) { }

   explicit socketstream(int s)
      : buf__(new socketbuf(s)) { std::iostream::rdbuf(buf__); }

   socketstream(const char hostname[], int port)
      : buf__(new socketbuf)
   { std::iostream::rdbuf(buf__); open(hostname, port); }

   socketbuf *rdbuf() { return buf__; }

   int open(const char hostname[], int port)
   {
      int err = buf__->open(hostname, port);
      if (err)
      {
         setstate(std::ios::failbit);
      }
      else
      {
         clear();
      }
      return err;
   }

   int close() { return buf__->close(); }

   bool is_open() { return buf__->is_open(); }

   virtual ~socketstream() { delete buf__; }
};


class socketserver
{
private:
   int listen_socket;

public:
   explicit socketserver(int port);

   bool good() { return (listen_socket >= 0); }

   int close();

   int accept();

   int accept(socketstream &sockstr);

   ~socketserver() { close(); }
};

#ifdef MFEM_USE_GNUTLS

class GnuTLS_status
{
protected:
   int res;

public:
   GnuTLS_status() : res(GNUTLS_E_SUCCESS) { }

   bool good() const { return (res == GNUTLS_E_SUCCESS); }

   void set_result(int result) { res = result; }

   int get_result() const { return res; }

   void print_on_error(const char *msg) const
   {
      if (good()) { return; }
      std::cout << "Error in " << msg << ": " << gnutls_strerror(res)
                << std::endl;
   }
};

class GnuTLS_global_state
{
protected:
   gnutls_dh_params_t dh_params;
   bool glob_init_ok;

   void generate_dh_params();

public:
   GnuTLS_global_state();
   ~GnuTLS_global_state();

   GnuTLS_status status;

   void set_log_level(int level)
   { if (status.good()) { gnutls_global_set_log_level(level); } }

   gnutls_dh_params_t get_dh_params()
   {
      if (!dh_params) { generate_dh_params(); }
      return dh_params;
   }
};

class GnuTLS_session_params
{
protected:
   gnutls_certificate_credentials_t my_cred;
   unsigned int my_flags;

public:
   GnuTLS_global_state &state;
   GnuTLS_status status;

   GnuTLS_session_params(GnuTLS_global_state &state,
                         const char *pubkey_file,
                         const char *privkey_file,
                         const char *trustedkeys_file,
                         unsigned int flags);
   ~GnuTLS_session_params()
   {
      if (my_cred) { gnutls_certificate_free_credentials(my_cred); }
   }

   gnutls_certificate_credentials_t get_cred() const { return my_cred; }
   unsigned int get_flags() const { return my_flags; }
};

class GnuTLS_socketbuf : public socketbuf
{
protected:
   GnuTLS_status status;
   gnutls_session_t session;
   bool session_started;

   const GnuTLS_session_params &params;
   gnutls_certificate_credentials_t my_cred; // same as params.my_cred

   int handshake();
   void start_session();
   void end_session();

public:
   GnuTLS_socketbuf(const GnuTLS_session_params &p)
      : session_started(false), params(p), my_cred(params.get_cred())
   { status.set_result(params.status.get_result()); }

   virtual ~GnuTLS_socketbuf() { close(); }

   bool gnutls_good() const { return status.good(); }

   /** Attach a new socket descriptor to the socketbuf.
       Returns the old socket descriptor which is NOT closed. */
   virtual int attach(int sd);

   virtual int open(const char hostname[], int port);

   virtual int close();

protected:
   virtual int sync();

   virtual int_type underflow();

   // Same as in the base class:
   // virtual int_type overflow(int_type c = traits_type::eof());

   virtual std::streamsize xsgetn(char_type *__s, std::streamsize __n);

   virtual std::streamsize xsputn(const char_type *__s, std::streamsize __n);
};

/// Secure socket stream, based on GnuTLS using OpenPGP/GnuPG keys.
class GnuTLS_socketstream : public socketstream
{
public:
   GnuTLS_socketstream(const GnuTLS_session_params &p)
      : socketstream(new GnuTLS_socketbuf(p))
   {
      if (((GnuTLS_socketbuf*)buf__)->gnutls_good())
      {
         clear();
      }
      else
      {
         setstate(std::ios::failbit);
      }
   }
};

#endif // MFEM_USE_GNUTLS


/// Socket stream connection to GLVis using connection type based on the MFEM
/// build-time option MFEM_USE_GNUTLS. This class is a shortcut for connecting
/// to GLVis regardless of the value of MFEM_USE_GNUTLS.
class GLVis_socketstream : public
#ifndef MFEM_USE_GNUTLS
   socketstream
#else
   GnuTLS_socketstream
#endif
{
#ifdef MFEM_USE_GNUTLS
protected:
   static int num_glvis_sockets;
   static GnuTLS_global_state *state;
   static GnuTLS_session_params *params;
   static GnuTLS_session_params &add_socket();
#endif

public:
   GLVis_socketstream();
   GLVis_socketstream(const char hostname[], int port);
   ~GLVis_socketstream();
};

}

#endif
