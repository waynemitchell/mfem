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

   int accept(socketstream &sockstr);

   ~socketserver() { close(); }
};

#ifdef MFEM_USE_GNUTLS

class gnutls_socketbuf : public socketbuf
{
protected:
   bool gnutls_ok;
   int gnutls_res;
   unsigned int gnutls_flags;
   gnutls_session_t session;

   gnutls_certificate_credentials_t my_cred;
   gnutls_dh_params_t dh_params;
   bool glob_init_ok, my_cred_ok, dh_params_ok;

   void check_result(int result)
   {
      gnutls_res = result;
      gnutls_ok = (result == GNUTLS_E_SUCCESS);
   }

   void print_gnutls_err(const char *msg)
   {
      if (gnutls_ok) { return; }
      std::cout << "Error in " << msg << ": " << gnutls_strerror(gnutls_res)
                << std::endl;
   }

   int handshake();
   void start_session();
   void end_session();

public:
   gnutls_socketbuf(unsigned int flags, const char *pubkey_file,
                    const char *privkey_file, const char *trustedkeys_file);

   virtual ~gnutls_socketbuf();

   bool gnutls_good() const { return gnutls_ok; }

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

/// Secure socket stream, based on GNUTLS using OpenPGP/GnuPG keys.
class gnutls_socketstream : public socketstream
{
public:
   gnutls_socketstream(const char *pubkey_file, const char *privkey_file,
                       const char *trustedkeys_file,
                       unsigned int flags = GNUTLS_CLIENT)
      : socketstream(new gnutls_socketbuf(flags, pubkey_file,
                                          privkey_file, trustedkeys_file))
   {
      if (((gnutls_socketbuf*)buf__)->gnutls_good())
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

}

#endif
