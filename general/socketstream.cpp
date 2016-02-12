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

#ifdef _WIN32
// Turn off CRT deprecation warnings for strerror (VS 2013)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "socketstream.hpp"

#include <cstring>      // memset, memcpy, strerror
#include <cerrno>       // errno
#ifndef _WIN32
#include <netdb.h>      // gethostbyname
#include <arpa/inet.h>  // htons
#include <sys/types.h>  // socket, setsockopt, connect, recv, send
#include <sys/socket.h> // socket, setsockopt, connect, recv, send
#include <unistd.h>     // close
#include <netinet/in.h> // sockaddr_in
#define closesocket (::close)
#else
#include <winsock.h>
typedef int ssize_t;
// Link with ws2_32.lib
#pragma comment(lib, "ws2_32.lib")
#endif

namespace mfem
{

int socketbuf::attach(int sd)
{
   int old_sd = socket_descriptor;
   pubsync();
   socket_descriptor = sd;
   setg(NULL, NULL, NULL);
   setp(obuf, obuf + buflen);
   return old_sd;
}

int socketbuf::open(const char hostname[], int port)
{
   struct sockaddr_in  sa;
   struct hostent     *hp;

   close();
   setg(NULL, NULL, NULL);
   setp(obuf, obuf + buflen);

   hp = gethostbyname(hostname);
   if (hp == NULL)
   {
      socket_descriptor = -3;
      return -1;
   }
   memset(&sa, 0, sizeof(sa));
   memcpy((char *)&sa.sin_addr, hp->h_addr, hp->h_length);
   sa.sin_family = hp->h_addrtype;
   sa.sin_port = htons(port);
   socket_descriptor = socket(hp->h_addrtype, SOCK_STREAM, 0);
   if (socket_descriptor < 0)
   {
      return -1;
   }

#if defined __APPLE__
   // OS X does not support the MSG_NOSIGNAL option of send().
   // Instead we can use the SO_NOSIGPIPE socket option.
   int on = 1;
   if (setsockopt(socket_descriptor, SOL_SOCKET, SO_NOSIGPIPE,
                  (char *)(&on), sizeof(on)) < 0)
   {
      closesocket(socket_descriptor);
      socket_descriptor = -2;
      return -1;
   }
#endif

   if (connect(socket_descriptor,
               (const struct sockaddr *)&sa, sizeof(sa)) < 0)
   {
      closesocket(socket_descriptor);
      socket_descriptor = -2;
      return -1;
   }
   return 0;
}

int socketbuf::close()
{
   if (is_open())
   {
      pubsync();
      int err = closesocket(socket_descriptor);
      socket_descriptor = -1;
      return err;
   }
   return 0;
}

int socketbuf::sync()
{
   ssize_t bw, n = pptr() - pbase();
   // std::cout << "[socketbuf::sync n=" << n << ']' << std::endl;
   while (n > 0)
   {
#ifdef MSG_NOSIGNAL
      bw = send(socket_descriptor, pptr() - n, n, MSG_NOSIGNAL);
#else
      bw = send(socket_descriptor, pptr() - n, n, 0);
#endif
      if (bw < 0)
      {
#ifdef MFEM_DEBUG
         std::cout << "Error in send(): " << strerror(errno) << std::endl;
#endif
         setp(pptr() - n, obuf + buflen);
         pbump(n);
         return -1;
      }
      n -= bw;
   }
   setp(obuf, obuf + buflen);
   return 0;
}

socketbuf::int_type socketbuf::underflow()
{
   // assuming (gptr() < egptr()) is false
   ssize_t br = recv(socket_descriptor, ibuf, buflen, 0);
   // std::cout << "[socketbuf::underflow br=" << br << ']'
   //           << std::endl;
   if (br <= 0)
   {
#ifdef MFEM_DEBUG
      if (br < 0)
      {
         std::cout << "Error in recv(): " << strerror(errno) << std::endl;
      }
#endif
      setg(NULL, NULL, NULL);
      return traits_type::eof();
   }
   setg(ibuf, ibuf, ibuf + br);
   return traits_type::to_int_type(*ibuf);
}

socketbuf::int_type socketbuf::overflow(int_type c)
{
   if (sync() < 0)
   {
      return traits_type::eof();
   }
   if (traits_type::eq_int_type(c, traits_type::eof()))
   {
      return traits_type::not_eof(c);
   }
   *pptr() = traits_type::to_char_type(c);
   pbump(1);
   return c;
}

std::streamsize socketbuf::xsgetn(char_type *__s, std::streamsize __n)
{
   // std::cout << "[socketbuf::xsgetn __n=" << __n << ']'
   //           << std::endl;
   const std::streamsize bn = egptr() - gptr();
   if (__n <= bn)
   {
      traits_type::copy(__s, gptr(), __n);
      gbump(__n);
      return __n;
   }
   traits_type::copy(__s, gptr(), bn);
   setg(NULL, NULL, NULL);
   std::streamsize remain = __n - bn;
   char_type *end = __s + __n;
   ssize_t br;
   while (remain > 0)
   {
      br = recv(socket_descriptor, end - remain, remain, 0);
      if (br <= 0)
      {
#ifdef MFEM_DEBUG
         if (br < 0)
         {
            std::cout << "Error in recv(): " << strerror(errno) << std::endl;
         }
#endif
         return (__n - remain);
      }
      remain -= br;
   }
   return __n;
}

std::streamsize socketbuf::xsputn(const char_type *__s, std::streamsize __n)
{
   // std::cout << "[socketbuf::xsputn __n=" << __n << ']'
   //           << std::endl;
   if (pptr() + __n <= epptr())
   {
      traits_type::copy(pptr(), __s, __n);
      pbump(__n);
      return __n;
   }
   if (sync() < 0)
   {
      return 0;
   }
   ssize_t bw;
   std::streamsize remain = __n;
   const char_type *end = __s + __n;
   while (remain > buflen)
   {
#ifdef MSG_NOSIGNAL
      bw = send(socket_descriptor, end - remain, remain, MSG_NOSIGNAL);
#else
      bw = send(socket_descriptor, end - remain, remain, 0);
#endif
      if (bw < 0)
      {
#ifdef MFEM_DEBUG
         std::cout << "Error in send(): " << strerror(errno) << std::endl;
#endif
         return (__n - remain);
      }
      remain -= bw;
   }
   if (remain > 0)
   {
      traits_type::copy(pptr(), end - remain, remain);
      pbump(remain);
   }
   return __n;
}


socketserver::socketserver(int port)
{
   listen_socket = socket(PF_INET, SOCK_STREAM, 0); // tcp socket
   if (listen_socket < 0)
   {
      return;
   }
   int on = 1;
   if (setsockopt(listen_socket, SOL_SOCKET, SO_REUSEADDR,
                  (char *)(&on), sizeof(on)) < 0)
   {
      closesocket(listen_socket);
      listen_socket = -2;
      return;
   }
   struct sockaddr_in sa;
   memset(&sa, 0, sizeof(sa));
   sa.sin_family = AF_INET;
   sa.sin_port = htons(port);
   sa.sin_addr.s_addr = INADDR_ANY;
   if (bind(listen_socket, (const struct sockaddr *)&sa, sizeof(sa)))
   {
      closesocket(listen_socket);
      listen_socket = -3;
      return;
   }
   const int backlog = 4;
   if (listen(listen_socket, backlog) < 0)
   {
      closesocket(listen_socket);
      listen_socket = -4;
      return;
   }
}

int socketserver::close()
{
   if (!good())
   {
      return 0;
   }
   int err = closesocket(listen_socket);
   listen_socket = -1;
   return err;
}

int socketserver::accept(socketstream &sockstr)
{
   if (!good())
   {
      return -1;
   }
   int socketd = ::accept(listen_socket, NULL, NULL);
   if (socketd >= 0)
   {
      sockstr.rdbuf()->close();
      sockstr.rdbuf()->attach(socketd);
   }
   return sockstr.rdbuf()->getsocketdescriptor();
}

#ifdef MFEM_USE_GNUTLS

static int mfem_gnutls_verify_callback(gnutls_session_t session)
{
   unsigned int status;
   int ret;
   gnutls_certificate_type_t type;
   gnutls_datum_t out;
   const char *hostname;

   hostname = (const char *) gnutls_session_get_ptr(session);
   ret = gnutls_certificate_verify_peers3(session, hostname, &status);
   if (ret < 0)
   {
      std::cout << "Error in gnutls_certificate_verify_peers3:"
                << gnutls_strerror(ret) << std::endl;
      return GNUTLS_E_CERTIFICATE_ERROR;
   }

   type = gnutls_certificate_type_get(session);
   ret = gnutls_certificate_verification_status_print(status, type, &out, 0);
   if (ret < 0)
   {
      std::cout << "Error in gnutls_certificate_verification_status_print:"
                << gnutls_strerror(ret) << std::endl;
      return GNUTLS_E_CERTIFICATE_ERROR;
   }
   std::cout << out.data << std::endl;
   gnutls_free(out.data);

   return status ? GNUTLS_E_CERTIFICATE_ERROR : 0;
}

static void mfem_gnutls_log_server(int level, const char *str)
{
   std::cout << "server |<" << level << ">| " << str << std::flush;
}

static void mfem_gnutls_log_client(int level, const char *str)
{
   std::cout << "client |<" << level << ">| " << str << std::flush;
}

gnutls_socketbuf::gnutls_socketbuf(unsigned int flags, const char *pubkey_file,
                                   const char *privkey_file,
                                   const char *trustedkeys_file)
{
   check_result(gnutls_global_init());
   print_gnutls_err("gnutls_global_init");
   glob_init_ok = gnutls_ok;

   gnutls_flags = gnutls_ok ? flags : 0;

#if 1
   // Enable logging
   if (gnutls_ok)
   {
      if (flags & GNUTLS_SERVER)
      {
         gnutls_global_set_log_function(mfem_gnutls_log_server);
      }
      else
      {
         gnutls_global_set_log_function(mfem_gnutls_log_client);
      }
      gnutls_global_set_log_level(1000);
   }
#endif

   // allocate my_cred
   if (gnutls_ok)
   {
      check_result(
         gnutls_certificate_allocate_credentials(&my_cred));
      print_gnutls_err("gnutls_certificate_allocate_credentials");
   }
   my_cred_ok = gnutls_ok;

   if (gnutls_ok)
   {
      check_result(
         gnutls_certificate_set_openpgp_key_file(
            my_cred, pubkey_file, privkey_file, GNUTLS_OPENPGP_FMT_RAW));
      print_gnutls_err("gnutls_certificate_set_openpgp_key_file");
   }

   if (gnutls_ok)
   {
      /*
      gnutls_certificate_set_pin_function(
         my_cred,
         (gnutls_pin_callback_t) fn,
         (void *) userdata);
      */
   }

   if (gnutls_ok)
   {
      check_result(
         gnutls_certificate_set_openpgp_keyring_file(
            my_cred, trustedkeys_file, GNUTLS_OPENPGP_FMT_RAW));
      print_gnutls_err("gnutls_certificate_set_openpgp_keyring_file");
   }

   dh_params_ok = false;
   if (gnutls_ok && (flags & GNUTLS_SERVER))
   {
      check_result(gnutls_dh_params_init(&dh_params));
      print_gnutls_err("gnutls_dh_params_init");
      dh_params_ok = gnutls_ok;
      if (gnutls_ok)
      {
         unsigned bits = gnutls_sec_param_to_pk_bits(GNUTLS_PK_DH,
                                                     GNUTLS_SEC_PARAM_LEGACY);
         // TODO: This function is slow! Run it only once.
         std::cout << "calling gnutls_dh_params_generate2 ..." << std::flush;
         check_result(gnutls_dh_params_generate2(dh_params, bits));
         std::cout << " done." << std::endl;
         print_gnutls_err("gnutls_dh_params_generate2");
         if (gnutls_ok)
         {
            gnutls_certificate_set_dh_params(my_cred, dh_params);
         }
      }
   }
}

gnutls_socketbuf::~gnutls_socketbuf()
{
   close();
   if (dh_params_ok) { gnutls_dh_params_deinit(dh_params); }
   if (my_cred_ok) { gnutls_certificate_free_credentials(my_cred); }
   if (glob_init_ok) { gnutls_global_deinit(); }
}

int gnutls_socketbuf::handshake()
{
   int err;
   do
   {
      err = gnutls_handshake(session);
      check_result(err);
      if (gnutls_ok)
      {
#if 1
         std::cout << "handshake successful, TLS version is "
                   << gnutls_protocol_get_name(
                      gnutls_protocol_get_version(session)) << std::endl;
#endif
         return 0;
      }
   }
   while (err == GNUTLS_E_INTERRUPTED || err == GNUTLS_E_AGAIN);
#ifdef MFEM_DEBUG
   print_gnutls_err("gnutls_handshake");
#endif
   return err;
}

void gnutls_socketbuf::start_session()
{
   // check for valid 'socket_descriptor'
   if (!is_open()) { return; }

   if (gnutls_ok)
   {
      check_result(gnutls_init(&session, gnutls_flags));
      print_gnutls_err("gnutls_init");
   }

   bool session_ok = gnutls_ok;
   if (gnutls_ok)
   {
      check_result(
         gnutls_priority_set_direct(
            session,
            "NONE:+VERS-TLS1.2:+CIPHER-ALL:+MAC-ALL:+SIGN-ALL:+COMP-ALL:"
            "+KX-ALL:+CTYPE-OPENPGP:+CURVE-ALL", NULL));
      print_gnutls_err("gnutls_priority_set_direct");
   }

   if (gnutls_ok)
   {
      // set session credentials
      check_result(
         gnutls_credentials_set(
            session, GNUTLS_CRD_CERTIFICATE, my_cred));
      print_gnutls_err("gnutls_credentials_set");
   }

   if (gnutls_ok)
   {
      const char *hostname = NULL; // no hostname verificaion
      gnutls_session_set_ptr(session, (void*)hostname);
      gnutls_certificate_set_verify_function(
         my_cred, mfem_gnutls_verify_callback);
      if (gnutls_flags & GNUTLS_SERVER)
      {
         // require clients to send certificate:
         gnutls_certificate_server_set_request(session, GNUTLS_CERT_REQUIRE);
      }
      gnutls_handshake_set_timeout(
         session, GNUTLS_DEFAULT_HANDSHAKE_TIMEOUT);
   }

   if (gnutls_ok)
   {
      gnutls_transport_set_int(session, socket_descriptor);

      handshake();
   }

   if (!gnutls_ok)
   {
      if (session_ok) { gnutls_deinit(session); }
      socketbuf::close();
   }
}

void gnutls_socketbuf::end_session()
{
   // check for valid 'socket_descriptor'
   if (!is_open()) { return; }

   int err;
   do
   {
      err = gnutls_bye(session, GNUTLS_SHUT_RDWR);
      check_result(err);
      if (gnutls_ok) { return; }
   }
   while (err == GNUTLS_E_AGAIN || err == GNUTLS_E_INTERRUPTED);
   print_gnutls_err("gnutls_bye");

   gnutls_deinit(session);
}

int gnutls_socketbuf::attach(int sd)
{
   end_session();

   int old_sd = socketbuf::attach(sd);

   start_session();

   return old_sd;
}

int gnutls_socketbuf::open(const char hostname[], int port)
{
   int err = socketbuf::open(hostname, port); // calls close()
   if (err) { return err; }

   start_session();

   return gnutls_ok ? 0 : -100;
}

/*
   // This function has the similar semantics with send(). The only
   // difference is that it accepts a GnuTLS session, and uses different
   // error codes. Note that if the send buffer is full, send() will block
   // this function.
   // If GNUTLS_E_INTERRUPTED or GNUTLS_E_AGAIN is returned, you must call
   // this function again, with the exact same parameters; alternatively you
   // could provide a NULL pointer for data, and 0 for size.
   // Returns: The number of bytes sent, or a negative error code. The number
   // of bytes sent might be less than data_size. The maximum number of bytes
   // this function can send in a single call depends on the negotiated
   // maximum record size.

   ssize_t gnutls_record_send(gnutls_session_t session, const void *data,
                              size_t data_size);

   // This function has the similar semantics with recv(). The only
   // difference is that it accepts a GnuTLS session, and uses different
   // error codes. In the special case that the peer requests a
   // renegotiation, the caller will receive an error code of
   // GNUTLS_E_REHANDSHAKE. In case of a client, this message may be simply
   // ignored, replied with an alert GNUTLS_A_NO_RENEGOTIATION, or replied
   // with a new handshake, depending on the client’s will. A server
   // receiving this error code can only initiate a new handshake or
   // terminate the session.
   // If EINTR is returned by the internal push function (the default is
   // recv()) then GNUTLS_E_INTERRUPTED will be returned. If
   // GNUTLS_E_INTERRUPTED or GNUTLS_E_AGAIN is returned, you must call this
   // function again to get the data. See also gnutls_record_get_direction().
   // Returns: The number of bytes received and zero on EOF (for stream
   // connections). A negative error code is returned in case of an error.
   // The number of bytes received might be less than the requested
   // data_size.

   ssize_t gnutls_record_recv(gnutls_session_t session, void *data,
                              size_t data_size);

   // This function checks if there are unread data in the gnutls buffers. If
   // the return value is non-zero the next call to gnutls_record_recv() is
   // guaranteed not to block. Returns the size of the data or zero.

   size_t gnutls_record_check_pending(gnutls_session_t session);

   // Terminates the current TLS/SSL connection. The connection should have
   // been initiated using gnutls_handshake(). how should be one of
   // GNUTLS_SHUT_RDWR, GNUTLS_SHUT_WR.
   // In case of GNUTLS_SHUT_RDWR the TLS session gets terminated and further
   // receives and sends will be disallowed. If the return value is zero you
   // may continue using the underlying transport layer. GNUTLS_SHUT_RDWR
   // sends an alert containing a close request and waits for the peer to
   // reply with the same message.
   // In case of GNUTLS_SHUT_WR the TLS session gets terminated and further
   // sends will be disallowed. In order to reuse the connection you should
   // wait for an EOF from the peer. GNUTLS_SHUT_WR sends an alert containing
   // a close request.
   // Note that not all implementations will properly terminate a TLS
   // connection. Some of them, usually for performance reasons, will
   // terminate only the underlying transport layer, and thus not
   // distinguishing between a malicious party prematurely terminating the
   // connection and normal termination.
   // This function may also return GNUTLS_E_AGAIN or GNUTLS_E_INTERRUPTED;
   // cf. gnutls_record_get_direction().
   // Returns: GNUTLS_E_SUCCESS on success, or an error code, see function
   // documentation for entire semantics.

   int gnutls_bye(gnutls_session_t session, gnutls_close_request_t how);

   // This function clears all buffers associated with the session. This
   // function will also remove session data from the session database if the
   // session was terminated abnormally.

   void gnutls_deinit(gnutls_session_t session);

   // If called, gnutls_record_send() will no longer send any records. Any
   // sent records will be cached until gnutls_record_uncork() is called.

   void gnutls_record_cork(gnutls_session_t session);

   // This resets the effect of gnutls_record_cork(), and flushes any pending
   // data. If the GNUTLS_RECORD_WAIT flag is specified then this function
   // will block until the data is sent or a fatal error occurs (i.e., the
   // function will retry on GNUTLS_E_AGAIN and GNUTLS_E_INTERRUPTED).
   // If the flag GNUTLS_RECORD_WAIT is not specified and the function is
   // interrupted then the GNUTLS_E_AGAIN or GNUTLS_E_INTERRUPTED errors will
   // be returned. To obtain the data left in the corked buffer use
   // gnutls_record_check_corked().
   // Returns: On success the number of transmitted data is returned, or
   // otherwise a negative error code.
   // flags: Could be zero or GNUTLS_RECORD_WAIT

   int gnutls_record_uncork(gnutls_session_t session, unsigned int flags);
*/

int gnutls_socketbuf::close()
{
   end_session();

   int err = socketbuf::close();

   return gnutls_ok ? err : -100;
}

int gnutls_socketbuf::sync()
{
   ssize_t bw, n = pptr() - pbase();
   while (n > 0)
   {
      bw = gnutls_record_send(session, pptr() - n, n);
      if (bw == GNUTLS_E_INTERRUPTED || bw == GNUTLS_E_AGAIN) { continue; }
      if (bw < 0)
      {
         check_result((int)bw);
#ifdef MFEM_DEBUG
         print_gnutls_err("gnutls_record_send");
#endif
         setp(pptr() - n, obuf + buflen);
         pbump(n);
         return -1;
      }
      n -= bw;
   }
   setp(obuf, obuf + buflen);
   return 0;
}

gnutls_socketbuf::int_type gnutls_socketbuf::underflow()
{
   ssize_t br;
   do
   {
      br = gnutls_record_recv(session, ibuf, buflen);
      if (br == GNUTLS_E_REHANDSHAKE)
      {
         continue; // TODO: replace with re-handshake
      }
   }
   while (br == GNUTLS_E_INTERRUPTED || br == GNUTLS_E_AGAIN);
   if (br <= 0)
   {
      if (br < 0)
      {
         check_result((int)br);
#ifdef MFEM_DEBUG
         print_gnutls_err("gnutls_record_recv");
#endif
      }
      setg(NULL, NULL, NULL);
      return traits_type::eof();
   }
   setg(ibuf, ibuf, ibuf + br);
   return traits_type::to_int_type(*ibuf);
}

std::streamsize gnutls_socketbuf::xsgetn(char_type *__s, std::streamsize __n)
{
   const std::streamsize bn = egptr() - gptr();
   if (__n <= bn)
   {
      traits_type::copy(__s, gptr(), __n);
      gbump(__n);
      return __n;
   }
   traits_type::copy(__s, gptr(), bn);
   setg(NULL, NULL, NULL);
   std::streamsize remain = __n - bn;
   char_type *end = __s + __n;
   ssize_t br;
   while (remain > 0)
   {
      do
      {
         br = gnutls_record_recv(session, end - remain, remain);
         if (br == GNUTLS_E_REHANDSHAKE)
         {
            continue; // TODO: replace with re-handshake
         }
      }
      while (br == GNUTLS_E_INTERRUPTED || br == GNUTLS_E_AGAIN);
      if (br <= 0)
      {
         if (br < 0)
         {
            check_result((int)br);
#ifdef MFEM_DEBUG
            print_gnutls_err("gnutls_record_recv");
#endif
         }
         return (__n - remain);
      }
      remain -= br;
   }
   return __n;
}

std::streamsize gnutls_socketbuf::xsputn(const char_type *__s,
                                         std::streamsize __n)
{
   if (pptr() + __n <= epptr())
   {
      traits_type::copy(pptr(), __s, __n);
      pbump(__n);
      return __n;
   }
   if (sync() < 0)
   {
      return 0;
   }
   ssize_t bw;
   std::streamsize remain = __n;
   const char_type *end = __s + __n;
   while (remain > buflen)
   {
      bw = gnutls_record_send(session, end - remain, remain);
      if (bw == GNUTLS_E_INTERRUPTED || bw == GNUTLS_E_AGAIN) { continue; }
      if (bw < 0)
      {
         check_result((int)bw);
#ifdef MFEM_DEBUG
         print_gnutls_err("gnutls_record_send");
#endif
         return (__n - remain);
      }
      remain -= bw;
   }
   if (remain > 0)
   {
      traits_type::copy(pptr(), end - remain, remain);
      pbump(remain);
   }
   return __n;
}

#endif // MFEM_USE_GNUTLS

}
