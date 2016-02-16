
#include "mfem.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

using namespace std;

int main(int argc, char *argv[])
{
   string home_dir(getenv("HOME"));
   string server_dir = home_dir + "/.config/glvis/server/";
   string pubkey  = server_dir + "pubring.gpg";
   string privkey = server_dir + "secring.gpg";
   string trustedkeys = server_dir + "trusted-clients.gpg";

   cout << "pubkey      = " << pubkey << '\n'
        << "privkey     = " << privkey << '\n'
        << "trustedkeys = " << trustedkeys << endl;

   int port = 19916;

   mfem::GnuTLS_global_state state;
   // state.set_log_level(1000);
   mfem::GnuTLS_session_params params(state, pubkey.c_str(), privkey.c_str(),
                                      trustedkeys.c_str(), GNUTLS_SERVER);
   mfem::GnuTLS_socketstream isock(params);
   if (!isock.good())
   {
      cout << "GnuTLS initialization failed!" << endl;
      return 1;
   }
   mfem::socketserver server(port);
   if (server.good())
   {
      cout << "Waiting for connection on port " << port << " ..." << endl;
      while (server.accept(isock) < 0)
      {
         cout << "Unsuccessful connection." << endl;
         cout << "Waiting for another connection ..." << endl;
      }
      cout << "Connection successful." << endl;
      string line;
      while (isock.good())
      {
         getline(isock, line);
         if (isock.eof())
         {
            cout << "reached EOF." << endl;
         }
         cout << "received " << line.size() << " bytes." << endl;
         if (line.size() == 0)
         {
            cout << "LINE: (empty)" << endl;
         }
         else if (line.size() <= 4*1024)
         {
            cout << "LINE: " << line << endl;
         }
         else
         {
            cout << "LINE: (more than 4 KiB)" << endl;
         }
      }
      cout << "End of connection." << endl;
   }
   else
   {
      cout << "Could not establish a server on port " << port << endl;
   }
   return 0;
}
