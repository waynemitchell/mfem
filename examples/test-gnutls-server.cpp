
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

   int port = 19917;

   mfem::gnutls_socketstream isock(pubkey.c_str(), privkey.c_str(),
                                   trustedkeys.c_str(), GNUTLS_SERVER);
   if (!isock.good())
   {
      cout << "GnuTLS initialization failed!" << endl;
      return 1;
   }
   mfem::socketserver server(port);
   if (server.good())
   {
      cout << "Waiting for connection ..." << flush;
      while (server.accept(isock) < 0)
      {
         cout << " unsuccessful connection." << endl;
         cout << "Waiting for another connection ..." << flush;
      }
      cout << " connection successful." << endl;
      string line;
      while (isock.good())
      {
         getline(isock, line);
         cout << "LINE: " << line << endl;
      }
      cout << "End of connection." << endl;
   }
   else
   {
      cout << "Could not establish a server on port " << port << endl;
   }
   return 0;
}
