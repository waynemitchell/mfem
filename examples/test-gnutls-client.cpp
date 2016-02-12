
#include "mfem.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

using namespace std;

int main(int argc, char *argv[])
{
   string home_dir(getenv("HOME"));
   string client_dir = home_dir + "/.config/glvis/client/";
   string pubkey  = client_dir + "pubring.gpg";
   string privkey = client_dir + "secring.gpg";
   string trustedkeys = client_dir + "trusted-servers.gpg";

   cout << "pubkey      = " << pubkey << '\n'
        << "privkey     = " << privkey << '\n'
        << "trustedkeys = " << trustedkeys << endl;

   const char *hostname = "localhost";
   int port = 19917;

   mfem::gnutls_socketstream osock(pubkey.c_str(), privkey.c_str(),
                                   trustedkeys.c_str());
   if (!osock.good())
   {
      cout << "GnuTLS initialization failed!" << endl;
      return 1;
   }
   osock.open(hostname, port);
   if (osock.good())
   {
      cout << "Connection established." << endl;
      string line;
      while (cin.good())
      {
         getline(cin, line);
         osock << line << endl;
      }
      cout << "Closing connection." << endl;
   }
   else
   {
      cout << "Could not connect to server " << hostname
           << " on port " << port << endl;
   }
   return 0;
}
