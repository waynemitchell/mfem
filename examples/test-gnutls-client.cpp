
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
   int port = 19916;

   mfem::GnuTLS_global_state state;
   // state.set_log_level(1000);
   mfem::GnuTLS_session_params params(state, pubkey.c_str(), privkey.c_str(),
                                      trustedkeys.c_str(), GNUTLS_CLIENT);
   mfem::GnuTLS_socketstream osock(params);
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
#if 0
      while (cin.good())
      {
         getline(cin, line);
         if (cin.eof())
         {
            break;
         }
         cout << "sending " << line.size() << " bytes." << endl;
         osock << line << endl;
      }
#else
      double MiB = 1024.*1024.;
      size_t num_bytes = 1024*1024*1024;
      line.resize(num_bytes, 'A');
      cout << "sending " << num_bytes << " bytes." << endl;
      mfem::tic_toc.Start();
      osock << line << flush;
      mfem::tic_toc.Stop();
      cout << "it took " << mfem::tic_toc.RealTime() << " s ("
           << num_bytes/MiB/mfem::tic_toc.RealTime() << " MiB/s)" << endl;
#endif
      cout << "Closing connection." << endl;
   }
   else
   {
      cout << "Could not connect to server " << hostname
           << " on port " << port << endl;
   }
   return 0;
}
