//                                MFEM Example 5
//
// Compile with: make ex5
//
// Sample runs:  ex5 ../data/square-disc.mesh
//               ex5 ../data/star.mesh
//               ex5 ../data/escher.mesh
//               ex5 ../data/fichera.mesh
//
// Description: Time dependent problem. Right now this is just a demo how to
//              visualize data dynamically using a socket connection to a GLVis
//              server.

#include <fstream>
#include <limits>
#include <time.h>     // nanosleep
#include <signal.h>   // signal
#include "mfem.hpp"

double loc_time;
double time_dep_u(Vector &);

int interrupted = 0;
void interrupt_handler(int signum)
{
   interrupted++;
}


int main (int argc, char *argv[])
{
   Mesh *mesh;

   if (argc == 1)
   {
      cout << "\nUsage: " << argv[0] << " <mesh_file>\n" << endl;
      return 1;
   }

   ifstream imesh(argv[1]);
   if (!imesh)
   {
      cerr << "\nCan not open mesh file: " << argv[1] << '\n' << endl;
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();

   {
      int ref_levels =
         (int)floor(log(5000./mesh->GetNE())/log(2.)/mesh->Dimension());
      for (int l = 0; l < ref_levels; l++)
         mesh->UniformRefinement();
   }

   // FiniteElementCollection *fec = new LinearFECollection;
   FiniteElementCollection *fec = new QuadraticFECollection;
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

   GridFunction x(fespace);
   FunctionCoefficient u(time_dep_u);

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   if (sol_sock.is_open())
   {
      struct timespec req;
      req.tv_sec  = 0;
      req.tv_nsec = 20 * 1000000; // sleep for 0.02 seconds

      signal(SIGINT, interrupt_handler);

      sol_sock.precision(8);
      for (loc_time = 0.0; !interrupted && sol_sock.good(); loc_time += 0.05)
      {
         if (fabs(loc_time - round(loc_time)) < 1e-3)
            cout << "time: " << loc_time << endl;
         x.ProjectCoefficient(u);
         sol_sock << "solution\n";
         mesh->Print(sol_sock);
         x.Save(sol_sock);
         sol_sock << flush;

         // cout << "press enter ... " << flush;
         // cin.ignore(numeric_limits<streamsize>::max(), '\n');
         nanosleep(&req, NULL);
      }
      if (!interrupted)
         cout << "Connection closed by server." << endl;
      else
         cout << "Terminating." << endl;

      signal(SIGINT, SIG_DFL);
   }
   else
   {
      cout << "Unable to connect to server at " << vishost << ':' << visport
           << endl;
   }

   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}

double time_dep_u(Vector &p)
{
   // double x = p(0);
   double x = p.Norml2();

   return sin(M_PI*(x - loc_time));
}
