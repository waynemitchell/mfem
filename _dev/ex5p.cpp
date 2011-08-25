//                                MFEM Example 5
//
// Compile with: make ex5p
//
// Sample runs:  mpirun -np 2 ex5p ../data/square-disc.mesh
//               mpirun -np 2 ex5p ../data/star.mesh
//               mpirun -np 2 ex5p ../data/escher.mesh
//               mpirun -np 2 ex5p ../data/fichera.mesh
//
// Description: Time dependent problem. Right now this is just a demo how to
//              visualize data dynamically using socket connections to a GLVis
//              server.

#include <fstream>
#include <limits>
#include <time.h>     // nanosleep
#include <signal.h>   // signal
#include "mfem.hpp"

double loc_time;
double time_dep_u(Vector &);

int main (int argc, char *argv[])
{
   int num_procs, myid;

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   Mesh *mesh;

   if (argc == 1)
   {
      if (myid == 0)
         cout << "\nUsage: mpirun -np <np> " << argv[0] << " <mesh_file>\n"
              << endl;
      MPI_Finalize();
      return 1;
   }

   ifstream imesh(argv[1]);
   if (!imesh)
   {
      if (myid == 0)
         cerr << "\nCan not open mesh file: " << argv[1] << '\n' << endl;
      MPI_Finalize();
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();

   {
      int ref_levels =
         (int)floor(log(1000./mesh->GetNE())/log(2.)/mesh->Dimension());
      for (int l = 0; l < ref_levels; l++)
         mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = 1;
      for (int l = 0; l < par_ref_levels; l++)
         pmesh->UniformRefinement();
   }

   // FiniteElementCollection *fec = new LinearFECollection;
   FiniteElementCollection *fec = new QuadraticFECollection;
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);

   ParGridFunction x(fespace);
   FunctionCoefficient u(time_dep_u);

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   if (sol_sock.is_open())
   {
      struct timespec req;
      req.tv_sec  = 0;
      req.tv_nsec = 20 * 1000000; // sleep for 0.02 seconds

      sol_sock.precision(8);
      for (loc_time = 0.0; sol_sock.good(); loc_time += 0.05)
      {
         if (fabs(loc_time - round(loc_time)) < 1e-3)
            if (myid == 0)
               cout << "time: " << loc_time << endl;
         x.ProjectCoefficient(u);
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock << "solution\n";
         pmesh->Print(sol_sock);
         x.Save(sol_sock);
         sol_sock << flush;

         // cout << "press enter ... " << flush;
         // cin.ignore(numeric_limits<streamsize>::max(), '\n');
         nanosleep(&req, NULL);
      }
      cout << "Process " << myid << '/' << num_procs
           << " : connection closed by server." << endl;
   }
   else
   {
      cout << "Process " << myid << '/' << num_procs
           << " : unable to connect to server at " << vishost << ':' << visport
           << endl;
   }

   delete fespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();

   return 0;
}

double time_dep_u(Vector &p)
{
   // double x = p(0);
   double x = p.Norml2();

   return sin(M_PI*(x - loc_time));
}
