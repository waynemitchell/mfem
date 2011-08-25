#include <fstream>
#include <limits>
#include "mfem.hpp"

int main(int argc, char *argv[])
{
   Mesh *mesh;

   if (argc == 1)
   {
      cout << "\nUsage: view_basis <mesh_file>\n" << endl;
      return 1;
   }

   // 1. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral or hexahedral elements with the same code.
   ifstream imesh(argv[1]);
   if (!imesh)
   {
      cerr << "\nCan not open mesh file: " << argv[1] << '\n' << endl;
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();

   // 2. Refine the mesh uniformly.
   {
      int ref_levels;
      mesh->PrintCharacteristics();
      cout << "Enter ref. levels = " << flush;
      cin >> ref_levels;
      for (int l = 0; l < ref_levels; l++)
         mesh->UniformRefinement();
      cout << endl;
      mesh->PrintCharacteristics();
   }

   int p;
   do
   {
      cout << "Enter p = " << flush;
      cin >> p;
   }
   while (p < 0 || p > 32);

   FiniteElementCollection *fec;
   fec = new H1_FECollection(p, mesh->Dimension());
   // fec = new DiscontFECollection(p, mesh->Dimension());
   // fec = new RT_FECollection(p, mesh->Dimension());

   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

   GridFunction x(fespace);
   x = 0.;

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);

   cin.ignore(numeric_limits<streamsize>::max(), '\n');
   for (int i = 0; true; )
   {
      x(i) = 1.;

      cout << "displaying basis function No " << i << " ... " << flush;

      sol_sock << "solution\n";
      mesh->Print(sol_sock);
      x.Save(sol_sock);
      sol_sock << flush;

      cout << endl;

      if (!sol_sock.good())
      {
         cout << "connection closed by server." << endl;
         break;
      }

      x(i) = 0.;

      if (++i == x.Size())
         break;

      cout << "press enter ... " << flush;
      cin.ignore(numeric_limits<streamsize>::max(), '\n');
   }

   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}
