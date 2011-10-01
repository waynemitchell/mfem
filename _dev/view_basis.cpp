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
      if (mesh->NURBSext)
      {
         char c;
         do
         {
            mesh->PrintCharacteristics();
            mesh->NURBSext->PrintCharacteristics(cout);
            cout <<
               "0) continue\n"
               "1) degree elevate\n"
               "2) knot insert\n"
               "3) print the mesh\n"
               " --> " << flush;
            cin >> c;
            switch (c)
            {
            case '1':
            {
               int t;
               cout << "\nenter new order --> " << flush;
               cin >> t;
               t -= mesh->NURBSext->GetOrder();
               if (t > 0)
                  mesh->DegreeElevate(t);
               break;
            }
            case '2':
            {
               int k;
               Array<KnotVector *> kv(mesh->NURBSext->GetNKV());
               cout << "modify which knot vector (1-" << kv.Size() << ") --> "
                    << flush;
               cin >> k;
               if (k < 1 || k > kv.Size())
                  break;
               cout << "enter knot vector " << k << " --> " << flush;
               k--;
               kv[k] = new KnotVector(cin);
               for (int i = 0; i < kv.Size(); i++)
               {
                  if (i != k)
                     kv[i] = new KnotVector(*mesh->NURBSext->GetKnotVector(i));
               }
               mesh->KnotInsert(kv);
               for (int i = 0; i < kv.Size(); i++)
                  delete kv[i];
               break;
            }
            case '3':
            {
               string fname;
               cout << "enter filename ('t' for terminal) --> " << flush;
               cin >> fname;
               if (fname == "t")
                  mesh->Print(cout);
               else
               {
                  ofstream out(fname.c_str());
                  int prec;
                  cout << "enter precision --> " << flush;
                  cin >> prec;
                  out.precision(prec);
                  mesh->Print(out);
               }
               break;
            }
            }
         }
         while (c != '0');
      }

      int ref_levels;
      mesh->PrintCharacteristics();
      if (mesh->NURBSext)
         mesh->NURBSext->PrintCharacteristics(cout);
      cout << "enter ref. levels --> " << flush;
      cin >> ref_levels;
      for (int l = 0; l < ref_levels; l++)
         mesh->UniformRefinement();
      cout << endl;
      mesh->PrintCharacteristics();
      if (mesh->NURBSext)
         mesh->NURBSext->PrintCharacteristics(cout);
   }

   char c = '1';
   int min_p = 0;
   if (mesh->NURBSext)
   {
      cout <<
         "0) NURBS basis functions\n"
         "1) classical FEM basis functions\n"
         " --> " << flush;
      cin >> c;
      if (c == '0')
         min_p = mesh->NURBSext->GetOrder();
   }

   int p;
   do
   {
      cout << "enter order --> " << flush;
      cin >> p;
   }
   while (p < min_p || p > 32);

   FiniteElementCollection *fec;
   switch(c)
   {
   case '0':
      fec = new NURBSFECollection(p);
      break;
   case '1':
      if (p > 0)
         fec = new H1_FECollection(p, mesh->Dimension());
      else
         fec = new L2_FECollection(p, mesh->Dimension());
      // fec = new RT_FECollection(p, mesh->Dimension());
      break;
   }

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

      cout << "displaying basis function " << i + 1
           << " / " << x.Size() << " ... " << flush;

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
