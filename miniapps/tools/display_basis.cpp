#include "mfem.hpp"
#include "../common/fem_extras.hpp"
#include "../common/mesh_extras.hpp"
#include <vector>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::miniapps;

static int t_ = 0;

void transformation(const Vector &p, Vector &v)
{
   int dim = p.Size();
   v.SetSize(dim);

   switch (t_)
   {
      case 1:
         v[0] = 0.5 * p[0];
         if ( dim > 1 )
         {
            v[0] += 0.125*(sqrt(5.0) - 1.0) * p[1];
            v[1] = sqrt(0.0625 * (5.0 + sqrt(5.0))) * p[1];
         }
         if ( dim > 2 ) { v[2] = 0.5 * p[2]; }
         break;
      case 2:
         v[0] = 0.125 * (sqrt(5.0) - 1.0) * p[0];
         if ( dim > 1 )
         {
            v[0] -= 0.125*(sqrt(5.0) + 1.0) * p[1];
            v[1] = 0.125 * sqrt(0.5 * (5.0 + sqrt(5.0))) *
                   (2.0 * p[0] + (sqrt(5.0) - 1.0) * p[1]);
         }
         if ( dim > 2 ) { v[2] = 0.5 * p[2]; }
         break;
   }
}

struct VisWinLayout
{
   int nx;
   int ny;
   int w;
   int h;
};


int update_basis(vector<socketstream*> & sock, const VisWinLayout & vwl,
                 Element::Type e, int f, int order);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   Element::Type e = Element::TRIANGLE;
   int f = 0;

   VisWinLayout vwl;
   vwl.nx = 5;
   vwl.ny = 3;
   vwl.w  = 250;
   vwl.h  = 250;

   int order = 1;
   bool visualization = true;
   bool visit = false;

   OptionsParser args(argc, argv);
   args.AddOption(&f, "-f", "--fe-type",
                  "Finite element type:\n"
                  "\t  0 -> H1\n"
                  "\t  1 -> ND\n"
                  "\t  2 -> RT\n"
                  "\t  3 -> L2");
   args.AddOption(&t_, "-t", "--trans",
                  "Coordinate transformation");
   args.AddOption(&vwl.nx, "-nx", "--num-win-x",
                  "Number of Viz windows in X");
   args.AddOption(&vwl.ny, "-ny", "--num-win-y",
                  "Number of Viz windows in y");
   args.AddOption(&vwl.w, "-w", "--width",
                  "Width of Viz windows");
   args.AddOption(&vwl.h, "-h", "--height",
                  "Height of Viz windows");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   {
      args.PrintOptions(cout);
   }

   // 2. Define sockets for visualization
   vector<socketstream*> sock;

   // 3. Collect user input
   bool print_char = true;
   while (true)
   {
      if (print_char)
      {
         cout << "Basis function order:  " << order << endl;
      }
      update_basis(sock, vwl, e, f, order);

      print_char = false;
      cout << endl;
      cout << "What would you like to do?\n"
           "q) Quit\n"
           "e) Change Element Type\n"
           "o) Change Basis Order\n"
           "--> " << flush;
      char mk;
      cin >> mk;

      if (mk == 'q')
      {
         break;
      }
      if (mk == 'e')
      {
         int eInt = 0;
         cout << "valid element types:\n"
              "1) Segment\n"
              "2) Triangle\n"
              "3) Quadrilateral\n"
              "4) Tetrahedron\n"
              "5) Hexhedron\n";
         cout << "enter new element type --> " << flush;
         cin >> eInt;
         if ( eInt > 0 && eInt <= 5 )
         {
            e = (Element::Type)eInt;
         }
         else
         {
            cout << "invalid element type \"" << eInt << "\"." << endl;
         }
      }
      if (mk == 'o')
      {
         cout << "enter new basis function order --> " << flush;
         cin >> order;
      }
   }

   // 4. Delete sockets
   cout << "Exiting" << endl;

   for (unsigned int i=0; i<sock.size(); i++)
   {
      delete sock[i];
   }

   // 5. Exit
   return 0;
}

int
update_basis(vector<socketstream*> & sock,  const VisWinLayout & vwl,
             Element::Type e, int f, int order)
{
   bool vec = false;

   Mesh *mesh;
   ElementMeshStream imesh(e);
   if (!imesh)
   {
      {
         cerr << "\nProblem with meshstream object\n" << endl;
      }
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   int dim = mesh->Dimension();
   // int sdim = mesh->SpaceDimension();

   if ( t_ > 0 )
   {
      mesh->Transform(transformation);
   }

   // cout << "Building Finite Element Collection" << endl;
   FiniteElementCollection * FEC = NULL;
   switch (f)
   {
      case 0:
         FEC = new H1_FECollection(order, dim);
         vec = false;
         break;
      case 1:
         FEC = new ND_FECollection(order, dim);
         vec = true;
         break;
      case 2:
         FEC = new RT_FECollection(order, dim);
         vec = true;
         break;
      case 3:
         FEC = new L2_FECollection(order, dim);
         vec = false;
         break;
   }

   // cout << "Building Finite Element Space" << endl;
   FiniteElementSpace FESpace(mesh, FEC);

   int ndof = FESpace.GetFE(0)->GetDof();

   Array<int> vdofs;
   FESpace.GetElementVDofs(0,vdofs);

   char vishost[] = "localhost";
   int  visport   = 19916;

   int offx = vwl.w+10, offy = vwl.h+45; // window offsets

   // cout << "Setting up sockets" << endl;
   int nsock = sock.size();
   for (int i=ndof; i<nsock; i++)
   {
      *sock[i] << "keys q";
      delete sock[i];
   }
   sock.resize(ndof);
   for (int i=nsock; i<ndof; i++)
   {
      sock[i] = new socketstream; sock[i]->precision(8);
   }

   // cout << "Building Grid Functions" << endl;
   GridFunction **    x = new GridFunction*[ndof];
   for (int i=0; i<ndof; i++)
   {
      x[i]    = new GridFunction(&FESpace);
      *x[i] = 0.0;
      if ( vdofs[i] < 0 )
      {
         (*x[i])(-1-vdofs[i]) = -1.0;
      }
      else
      {
         (*x[i])(vdofs[i]) = 1.0;
      }
   }

   for (int j=0; j<order; j++)
   {
      mesh->UniformRefinement();
      FESpace.Update();

      for (int i=0; i<ndof; i++)
      {
         x[i]->Update();
      }
   }

   // cout << "Visualizing Basis Functions" << endl;
   for (int i=0; i<ndof; i++)
   {
      ostringstream oss;
      oss << "DoF " << i + 1;
      VisualizeField(*sock[i], vishost, visport, *x[i], oss.str().c_str(),
                     (i % vwl.nx) * offx, ((i / vwl.nx) % vwl.ny) * offy,
                     vwl.w, vwl.h,
                     vec);
   }

   for (int i=0; i<ndof; i++)
   {
      delete x[i];
   }
   delete [] x;

   delete FEC;
   delete mesh;

   return 0;
}
