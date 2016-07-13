#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class meshstream : public stringstream
{
public:
   meshstream(int dim, int e);
};

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

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    GridFunction &gf, const char *title,
                    int x = 0, int y = 0, int w = 400, int h = 400,
                    bool vec = false);

int main(int argc, char *argv[])
{
   // 2. Parse command-line options.
   // const char *mesh_file = "../data/inline-quad.mesh";
   int d = 2;
   int e = 1;
   int f = 0;
   int nx = 5;
   int ny = 3;
   int Ww = 250, Wh = 250; // window size
   int vec = false;
   int order = 1;
   bool visualization = true;
   bool visit = false;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&d, "-d", "--dimension",
                  "Space Dimension");
   args.AddOption(&e, "-e", "--elem-type",
                  "If d == 1, e = 0\n"
                  "   d == 2, e = 0 -> triangle, e = 1 -> quadrilateral\n"
                  "   d == 3, e = 0 -> tetrahedron, e = 1 -> hexahedron");
   args.AddOption(&f, "-f", "--fe-type",
                  "Finite Element Type\n"
                  "  0 -> H1, 1 -> ND, 2 -> RT, 3 -> L2");
   args.AddOption(&t_, "-t", "--trans",
                  "Coordinate Transform");
   args.AddOption(&nx, "-nx", "--num-win-x",
                  "Number of Viz Windows in X");
   args.AddOption(&ny, "-ny", "--num-win-y",
                  "Number of Viz Windows in y");
   args.AddOption(&Ww, "-w", "--width",
                  "Width of Viz Windows");
   args.AddOption(&Wh, "-h", "--height",
                  "Height of Viz Windows");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.Parse();
   if (!args.Good())
   {
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   {
      args.PrintOptions(cout);
   }

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh;
   meshstream imesh(d, e);
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

   cout << "Building Finite Element Collection" << endl;
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

   cout << "Building Finite Element Space" << endl;
   FiniteElementSpace FESpace(mesh, FEC);

   int ndof = FESpace.GetFE(0)->GetDof();

   Array<int> vdofs;
   FESpace.GetElementVDofs(0,vdofs);

   char vishost[] = "localhost";
   int  visport   = 19916;

   int offx = Ww+10, offy = Wh+45; // window offsets

   cout << "Building Grid Functions" << endl;
   socketstream ** sock = new socketstream*[ndof];;
   GridFunction **    x = new GridFunction*[ndof];
   for (int i=0; i<ndof; i++)
   {
      sock[i] = new socketstream; sock[i]->precision(8);
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

   cout << "Visualizing Basis Functions" << endl;
   for (int i=0; i<ndof; i++)
   {
      ostringstream oss;
      oss << "DoF " << i + 1;
      VisualizeField(*sock[i], vishost, visport, *x[i], oss.str().c_str(),
                     (i % nx) * offx, ((i / nx) % ny) * offy, Ww, Wh,
                     vec);
   }

   cout << "Exiting" << endl;
   // 13. Exit
   return 0;
}

double scalar_func(const Vector &x)
{
   double d = -sin(x[0]);
   if ( x.Size() > 1 ) { d *= cos(x[1]); d += cos(x[0])*cos(x[1]); }
   if ( x.Size() > 2 ) { d *= sin(x[2]); d += sin(x[0])*sin(x[1])*cos(x[2]); }
   return d;
}

void vector_func(const Vector &x, Vector &v)
{
   v.SetSize(x.Size());
   v[0] = cos(x[0]);
   if ( x.Size() > 1 )
   {
      v[0] *= cos(x[1]);
      v[1] = cos(x[0])*sin(x[1]);
   }
   if ( x.Size() > 2 )
   {
      v[0] *= sin(x[2]);
      v[1] *= sin(x[2]);
      v[2] = sin(x[0])*sin(x[1])*sin(x[2]);
   }
}

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    GridFunction &gf, const char *title,
                    int x, int y, int w, int h, bool vec)
{
   Mesh &mesh = *gf.FESpace()->GetMesh();

   bool newly_opened = false;
   int connection_failed;

   do
   {
      {
         if (!sock.is_open() || !sock)
         {
            sock.open(vishost, visport);
            sock.precision(8);
            newly_opened = true;
         }
         sock << "solution\n";
      }

      mesh.Print(sock);
      gf.Save(sock);

      if (newly_opened)
      {
         sock << "window_title '" << title << "'\n"
              << "window_geometry "
              << x << " " << y << " " << w << " " << h << "\n"
              << "keys maAc";
         if ( vec ) { sock << "v"; }
         sock << endl;
      }

      {
         connection_failed = !sock && !newly_opened;
      }
   }
   while (connection_failed);
}

meshstream::meshstream(int dim, int e)
{
   *this << "MFEM mesh v1.0" << endl
         << "dimension" << endl << dim << endl
         << "elements" << endl << 1 << endl;
   switch (dim)
   {
      case 1:
         *this << "1 1 0 1" << endl
               << "boundary" << endl << 2 << endl
               << "1 0 0" << endl
               << "1 0 1" << endl
               << "vertices" << endl
               << 2 << endl
               << 1 << endl
               << 0 << endl
               << 1 << endl;
         break;
      case 2:
         if ( e == 0 )
         {
            *this << "1 2 0 1 2" << endl
                  << "boundary" << endl << 3 << endl
                  << "1 1 0 1" << endl
                  << "1 1 1 2" << endl
                  << "1 1 2 0" << endl
                  << "vertices" << endl
                  << "3" << endl
                  << "2" << endl
                  << "0 0" << endl
                  << "1 0" << endl
                  << "0 1" << endl;
         }
         else
         {
            *this << "1 3 0 1 2 3" << endl
                  << "boundary" << endl << 4 << endl
                  << "1 1 0 1" << endl
                  << "1 1 1 2" << endl
                  << "1 1 2 3" << endl
                  << "1 1 3 0" << endl
                  << "vertices" << endl
                  << "4" << endl
                  << "2" << endl
                  << "0 0" << endl
                  << "1 0" << endl
                  << "1 1" << endl
                  << "0 1" << endl;
         }
         break;
      case 3:
         if ( e == 0 )
         {
            *this << "1 4 0 1 2 3" << endl
                  << "boundary" << endl << 4 << endl
                  << "1 2 0 2 1" << endl
                  << "1 2 1 2 3" << endl
                  << "1 2 2 0 3" << endl
                  << "1 2 0 1 3" << endl
                  << "vertices" << endl
                  << "4" << endl
                  << "3" << endl
                  << "0 0 0" << endl
                  << "1 0 0" << endl
                  << "0 1 0" << endl
                  << "0 0 1" << endl;
         }
         else
         {
            *this << "1 5 0 1 2 3 4 5 6 7" << endl
                  << "boundary" << endl << 6 << endl
                  << "1 3 0 3 2 1" << endl
                  << "1 3 4 5 6 7" << endl
                  << "1 3 0 1 5 4" << endl
                  << "1 3 1 2 6 5" << endl
                  << "1 3 2 3 7 6" << endl
                  << "1 3 3 0 4 7" << endl
                  << "vertices" << endl
                  << "8" << endl
                  << "3" << endl
                  << "0 0 0" << endl
                  << "1 0 0" << endl
                  << "1 1 0" << endl
                  << "0 1 0" << endl
                  << "0 0 1" << endl
                  << "1 0 1" << endl
                  << "1 1 1" << endl
                  << "0 1 1" << endl;
         }
         break;
   }

}
