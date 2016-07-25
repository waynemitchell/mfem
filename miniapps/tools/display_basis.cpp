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

string   elemTypeStr(const Element::Type & eType);
inline bool elemIs1D(const Element::Type & eType);
inline bool elemIs2D(const Element::Type & eType);
inline bool elemIs3D(const Element::Type & eType);

string   basisTypeStr(char bType);
inline bool basisIs1D(char bType);
inline bool basisIs2D(char bType);
inline bool basisIs3D(char bType);

string mapTypeStr(int mType);

int update_basis(vector<socketstream*> & sock, const VisWinLayout & vwl,
                 Element::Type e, char bType, int bOrder, int mType);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   Element::Type eType  = Element::TRIANGLE;
   char          bType  = 'h';
   int           bOrder = 2;
   int           mType  = 0;

   int eInt = -1;
   int bInt = -1;

   VisWinLayout vwl;
   vwl.nx = 5;
   vwl.ny = 3;
   vwl.w  = 250;
   vwl.h  = 250;

   bool visualization = true;
   bool visit = false;

   OptionsParser args(argc, argv);
   args.AddOption(&eInt, "-e", "--elem-type",
                  "Element Type: (1-Segment, 2-Triangle, 3-Quadrilateral, "
                  "4-Tetrahedron, 5-Hexahedron)");
   args.AddOption(&bInt, "-b", "--basis-type",
                  "Basis Function Type (0-H1, 1-Nedelec, 2-Raviart-Thomas, "
                  "3-L2, 4-Fixed Order Cont., 5-Gaussian Discontinuous (2D), "
                  "6-Crouzeix-Raviart)");
   args.AddOption(&bOrder, "-o", "--order", "Basis function order");
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
   if ( eInt > 0 && eInt < 6 )
   {
      eType = (Element::Type)eInt;
   }
   switch (bInt)
   {
      case 0:
         bType = 'h';
         break;
      case 1:
         bType = 'n';
         break;
      case 2:
         bType = 'r';
         break;
      case 3:
         bType = 'l';
         break;
      case 4:
         bType = 'f';
         break;
      case 5:
         bType = 'g';
         break;
      case 6:
         bType = 'c';
         break;
      default:
         bType = 'h';
   }

   // 2. Define sockets for visualization
   vector<socketstream*> sock;

   // 3. Collect user input
   bool print_char = true;
   while (true)
   {
      if (print_char)
      {
         cout << endl;
         cout << "Element Type:          " << elemTypeStr(eType) << endl;
         cout << "Basis Type:            " << basisTypeStr(bType) << endl;;
         cout << "Basis function order:  " << bOrder << endl;
         cout << "Map Type:              " << mapTypeStr(mType) << endl;
      }
      if ( update_basis(sock, vwl, eType, bType, bOrder, mType) )
      {
         cerr << "Invalid combination of basis info (try again)" << endl;
      }

      print_char = false;
      cout << endl;
      cout << "What would you like to do?\n"
           "q) Quit\n"
           "e) Change Element Type\n"
           "b) Change Basis Type\n";
      if ( bType == 'h' || bType == 'p' || bType == 'n' || bType == 'r' ||
           bType == 'l' || bType == 'f' || bType == 'g' )
      {
         cout << "o) Change Basis Order\n";
      }
      if ( bType == 'l' )
      {
         cout << "m) Change Map Type\n";
      }
      cout << "--> " << flush;
      char mk;
      cin >> mk;

      if (mk == 'q')
      {
         break;
      }
      if (mk == 'e')
      {
         eInt = 0;
         cout << "valid element types:\n";
         if ( basisIs1D(bType) )
         {
            cout <<
                 "1) Segment\n";
         }
         if ( basisIs2D(bType) )
         {
            cout <<
                 "2) Triangle\n"
                 "3) Quadrilateral\n";
         }
         if ( basisIs3D(bType) )
         {
            cout <<
                 "4) Tetrahedron\n"
                 "5) Hexhedron\n";
         }
         cout << "enter new element type --> " << flush;
         cin >> eInt;
         if ( eInt <= 0 || eInt > 5 )
         {
            cout << "invalid element type \"" << eInt << "\"" << endl << flush;
         }
         else if ( (elemIs1D((Element::Type)eInt) && basisIs1D(bType)) ||
                   (elemIs2D((Element::Type)eInt) && basisIs2D(bType)) ||
                   (elemIs3D((Element::Type)eInt) && basisIs3D(bType)) )
         {
            eType = (Element::Type)eInt;
            print_char = true;
         }
         else
         {
            cout << "invalid element type \"" << eInt <<
                 "\" for basis type \"" << basisTypeStr(bType) << "\"." << endl;
         }
      }
      if (mk == 'b')
      {
         char bChar = 0;
         cout << "valid basis types:\n";
         cout << "h) H1 Finite Element\n";
         // cout << "p) H1 Positive Finite Element\n";
         if ( elemIs2D(eType) || elemIs3D(eType) )
         {
            cout << "n) Nedelec Finite Element\n";
            cout << "r) Raviart-Thomas Finite Element\n";
         }
         cout << "l) L2 Finite Element\n";
         if ( elemIs1D(eType) || elemIs2D(eType) )
         {
            cout << "c) Crouzeix-Raviart Finite Element\n";
         }
         cout << "f) Fixed Order Continuous Finite Element\n";
         if ( elemIs2D(eType) )
         {
            cout << "g) Gauss Discontinuous Finite Element\n";
         }
         cout << "enter new basis type --> " << flush;
         cin >> bChar;
         if ( bChar == 'h' || bChar == 'p' || bChar == 'l' || bChar == 'f' ||
              ((bChar == 'n' || bChar == 'r') &&
               (elemIs2D(eType) || elemIs3D(eType))) ||
              (bChar == 'c' && (elemIs1D(eType) || elemIs2D(eType))) ||
              (bChar == 'g' && elemIs2D(eType)))
         {
            bType = bChar;
            if ( bType == 'h' )
            {
               mType = FiniteElement::VALUE;
            }
            else if ( bType == 'p' )
            {
               mType = FiniteElement::VALUE;
            }
            else if ( bType == 'n' )
            {
               mType = FiniteElement::H_CURL;
            }
            else if ( bType == 'r' )
            {
               mType = FiniteElement::H_DIV;
            }
            else if ( bType == 'l' )
            {
               if ( mType != FiniteElement::VALUE &&
                    mType != FiniteElement::INTEGRAL )
               {
                  mType = FiniteElement::VALUE;
               }
            }
            else if ( bType == 'c' )
            {
               bOrder = 1;
               mType  = FiniteElement::VALUE;
            }
            else if ( bType == 'f' )
            {
               if ( bOrder < 1 || bOrder > 3)
               {
                  bOrder = 1;
               }
               mType  = FiniteElement::VALUE;
            }
            else if ( bType == 'g' )
            {
               if ( bOrder < 1 || bOrder > 2)
               {
                  bOrder = 1;
               }
               mType  = FiniteElement::VALUE;
            }
            print_char = true;
         }
         else
         {
            cout << "invalid basis type \"" << bChar << "\"." << endl;
         }
      }
      if (mk == 'm' && bType == 'l')
      {
         int mInt = 0;
         cout << "valid map types:\n"
              "0) VALUE\n"
              "1) INTEGRAL\n";
         cout << "enter new map type --> " << flush;
         cin >> mInt;
         if (mInt >=0 && mInt <= 1)
         {
            mType = mInt;
            print_char = true;
         }
         else
         {
            cout << "invalid map type \"" << mInt << "\"." << endl;
         }
      }
      if (mk == 'o')
      {
         int oInt = 1;
         int oMin = ( bType == 'h' || bType == 'p' || bType == 'n' ||
                      bType == 'f' || bType == 'g')?1:0;
         int oMax = -1;
         switch (bType)
         {
            case 'g':
               oMax = 2;
               break;
            case 'f':
               oMax = 3;
               break;
            default:
               oMax = -1;
         }
         cout << "basis function order must be >= " << oMin;
         if ( oMax >= 0 )
         {
            cout << " and <= " << oMax;
         }
         cout << endl;
         cout << "enter new basis function order --> " << flush;
         cin >> oInt;
         if ( oInt >= oMin && oInt <= (oMax>=0)?oMax:oInt )
         {
            bOrder = oInt;
            print_char = true;
         }
         else
         {
            cout << "invalid basis order \"" << oInt << "\"." << endl;
         }
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

string elemTypeStr(const Element::Type & eType)
{
   switch (eType)
   {
      case Element::POINT:
         return "POINT";
         break;
      case Element::SEGMENT:
         return "SEGMENT";
         break;
      case Element::TRIANGLE:
         return "TRIANGLE";
         break;
      case Element::QUADRILATERAL:
         return "QUADRILATERAL";
         break;
      case Element::TETRAHEDRON:
         return "TETRAHEDRON";
         break;
      case Element::HEXAHEDRON:
         return "HEXAHEDRON";
         break;
      default:
         return "INVALID";
         break;
   };
}

bool
elemIs1D(const Element::Type & eType)
{
   return eType == Element::SEGMENT;
}

bool
elemIs2D(const Element::Type & eType)
{
   return eType == Element::TRIANGLE || eType == Element::QUADRILATERAL;
}

bool
elemIs3D(const Element::Type & eType)
{
   return eType == Element::TETRAHEDRON || eType == Element::HEXAHEDRON;
}

string
basisTypeStr(char bType)
{
   switch (bType)
   {
      case 'h':
         return "Continuous (H1)";
         break;
      case 'p':
         return "Continuous Positive (H1)";
         break;
      case 'n':
         return "Nedelec";
         break;
      case 'r':
         return "Raviart-Thomas";
         break;
      case 'l':
         return "Discontinuous (L2)";
         break;
      case 'f':
         return "Fixed Order Continuous";
         break;
      case 'g':
         return "Gaussian Discontinuous";
         break;
      case 'c':
         return "Crouzeix-Raviart";
         break;
      default:
         return "INVALID";
         break;
   };
}

bool
basisIs1D(char bType)
{
   return bType == 'h' || bType == 'p' || bType == 'l' || bType == 'c' ||
          bType == 'f';
}

bool
basisIs2D(char bType)
{
   return bType == 'h' || bType == 'p' || bType == 'n' || bType == 'r' ||
          bType == 'l' || bType == 'c' || bType == 'f' || bType == 'g';
}

bool
basisIs3D(char bType)
{
   return bType == 'h' || bType == 'p' || bType == 'n' || bType == 'r' ||
          bType == 'f' || bType == 'l';
}

string
mapTypeStr(int mType)
{
   switch (mType)
   {
      case FiniteElement::VALUE:
         return "VALUE";
         break;
      case FiniteElement::H_CURL:
         return "H_CURL";
         break;
      case FiniteElement::H_DIV:
         return "H_DIV";
         break;
      case FiniteElement::INTEGRAL:
         return "INTEGRAL";
         break;
      default:
         return "INVALID";
         break;
   }
}

int
update_basis(vector<socketstream*> & sock,  const VisWinLayout & vwl,
             Element::Type e, char bType, int bOrder, int mType)
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
   // int geom = mesh->GetElementBaseGeometry(0);
   // int sdim = mesh->SpaceDimension();

   if ( t_ > 0 )
   {
      mesh->Transform(transformation);
   }

   // cout << "Building Finite Element Collection" << endl;
   FiniteElementCollection * FEC = NULL;
   switch (bType)
   {
      case 'h':
         FEC = new H1_FECollection(bOrder, dim);
         vec = false;
         break;
      case 'p':
         FEC = new H1Pos_FECollection(bOrder, dim);
         vec = false;
         break;
      case 'n':
         FEC = new ND_FECollection(bOrder, dim);
         vec = true;
         break;
      case 'r':
         FEC = new RT_FECollection(bOrder, dim);
         vec = true;
         break;
      case 'l':
         FEC = new L2_FECollection(bOrder, dim, L2_FECollection::GaussLegendre,
                                   mType);
         vec = false;
         break;
      case 'c':
         FEC = new CrouzeixRaviartFECollection();
         break;
      case 'f':
         if ( bOrder == 1 )
         {
            FEC = new LinearFECollection();
         }
         else if ( bOrder == 2 )
         {
            FEC = new QuadraticFECollection();
         }
         else if ( bOrder == 3 )
         {
            FEC = new CubicFECollection();
         }
         break;
      case 'g':
         if ( bOrder == 1 )
         {
            FEC = new GaussLinearDiscont2DFECollection();
         }
         else if ( bOrder == 2 )
         {
            FEC = new GaussQuadraticDiscont2DFECollection();
         }
         break;
      case 'd':
         FEC = new DG_Interface_FECollection(bOrder, dim);
         vec = true;
         break;
   }
   if ( FEC == NULL)
   {
      return 1;
   }
   // cout << "Building Finite Element Space" << endl;
   FiniteElementSpace FESpace(mesh, FEC);

   // int ndof = FESpace.GetFE(0)->GetDof();
   int ndof = FESpace.GetVSize();

   Array<int> vdofs;
   FESpace.GetElementVDofs(0,vdofs);

   char vishost[] = "localhost";
   int  visport   = 19916;

   int offx = vwl.w+10, offy = vwl.h+45; // window offsets

   int nsock = sock.size();
   for (int i=0; i<nsock; i++)
   {
      *sock[i] << "keys q";
      delete sock[i];
   }

   sock.resize(ndof);
   for (int i=0; i<ndof; i++)
   {
      sock[i] = new socketstream; sock[i]->precision(8);
   }

   GridFunction **    x = new GridFunction*[ndof];
   for (int i=0; i<ndof; i++)
   {
      x[i]  = new GridFunction(&FESpace);
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

   int ref = 0;
   int exOrder = ( bType == 'p' && false )?3:0;
   while ( 1<<ref < bOrder + exOrder || ref == 0 )
   {
      mesh->UniformRefinement();
      FESpace.Update();

      for (int i=0; i<ndof; i++)
      {
         x[i]->Update();
      }
      ref++;
   }

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
