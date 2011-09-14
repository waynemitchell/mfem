#include <cmath>
#include <cstdlib>

#include <fstream>

#include "../mfem.hpp"

int       kx, ky, kz;
int       vspace;
int       dim;
Mesh     *mesh;

void refine();

double poly (Vector &x)
{
   double d = 1.;

   if (kx) d *= pow (x(0), kx);
   if (ky) d *= pow (x(1), ky);
   if (kz) d *= pow (x(2), kz);

   return d;
}

void vpoly (const Vector &, Vector &);


int main (int argc, char *argv[])
{

   if (argc < 2)
   {
      cout << "usage: " << argv[0] << " meshfile" << endl;
      return 1;
   }

   ifstream meshin(argv[1]);
   if (meshin)
   {
      cout << "Reading the mesh ... ( " << flush;
      mesh = new Mesh(meshin, 1, 1);
      cout << " ) done." << endl;
      dim = mesh->Dimension();
   }
   else
   {
      cerr << "Can not open input mesh file." << endl;
      return 3;
   }
   meshin.close();

   FunctionCoefficient  polynomial (poly);

   const int d = 3; // highest degree polynomials to test
   int i, j, k;

   int poly_type;

   LinearFECollection    fec_1;
   QuadraticFECollection fec_2;
   CubicFECollection     fec_3;

   FiniteElementCollection  *fec[] = { &fec_1, &fec_2, &fec_3 };
   FiniteElementSpace       *cfes;
   GridFunction             *cpoly;

   cout << "Enter seed for srandom : " << flush;
   cin >> i;
   srandom (i);

   mesh -> PrintCharacteristics();
   mesh -> UseTwoLevelState (1);
   refine();
   mesh -> PrintCharacteristics();

   FiniteElementSpace  *ffes;
   GridFunction        *fpoly;
   Vector              *ipoly;
   SparseMatrix        *restr;

   //  Test if the interpolation preserves the polys { x^i y^j z^k }
   int dd, dz;
   double max;
   for (int c = 0; c < d; c++)
   {
      mesh -> SetState (Mesh::TWO_LEVEL_COARSE);
      cfes  = new FiniteElementSpace (mesh, fec[c]);
      cpoly = new GridFunction (cfes);
      if (c == 0) poly_type = cfes -> GetFE (0) -> Space();

      mesh -> SetState (Mesh::TWO_LEVEL_FINE);
      ffes   = new FiniteElementSpace (mesh, fec[c]);
      fpoly  = new GridFunction (ffes);
      ipoly  = new Vector (ffes -> GetVSize());
      // no need to finalize restr
      restr = ffes -> GlobalRestrictionMatrix (cfes);

      //---------------------------------------------------------
      dd = c+1;
      dz = (dim == 2) ? 0 : dd;
      cout << "Testing ";
      if (poly_type == FunctionSpace::Pk) cout << "P_"; else cout << "Q_";
      cout << dd << " elements ..." << endl;
      for (i = 0; i <= dd; i++)
         for (j = 0; j <= dd; j++)
            for (k = 0; k <= dz; k++)
            {
               if (poly_type == FunctionSpace::Pk && i+j+k > dd)
                  continue;
               kx = i; ky = j; kz = k;
               mesh -> SetState (Mesh::TWO_LEVEL_COARSE);
               cpoly -> ProjectCoefficient (polynomial);
               mesh -> SetState (Mesh::TWO_LEVEL_FINE);
               fpoly -> ProjectCoefficient (polynomial);
               restr -> MultTranspose (*cpoly, *ipoly);
               *ipoly -= *fpoly;
               max = ipoly -> Normlinf();
               cout << "Max. norm error for x^" << i << " y^" << j;
               if (dim == 3)
                  cout << " z^" << k;
               cout << " = " << max;
               if (max > 1.0e-12) cout << "  Oops!" << " (" << i+j+k << ")";
               cout << endl;
            }
      delete restr;
      delete ipoly;
      delete fpoly;
      delete ffes;
      delete cpoly;
      delete cfes;
   }

   VectorFunctionCoefficient vpolynomial (dim, vpoly);

   //------------------------------------------------------------
   RT0_2DFECollection       fec_rt0_2d;
   RT0_3DFECollection       fec_rt0_3d;
   FiniteElementCollection *fec_rt0;

   fec_rt0 =
      (dim == 2)?
      (FiniteElementCollection*)(&fec_rt0_2d):
      (FiniteElementCollection*)(&fec_rt0_3d);
   vspace = (poly_type == FunctionSpace::Pk) ? 0 : 1;

   mesh -> SetState (Mesh::TWO_LEVEL_COARSE);
   cfes  = new FiniteElementSpace (mesh, fec_rt0);
   cpoly = new GridFunction (cfes);

   mesh -> SetState (Mesh::TWO_LEVEL_FINE);
   ffes  = new FiniteElementSpace (mesh, fec_rt0);
   fpoly = new GridFunction (ffes);
   ipoly = new Vector (ffes -> GetVSize());
   // no need to finalize restr
   restr = ffes -> GlobalRestrictionMatrix (cfes);

   cout << "Testing RT0 ..." << endl;
   dd = cfes->GetFE(0)->GetDof();
   for (kx = 0; kx < dd; kx++)
   {
      mesh -> SetState (Mesh::TWO_LEVEL_COARSE);
      cpoly -> ProjectCoefficient (vpolynomial);
      mesh -> SetState (Mesh::TWO_LEVEL_FINE);
      fpoly -> ProjectCoefficient (vpolynomial);
      restr -> MultTranspose (*cpoly, *ipoly);
      *ipoly -= *fpoly;
      max = ipoly -> Normlinf();
      cout << "Max. norm error for phi_" << kx << " = " << max;
      if (max > 1.0e-12) cout << "  Oops!";
      cout << endl;
   }
   delete restr;
   delete ipoly;
   delete fpoly;
   delete ffes;
   delete cpoly;
   delete cfes;

nedelec_label:
   //------------------------------------------------------------
   ND1_3DFECollection fec_nd1;

   if (dim == 2)
      return 0;

   vspace = (poly_type == FunctionSpace::Pk) ? 2 : 3;

   mesh -> SetState (Mesh::TWO_LEVEL_COARSE);
   cfes  = new FiniteElementSpace (mesh, &fec_nd1);
   cpoly = new GridFunction (cfes);

   mesh -> SetState (Mesh::TWO_LEVEL_FINE);
   ffes  = new FiniteElementSpace (mesh, &fec_nd1);
   fpoly = new GridFunction (ffes);
   ipoly = new Vector (ffes -> GetVSize());
   // no need to finalize restr
   restr = ffes -> GlobalRestrictionMatrix (cfes);

   cout << "Testing Nedelec1 ..." << endl;
   dd = cfes->GetFE(0)->GetDof();
   for (kx = 0; kx < dd; kx++)
   {
      mesh -> SetState (Mesh::TWO_LEVEL_COARSE);
      cpoly -> ProjectCoefficient (vpolynomial);
      mesh -> SetState (Mesh::TWO_LEVEL_FINE);
      fpoly -> ProjectCoefficient (vpolynomial);
      restr -> MultTranspose (*cpoly, *ipoly);
      *ipoly -= *fpoly;
      max = ipoly -> Normlinf();
      cout << "Max. norm error for phi_" << kx << " = " << max;
      if (max > 1.0e-12) cout << "  Oops!";
      cout << endl;
   }
   delete restr;
   delete ipoly;
   delete fpoly;
   delete ffes;
   delete cpoly;
   delete cfes;

   return 0;
}

void refine()
{
   switch (mesh -> GetElementType (0))
   {
   case Element::TRIANGLE:
   case Element::TETRAHEDRON:
   {
      // refine "half" of the elements, randomly selected
      Array<int> marked_el;
      {
         int i, j, ne = mesh -> GetNE();
         Array<int> el_marker (ne);
         for (i = 0; i < ne; i++)
            el_marker[i] = 0;
         for (i = 0; i < (ne+1)/2; i++)
         {
            j = ((double)random() / (RAND_MAX + 1.0)) * ne;
            el_marker[j] = 1;
         }
         j = 0;
         for (i = 0; i < ne; i++)
            if (el_marker[i] == 1)
               j++;
         marked_el.SetSize (j);
         j = 0;
         for (i = 0; i < ne; i++)
            if (el_marker[i] == 1)
               marked_el[j++] = i;
      }
      cout << "Refining " << marked_el.Size() << " / " << mesh -> GetNE()
           << " elements ..." << endl;
      mesh -> LocalRefinement (marked_el);
   }
   break;

   default:
      cout << "Refining all elements ..." << endl;
      mesh -> UniformRefinement();
      break;
   }
}

void vpoly (const Vector &w, Vector &v)
{
   double x, y, z;

   x = w(0); y = w(1); if (dim == 3) z = w(2);

   if (vspace/2 == 0) // RT0
   {
      if (dim == 2)
      {
         if (vspace%2 == 0) // triangles
            switch (kx)
            {
            case 0: v(0) = 1.; v(1) = 0.; break;
            case 1: v(0) = 0.; v(1) = 1.; break;
            case 2: v(0) = x;  v(1) = y;  break;
            }
         else // quadrilaterals
            switch (kx)
            {
            case 0: v(0) = 1.; v(1) = 0.; break;
            case 1: v(0) = 0.; v(1) = 1.; break;
            case 2: v(0) = x;  v(1) = 0.; break;
            case 3: v(0) = 0.; v(1) = y;  break;
            }
      }
      else
      {
         if (vspace%2 == 0) // tetrahera
            switch (kx)
            {
            case 0: v(0) = 1.; v(1) = 0.; v(2) = 0.; break;
            case 1: v(0) = 0.; v(1) = 1.; v(2) = 0.; break;
            case 2: v(0) = 0.; v(1) = 0.; v(2) = 1.; break;
            case 3: v(0) = x;  v(1) = y;  v(2) = z;  break;
            }
         else // hexahedra
            switch (kx)
            {
            case 0: v(0) = 1.; v(1) = 0.; v(2) = 0.; break;
            case 1: v(0) = 0.; v(1) = 1.; v(2) = 0.; break;
            case 2: v(0) = 0.; v(1) = 0.; v(2) = 1.; break;
            case 3: v(0) = x;  v(1) = 0.; v(2) = 0.; break;
            case 4: v(0) = 0.; v(1) = y;  v(2) = 0.; break;
            case 5: v(0) = 0.; v(1) = 0.; v(2) = z;  break;
            }
      }
   }
   else // Nedelec1
   {
      if (vspace%2 == 0) // tetrahedra
         switch (kx)
         {
         case 0: v(0) = 1.; v(1) = 0.; v(2) = 0.; break;
         case 1: v(0) = 0.; v(1) = 1.; v(2) = 0.; break;
         case 2: v(0) = 0.; v(1) = 0.; v(2) = 1.; break;
         case 3: v(0) = 0.; v(1) = -z; v(2) = y;  break;
         case 4: v(0) = z;  v(1) = 0.; v(2) = -x; break;
         case 5: v(0) = -y; v(1) = x;  v(2) = 0.; break;
         }
      else // hexahedra
         switch (kx)
         {
         case  0: v(0) = 1.;  v(1) = 0.;  v(2) = 0.;  break;
         case  1: v(0) = 0.;  v(1) = 1.;  v(2) = 0.;  break;
         case  2: v(0) = 0.;  v(1) = 0.;  v(2) = 1.;  break;
         case  3: v(0) = y;   v(1) = 0.;  v(2) = 0.;  break;
         case  4: v(0) = z;   v(1) = 0.;  v(2) = 0.;  break;
         case  5: v(0) = y*z; v(1) = 0.;  v(2) = 0.;  break;
         case  6: v(0) = 0.;  v(1) = x;   v(2) = 0.;  break;
         case  7: v(0) = 0.;  v(1) = z;   v(2) = 0.;  break;
         case  8: v(0) = 0.;  v(1) = x*z; v(2) = 0.;  break;
         case  9: v(0) = 0.;  v(1) = 0.;  v(2) = x;   break;
         case 10: v(0) = 0.;  v(1) = 0.;  v(2) = y;   break;
         case 11: v(0) = 0.;  v(1) = 0.;  v(2) = x*y; break;
         }
   }
}
