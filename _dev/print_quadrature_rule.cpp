#include "mfem.hpp"

int main()
{
   int Geom, Order;

   cout << "Choose geometry:\n";
   for (Geom = 1; Geom < Geometry::NumGeom; Geom++)
      cout << Geom << ") " << Geometry::Name[Geom] << '\n';
   cout << " --> " << flush;
   cin >> Geom;
   cout << "Enter order = " << flush;
   cin >> Order;

   DenseMatrix pm;
   Geometries.GetPerfPointMat(Geom, pm);
   int dim = pm.Height();

   cout.precision(18);
   cout << scientific << showpos;

   const IntegrationRule &ir = IntRules.Get(Geom, Order);

   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      cout << ip.x;
      if (dim > 1)
      {
         cout << ", " << ip.y;
         if (dim > 2)
            cout << ", " << ip.z;
      }
      cout << ", " << ip.weight << ",\n";
   }
}
