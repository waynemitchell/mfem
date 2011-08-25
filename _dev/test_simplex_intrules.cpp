
#include "mfem.hpp"

double poly2d(const IntegrationPoint &ip, int m, int n)
{
   return pow(ip.x, m)*pow(ip.y, n);
   // integral over the reference triangle is
   // m!n!/(m+n+2)! = 1/binom(m+n,m)/(m+n+1)/(m+n+2)
}

double apoly2d(const IntegrationPoint &ip, int i, int j, int k)
{
   return pow(1. - ip.x - ip.y, i)*pow(ip.x, j)*pow(ip.y, k);
   // integral over the reference triangle is (with p = i+j+k)
   // i!j!k!/(p+2)! = 1/binom(p,i+j)/binom(i+j,i)/(p+1)/(p+2)
}

double poly3d(const IntegrationPoint &ip, int l, int m, int n)
{
   return pow(ip.x, l)*pow(ip.y, m)*pow(ip.z, n);
   // integral over the reference tetrahedron is (with p = l+m+n)
   // l!m!n!/(p+3)! = 1/binom(p,l+m)/binom(l+m,l)/(p+1)/(p+2)/(p+3)
}

const int maxn = 32;
int binom_array[maxn+1][maxn+1];

void binom_init()
{
   for (int n = 0; n <= maxn; n++)
   {
      binom_array[n][0] = binom_array[n][n] = 1;
      for (int k = 1; k < n; k++)
         binom_array[n][k] = binom_array[n-1][k] + binom_array[n-1][k-1];
   }
}

int binom(int n, int k)
{
   return binom_array[n][k];
}

int main()
{
   const double btol = 1e-12;

   double maxrelerr, maxrelerr_order;

   binom_init();

   cout << "TRIANGLE:\n";
   maxrelerr = 0.;
   for (int order = 0; order <= 25; order++)
   {
      const IntegrationRule &ir = IntRules.Get(Geometry::TRIANGLE, order);

      int type = 0;
      // type & 1 --> has negative weights
      // type & 2 --> has points outside the domain
      // type & 4 --> has points on the boundary
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         if (ip.weight <= 0.0)
            type |= 1;
         if (ip.x < -btol || ip.y < -btol || 1. - ip.x - ip.y < -btol)
            type |= 2;
         else if (ip.x < btol || ip.y < btol || 1. - ip.x - ip.y < btol)
            type |= 4;
      }

      cout << "order = " << setw(2) << order << " | type = "
           << char((type & 1) ? 'N' : 'P')
           << char((type & 2) ? 'O' : ((type & 4)? 'B' : 'I')) << '\n';

      maxrelerr_order = 0.;
#if 1
      // using the monomial basis: x^m y^n, 0 <= m+n <= order
      for (int p = 0; p <= order; p++)
      {
         for (int m = p; m >= 0; m--)
         {
            int n = p - m;

            double integral = 0.0;
            for (int i = 0; i < ir.GetNPoints(); i++)
            {
               const IntegrationPoint &ip = ir.IntPoint(i);
               integral += ip.weight*poly2d(ip, m, n);
            }

            double exact = 1.0/binom(p, m)/(p + 1)/(p + 2);
            double relerr = 1. - integral/exact;

            maxrelerr_order = fmax(maxrelerr_order, fabs(relerr));

            cout << "order = " << setw(2) << order
                 << " | m = " << setw(2) << m
                 << " | n = " << setw(2) << n
                 << " | relerr = " << setw(12) << relerr << '\n';
         }
      }
#else
      // the basis (1-x-y)^i x^j y^k, i+j+k = order
      int p = order;
      for (int i = 0; i <= p; i++)
      {
         for (int j = 0; i + j <= p; j++)
         {
            int k = p - i - j;

            double integral = 0.0;
            for (int n = 0; n < ir.GetNPoints(); n++)
            {
               const IntegrationPoint &ip = ir.IntPoint(n);
               integral += ip.weight*apoly2d(ip, i, j, k);
            }

            double exact = 1.0/binom(p, i+j)/binom(i+j, i)/(p+1)/(p+2);
            double relerr = 1. - integral/exact;

            maxrelerr_order = fmax(maxrelerr_order, fabs(relerr));

            cout << "order = " << setw(2) << order
                 << " | i = " << setw(2) << i
                 << " | j = " << setw(2) << j
                 << " | k = " << setw(2) << k
                 << " | relerr = " << setw(12) << relerr << '\n';
         }
      }
#endif

      cout << "order = " << setw(2) << order
           << " | maxrelerr_order = " << maxrelerr_order << '\n';

      maxrelerr = fmax(maxrelerr, maxrelerr_order);
   }

   cout << "maxrelerr = " << maxrelerr << '\n' << endl;

   cout << "TETRAHEDRON:\n";
   maxrelerr = 0.;
   for (int order = 0; order <= 21; order++)
   {
      const IntegrationRule &ir = IntRules.Get(Geometry::TETRAHEDRON, order);

      int type = 0;
      // type & 1 --> has negative weights
      // type & 2 --> has points outside the domain
      // type & 4 --> has points on the boundary
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         if (ip.weight <= 0.0)
            type |= 1;
         if (ip.x < -btol || ip.y < -btol || ip.z < -btol ||
             1. - ip.x - ip.y - ip.z < -btol)
            type |= 2;
         else if (ip.x < btol || ip.y < btol || ip.z < btol ||
                  1. - ip.x - ip.y - ip.z < btol)
            type |= 4;
      }

      cout << "order = " << setw(2) << order << " | type = "
           << char((type & 1) ? 'N' : 'P')
           << char((type & 2) ? 'O' : ((type & 4)? 'B' : 'I')) << '\n';

      maxrelerr_order = 0.;
      for (int p = 0; p <= order; p++)
      {
         for (int l = p; l >= 0; l--)
         {
            for (int m = p - l; m >= 0; m--)
            {
               int n = p - l - m;

               double integral = 0.0;
               for (int i = 0; i < ir.GetNPoints(); i++)
               {
                  const IntegrationPoint &ip = ir.IntPoint(i);
                  integral += ip.weight*poly3d(ip, l, m, n);
               }

               double exact = 1.0/binom(p,l+m)/binom(l+m,l)/(p+1)/(p+2)/(p+3);
               double relerr = 1. - integral/exact;

               maxrelerr_order = fmax(maxrelerr_order, fabs(relerr));

               cout << "order = " << setw(2) << order
                    << " | l = " << setw(2) << l
                    << " | m = " << setw(2) << m
                    << " | n = " << setw(2) << n
                    << " | relerr = " << setw(12) << relerr << '\n';
            }
         }
      }

      cout << "order = " << setw(2) << order
           << " | maxrelerr_order = " << maxrelerr_order << '\n';

      maxrelerr = fmax(maxrelerr, maxrelerr_order);
   }

   cout << "maxrelerr = " << maxrelerr << endl;

   return 0;
}
