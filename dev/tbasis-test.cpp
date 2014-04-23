
#include <time.h>
#include "tbasis.hpp"
#include "mfem.hpp"

using namespace std;

const int p = 3;
const int q = 6;
const int n = 4;

void init(TensorProductBasis<2, p, q, n> &T2,
          Array<TMatrix<p+1, p+1> >      &A,
          Array<TMatrix<q, q> >          &B);

void test_direct_2D(TensorProductBasis<2, p, q, n> &T2,
                    Array<TMatrix<p+1, p+1> >      &A,
                    Array<TMatrix<q, q> >          &B);

void test_indirect_2D(TensorProductBasis<2, p, q, n> &T2,
                      Array<TMatrix<p+1, p+1> >      &A,
                      Array<TMatrix<q, q> >          &B,
                      const int *elem_dof_2D);

int main(int argc, char *argv[])
{
   const int MiB = 1024*1024;
   int mem_limit = 1024*MiB;
   // int mem_limit = 128*MiB;
   int z = max(p+1, q);
   int m = mem_limit/(n*z*z*sizeof(double));

   TensorProductBasis<2, p, q, n> T2;
   TensorProductBasis<3, p, q, n> T3;

   Array<TMatrix<p+1, p+1> > A(m*n);
   Array<TMatrix<q, q> >     B(m*n);

   Array<int> elem_dof_2D(m*n*(p+1)*(p+1));

   // init elem_dof_2D
   for (int i = 0; i < m*n*(p+1)*(p+1); i++)
      elem_dof_2D[i] = i;

   cout << endl
        << "p = " << p << endl
        << "q = " << q << endl
        << "n = " << n << endl
        << "m = " << m << endl
        << "size of A = " << double(sizeof(TMatrix<p+1, p+1>)*m*n)/MiB
        << " MiB" << endl
        << "size of B = " << double(sizeof(TMatrix<q, q>)*m*n)/MiB
        << " MiB" << endl
        << "size of elem_dof = "
        << double(sizeof(int)*elem_dof_2D.Size())/MiB
        << " MiB" << endl
        << endl;

   init(T2, A, B);
   test_direct_2D(T2, A, B);
   test_indirect_2D(T2, A, B, elem_dof_2D);

   cout << "----------------------------------------" << endl << endl;

   init(T2, A, B);
   test_direct_2D(T2, A, B);
   test_indirect_2D(T2, A, B, elem_dof_2D);

   return 0;
}

void init(TensorProductBasis<2, p, q, n> &T2,
          Array<TMatrix<p+1, p+1> >      &A,
          Array<TMatrix<q, q> >          &B)
{
   int mn = A.Size();
   cout << "init ..." << flush;
   srand(time(NULL));
   T2.I.Random();
   for (int i = 0; i < mn; i++)
   {
      A[i].Random();
      B[i].Set(0.0);
   }
   cout << endl << endl;
}

void test_direct_2D(TensorProductBasis<2, p, q, n> &T2,
                    Array<TMatrix<p+1, p+1> >      &A,
                    Array<TMatrix<q, q> >          &B)
{
   int m = A.Size()/n;
   double rtime, flops;

   tic();
#if 1
   for (int k = 0; k < m; k++)
      T2.Calc(&A[k*n].data[0][0], &B[k*n].data[0][0]);
#else
   // always use the same B[]s
   for (int k = 0; k < m; k++)
      T2.Calc(&A[k*n].data[0][0], &B[0].data[0][0]);
#endif
   rtime = tic_toc.RealTime();
   flops = double(q*(p+1)*(p+1) + q*(p+1)*q)*m*n;
   cout << "Calc 2D direct:     rtime = " << rtime << " s" << endl;
   cout << "Calc 2D direct:    Gflops = " << flops/1e9 << endl;
   cout << "Calc 2D direct:  Gflops/s = " << flops/rtime/1e9 << endl;

   cout << endl;

   tic();
   const bool Add = true;
#if 1
   for (int k = 0; k < m; k++)
      T2.CalcT<Add>(&B[k*n].data[0][0], &A[k*n].data[0][0]);
#else
   // always use the same B[]s
   for (int k = 0; k < m; k++)
      T2.CalcT<Add>(&B[0].data[0][0], &A[k*n].data[0][0]);
#endif
   rtime = tic_toc.RealTime();
   flops = double(q*q*(p+1) + (p+1)*q*(p+1))*m*n;
   cout << "CalcT 2D direct:    rtime = " << rtime << " s" << endl;
   cout << "CalcT 2D direct:   Gflops = " << flops/1e9 << endl;
   cout << "CalcT 2D direct: Gflops/s = " << flops/rtime/1e9 << endl;

   cout << endl;
}

void test_indirect_2D(TensorProductBasis<2, p, q, n> &T2,
                      Array<TMatrix<p+1, p+1> >      &A,
                      Array<TMatrix<q, q> >          &B,
                      const int *elem_dof_2D)
{
   int m = A.Size()/n;
   double rtime, flops;

   tic();
#if 1
   for (int k = 0; k < m; k++)
      T2.Calc(&elem_dof_2D[k*(p+1)*(p+1)],
              &A[0].data[0][0], &B[k*n].data[0][0]);
#else
   // always use the same B[]s
   for (int k = 0; k < m; k++)
      T2.Calc(&elem_dof_2D[k*(p+1)*(p+1)],
              &A[0].data[0][0], &B[0].data[0][0]);
#endif
   rtime = tic_toc.RealTime();
   flops = double(q*(p+1)*(p+1) + q*(p+1)*q)*m*n;
   cout << "Calc 2D indirect:     rtime = " << rtime << " s" << endl;
   cout << "Calc 2D indirect:    Gflops = " << flops/1e9 << endl;
   cout << "Calc 2D indirect:  Gflops/s = " << flops/rtime/1e9 << endl;

   cout << endl;

   tic();
   const bool Add = true;
#if 1
   for (int k = 0; k < m; k++)
      T2.CalcT<Add>(&elem_dof_2D[k*(p+1)*(p+1)],
                    &B[k*n].data[0][0], &A[0].data[0][0]);
#else
   // always use the same B[]s
   for (int k = 0; k < m; k++)
      T2.CalcT<Add>(&elem_dof_2D[k*(p+1)*(p+1)],
                    &B[0].data[0][0], &A[0].data[0][0]);
#endif
   rtime = tic_toc.RealTime();
   flops = double(q*q*(p+1) + (p+1)*q*(p+1))*m*n;
   cout << "CalcT 2D indirect:    rtime = " << rtime << " s" << endl;
   cout << "CalcT 2D indirect:   Gflops = " << flops/1e9 << endl;
   cout << "CalcT 2D indirect: Gflops/s = " << flops/rtime/1e9 << endl;

   cout << endl;
}
