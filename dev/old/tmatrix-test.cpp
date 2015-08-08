
#include <time.h>
#include "mfem.hpp"
#include "tmatrix.hpp"

using namespace std;
using namespace mfem;


// #define TEST_EIGEN 1
#if defined(TEST_EIGEN)
#include <Eigen/Dense>
#endif


// matrix size
const int S = 5;

// number of matrices to multiply in an inner loop
const int M = 1;


void template_test(TMatrix<S,S> &A,
                   Array<TMatrix<S,S> > &B,
                   Array<TMatrix<S,S> > &C,
                   Array<double> &gflops);

void mfem_test(TMatrix<S,S> &A,
               Array<TMatrix<S,S> > &B,
               Array<TMatrix<S,S> > &C);

void smatrix_test();

#if defined(TEST_EIGEN)
void eigen_test1(Eigen::Matrix<double, S, S> &A,
                 Eigen::Matrix<double, S, S> *B,
                 Eigen::Matrix<double, S, S> *C, int num_iter);
void eigen_test2(Eigen::Matrix<double, S, S> &A,
                 Eigen::Matrix<double, S, S> *B,
                 Eigen::Matrix<double, S, S> *C, int num_iter);
#endif


int main(int argc, char *argv[])
{
   const int MiB = 1024*1024;
   // int mem_limit = 1024*MiB; // use ~1 GiB memory for B and C each
   int mem_limit = 512*MiB;  // use ~512 MiB memory for B and C each
   // int mem_limit = 128*MiB;  // use ~128 MiB memory for B and C each
   // int mem_limit = 16*MiB;   // use ~16 MiB memory for B and C each
   // int mem_limit = MiB;      // use ~1 MiB memory for B and C each
   int num_iter = (mem_limit/(M*S*S*sizeof(double)))*M;

   TMatrix<S,S> A;
   Array<TMatrix<S,S> > B(num_iter), C(num_iter);

   cout << "S = " << S << "\nM = " << M << endl;
   cout << "A   is " << S << " x " << S << " (" << sizeof(A) << " bytes = "
        << S << " x " << S << " doubles + " << sizeof(A)-S*S*sizeof(double)
        << " bytes)\n"
        << "B[] is " << S << " x " << S << " x " << num_iter << " ("
        << num_iter*sizeof(A)/double(MiB) << " MiB)\n"
        << "C[] is " << S << " x " << S << " x " << num_iter << " ("
        << num_iter*sizeof(A)/double(MiB) << " MiB)" << endl;

   // init A, B[] with random values
   srand(time(NULL));
   A.Random();
   cout << "Init B[] ..." << flush;
   for (int k = 0; k < num_iter; k++)
      B[k].Random();
   cout << endl;
   // init C[] with zeros (make sure allocation is done?)
   cout << "Init C[] ..." << flush;
   for (int k = 0; k < num_iter; k++)
      C[k].Set(0.0);
   cout << endl;

   cout << endl;

   Array<double> gflops(8);

   gflops = 0.0;
   for (int i = 0; i < 4; i++)
      template_test(A, B, C, gflops);
   cout << "Summary: max GFlops:\n"
        << "  A.B[]     = " << gflops[0] << '\n'
        << "  B[].A     = " << gflops[1] << '\n'
        << "  A^t.B[]   = " << gflops[2] << '\n'
        << "  B[]^t.A   = " << gflops[3] << '\n'
        << "  A.B[]^t   = " << gflops[4] << '\n'
        << "  B[].A^t   = " << gflops[5] << '\n'
        << "  A^t.B[]^t = " << gflops[6] << '\n'
        << "  B[]^t.A^t = " << gflops[7] << '\n'
        << endl;
#if 0
#ifndef MFEM_USE_LAPACK
   cout << "using MFEM (no LAPACK) ...\n" << endl;
#else
   cout << "using LAPACK (through MFEM) ...\n" << endl;
#endif

   mfem_test(A, B, C);

   mfem_test(A, B, C);
#endif

   // smatrix_test();

#if defined(TEST_EIGEN)
   B.DeleteAll();
   C.DeleteAll();

   cout << endl << "using Eigen ...\n" << endl;

   {
      Eigen::Matrix<double, S, S>  A;
      Eigen::Matrix<double, S, S> *B;
      Eigen::Matrix<double, S, S> *C;

      B = new Eigen::Matrix<double, S, S>[num_iter];
      C = new Eigen::Matrix<double, S, S>[num_iter];

      cout << "A is " << S << " x " << S << " (" << sizeof(A) << " bytes = "
           << S << " x " << S << " doubles + " << sizeof(A)-S*S*sizeof(double)
           << " bytes)\n" << endl;

      for (int i = 0; i < S; i++)
         for (int j = 0; j < S; j++)
            A(i,j) = rand() / (RAND_MAX + 1.0);
      cout << "Init B[] ..." << flush;
      for (int k = 0; k < num_iter; k++)
         for (int i = 0; i < S; i++)
            for (int j = 0; j < S; j++)
               B[k](i,j) = rand() / (RAND_MAX + 1.0);
      cout << endl;
      // init C[] with zeros (make sure allocation is done?)
      cout << "Init C[] ..." << flush;
      for (int k = 0; k < num_iter; k++)
         for (int i = 0; i < S; i++)
            for (int j = 0; j < S; j++)
               C[k](i,j) = 0.0;
      cout << endl << endl;

      eigen_test1(A, B, C, num_iter);
      eigen_test1(A, B, C, num_iter);
      eigen_test2(A, B, C, num_iter);
      eigen_test2(A, B, C, num_iter);

      delete [] C;
      delete [] B;
   }
#endif

   return 0;
}

void template_test(TMatrix<S,S> &A,
                   Array<TMatrix<S,S> > &B,
                   Array<TMatrix<S,S> > &C,
                   Array<double> &gflops)
{
   int num_iter = B.Size();
   double utime, rtime, flops;

   TMatrix<S,S> *Bk, *Ck;

   tic();
   if (M == 1)
   {
      for (int k = 0; k < num_iter; k++)
         Mult(A, B[k], C[k]);
   }
   else
   {
      for (int k = 0; k < num_iter/M; k++)
      {
         Bk = &B[k*M];
         Ck = &C[k*M];
         for (int l = 0; l < M; l++)
            Mult(A, Bk[l], Ck[l]);
      }
   }
   tic_toc.Stop();
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   flops = double(S)*S*S*num_iter;
   cout << "Mult  A.B[]    utime: " << utime << " s" << endl;
   cout << "Mult  A.B[]    rtime: " << rtime << " s" << endl;
   cout << "Mult  A.B[]   Gflops: " << flops/1e9 << endl;
   cout << "Mult  A.B[] Gflops/s: " << flops/rtime/1e9 << endl;
   gflops[0] = max(gflops[0], flops/rtime/1e9);

   cout << endl;

   tic();
   if (M == 1)
   {
      for (int k = 0; k < num_iter; k++)
         Mult(B[k], A, C[k]);
   }
   else
   {
      for (int k = 0; k < num_iter/M; k++)
      {
         Bk = &B[k*M];
         Ck = &C[k*M];
         for (int l = 0; l < M; l++)
            Mult(Bk[l], A, Ck[l]);
      }
   }
   tic_toc.Stop();
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   flops = double(S)*S*S*num_iter;
   cout << "Mult  B[].A    utime: " << utime << " s" << endl;
   cout << "Mult  B[].A    rtime: " << rtime << " s" << endl;
   cout << "Mult  B[].A   Gflops: " << flops/1e9 << endl;
   cout << "Mult  B[].A Gflops/s: " << flops/rtime/1e9 << endl;
   gflops[1] = max(gflops[1], flops/rtime/1e9);

   cout << endl;

   tic();
   if (M == 1)
   {
      for (int k = 0; k < num_iter; k++)
         MultAtB(A, B[k], C[k]);
   }
   else
   {
      for (int k = 0; k < num_iter/M; k++)
      {
         Bk = &B[k*M];
         Ck = &C[k*M];
         for (int l = 0; l < M; l++)
            MultAtB(A, Bk[l], Ck[l]);
      }
   }
   tic_toc.Stop();
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   flops = double(S)*S*S*num_iter;
   cout << "Mult  A^t.B[]    utime: " << utime << " s" << endl;
   cout << "Mult  A^t.B[]    rtime: " << rtime << " s" << endl;
   cout << "Mult  A^t.B[]   Gflops: " << flops/1e9 << endl;
   cout << "Mult  A^t.B[] Gflops/s: " << flops/rtime/1e9 << endl;
   gflops[2] = max(gflops[2], flops/rtime/1e9);

   cout << endl;

   tic();
   if (M == 1)
   {
      for (int k = 0; k < num_iter; k++)
         MultAtB(B[k], A, C[k]);
   }
   else
   {
      for (int k = 0; k < num_iter/M; k++)
      {
         Bk = &B[k*M];
         Ck = &C[k*M];
         for (int l = 0; l < M; l++)
            MultAtB(Bk[l], A, Ck[l]);
      }
   }
   tic_toc.Stop();
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   flops = double(S)*S*S*num_iter;
   cout << "Mult  B[]^t.A    utime: " << utime << " s" << endl;
   cout << "Mult  B[]^t.A    rtime: " << rtime << " s" << endl;
   cout << "Mult  B[]^t.A   Gflops: " << flops/1e9 << endl;
   cout << "Mult  B[]^t.A Gflops/s: " << flops/rtime/1e9 << endl;
   gflops[3] = max(gflops[3], flops/rtime/1e9);

   cout << endl;

   tic();
   if (M == 1)
   {
      for (int k = 0; k < num_iter; k++)
         MultABt(A, B[k], C[k]);
   }
   else
   {
      for (int k = 0; k < num_iter/M; k++)
      {
         Bk = &B[k*M];
         Ck = &C[k*M];
         for (int l = 0; l < M; l++)
            MultABt(A, Bk[l], Ck[l]);
      }
   }
   tic_toc.Stop();
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   flops = double(S)*S*S*num_iter;
   cout << "Mult  A.B[]^t    utime: " << utime << " s" << endl;
   cout << "Mult  A.B[]^t    rtime: " << rtime << " s" << endl;
   cout << "Mult  A.B[]^t   Gflops: " << flops/1e9 << endl;
   cout << "Mult  A.B[]^t Gflops/s: " << flops/rtime/1e9 << endl;
   gflops[4] = max(gflops[4], flops/rtime/1e9);

   cout << endl;

   tic();
   if (M == 1)
   {
      for (int k = 0; k < num_iter; k++)
         MultABt(B[k], A, C[k]);
   }
   else
   {
      for (int k = 0; k < num_iter/M; k++)
      {
         Bk = &B[k*M];
         Ck = &C[k*M];
         for (int l = 0; l < M; l++)
            MultABt(Bk[l], A, Ck[l]);
      }
   }
   tic_toc.Stop();
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   flops = double(S)*S*S*num_iter;
   cout << "Mult  B[].A^t    utime: " << utime << " s" << endl;
   cout << "Mult  B[].A^t    rtime: " << rtime << " s" << endl;
   cout << "Mult  B[].A^t   Gflops: " << flops/1e9 << endl;
   cout << "Mult  B[].A^t Gflops/s: " << flops/rtime/1e9 << endl;
   gflops[5] = max(gflops[5], flops/rtime/1e9);

   cout << endl;

   tic();
   if (M == 1)
   {
      for (int k = 0; k < num_iter; k++)
         MultAtBt(A, B[k], C[k]);
   }
   else
   {
      for (int k = 0; k < num_iter/M; k++)
      {
         Bk = &B[k*M];
         Ck = &C[k*M];
         for (int l = 0; l < M; l++)
            MultAtBt(A, Bk[l], Ck[l]);
      }
   }
   tic_toc.Stop();
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   flops = double(S)*S*S*num_iter;
   cout << "Mult  A^t.B[]^t    utime: " << utime << " s" << endl;
   cout << "Mult  A^t.B[]^t    rtime: " << rtime << " s" << endl;
   cout << "Mult  A^t.B[]^t   Gflops: " << flops/1e9 << endl;
   cout << "Mult  A^t.B[]^t Gflops/s: " << flops/rtime/1e9 << endl;
   gflops[6] = max(gflops[6], flops/rtime/1e9);

   cout << endl;

   tic();
   if (M == 1)
   {
      for (int k = 0; k < num_iter; k++)
         MultAtBt(B[k], A, C[k]);
   }
   else
   {
      for (int k = 0; k < num_iter/M; k++)
      {
         Bk = &B[k*M];
         Ck = &C[k*M];
         for (int l = 0; l < M; l++)
            MultAtBt(Bk[l], A, Ck[l]);
      }
   }
   tic_toc.Stop();
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   flops = double(S)*S*S*num_iter;
   cout << "Mult  B[]^t.A^t    utime: " << utime << " s" << endl;
   cout << "Mult  B[]^t.A^t    rtime: " << rtime << " s" << endl;
   cout << "Mult  B[]^t.A^t   Gflops: " << flops/1e9 << endl;
   cout << "Mult  B[]^t.A^t Gflops/s: " << flops/rtime/1e9 << endl;
   gflops[7] = max(gflops[7], flops/rtime/1e9);

   cout << endl;
}

void mfem_test(TMatrix<S,S> &A,
               Array<TMatrix<S,S> > &B,
               Array<TMatrix<S,S> > &C)
{
   int num_iter = B.Size();
   double utime, rtime, flops;

   DenseMatrix mfem_A(&A.data[0][0], S, S);
   DenseMatrix mfem_B(&B[0].data[0][0], S, S*num_iter);
   DenseMatrix mfem_C(&C[0].data[0][0], S, S*num_iter);

   tic();
   Mult(mfem_A, mfem_B, mfem_C);
   tic_toc.Stop();
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   flops = double(S)*S*S*num_iter;
   cout << "Mult mfem_A.mfem_B    utime: " << utime << " s" << endl;
   cout << "Mult mfem_A.mfem_B    rtime: " << rtime << " s" << endl;
   cout << "Mult mfem_A.mfem_B   Gflops: " << flops/1e9 << endl;
   cout << "Mult mfem_A.mfem_B Gflops/s: " << flops/rtime/1e9 << endl;

   cout << endl;

   tic();
   MultAtB(mfem_A, mfem_B, mfem_C);
   tic_toc.Stop();
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   flops = double(S)*S*S*num_iter;
   cout << "Mult mfem_A^t.mfem_B    utime: " << utime << " s" << endl;
   cout << "Mult mfem_A^t.mfem_B    rtime: " << rtime << " s" << endl;
   cout << "Mult mfem_A^t.mfem_B   Gflops: " << flops/1e9 << endl;
   cout << "Mult mfem_A^t.mfem_B Gflops/s: " << flops/rtime/1e9 << endl;

   cout << endl;

   mfem_C.ClearExternalData();
   mfem_B.ClearExternalData();
   mfem_A.ClearExternalData();
}

void smatrix_test()
{
   double A_data[6] = { 1., 2., 3., 4., 5., 6. }; // 2 x 3
   double B_data[12] = { 7., 8., 9., 10., 11., 12.,
                         13., 14., 15., 16., 17., 18. }; // 3 x 4
   double C_data[8]; // 2 x 4

   Mult<2,3,4,1,2,1,3,1,2,false>(A_data, B_data, C_data);

   SMatrix<4,2,2,1> C;
   C.data = C_data;

   for (int i = 0; i < C.NumRows(); i++)
   {
      for (int j = 0; j < C.NumCols(); j++)
         cout << ' ' << C(i,j);
      cout << endl;
   }
   cout << endl;

   TMatrix<2,2> A, B, V;
   A.data[0][0] = 1.;
   A.data[1][0] = 2.;
   A.data[0][1] = 3.;
   A.data[1][1] = 4.;

   B.data[0][0] = 5.;
   B.data[1][0] = 6.;
   B.data[0][1] = 7.;
   B.data[1][1] = 8.;

   Mult(A, B, V);
   cout << "A.B =" << endl;
   for (int i = 0; i < 2; i++)
   {
      for (int j = 0; j < 2; j++)
         cout << ' ' << V.data[j][i];
      cout << endl;
   }

   MultAtB(A, B, V);
   cout << "A^t.B =" << endl;
   for (int i = 0; i < 2; i++)
   {
      for (int j = 0; j < 2; j++)
         cout << ' ' << V.data[j][i];
      cout << endl;
   }

   MultABt(A, B, V);
   cout << "A.B^t =" << endl;
   for (int i = 0; i < 2; i++)
   {
      for (int j = 0; j < 2; j++)
         cout << ' ' << V.data[j][i];
      cout << endl;
   }

   MultAtBt(A, B, V);
   cout << "A^t.B^t =" << endl;
   for (int i = 0; i < 2; i++)
   {
      for (int j = 0; j < 2; j++)
         cout << ' ' << V.data[j][i];
      cout << endl;
   }
}

#if defined(TEST_EIGEN)
void eigen_test1(Eigen::Matrix<double, S, S> &A,
                 Eigen::Matrix<double, S, S> *B,
                 Eigen::Matrix<double, S, S> *C, int num_iter)
{
   double utime, rtime, flops;

   // Using dynamic sizes, even though we know size of matrix.
   // This just creates a view into the matrix.
   //Eigen::Map<Eigen::MatrixXd> eigenB(&B[0](0,0), S, S*num_iter);
   //Eigen::Map<Eigen::MatrixXd> eigenC(&C[0](0,0), S, S*num_iter);
   Eigen::Map<Eigen::Matrix<double, S, Eigen::Dynamic> > eigenB(&B[0](0,0), S, S*num_iter);
   Eigen::Map<Eigen::Matrix<double, S, Eigen::Dynamic> > eigenC(&C[0](0,0), S, S*num_iter);

   tic();
   eigenC = A*eigenB;
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   flops = double(S)*S*S*num_iter;
   cout << "Mult A.eigen_B    utime: " << utime << " s" << endl;
   cout << "Mult A.eigen_B    rtime: " << rtime << " s" << endl;
   cout << "Mult A.eigen_B   Gflops: " << flops/1e9 << endl;
   cout << "Mult A.eigen_B Gflops/s: " << flops/rtime/1e9 << endl;

   cout << endl;

   tic();
   eigenC = A.transpose()*eigenB;
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   flops = double(S)*S*S*num_iter;
   cout << "Mult A^t.eigen_B    utime: " << utime << " s" << endl;
   cout << "Mult A^t.eigen_B    rtime: " << rtime << " s" << endl;
   cout << "Mult A^t.eigen_B   Gflops: " << flops/1e9 << endl;
   cout << "Mult A^t.eigen_B Gflops/s: " << flops/rtime/1e9 << endl;

   cout << endl;
}

void eigen_test2(Eigen::Matrix<double, S, S> &A,
                 Eigen::Matrix<double, S, S> *B,
                 Eigen::Matrix<double, S, S> *C, int num_iter)
{
   double utime, rtime, flops;

   Eigen::Matrix<double, S, S> *Bk, *Ck;

   tic();
   if (M == 1)
   {
      for (int k = 0; k < num_iter; k++)
         C[k] = A*B[k];
   }
   else
   {
      for (int k = 0; k < num_iter/M; k++)
      {
         Bk = &B[k*M];
         Ck = &C[k*M];
         for (int l = 0; l < M; l++)
            Ck[l] = A*Bk[l];
      }
   }
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   flops = double(S)*S*S*num_iter;
   cout << "Mult A.B[]    utime: " << utime << " s" << endl;
   cout << "Mult A.B[]    rtime: " << rtime << " s" << endl;
   cout << "Mult A.B[]   Gflops: " << flops/1e9 << endl;
   cout << "Mult A.B[] Gflops/s: " << flops/rtime/1e9 << endl;

   cout << endl;

   tic();
   if (M == 1)
   {
      for (int k = 0; k < num_iter; k++)
         C[k] = A.transpose()*B[k];
   }
   else
   {
      for (int k = 0; k < num_iter/M; k++)
      {
         Bk = &B[k*M];
         Ck = &C[k*M];
         for (int l = 0; l < M; l++)
            Ck[l] = A.transpose()*Bk[l];
      }
   }
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   flops = double(S)*S*S*num_iter;
   cout << "Mult A^t.B[]    utime: " << utime << " s" << endl;
   cout << "Mult A^t.B[]    rtime: " << rtime << " s" << endl;
   cout << "Mult A^t.B[]   Gflops: " << flops/1e9 << endl;
   cout << "Mult A^t.B[] Gflops/s: " << flops/rtime/1e9 << endl;

   cout << endl;

   tic();
   if (M == 1)
   {
      for (int k = 0; k < num_iter; k++)
         C[k] = A*B[k].transpose();
   }
   else
   {
      for (int k = 0; k < num_iter/M; k++)
      {
         Bk = &B[k*M];
         Ck = &C[k*M];
         for (int l = 0; l < M; l++)
            Ck[l] = A*Bk[l].transpose();
      }
   }
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   flops = double(S)*S*S*num_iter;
   cout << "Mult A.B[]^t    utime: " << utime << " s" << endl;
   cout << "Mult A.B[]^t    rtime: " << rtime << " s" << endl;
   cout << "Mult A.B[]^t   Gflops: " << flops/1e9 << endl;
   cout << "Mult A.B[]^t Gflops/s: " << flops/rtime/1e9 << endl;

   cout << endl;

   tic();
   if (M == 1)
   {
      for (int k = 0; k < num_iter; k++)
         C[k] = A.transpose()*B[k].transpose();
   }
   else
   {
      for (int k = 0; k < num_iter/M; k++)
      {
         Bk = &B[k*M];
         Ck = &C[k*M];
         for (int l = 0; l < M; l++)
            Ck[l] = A.transpose()*Bk[l].transpose();
      }
   }
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   flops = double(S)*S*S*num_iter;
   cout << "Mult A^t.B[]^t    utime: " << utime << " s" << endl;
   cout << "Mult A^t.B[]^t    rtime: " << rtime << " s" << endl;
   cout << "Mult A^t.B[]^t   Gflops: " << flops/1e9 << endl;
   cout << "Mult A^t.B[]^t Gflops/s: " << flops/rtime/1e9 << endl;

   cout << endl;
}
#endif
