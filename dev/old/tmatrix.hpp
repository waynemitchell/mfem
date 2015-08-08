
#ifndef TEST_TMATRIX
#define TEST_TMATRIX

#include <stdlib.h>


// Fixed size matrix view of double* with arbitrary fixed row and column strides
// entry (i,j) of the matrix is at offset i*RowStride+j*ColStride
template <int Height, int Width, int RowStride, int ColStride>
class SMatrix
{
public:
   double *data;

   int NumRows() const
   {
      return Height;
   }

   int NumCols() const
   {
      return Width;
   }

   double &operator()(const int i, const int j)
   {
      return data[i * RowStride + j * ColStride];
   }

   const double &operator()(const int i, const int j) const
   {
      return data[i * RowStride + j * ColStride];
   }

   void Set(const double val)
   {
      if (RowStride < ColStride)
      {
         for (int j = 0; j < Width; j++)
            for (int i = 0; i < Height; i++)
               operator()(i,j) = val;
      }
      else
      {
         for (int i = 0; i < Height; i++)
            for (int j = 0; j < Width; j++)
               operator()(i,j) = val;
      }
   }
};


// Matrix multiplication C {=,+=} A.B for SMatrix'es.
// This version can be used to implement A^t.B, A.B^t, and A^t.B^t as well.
template <int HeightA, int WidthA, int WidthB, int RowStrideA, int ColStrideA,
          int RowStrideB, int ColStrideB, int RowStrideC, int ColStrideC,
          bool Add>
inline void Mult(const SMatrix<HeightA,WidthA,RowStrideA,ColStrideA> &A,
                 const SMatrix<WidthA,WidthB,RowStrideB,ColStrideB> &B,
                 SMatrix<HeightA,WidthB,RowStrideC,ColStrideC> &C)
{
   if (!Add)
      C.Set(0.0);

   for (int iA = 0; iA < HeightA; iA++)
      for (int jB = 0; jB < WidthB; jB++)
         for (int k = 0; k < WidthA; k++)
            C(iA,jB) += A(iA,k) * B(k,jB);
}

// Matrix multiplication C {=,+=} A.B for double*.
// This version can be used to implement A^t.B, A.B^t, and A^t.B^t as well.
template <int HeightA, int WidthA, int WidthB, int RowStrideA, int ColStrideA,
          int RowStrideB, int ColStrideB, int RowStrideC, int ColStrideC,
          bool Add>
inline void Mult(const double *A, const double *B, double *C)
{
#if 1
   Mult<HeightA, WidthA, WidthB, RowStrideA, ColStrideA, RowStrideB,
        ColStrideB, RowStrideC, ColStrideC, Add>(
           (const SMatrix<HeightA,WidthA,RowStrideA,ColStrideA> &)(A),
           (const SMatrix<WidthA,WidthB,RowStrideB,ColStrideB> &)(B),
           (SMatrix<HeightA,WidthB,RowStrideC,ColStrideC> &)(C));
#else
   if (!Add)
   {
      if (RowStrideC < ColStrideC)
      {
         for (int i = 0; i < HeightA; i++)
            for (int j = 0; j < WidthB; j++)
               C[i * RowStrideC + j * ColStrideC] = 0.0;
      }
      else
      {
         for (int j = 0; j < WidthB; j++)
            for (int i = 0; i < HeightA; i++)
               C[i * RowStrideC + j * ColStrideC] = 0.0;
      }
   }

   for (int iA = 0; iA < HeightA; iA++)
      for (int jB = 0; jB < WidthB; jB++)
         for (int k = 0; k < WidthA; k++)
            C[iA * RowStrideC + jB * ColStrideC] +=
               A[iA * RowStrideA + k * ColStrideA] *
               B[k * RowStrideB + jB * ColStrideB];
#endif
}


// Class for a fixed size matrix with column-major storage
template <int Height, int Width>
class TMatrix
{
public:
   double data[Width][Height]; // column-major storage

   void Set(const double val)
   {
      for (int j = 0; j < Width; j++)
         for (int i = 0; i < Height; i++)
            data[j][i] = val;
   }

   void Random()
   {
      for (int j = 0; j < Width; j++)
         for (int i = 0; i < Height; i++)
            data[j][i] = rand() / (RAND_MAX + 1.0);
   }

   inline double Det() const;

   inline void CalcAdjugate(TMatrix<Height,Width> &adj) const;

   // Given the adjugate matrix, compute the determinant using the identity
   // det(A) I = A.adj(A) which is more efficient than Det() for 3x3 and
   // larger matrices
   inline double Det(const TMatrix<Height,Width> &adj) const;

   void Scale(const double scale)
   {
      for (int j = 0; j < Width; j++)
         for (int i = 0; i < Height; i++)
            data[j][i] *= scale;
   }
};


template <> inline double TMatrix<1,1>::Det() const
{
   return data[0][0];
}

template <> inline double TMatrix<2,2>::Det() const
{
   return data[0][0]*data[1][1] - data[1][0]*data[0][1];
}

template <> inline double TMatrix<3,3>::Det() const
{
   return (data[0][0]*(data[1][1]*data[2][2] - data[1][2]*data[2][1]) -
           data[0][1]*(data[1][0]*data[2][2] - data[1][2]*data[2][0]) +
           data[0][2]*(data[1][0]*data[2][1] - data[1][1]*data[2][0]));
}

template <> inline void TMatrix<1,1>::CalcAdjugate(TMatrix<1,1> &adj) const
{
   adj.data[0][0] = 1.;
}

template <> inline void TMatrix<2,2>::CalcAdjugate(TMatrix<2,2> &adj) const
{
   adj.data[0][0] =  data[1][1];
   adj.data[1][0] = -data[1][0];
   adj.data[0][1] = -data[0][1];
   adj.data[1][1] =  data[0][0];
}

template <> inline void TMatrix<3,3>::CalcAdjugate(TMatrix<3,3> &adj) const
{
   adj.data[0][0] = data[1][1]*data[2][2] - data[2][1]*data[1][2];
   adj.data[1][0] = data[2][0]*data[1][2] - data[1][0]*data[2][2];
   adj.data[2][0] = data[1][0]*data[2][1] - data[2][0]*data[1][1];
   adj.data[0][1] = data[2][1]*data[0][2] - data[0][1]*data[2][2];
   adj.data[1][1] = data[0][0]*data[2][2] - data[2][0]*data[0][2];
   adj.data[2][1] = data[2][0]*data[0][1] - data[0][0]*data[2][1];
   adj.data[0][2] = data[0][1]*data[1][2] - data[1][1]*data[0][2];
   adj.data[1][2] = data[1][0]*data[0][2] - data[0][0]*data[1][2];
   adj.data[2][2] = data[0][0]*data[1][1] - data[1][0]*data[0][1];
}

template <> inline double TMatrix<1,1>::Det(const TMatrix<1,1> &adj) const
{
   return Det();
}

template <> inline double TMatrix<2,2>::Det(const TMatrix<2,2> &adj) const
{
   return Det();
}

template <> inline double TMatrix<3,3>::Det(const TMatrix<3,3> &adj) const
{
   return (data[0][0]*adj.data[0][0] + data[0][1]*adj.data[1][0] +
           data[0][2]*adj.data[2][0]);
}


// Matrix multiplication, C += A.B
template <int HeightA, int WidthA, int WidthB>
inline void AddMult(const TMatrix<HeightA, WidthA> &A,
                    const TMatrix<WidthA, WidthB> &B,
                    TMatrix<HeightA, WidthB> &C)
{
#if 1
   for (int iA = 0; iA < HeightA; iA++)
      for (int jB = 0; jB < WidthB; jB++)
         for (int k = 0; k < WidthA; k++)
            C.data[jB][iA] += A.data[k][iA] * B.data[jB][k];
#elif 0
   for (int iA = 0; iA < HeightA; iA++)
      for (int jB = 0; jB < WidthB; jB++)
      {
         register double d = 0.0;
         for (int k = 0; k < WidthA; k++)
            d += A.data[k][iA] * B.data[jB][k];
         C.data[jB][iA] += d;
      }
#else
   Mult<HeightA,WidthA,WidthB,1,HeightA,1,WidthA,1,HeightA,true>(
      &A.data[0][0], &B.data[0][0], &C.data[0][0]);
#endif
}

// Matrix multiplication, C = A.B
template <int HeightA, int WidthA, int WidthB>
inline void Mult(const TMatrix<HeightA, WidthA> &A,
                 const TMatrix<WidthA, WidthB> &B,
                 TMatrix<HeightA, WidthB> &C)
{
#if 1
   C.Set(0.0);
   AddMult<HeightA, WidthA, WidthB>(A, B, C);
#else
   Mult<HeightA,WidthA,WidthB,1,HeightA,1,WidthA,1,HeightA,false>(
      &A.data[0][0], &B.data[0][0], &C.data[0][0]);
#endif
}


// Matrix multiplication, C += A^t.B
template <int HeightA, int WidthA, int WidthB>
inline void AddMultAtB(const TMatrix<HeightA, WidthA> &A,
                       const TMatrix<HeightA, WidthB> &B,
                       TMatrix<WidthA, WidthB> &C)
{
#if 1
   for (int jA = 0; jA < WidthA; jA++)
      for (int jB = 0; jB < WidthB; jB++)
         for (int k = 0; k < HeightA; k++)
            C.data[jB][jA] += A.data[jA][k] * B.data[jB][k];
#elif 0
   for (int jA = 0; jA < WidthA; jA++)
      for (int jB = 0; jB < WidthB; jB++)
      {
         double d = 0.0;
         for (int k = 0; k < HeightA; k++)
            d += A.data[jA][k] * B.data[jB][k];
         C.data[jB][jA] += d;
      }
#else
   Mult<WidthA,HeightA,WidthB,HeightA,1,1,HeightA,1,WidthA,true>(
      &A.data[0][0], &B.data[0][0], &C.data[0][0]);
#endif
}

// Matrix multiplication, C = A^t.B
template <int HeightA, int WidthA, int WidthB>
inline void MultAtB(const TMatrix<HeightA, WidthA> &A,
                    const TMatrix<HeightA, WidthB> &B,
                    TMatrix<WidthA, WidthB> &C)
{
#if 1
   C.Set(0.0);
   AddMultAtB<HeightA, WidthA, WidthB>(A, B, C);
#else
   Mult<WidthA,HeightA,WidthB,HeightA,1,1,HeightA,1,WidthA,false>(
      &A.data[0][0], &B.data[0][0], &C.data[0][0]);
#endif
}

// Matrix multiplication, C += A.B^t
template <int HeightA, int WidthA, int HeightB>
inline void AddMultABt(const TMatrix<HeightA, WidthA> &A,
                       const TMatrix<HeightB, WidthA> &B,
                       TMatrix<HeightA, HeightB> &C)
{
#if 1
   for (int iA = 0; iA < HeightA; iA++)
      for (int iB = 0; iB < HeightB; iB++)
         for (int k = 0; k < WidthA; k++)
            C.data[iB][iA] += A.data[k][iA] * B.data[k][iB];
#elif 0
   for (int iA = 0; iA < HeightA; iA++)
      for (int k = 0; k < WidthA; k++)
         for (int iB = 0; iB < HeightB; iB++)
            C.data[iB][iA] += A.data[k][iA] * B.data[k][iB];
#elif 0
   for (int k = 0; k < WidthA; k++)
      for (int iA = 0; iA < HeightA; iA++)
         for (int iB = 0; iB < HeightB; iB++)
            C.data[iB][iA] += A.data[k][iA] * B.data[k][iB];
#elif 0
   for (int k = 0; k < WidthA; k++)
      for (int iB = 0; iB < HeightB; iB++)
         for (int iA = 0; iA < HeightA; iA++)
            C.data[iB][iA] += A.data[k][iA] * B.data[k][iB];
#elif 0
   for (int iB = 0; iB < HeightB; iB++)
      for (int k = 0; k < WidthA; k++)
         for (int iA = 0; iA < HeightA; iA++)
            C.data[iB][iA] += A.data[k][iA] * B.data[k][iB];
#elif 0
   for (int iB = 0; iB < HeightB; iB++)
      for (int iA = 0; iA < HeightA; iA++)
         for (int k = 0; k < WidthA; k++)
            C.data[iB][iA] += A.data[k][iA] * B.data[k][iB];
#elif 0
   for (int iA = 0; iA < HeightA; iA++)
      for (int iB = 0; iB < HeightB; iB++)
      {
         double d = 0.0;
         for (int k = 0; k < WidthA; k++)
            d += A.data[k][iA] * B.data[k][iB];
         C.data[iB][iA] += d;
      }
#else
   Mult<HeightA,WidthA,HeightB,1,HeightA,HeightB,1,1,HeightA,true>(
      &A.data[0][0], &B.data[0][0], &C.data[0][0]);
#endif
}

// Matrix multiplication, C = A.B^t
template <int HeightA, int WidthA, int HeightB>
inline void MultABt(const TMatrix<HeightA, WidthA> &A,
                    const TMatrix<HeightB, WidthA> &B,
                    TMatrix<HeightA, HeightB> &C)
{
#if 1
   C.Set(0.0);
   AddMultABt<HeightA, WidthA, HeightB>(A, B, C);
#else
   Mult<HeightA,WidthA,HeightB,1,HeightA,HeightB,1,1,HeightA,false>(
      &A.data[0][0], &B.data[0][0], &C.data[0][0]);
#endif
}


// Matrix multiplication, C += A^t.B^t
template <int HeightA, int WidthA, int HeightB>
inline void AddMultAtBt(const TMatrix<HeightA, WidthA> &A,
                        const TMatrix<HeightB, HeightA> &B,
                        TMatrix<WidthA, HeightB> &C)
{
#if 1
   for (int jA = 0; jA < WidthA; jA++)
      for (int iB = 0; iB < HeightB; iB++)
         for (int k = 0; k < HeightA; k++)
            C.data[iB][jA] += A.data[jA][k] * B.data[k][iB];
#elif 0
   for (int jA = 0; jA < WidthA; jA++)
      for (int iB = 0; iB < HeightB; iB++)
      {
         double d = 0.0;
         for (int k = 0; k < HeightA; k++)
            d += A.data[jA][k] * B.data[k][iB];
         C.data[iB][jA] += d;
      }
#else
   Mult<WidthA,HeightA,HeightB,HeightA,1,HeightB,1,1,WidthA,true>(
      &A.data[0][0], &B.data[0][0], &C.data[0][0]);
#endif
}

// Matrix multiplication, C = A^t.B^t
template <int HeightA, int WidthA, int HeightB>
inline void MultAtBt(const TMatrix<HeightA, WidthA> &A,
                     const TMatrix<HeightB, HeightA> &B,
                     TMatrix<WidthA, HeightB> &C)
{
#if 1
   C.Set(0.0);
   AddMultAtBt<HeightA, WidthA, HeightB>(A, B, C);
#else
   Mult<WidthA,HeightA,HeightB,HeightA,1,HeightB,1,1,WidthA,false>(
      &A.data[0][0], &B.data[0][0], &C.data[0][0]);
#endif
}

#endif
