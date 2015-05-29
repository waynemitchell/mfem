// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_TEMPLATE
#define MFEM_TEMPLATE

#include "mfem.hpp"
#include <cstdlib> // std::rand()
#include <ctime>   // std::time()

namespace mfem
{

template <int S>
struct TVector
{
public:
   static const int size = S;
   double data[size];

   double &operator[](int i) { return data[i]; }
   const double &operator[](int i) const { return data[i]; }

   void Set(const double d)
   {
      for (int i = 0; i < size; i++)
      {
         data[i] = d;
      }
   }

   void Set(const TVector<size> &v)
   {
      for (int i = 0; i < size; i++)
      {
         data[i] = v.data[i];
      }
   }

   void Set(const double *v)
   {
      for (int i = 0; i < size; i++)
      {
         data[i] = v[i];
      }
   }

   void Assemble(double *v) const
   {
      for (int i = 0; i < size; i++)
      {
         v[i] += data[i];
      }
   }

   void Random()
   {
      for (int i = 0; i < size; i++)
      {
         data[i] = std::rand() / (RAND_MAX + 1.0);
      }
   }

   void Scale(const double scale)
   {
      for (int i = 0; i < size; i++)
      {
         data[i] *= scale;
      }
   }
};

template <int N1, int N2>
struct TMatrix : public TVector<N1*N2>
{
   using TVector<N1*N2>::size;
   using TVector<N1*N2>::data;

   static inline int ind(int i1, int i2) { return (i1+N1*i2); }

   double &operator()(int i, int j) { return data[ind(i,j)]; }
   const double &operator()(int i, int j) const { return data[ind(i,j)]; }

   double &at(int i, int j) { return data[ind(i,j)]; }
   const double &at(int i, int j) const { return data[ind(i,j)]; }

   // operator()(int) returns a column of the matrix
   TVector<N1> &operator()(int i2)
   { return reinterpret_cast<TVector<N1> &>(at(0,i2)); }
   const TVector<N1> &operator()(int i2) const
   { return reinterpret_cast<const TVector<N1> &>(at(0,i2)); }

   template <bool Add>
   void Mult(const TVector<N2> &x, TVector<N1> &y) const
   {
      if (!Add) { y.Set(0.0); }
      for (int i1 = 0; i1 < N1; i1++)
      {
         for (int i2 = 0; i2 < N2; i2++)
         {
            y[i1] += at(i1,i2) * x[i2];
         }
      }
   }

   template <bool Add>
   void MultTranspose(const TVector<N1> &x, TVector<N2> &y) const
   {
      if (!Add) { y.Set(0.0); }
      for (int i2 = 0; i2 < N2; i2++)
      {
         for (int i1 = 0; i1 < N1; i1++)
         {
            y[i2] += at(i1,i2) * x[i1];
         }
      }
   }

   inline double Det() const;

   inline void CalcAdjugate(TMatrix<N1,N2> &adj) const;

   // Given the adjugate matrix, compute the determinant using the identity
   // det(A) I = A.adj(A) which is more efficient than Det() for 3x3 and
   // larger matrices
   inline double Det(const TMatrix<N1,N2> &adj) const;
};

template <> inline double TMatrix<1,1>::Det() const
{
   return at(0,0);
}

template <> inline double TMatrix<2,2>::Det() const
{
   return at(0,0)*at(1,1) - at(1,0)*at(0,1);
}

template <> inline double TMatrix<3,3>::Det() const
{
   return (at(0,0)*(at(1,1)*at(2,2) - at(2,1)*at(1,2)) -
           at(1,0)*(at(0,1)*at(2,2) - at(2,1)*at(0,2)) +
           at(2,0)*(at(0,1)*at(1,2) - at(1,1)*at(0,2)));
}

template <> inline void TMatrix<1,1>::CalcAdjugate(TMatrix<1,1> &adj) const
{
   adj(0,0) = 1.;
}

template <> inline void TMatrix<2,2>::CalcAdjugate(TMatrix<2,2> &adj) const
{
   adj(0,0) =  at(1,1);
   adj(0,1) = -at(0,1);
   adj(1,0) = -at(1,0);
   adj(1,1) =  at(0,0);
}

template <> inline void TMatrix<3,3>::CalcAdjugate(TMatrix<3,3> &adj) const
{
   adj(0,0) = at(1,1)*at(2,2) - at(1,2)*at(2,1);
   adj(0,1) = at(0,2)*at(2,1) - at(0,1)*at(2,2);
   adj(0,2) = at(0,1)*at(1,2) - at(0,2)*at(1,1);
   adj(1,0) = at(1,2)*at(2,0) - at(1,0)*at(2,2);
   adj(1,1) = at(0,0)*at(2,2) - at(0,2)*at(2,0);
   adj(1,2) = at(0,2)*at(1,0) - at(0,0)*at(1,2);
   adj(2,0) = at(1,0)*at(2,1) - at(1,1)*at(2,0);
   adj(2,1) = at(0,1)*at(2,0) - at(0,0)*at(2,1);
   adj(2,2) = at(0,0)*at(1,1) - at(0,1)*at(1,0);
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
   return at(0,0)*adj(0,0) + at(1,0)*adj(0,1) + at(2,0)*adj(0,2);
}

template <int N1, int N2, int N3>
struct TTensor : TVector<N1*N2*N3>
{
   using TVector<N1*N2*N3>::size;
   using TVector<N1*N2*N3>::data;

   static inline int ind(int i1, int i2, int i3)
   { return (i1+N1*(i2+N2*i3)); }

   double &operator()(int i, int j, int k) { return data[ind(i,j,k)]; }
   const double &operator()(int i, int j, int k) const
   { return data[ind(i,j,k)]; }

   // operator()(int) returns a sub-matrix of the tensor by fixing the given
   // last index
   TMatrix<N1,N2> &operator()(int i3)
   { return reinterpret_cast<TMatrix<N1,N2> &>(operator()(0,0,i3)); }
   const TMatrix<N1,N2> &operator()(int i3) const
   { return reinterpret_cast<const TMatrix<N1,N2> &>(operator()(0,0,i3)); }
};

template <int N1, int N2, int N3, int N4>
struct TTensor4 : TVector<N1*N2*N3*N4>
{
   using TVector<N1*N2*N3*N4>::size;
   using TVector<N1*N2*N3*N4>::data;

   static inline int ind(int i1, int i2, int i3, int i4)
   { return (i1+N1*(i2+N2*(i3+N3*i4))); }

   double &operator()(int i, int j, int k, int l)
   { return data[ind(i,j,k,l)]; }
   const double &operator()(int i, int j, int k, int l) const
   { return data[ind(i,j,k,l)]; }

   // operator()(int) returns a rank-3 sub-tensor of the tensor by fixing the
   // given last index
   TTensor<N1,N2,N3> &operator()(int i4)
   { return reinterpret_cast<TTensor<N1,N2,N3> &>(operator()(0,0,0,i4)); }
   const TTensor<N1,N2,N3> &operator()(int i4) const
   { return reinterpret_cast<const TTensor<N1,N2,N3> &>(operator()(0,0,0,i4)); }
};

template <int Dim, int N>
struct SquareTensor;

template <int N>
struct SquareTensor<1, N> : public TVector<N> { };

template <int N>
struct SquareTensor<2, N> : public TMatrix<N, N> { };

template <int N>
struct SquareTensor<3, N> : public TTensor<N, N, N> { };

template <int N>
struct SquareTensor<4, N> : public TTensor4<N, N, N, N> { };

template <int Dim, int N>
struct TCellData : public SquareTensor<Dim, N> { };


// Reshape (or "view as") functions

// All tensors are TVectors, so we do not need to reshape to TVector

// Reshape to (view as) TMatrix
template <int N1, int N2>
inline TMatrix<N1,N2> &Reshape(TVector<N1*N2> &tensor)
{
   return reinterpret_cast<TMatrix<N1,N2> &>(tensor);
}

template <int N1, int N2>
inline const TMatrix<N1,N2> &Reshape(const TVector<N1*N2> &tensor)
{
   return reinterpret_cast<const TMatrix<N1,N2> &>(tensor);
}

// Reshape to (view as) TTensor
template <int N1, int N2, int N3>
inline TTensor<N1,N2,N3> &Reshape(TVector<N1*N2*N3> &tensor)
{
   return reinterpret_cast<TTensor<N1,N2,N3> &>(tensor);
}

template <int N1, int N2, int N3>
inline const TTensor<N1,N2,N3> &Reshape(const TVector<N1*N2*N3> &tensor)
{
   return reinterpret_cast<const TTensor<N1,N2,N3> &>(tensor);
}

// Reshape to (view as) TTensor4
template <int N1, int N2, int N3, int N4>
inline TTensor4<N1,N2,N3,N4> &Reshape(TVector<N1*N2*N3*N4> &tensor)
{
   return reinterpret_cast<TTensor4<N1,N2,N3,N4> &>(tensor);
}

template <int N1, int N2, int N3, int N4>
inline const TTensor4<N1,N2,N3,N4> &Reshape(const TVector<N1*N2*N3*N4> &tensor)
{
   return reinterpret_cast<const TTensor4<N1,N2,N3,N4> &>(tensor);
}


// Tensor multiplication

// C_{i,j,k,l}  {=|+=}  \sum_s A_{i,s,j} B_{k,s,l}
// The string '1234' in the name indicates the order of the ijkl indices
// in the output tensor C.
template <int Impl, bool Add, int A1, int A2, int A3, int B1, int B3>
inline
void Mult_1234(const TTensor<A1,A2,A3> &A, const TTensor<B1,A2,B3> &B,
               TTensor4<A1,A3,B1,B3> &C)
{
#define IND3(A,A1,A2,i1,i2,i3) \
   ((A).data[(i1)+(A1)*((i2)+(A2)*(i3))])
#define IND4(A,A1,A2,A3,i1,i2,i3,i4) \
   ((A).data[(i1)+(A1)*((i2)+(A2)*((i3)+(A3)*(i4)))])

   if (Impl == 0)
   {
      if (!Add) { C.Set(0.0); }

      for (int i = 0; i < A1; i++)
      {
         for (int j = 0; j < A3; j++)
         {
            for (int k = 0; k < B1; k++)
            {
               for (int l = 0; l < B3; l++)
               {
                  for (int s = 0; s < A2; s++)
                  {
                     // C(i,j,k,l) += A(i,s,j) * B(k,s,l);
                     IND4(C,A1,A3,B1,i,j,k,l) +=
                        IND3(A,A1,A2,i,s,j) * IND3(B,B1,A2,k,s,l);
                  }
               }
            }
         }
      }
   }
   else if (Impl == 1)
   {
      if (!Add) { C.Set(0.0); }

      for (int s = 0; s < A2; s++)
      {
         for (int l = 0; l < B3; l++)
         {
            for (int k = 0; k < B1; k++)
            {
               for (int j = 0; j < A3; j++)
               {
                  for (int i = 0; i < A1; i++)
                  {
                     // C(i,j,k,l) += A(i,s,j) * B(k,s,l);
                     IND4(C,A1,A3,B1,i,j,k,l) +=
                        IND3(A,A1,A2,i,s,j) * IND3(B,B1,A2,k,s,l);
                  }
               }
            }
         }
      }
   }

#undef IND4
#undef IND3
}

// C_{i,k,l,j}  {=|+=}  \sum_s A_{i,s,j} B_{k,s,l}
// The string '1342' in the name indicates the order of the ijkl indices
// in the output tensor C.
template <bool Add, int A1, int A2, int A3, int B1, int B3>
inline
void Mult_1342(const TTensor<A1,A2,A3> &A, const TTensor<B1,A2,B3> &B,
               TTensor4<A1,B1,B3,A3> &C)
{
   if (!Add) { C.Set(0.0); }
#define IND3(A,A1,A2,i1,i2,i3) \
   ((A).data[(i1)+(A1)*((i2)+(A2)*(i3))])
#define IND4(A,A1,A2,A3,i1,i2,i3,i4) \
   ((A).data[(i1)+(A1)*((i2)+(A2)*((i3)+(A3)*(i4)))])
   for (int j = 0; j < A3; j++)
   {
      for (int l = 0; l < B3; l++)
      {
         for (int k = 0; k < B1; k++)
         {
            for (int i = 0; i < A1; i++)
            {
               for (int s = 0; s < A2; s++)
               {
                  // C(i,k,l,j) += A(i,s,j) * B(k,s,l);
                  IND4(C,A1,B1,B3,i,k,l,j) +=
                     IND3(A,A1,A2,i,s,j) * IND3(B,B1,A2,k,s,l);
               }
            }
         }
      }
   }
#undef IND4
#undef IND3
}

// C  {=|+=}  A.B
template <bool Add, int A1, int A2, int B2>
inline
void Mult_AB(const TMatrix<A1,A2> &A, const TMatrix<A2,B2> &B,
             TMatrix<A1,B2> &C)
{
   Mult_1234<0,Add>(Reshape<A1,A2,1>(A),
                    Reshape<1,A2,B2>(B),
                    Reshape<A1,1,1,B2>(C));
}

// C  {=|+=}  At.B
template <bool Add, int A1, int A2, int B2>
inline
void Mult_AtB(const TMatrix<A1,A2> &A, const TMatrix<A1,B2> &B,
              TMatrix<A2,B2> &C)
{
   Mult_1234<1,Add>(Reshape<1,A1,A2>(A),
                    Reshape<1,A1,B2>(B),
                    Reshape<1,A2,1,B2>(C));
}

// C  {=|+=}  A.Bt
template <bool Add, int A1, int A2, int B1>
inline
void Mult_ABt(const TMatrix<A1,A2> &A, const TMatrix<B1,A2> &B,
              TMatrix<A1,B1> &C)
{
   Mult_1234<0,Add>(Reshape<A1,A2,1>(A),
                    Reshape<B1,A2,1>(B),
                    Reshape<A1,1,B1,1>(C));
}

// C  {=|+=}  At.Bt
template <bool Add, int A1, int A2, int B1>
inline
void Mult_AtBt(const TMatrix<A1,A2> &A, const TMatrix<B1,A1> &B,
               TMatrix<A2,B1> &C)
{
   Mult_1234<0,Add>(Reshape<1,A1,A2>(A),
                    Reshape<B1,A1,1>(B),
                    Reshape<1,A2,B1,1>(C));
}

// C_{i,j,k}  {=|+=}  \sum_s A_{s,i} B_{s,j,k}
template <bool Add, int A1, int A2, int B2, int B3>
inline
void Mult_1_1(const TMatrix<A1,A2> &A, const TTensor<A1,B2,B3> &B,
              TTensor<A2,B2,B3> &C)
{
   Mult_AtB<Add>(Reshape<A1,A2>(A),
                 Reshape<A1,B2*B3>(B),
                 Reshape<A2,B2*B3>(C));
}

// C_{i,j,k}  {=|+=}  \sum_s A_{s,j} B_{i,s,k}
template <bool Add, int A1, int A2, int B1, int B3>
inline
void Mult_1_2(const TMatrix<A1,A2> &A, const TTensor<B1,A1,B3> &B,
              TTensor<B1,A2,B3> &C)
{
   Mult_1342<Add>(Reshape<B1,A1,B3>(B),
                  Reshape<1,A1,A2>(A),
                  Reshape<B1,1,A2,B3>(C));
}

// C_{i,j,k}  {=|+=}  \sum_s A_{s,k} B_{i,j,s}
template <bool Add, int A1, int A2, int B1, int B2>
inline
void Mult_1_3(const TMatrix<A1,A2> &A, const TTensor<B1,B2,A1> &B,
              TTensor<B1,B2,A2> &C)
{
   Mult_AB<Add>(Reshape<B1*B2,A1>(B),
                Reshape<A1,A2>(A),
                Reshape<B1*B2,A2>(C));
}

// C_{i,j,k}  {=|+=}  \sum_s A_{i,s} B_{s,j,k}
template <bool Add, int A1, int A2, int B2, int B3>
inline
void Mult_2_1(const TMatrix<A1,A2> &A, const TTensor<A2,B2,B3> &B,
              TTensor<A1,B2,B3> &C)
{
   Mult_AB<Add>(Reshape<A1,A2>(A),
                Reshape<A2,B2*B3>(B),
                Reshape<A1,B2*B3>(C));
}

// C_{i,j,k}  {=|+=}  \sum_s A_{j,s} B_{i,s,k}
template <bool Add, int A1, int A2, int B1, int B3>
inline
void Mult_2_2(const TMatrix<A1,A2> &A, const TTensor<B1,A2,B3> &B,
              TTensor<B1,A1,B3> &C)
{
   Mult_1342<Add>(Reshape<B1,A2,B3>(B),
                  Reshape<A1,A2,1>(A),
                  Reshape<B1,A1,1,B3>(C));
}

// C_{i,j,k}  {=|+=}  \sum_s A_{k,s} B_{i,j,s}
template <bool Add, int A1, int A2, int B1, int B2>
inline
void Mult_2_3(const TMatrix<A1,A2> &A, const TTensor<B1,B2,A2> &B,
              TTensor<B1,B2,A1> &C)
{
   Mult_ABt<Add>(Reshape<B1*B2,A2>(B),
                 Reshape<A1,A2>(A),
                 Reshape<B1*B2,A1>(C));
}

// C_{k,i,l,j}  {=|+=}  A_{s,i} A_{s,j} B_{k,s,l}
template <bool Add, int A1, int A2, int B1, int B3>
inline
void TensorAssemble(const TMatrix<A1,A2> &A, const TTensor<B1,A1,B3> &B,
                    TTensor4<B1,A2,B3,A2> &C)
{
   if (!Add) { C.Set(0.0); }
   for (int j = 0; j < A2; j++)
   {
      for (int l = 0; l < B3; l++)
      {
         for (int i = 0; i < A2; i++)
         {
            for (int k = 0; k < B1; k++)
            {
               for (int s = 0; s < A1; s++)
               {
                  C(k,i,l,j) += A(s,i) * A(s,j) * B(k,s,l);
               }
            }
         }
      }
   }
}

// D_{k,i,l,j}  {=|+=}  A_{s,i} B_{s,j} C_{k,s,l}
template <bool Add, int A1, int A2, int B2, int C1, int C3>
inline
void TensorAssemble(const TMatrix<A1,A2> &A,
                    const TMatrix<A1,B2> &B,
                    const TTensor<C1,A1,C3> &C,
                    TTensor4<C1,A2,C3,B2> &D)
{
   if (!Add) { D.Set(0.0); }
   for (int j = 0; j < B2; j++)
   {
      for (int l = 0; l < C3; l++)
      {
         for (int i = 0; i < A2; i++)
         {
            for (int k = 0; k < C1; k++)
            {
               for (int s = 0; s < A1; s++)
               {
                  D(k,i,l,j) += A(s,i) * B(s,j) * C(k,s,l);
               }
            }
         }
      }
   }
}


// Finite elements

void CalcShapeMatrix(const FiniteElement &fe, const IntegrationRule &ir,
                     double *B, const Array<int> *dof_map = NULL)
{
   // - B must be (nip x dof) with column major storage
   // - The inverse of dof_map is applied to reorder the local dofs.
   int nip = ir.GetNPoints();
   int dof = fe.GetDof();
   Vector shape(dof);

   for (int ip = 0; ip < nip; ip++)
   {
      fe.CalcShape(ir.IntPoint(ip), shape);
      for (int id = 0; id < dof; id++)
      {
         int orig_id = dof_map ? (*dof_map)[id] : id;
         B[ip+nip*id] = shape(orig_id);
      }
   }
}

void CalcGradTensor(const FiniteElement &fe, const IntegrationRule &ir,
                    double *G, const Array<int> *dof_map = NULL)
{
   // - G must be (nip x dim x dof) with column major storage
   // - The inverse of dof_map is applied to reorder the local dofs.
   int dim = fe.GetDim();
   int nip = ir.GetNPoints();
   int dof = fe.GetDof();
   DenseMatrix dshape(dof, dim);

   for (int ip = 0; ip < nip; ip++)
   {
      fe.CalcDShape(ir.IntPoint(ip), dshape);
      for (int id = 0; id < dof; id++)
      {
         int orig_id = dof_map ? (*dof_map)[id] : id;
         for (int d = 0; d < dim; d++)
         {
            G[ip+nip*(d+dim*id)] = dshape(orig_id, d);
         }
      }
   }
}

void CalcShapes(const FiniteElement &fe, const IntegrationRule &ir,
                double *B, double *G, const Array<int> *dof_map)
{
   if (B) { mfem::CalcShapeMatrix(fe, ir, B, dof_map); }
   if (G) { mfem::CalcGradTensor(fe, ir, G, dof_map); }
}

template <Geometry::Type G, int P>
class H1_FiniteElement;

struct H1_FiniteElement_Basis
{
   enum Type
   {
      GaussLobatto = 1, // Nodal basis, with nodes at the Gauss-Lobatto points
      Positive     = 2  // Positive basis, Bernstein polynomials
   };
};

template <int P>
class H1_FiniteElement<Geometry::SEGMENT, P>
{
public:
   static const Geometry::Type geom = Geometry::SEGMENT;
   static const int dim    = 1;
   static const int degree = P;
   static const int dofs   = P+1;

   static const bool tensor_prod = true;
   static const int  dofs_1d     = P+1;

   typedef TCellData<dim, dofs_1d> dof_data_type;
   // Type for run-time parameter for the constructor
   typedef H1_FiniteElement_Basis::Type parameter_type;

protected:
   const FiniteElement *my_fe;
   const Array<int> *my_dof_map;
   parameter_type type; // run-time specified basis type

public:
   H1_FiniteElement(
      const parameter_type type_ = H1_FiniteElement_Basis::GaussLobatto)
      : type(type_)
   {
      if (type == H1_FiniteElement_Basis::GaussLobatto)
      {
         H1_SegmentElement *fe = new H1_SegmentElement(P);
         my_fe = fe;
         my_dof_map = &fe->GetDofMap();
      }
      else if (type == H1_FiniteElement_Basis::Positive)
      {
         H1Pos_SegmentElement *fe = new H1Pos_SegmentElement(P);
         my_fe = fe;
         my_dof_map = &fe->GetDofMap();
      }
      else
      {
         MFEM_ABORT("invalid basis type!");
      }
   }
   ~H1_FiniteElement() { delete my_fe; }

   void CalcShapes(const IntegrationRule &ir, double *B, double *G) const
   {
      mfem::CalcShapes(*my_fe, ir, B, G, my_dof_map);
   }
   void Calc1DShapes(const IntegrationRule &ir, double *B, double *G) const
   {
      CalcShapes(ir, B, G);
   }
   const Array<int> *GetDofMap() const { return my_dof_map; }
};

template <int P>
class H1_FiniteElement<Geometry::TRIANGLE, P>
{
public:
   static const Geometry::Type geom = Geometry::TRIANGLE;
   static const int dim    = 2;
   static const int degree = P;
   static const int dofs   = ((P + 1)*(P + 2))/2;

   static const bool tensor_prod = false;

   typedef TVector<dofs> dof_data_type;
   // Type for run-time parameter for the constructor
   typedef H1_FiniteElement_Basis::Type parameter_type;

protected:
   const FiniteElement *my_fe;
   parameter_type type; // run-time specified basis type

public:
   H1_FiniteElement(
      const parameter_type type_ = H1_FiniteElement_Basis::GaussLobatto)
      : type(type_)
   {
      if (type == H1_FiniteElement_Basis::GaussLobatto)
      {
         my_fe = new H1_TriangleElement(P);
      }
      else if (type == H1_FiniteElement_Basis::Positive)
      {
         MFEM_ABORT("TODO: implement H1Pos_TriangleElement");
         // my_fe = new H1Pos_TriangleElement(P);
         my_fe = NULL;
      }
      else
      {
         MFEM_ABORT("invalid basis type!");
      }
   }
   ~H1_FiniteElement() { delete my_fe; }

   void CalcShapes(const IntegrationRule &ir, double *B, double *G) const
   {
      mfem::CalcShapes(*my_fe, ir, B, G, NULL);
   }
   const Array<int> *GetDofMap() const { return NULL; }
};

template <int P>
class H1_FiniteElement<Geometry::SQUARE, P>
{
public:
   static const Geometry::Type geom = Geometry::SQUARE;
   static const int dim     = 2;
   static const int degree  = P;
   static const int dofs    = (P+1)*(P+1);

   static const bool tensor_prod = true;
   static const int dofs_1d = P+1;

   typedef TCellData<dim, dofs_1d> dof_data_type;
   // Type for run-time parameter for the constructor
   typedef H1_FiniteElement_Basis::Type parameter_type;

protected:
   const FiniteElement *my_fe, *my_fe_1d;
   const Array<int> *my_dof_map;
   parameter_type type; // run-time specified basis type

public:
   H1_FiniteElement(
      const parameter_type type_ = H1_FiniteElement_Basis::GaussLobatto)
      : type(type_)
   {
      if (type == H1_FiniteElement_Basis::GaussLobatto)
      {
         H1_QuadrilateralElement *fe = new H1_QuadrilateralElement(P);
         my_fe = fe;
         my_dof_map = &fe->GetDofMap();
         my_fe_1d = new L2_SegmentElement(P, type);
      }
      else if (type == H1_FiniteElement_Basis::Positive)
      {
         H1Pos_QuadrilateralElement *fe = new H1Pos_QuadrilateralElement(P);
         my_fe = fe;
         my_dof_map = &fe->GetDofMap();
         my_fe_1d = new L2Pos_SegmentElement(P);
      }
      else
      {
         MFEM_ABORT("invalid basis type!");
      }
   }
   ~H1_FiniteElement() { delete my_fe; delete my_fe_1d; }

   void CalcShapes(const IntegrationRule &ir, double *B, double *G) const
   {
      mfem::CalcShapes(*my_fe, ir, B, G, my_dof_map);
   }
   void Calc1DShapes(const IntegrationRule &ir, double *B, double *G) const
   {
      mfem::CalcShapes(*my_fe_1d, ir, B, G, NULL);
   }
   const Array<int> *GetDofMap() const { return my_dof_map; }
};

template <int P>
class H1_FiniteElement<Geometry::TETRAHEDRON, P>
{
public:
   static const Geometry::Type geom = Geometry::TETRAHEDRON;
   static const int dim    = 3;
   static const int degree = P;
   static const int dofs   = ((P + 1)*(P + 2)*(P + 3))/6;

   static const bool tensor_prod = false;

   typedef TVector<dofs> dof_data_type;
   // Type for run-time parameter for the constructor
   typedef H1_FiniteElement_Basis::Type parameter_type;

protected:
   const FiniteElement *my_fe;
   parameter_type type; // run-time specified basis type

public:
   H1_FiniteElement(
      const parameter_type type_ = H1_FiniteElement_Basis::GaussLobatto)
      : type(type_)
   {
      if (type == H1_FiniteElement_Basis::GaussLobatto)
      {
         my_fe = new H1_TetrahedronElement(P);
      }
      else if (type == H1_FiniteElement_Basis::Positive)
      {
         MFEM_ABORT("TODO: implement H1Pos_TetrahedronElement");
         // my_fe = new H1Pos_TetrahedronElement(P);
         my_fe = NULL;
      }
      else
      {
         MFEM_ABORT("invalid basis type!");
      }
   }
   ~H1_FiniteElement() { delete my_fe; }

   void CalcShapes(const IntegrationRule &ir, double *B, double *G) const
   {
      mfem::CalcShapes(*my_fe, ir, B, G, NULL);
   }
   const Array<int> *GetDofMap() const { return NULL; }
};

template <int P>
class H1_FiniteElement<Geometry::CUBE, P>
{
public:
   static const Geometry::Type geom = Geometry::CUBE;
   static const int dim     = 3;
   static const int degree  = P;
   static const int dofs    = (P+1)*(P+1)*(P+1);

   static const bool tensor_prod = true;
   static const int dofs_1d = P+1;

   typedef TCellData<dim, dofs_1d> dof_data_type;
   // Type for run-time parameter for the constructor
   typedef H1_FiniteElement_Basis::Type parameter_type;

protected:
   const FiniteElement *my_fe, *my_fe_1d;
   const Array<int> *my_dof_map;
   parameter_type type; // run-time specified basis type

public:
   H1_FiniteElement(
      const parameter_type type_ = H1_FiniteElement_Basis::GaussLobatto)
      : type(type_)
   {
      if (type == H1_FiniteElement_Basis::GaussLobatto)
      {
         H1_HexahedronElement *fe = new H1_HexahedronElement(P);
         my_fe = fe;
         my_dof_map = &fe->GetDofMap();
         my_fe_1d = new L2_SegmentElement(P, type);
      }
      else if (type == H1_FiniteElement_Basis::Positive)
      {
         H1Pos_HexahedronElement *fe = new H1Pos_HexahedronElement(P);
         my_fe = fe;
         my_dof_map = &fe->GetDofMap();
         my_fe_1d = new L2Pos_SegmentElement(P);
      }
      else
      {
         MFEM_ABORT("invalid basis type!");
      }
   }
   ~H1_FiniteElement() { delete my_fe; delete my_fe_1d; }

   void CalcShapes(const IntegrationRule &ir, double *B, double *G) const
   {
      mfem::CalcShapes(*my_fe, ir, B, G, my_dof_map);
   }
   void Calc1DShapes(const IntegrationRule &ir, double *B, double *G) const
   {
      mfem::CalcShapes(*my_fe_1d, ir, B, G, NULL);
   }
   const Array<int> *GetDofMap() const { return my_dof_map; }
};


// Integration rules

template <Geometry::Type G, int Q, int Order>
class GenericIntegrationRule
{
public:
   static const Geometry::Type geom = G;
   static const int dim = Geometry::Constants<geom>::Dimension;
   static const int qpts = Q;
   static const int order = Order;

   static const bool tensor_prod = false;

   typedef TVector<qpts> qpt_data_type;

protected:
   TVector<qpts> weights;

public:
   GenericIntegrationRule()
   {
      const IntegrationRule &ir = GetIntRule();
      MFEM_ASSERT(ir.GetNPoints() == qpts, "quadrature rule mismatch");
      for (int j = 0; j < qpts; j++)
      {
         weights[j] = ir.IntPoint(j).weight;
      }
   }

   GenericIntegrationRule(const GenericIntegrationRule &ir)
   {
      weights.Set(ir.weights);
   }

   static const IntegrationRule &GetIntRule()
   {
      return IntRules.Get(geom, order);
   }

   void ApplyWeights(qpt_data_type &qpt_data) const
   {
      for (int j = 0; j < qpts; j++)
      {
         qpt_data.data[j] *= weights.data[j];
      }
   }
};

template <int Dim, int Q>
class TProductIntegrationRule_base;

template <int Q>
class TProductIntegrationRule_base<1, Q>
{
protected:
   TVector<Q> weights_1d;

public:
   void ApplyWeights(TVector<Q> &qpt_data) const
   {
      for (int j = 0; j < Q; j++)
      {
         qpt_data.data[j] *= weights_1d.data[j];
      }
   }
};

template <int Q>
class TProductIntegrationRule_base<2, Q>
{
protected:
   TVector<Q> weights_1d;

public:
   void ApplyWeights(TMatrix<Q,Q> &qpt_data) const
   {
      for (int j1 = 0; j1 < Q; j1++)
      {
         for (int j2 = 0; j2 < Q; j2++)
         {
            qpt_data(j1,j2) *= weights_1d.data[j1]*weights_1d.data[j2];
         }
      }
   }
};

template <int Q>
class TProductIntegrationRule_base<3, Q>
{
protected:
   TVector<Q> weights_1d;

public:
   void ApplyWeights(TTensor<Q,Q,Q> &qpt_data) const
   {
      for (int j1 = 0; j1 < Q; j1++)
      {
         for (int j2 = 0; j2 < Q; j2++)
         {
            for (int j3 = 0; j3 < Q; j3++)
            {
               qpt_data(j1,j2,j3) *=
                  weights_1d.data[j1]*weights_1d.data[j2]*weights_1d.data[j3];
            }
         }
      }
   }
};

template <int Dim, int Q, int Order>
class TProductIntegrationRule : public TProductIntegrationRule_base<Dim, Q>
{
public:
   static const Geometry::Type geom =
      ((Dim == 1) ? Geometry::SEGMENT :
       ((Dim == 2) ? Geometry::SQUARE : Geometry::CUBE));
   static const int dim = Dim;
   static const int qpts_1d = Q;
   static const int qpts = (Dim == 1) ? Q : ((Dim == 2) ? (Q*Q) : (Q*Q*Q));
   static const int order = Order;

   static const bool tensor_prod = true;

   typedef TCellData<dim, qpts_1d> qpt_data_type;

protected:
   using TProductIntegrationRule_base<Dim, Q>::weights_1d;

public:
   TProductIntegrationRule() { }

   TProductIntegrationRule(const TProductIntegrationRule &ir)
   {
      weights_1d.Set(ir.weights_1d);
      // std::cout << '\n' << _MFEM_FUNC_NAME << std::endl;
   }
};

template <int Dim, int Q>
class GaussIntegrationRule
   : public TProductIntegrationRule<Dim, Q, 2*Q-1>
{
public:
   typedef TProductIntegrationRule<Dim, Q, 2*Q-1> base_class;

   using base_class::geom;
   using base_class::order;
   using base_class::qpts_1d;

protected:
   using base_class::weights_1d;

public:
   GaussIntegrationRule()
   {
      const IntegrationRule &ir_1d = Get1DIntRule();
      MFEM_ASSERT(ir_1d.GetNPoints() == qpts_1d, "quadrature rule mismatch");
      for (int j = 0; j < qpts_1d; j++)
      {
         weights_1d.data[j] = ir_1d.IntPoint(j).weight;
      }
   }

   static const IntegrationRule &Get1DIntRule()
   {
      return IntRules.Get(Geometry::SEGMENT, order);
   }
   static const IntegrationRule &GetIntRule()
   {
      return IntRules.Get(geom, order);
   }
};

template <Geometry::Type G, int Order>
class TIntegrationRule;

template <int Order>
class TIntegrationRule<Geometry::SEGMENT, Order>
   : public GaussIntegrationRule<1, Order/2+1> { };

template <int Order>
class TIntegrationRule<Geometry::SQUARE, Order>
   : public GaussIntegrationRule<2, Order/2+1> { };

template <int Order>
class TIntegrationRule<Geometry::CUBE, Order>
   : public GaussIntegrationRule<3, Order/2+1> { };

// Triangle integration rules (based on intrules.cpp)
// These specializations define the number of quadrature points for each rule
// as a compile-time constant.
// TODO: add higher order rules
template <> class TIntegrationRule<Geometry::TRIANGLE, 0>
   : public GenericIntegrationRule<Geometry::TRIANGLE, 1, 0> { };
template <> class TIntegrationRule<Geometry::TRIANGLE, 1>
   : public GenericIntegrationRule<Geometry::TRIANGLE, 1, 1> { };
template <> class TIntegrationRule<Geometry::TRIANGLE, 2>
   : public GenericIntegrationRule<Geometry::TRIANGLE, 3, 2> { };
template <> class TIntegrationRule<Geometry::TRIANGLE, 3>
   : public GenericIntegrationRule<Geometry::TRIANGLE, 4, 3> { };
template <> class TIntegrationRule<Geometry::TRIANGLE, 4>
   : public GenericIntegrationRule<Geometry::TRIANGLE, 6, 4> { };
template <> class TIntegrationRule<Geometry::TRIANGLE, 5>
   : public GenericIntegrationRule<Geometry::TRIANGLE, 7, 5> { };
template <> class TIntegrationRule<Geometry::TRIANGLE, 6>
   : public GenericIntegrationRule<Geometry::TRIANGLE, 12, 6> { };
template <> class TIntegrationRule<Geometry::TRIANGLE, 7>
   : public GenericIntegrationRule<Geometry::TRIANGLE, 12, 7> { };

// Tetrahedron integration rules (based on intrules.cpp)
// These specializations define the number of quadrature points for each rule
// as a compile-time constant.
// TODO: add higher order rules
template <> class TIntegrationRule<Geometry::TETRAHEDRON, 0>
   : public GenericIntegrationRule<Geometry::TETRAHEDRON, 1, 0> { };
template <> class TIntegrationRule<Geometry::TETRAHEDRON, 1>
   : public GenericIntegrationRule<Geometry::TETRAHEDRON, 1, 1> { };
template <> class TIntegrationRule<Geometry::TETRAHEDRON, 2>
   : public GenericIntegrationRule<Geometry::TETRAHEDRON, 4, 2> { };
template <> class TIntegrationRule<Geometry::TETRAHEDRON, 3>
   : public GenericIntegrationRule<Geometry::TETRAHEDRON, 5, 3> { };
template <> class TIntegrationRule<Geometry::TETRAHEDRON, 4>
   : public GenericIntegrationRule<Geometry::TETRAHEDRON, 11, 4> { };
template <> class TIntegrationRule<Geometry::TETRAHEDRON, 5>
   : public GenericIntegrationRule<Geometry::TETRAHEDRON, 14, 5> { };
template <> class TIntegrationRule<Geometry::TETRAHEDRON, 6>
   : public GenericIntegrationRule<Geometry::TETRAHEDRON, 24, 6> { };
template <> class TIntegrationRule<Geometry::TETRAHEDRON, 7>
   : public GenericIntegrationRule<Geometry::TETRAHEDRON, 31, 7> { };


// Shape evaluators

template <class FE, class IR, bool TP>
class ShapeEvaluator_base;

template <class FE, class IR>
class ShapeEvaluator_base<FE, IR, false>
{
public:
   static const int DOF = FE::dofs;
   static const int NIP = IR::qpts;
   static const int DIM = FE::dim;

protected:
   TMatrix<NIP, DOF> B;
   TTensor<NIP, DIM, DOF> G;

public:
   typedef typename FE::dof_data_type dof_data_type;
   typedef typename IR::qpt_data_type qpt_data_type;
   typedef TMatrix<NIP,DIM> grad_qpt_data_type;
   typedef TMatrix<DOF,DOF> asm_data_type;

   ShapeEvaluator_base(const FE &fe)
   {
      fe.CalcShapes(IR::GetIntRule(), B.data, G.data);
      // std::cout << '\n' << _MFEM_FUNC_NAME << std::endl;
   }

   ShapeEvaluator_base(const ShapeEvaluator_base &se)
   {
      B.Set(se.B);
      G.Set(se.G);
      // std::cout << '\n' << _MFEM_FUNC_NAME << std::endl;
   }

   void Calc(const dof_data_type &dof_data,
             qpt_data_type       &qpt_data) const
   {
      B.template Mult<false>(dof_data, qpt_data);
   }

   template <bool Add>
   void CalcT(const qpt_data_type &qpt_data,
              dof_data_type       &dof_data) const
   {
      B.template MultTranspose<Add>(qpt_data, dof_data);
   }

   void CalcGrad(const dof_data_type &dof_data,
                 grad_qpt_data_type &grad_qpt_data) const
   {
      // grad_dof_data(nip,dim) = \sum_{dof} G(nip,dim,dof) x dof_data[dof]
      Reshape<NIP*DIM,DOF>(G).template Mult<false>(dof_data, grad_qpt_data);
   }

   template <bool Add>
   void CalcGradT(const grad_qpt_data_type &grad_qpt_data,
                  dof_data_type &dof_data) const
   {
      // dof_data[dof] = \sum_{nip,dim} G(nip,dim,dof) x grad_qpt_data(nip,dim)
      Reshape<NIP*DIM,DOF>(G).template MultTranspose<Add>(
         grad_qpt_data, dof_data);
   }

   void Assemble(const qpt_data_type &qpt_data, asm_data_type &M) const
   {
      // M = B^t . diag(qpt_data) . B
      TensorAssemble<false>(B, Reshape<1,NIP,1>(qpt_data),
                            Reshape<1,DOF,1,DOF>(M));
   }
};

template <int Dim, int DOF, int NIP>
class TProductShapeEvaluator;

template <int DOF, int NIP>
class TProductShapeEvaluator<1, DOF, NIP>
{
protected:
   static const int TDOF = DOF; // total dofs

   TMatrix<NIP, DOF> B_1d, G_1d;

public:
   typedef TVector<NIP> grad_qpt_data_type;

   TProductShapeEvaluator()
   {
      // std::cout << '\n' << _MFEM_FUNC_NAME << std::endl;
   }

   void Calc(const TCellData<1,DOF> &dof_data,
             TCellData<1,NIP>       &qpt_data) const
   {
      B_1d.template Mult<false>(dof_data, qpt_data);
   }

   template <bool Add>
   void CalcT(const TCellData<1,NIP> &qpt_data,
              TCellData<1,DOF>       &dof_data) const
   {
      B_1d.template MultTranspose<Add>(qpt_data, dof_data);
   }

   void CalcGrad(const TCellData<1,DOF> &dof_data,
                 grad_qpt_data_type &grad_qpt_data) const
   {
      G_1d.template Mult<false>(dof_data, grad_qpt_data);
   }

   template <bool Add>
   void CalcGradT(const grad_qpt_data_type &grad_qpt_data,
                  TCellData<1,DOF> &dof_data) const
   {
      G_1d.template MultTranspose<Add>(grad_qpt_data, dof_data);
   }

   void Assemble(const TCellData<1,NIP> &qpt_data, TMatrix<TDOF,TDOF> &M) const
   {
      // M = B^t . diag(qpt_data) . B
      TensorAssemble<false>(B_1d, Reshape<1,NIP,1>(qpt_data),
                            Reshape<1,DOF,1,DOF>(M));
   }
};

template <int DOF, int NIP>
class TProductShapeEvaluator<2, DOF, NIP>
{
protected:
   static const int TDOF = DOF*DOF; // total dofs

   TMatrix<NIP, DOF> B_1d, G_1d;

public:
   typedef TTensor<NIP,NIP,2> grad_qpt_data_type;

   TProductShapeEvaluator()
   {
      // std::cout << '\n' << _MFEM_FUNC_NAME << std::endl;
   }

   template <bool Dx, bool Dy>
   void Calc(const TMatrix<DOF,DOF> &dof_data,
             TMatrix<NIP,NIP>       &qpt_data) const
   {
      TMatrix<NIP, DOF> A;

      Mult_AB<false>(Dx ? G_1d : B_1d, dof_data, A);
      Mult_ABt<false>(A, Dy ? G_1d : B_1d, qpt_data);
   }
   void Calc(const TCellData<2,DOF> &dof_data,
             TCellData<2,NIP>       &qpt_data) const
   {
      Calc<false,false>(dof_data, qpt_data);
   }

   template <bool Dx, bool Dy, bool Add>
   void CalcT(const TMatrix<NIP,NIP> &qpt_data,
              TMatrix<DOF,DOF>       &dof_data) const
   {
      TMatrix<NIP, DOF> A;

      Mult_AB<false>(qpt_data, Dy ? G_1d : B_1d, A);
      Mult_AtB<Add>(Dx ? G_1d : B_1d, A, dof_data);
   }
   template <bool Add>
   void CalcT(const TCellData<2,NIP> &qpt_data,
              TCellData<2,DOF>       &dof_data) const
   {
      CalcT<false,false,Add>(qpt_data, dof_data);
   }

   void CalcGrad(const TCellData<2,DOF> &dof_data,
                 grad_qpt_data_type     &grad_qpt_data) const
   {
      Calc<true,false>(dof_data, grad_qpt_data(0));
      Calc<false,true>(dof_data, grad_qpt_data(1));
   }

   template <bool Add>
   void CalcGradT(const grad_qpt_data_type &grad_qpt_data,
                  TCellData<2,DOF>         &dof_data) const
   {
      CalcT<true,false, Add>(grad_qpt_data(0), dof_data);
      CalcT<false,true,true>(grad_qpt_data(1), dof_data);
   }

   void Assemble(const TCellData<2,NIP> &qpt_data, TMatrix<TDOF,TDOF> &M) const
   {
      TTensor<DOF,NIP,DOF> A;

      TensorAssemble<false>(Reshape<NIP,DOF>(B_1d),
                            Reshape<1,NIP,NIP>(qpt_data),
                            Reshape<1,DOF,NIP,DOF>(A));
      TensorAssemble<false>(Reshape<NIP,DOF>(B_1d),
                            Reshape<DOF,NIP,DOF>(A),
                            Reshape<DOF,DOF,DOF,DOF>(M));
#if 0
      // Clearer implementation with different B1_1d and B2_1d:
      TTensor<DOF1,NIP2,DOF1> A;
      TensorAssemble<false>(Reshape<NIP1,DOF1>(B1_1d),
                            Reshape<1,NIP1,NIP2>(qpt_data),
                            Reshape<1,DOF1,NIP2,DOF1>(A));
      TensorAssemble<false>(Reshape<NIP2,DOF2>(B2_1d),
                            Reshape<DOF1,NIP2,DOF1>(A),
                            Reshape<DOF1,DOF2,DOF1,DOF2>(M));
#endif
   }
};

template <int DOF, int NIP>
class TProductShapeEvaluator<3, DOF, NIP>
{
protected:
   static const int TDOF = DOF*DOF*DOF; // total dofs

   TMatrix<NIP, DOF> B_1d, G_1d;

public:
   typedef TTensor4<NIP,NIP,NIP,3> grad_qpt_data_type;

   TProductShapeEvaluator()
   {
      // std::cout << '\n' << _MFEM_FUNC_NAME << std::endl;
   }

   template <bool Dx, bool Dy, bool Dz>
   void Calc(const TTensor<DOF,DOF,DOF> &dof_data,
             TTensor<NIP,NIP,NIP>       &qpt_data) const
   {
      TTensor<NIP,DOF,DOF> QDD;
      TTensor<NIP,NIP,DOF> QQD;

      Mult_2_1<false>(Dx ? G_1d : B_1d, dof_data, QDD);
      Mult_2_2<false>(Dy ? G_1d : B_1d, QDD, QQD);
      Mult_2_3<false>(Dz ? G_1d : B_1d, QQD, qpt_data);
   }
   void Calc(const TCellData<3,DOF> &dof_data,
             TCellData<3,NIP>       &qpt_data) const
   {
      Calc<false,false,false>(dof_data, qpt_data);
   }

   template <bool Dx, bool Dy, bool Dz, bool Add>
   void CalcT(const TTensor<NIP,NIP,NIP> &qpt_data,
              TTensor<DOF,DOF,DOF>       &dof_data) const
   {
      TTensor<NIP,DOF,DOF> QDD;
      TTensor<NIP,NIP,DOF> QQD;

      Mult_1_3<false>(Dz ? G_1d : B_1d, qpt_data, QQD);
      Mult_1_2<false>(Dy ? G_1d : B_1d, QQD, QDD);
      Mult_1_1<Add>  (Dx ? G_1d : B_1d, QDD, dof_data);
   }
   template <bool Add>
   void CalcT(const TCellData<3,NIP> &qpt_data,
              TCellData<3,DOF>       &dof_data) const
   {
      CalcT<false,false,false,Add>(qpt_data, dof_data);
   }

   void CalcGrad(const TCellData<3,DOF> &dof_data,
                 grad_qpt_data_type     &grad_qpt_data) const
   {
      Calc<true,false,false>(dof_data, grad_qpt_data(0));
      Calc<false,true,false>(dof_data, grad_qpt_data(1));
      Calc<false,false,true>(dof_data, grad_qpt_data(2));
      // optimization: the x-transition (dof->nip) is done twice -- once for the
      // y-derivatives and second time for the z-derivatives.
   }

   template <bool Add>
   void CalcGradT(const grad_qpt_data_type &grad_qpt_data,
                  TCellData<3,DOF>         &dof_data) const
   {
      CalcT<true,false,false, Add>(grad_qpt_data(0), dof_data);
      CalcT<false,true,false,true>(grad_qpt_data(1), dof_data);
      CalcT<false,false,true,true>(grad_qpt_data(2), dof_data);
   }

   void Assemble(const TCellData<3,NIP> &qpt_data, TMatrix<TDOF,TDOF> &M) const
   {
      TTensor<DOF,NIP*NIP,DOF> A1;
      TTensor4<DOF,DOF,NIP,DOF*DOF> A2;

      TensorAssemble<false>(Reshape<NIP,DOF>(B_1d),
                            Reshape<1,NIP,NIP*NIP>(qpt_data),
                            Reshape<1,DOF,NIP*NIP,DOF>(A1));
      TensorAssemble<false>(Reshape<NIP,DOF>(B_1d),
                            Reshape<DOF,NIP,NIP*DOF>(A1),
                            Reshape<DOF,DOF,NIP*DOF,DOF>(A2));
      TensorAssemble<false>(Reshape<NIP,DOF>(B_1d),
                            Reshape<DOF*DOF,NIP,DOF*DOF>(A2),
                            Reshape<DOF*DOF,DOF,DOF*DOF,DOF>(M));
#if 0
      // Clearer implementation with different B1_1d, B2_1d and B3_1d:
      TTensor<DOF1,NIP2*NIP3,DOF1> A1;
      TTensor4<DOF1,DOF2,NIP3,DOF1*DOF2> A2;
      TensorAssemble<false>(Reshape<NIP1,DOF1>(B1_1d),
                            Reshape<1,NIP1,NIP2*NIP3>(qpt_data),
                            Reshape<1,DOF1,NIP2*NIP3,DOF1>(A1));
      TensorAssemble<false>(Reshape<NIP2,DOF2>(B2_1d),
                            Reshape<DOF1,NIP2,NIP3*DOF1>(A1),
                            Reshape<DOF1,DOF2,NIP3*DOF1,DOF2>(A2));
      TensorAssemble<false>(Reshape<NIP3,DOF3>(B3_1d),
                            Reshape<DOF1*DOF2,NIP3,DOF1*DOF2>(A2),
                            Reshape<DOF1*DOF2,DOF3,DOF1*DOF2,DOF3>(M));
#endif
   }
};

template <class FE, class IR>
class ShapeEvaluator_base<FE, IR, true>
   : public TProductShapeEvaluator<FE::dim, FE::dofs_1d, IR::qpts_1d>
{
protected:
   using TProductShapeEvaluator<FE::dim, FE::dofs_1d, IR::qpts_1d>::B_1d;
   using TProductShapeEvaluator<FE::dim, FE::dofs_1d, IR::qpts_1d>::G_1d;
   using TProductShapeEvaluator<FE::dim, FE::dofs_1d, IR::qpts_1d>::TDOF;

public:
   typedef typename FE::dof_data_type dof_data_type;
   typedef typename IR::qpt_data_type qpt_data_type;
   typedef TMatrix<TDOF, TDOF> asm_data_type;

   ShapeEvaluator_base(const FE &fe)
   {
      fe.Calc1DShapes(IR::Get1DIntRule(), B_1d.data, G_1d.data);
      // std::cout << '\n' << _MFEM_FUNC_NAME << std::endl;
   }

   ShapeEvaluator_base(const ShapeEvaluator_base &se)
   {
      B_1d.Set(se.B_1d);
      G_1d.Set(se.G_1d);
      // std::cout << '\n' << _MFEM_FUNC_NAME << std::endl;
   }
};

template <class FE, class IR>
class ShapeEvaluator
   : public ShapeEvaluator_base<FE, IR, FE::tensor_prod && IR::tensor_prod>
{
public:
   static const int dim  = FE::dim;
   static const int qpts = IR::qpts;
   static const bool tensor_prod = FE::tensor_prod && IR::tensor_prod;
   typedef FE FE_type;
   typedef IR IR_type;
   typedef ShapeEvaluator_base<FE, IR, tensor_prod> base_class;

   using typename base_class::dof_data_type;
   using typename base_class::qpt_data_type;
   using typename base_class::grad_qpt_data_type;
   using base_class::Calc;
   using base_class::CalcT;
   using base_class::CalcGrad;
   using base_class::CalcGradT;

   ShapeEvaluator(const FE &fe) : base_class(fe)
   {
      // std::cout << '\n' << _MFEM_FUNC_NAME << std::endl;
   }

   ShapeEvaluator(const ShapeEvaluator &se) : base_class(se)
   {
      // std::cout << '\n' << _MFEM_FUNC_NAME << std::endl;
   }

   template <int NumComp>
   void CalcVec(const dof_data_type (&vdof_data)[NumComp],
                qpt_data_type (&vqpt_data)[NumComp]) const
   {
      for (int i = 0; i < NumComp; i++)
      {
         Calc(vdof_data[i], vqpt_data[i]);
      }
   }

   template <bool Add, int NumComp>
   void CalcVecT(const qpt_data_type (&vqpt_data)[NumComp],
                 dof_data_type (&vdof_data)[NumComp]) const
   {
      for (int i = 0; i < NumComp; i++)
      {
         CalcT<Add>(vqpt_data[i], vdof_data[i]);
      }
   }

   template <int NumComp>
   void CalcVecGrad(const dof_data_type (&vdof_data)[NumComp],
                    grad_qpt_data_type (&vgrad_qpt_data)[NumComp]) const
   {
      for (int i = 0; i < NumComp; i++)
      {
         CalcGrad(vdof_data[i], vgrad_qpt_data[i]);
      }
   }

   template <bool Add, int NumComp>
   void CalcVecGradT(const grad_qpt_data_type (&vgrad_qpt_data)[NumComp],
                     dof_data_type (&vdof_data)[NumComp]) const
   {
      for (int i = 0; i < NumComp; i++)
      {
         CalcGradT<Add>(vgrad_qpt_data[i], vdof_data[i]);
      }
   }

   void GetPointGrad(int qpt_idx, const grad_qpt_data_type &grad_qpt_data,
                     TVector<dim> &grad) const
   {
      for (int i = 0; i < dim; i++)
      {
         grad.data[i] = grad_qpt_data.data[qpt_idx+qpts*i];
      }
   }

   template <int NumComp>
   void GetPointVec(int qpt_idx,
                    const qpt_data_type (&vqpt_data)[NumComp],
                    TVector<NumComp> &vec) const
   {
      for (int comp = 0; comp < NumComp; comp++)
      {
         vec.data[comp] = vqpt_data[comp].data[qpt_idx+qpts*comp];
      }
   }

   template <int NumComp>
   void GetPointVecGrad(int qpt_idx,
                        const grad_qpt_data_type (&vgrad_qpt_data)[NumComp],
                        TMatrix<NumComp,dim> &vgrad) const
   {
      for (int comp = 0; comp < NumComp; comp++)
      {
         for (int der = 0; der < dim; der++)
         {
            vgrad(comp,der) = vgrad_qpt_data[comp].data[qpt_idx+qpts*der];
         }
      }
   }
};


// Element-Dof-Operators

class IndexVectorizer
{
protected:
   Ordering::Type ordering;
   int num_comp, scalar_size;

public:
   IndexVectorizer(Ordering::Type ordering_, int num_comp_,
                   int scalar_size_)
      : ordering(ordering_),
        num_comp(num_comp_),
        scalar_size(scalar_size_)
   { }

   IndexVectorizer(const FiniteElementSpace &fes)
      : ordering(fes.GetOrdering()),
        num_comp(fes.GetVDim()),
        scalar_size(fes.GetNDofs())
   { }

   int NumComponents() const { return num_comp; }

   int VectorIndex(int scalar_idx, int comp_idx) const
   {
      if (ordering == Ordering::byNODES)
      {
         return scalar_idx + comp_idx * scalar_size;
      }
      else
      {
         return comp_idx + num_comp * scalar_idx;
      }
   }
};

template <Ordering::Type Ord>
class TIndexVectorizer_Ord
{
protected:
   int num_comp, scalar_size;

public:
   TIndexVectorizer_Ord(int num_comp_, int scalar_size_)
      : num_comp(num_comp_),
        scalar_size(scalar_size_)
   { }

   TIndexVectorizer_Ord(const FiniteElementSpace &fes)
      : num_comp(fes.GetVDim()),
        scalar_size(fes.GetNDofs())
   {
      MFEM_ASSERT(fes.GetOrdering() == Ord, "ordering mismatch");
   }

   int NumComponents() const { return num_comp; }

   int VectorIndex(int scalar_idx, int comp_idx) const
   {
      if (Ord == Ordering::byNODES)
      {
         return scalar_idx + comp_idx * scalar_size;
      }
      else
      {
         return comp_idx + num_comp * scalar_idx;
      }
   }
};

template <Ordering::Type Ord, int NumComp>
class TIndexVectorizer
{
protected:
   int scalar_size;

public:
   explicit TIndexVectorizer(int scalar_size_)
      : scalar_size(scalar_size_) { }

   TIndexVectorizer(const FiniteElementSpace &fes)
      : scalar_size(fes.GetNDofs())
   {
      MFEM_ASSERT(fes.GetOrdering() == Ord, "ordering mismatch");
      MFEM_ASSERT(fes.GetVDim() == NumComp, "vdim mismatch");
   }

   int NumComponents() const { return NumComp; }

   int VectorIndex(int scalar_idx, int comp_idx) const
   {
      if (Ord == Ordering::byNODES)
      {
         return scalar_idx + comp_idx * scalar_size;
      }
      else
      {
         return comp_idx + NumComp * scalar_idx;
      }
   }
};

template <typename FE>
class Table_ElementDofOperator
{
protected:
   const int *el_dof_list, *loc_dof_list;
   bool own_list;

public:
   typedef typename FE::dof_data_type dof_data_type;

   Table_ElementDofOperator(const FE &fe, const FiniteElementSpace &fes)
   {
      const Array<int> *loc_dof_map = fe.GetDofMap();
      fes.BuildElementToDofTable();
      const Table &el_dof = fes.GetElementToDofTable();
      MFEM_ASSERT(el_dof.Size_of_connections() == el_dof.Size() * FE::dofs,
                  "the element-to-dof Table is not compatible with this FE!");
      int num_dofs = el_dof.Size() * FE::dofs;
      if (!loc_dof_map)
      {
         // no local dof reordering
         el_dof_list = el_dof.GetJ();
         own_list = false;
      }
      else
      {
         // reorder the local dofs according to loc_dof_map
         int *el_dof_list_ = new int[num_dofs];
         const int *loc_dof_map_ = loc_dof_map->GetData();
         for (int i = 0; i < el_dof.Size(); i++)
         {
            MFEM_ASSERT(el_dof.RowSize(i) == FE::dofs,
                        "incompatible element-to-dof Table!");
            for (int j = 0; j < FE::dofs; j++)
            {
               el_dof_list_[j+FE::dofs*i] =
                  el_dof.GetJ()[loc_dof_map_[j]+FE::dofs*i];
            }
         }
         el_dof_list = el_dof_list_;
         own_list = true;
      }
      loc_dof_list = el_dof_list; // point to element 0
   }

   // Shallow copy constructor
   Table_ElementDofOperator(const Table_ElementDofOperator &orig)
      : el_dof_list(orig.el_dof_list),
        loc_dof_list(orig.loc_dof_list),
        own_list(false)
   { }

   ~Table_ElementDofOperator() { if (own_list) { delete el_dof_list; } }

   void SetElement(int elem_idx)
   {
      loc_dof_list = el_dof_list + elem_idx * FE::dofs;
   }

   void Extract(const double glob_dof_data[], dof_data_type &dof_data) const
   {
      for (int i = 0; i < FE::dofs; i++)
      {
         dof_data.data[i] = glob_dof_data[loc_dof_list[i]];
      }
   }

   void Assemble(const dof_data_type &dof_data, double glob_dof_data[]) const
   {
      for (int i = 0; i < FE::dofs; i++)
      {
         glob_dof_data[loc_dof_list[i]] += dof_data.data[i];
      }
   }

   template <typename IdxVectorizer>
   void VectorExtract(IdxVectorizer &iv,
                      const double glob_vdof_data[],
                      dof_data_type vdof_data[]) const
   {
      const int nc = iv.NumComponents();
      for (int j = 0; j < nc; j++)
      {
         for (int i = 0; i < FE::dofs; i++)
         {
            vdof_data[j].data[i] =
               glob_vdof_data[iv.VectorIndex(loc_dof_list[i], j)];
         }
      }
   }

   template <typename IdxVectorizer>
   void VectorAssemble(IdxVectorizer &iv,
                       const dof_data_type vdof_data[],
                       double glob_vdof_data[]) const
   {
      const int nc = iv.NumComponents();
      for (int j = 0; j < nc; j++)
      {
         for (int i = 0; i < FE::dofs; i++)
         {
            glob_vdof_data[iv.VectorIndex(loc_dof_list[i], j)] +=
               vdof_data[j].data[i];
         }
      }
   }
};

template <typename FE>
class DG_ElementDofOperator
{
protected:
   int offset;

public:
   typedef typename FE::dof_data_type dof_data_type;

   DG_ElementDofOperator(const FE &fe, const FiniteElementSpace &fes)
   {
      MFEM_ASSERT(fes.GetNDofs() == fes.GetNE() * FE::dofs,
                  "the FE space is not compatible with this FE!");
      offset = 0;
   }

   DG_ElementDofOperator(const DG_ElementDofOperator &orig)
      : offset(orig.offset) { }

   void SetElement(int elem_idx)
   {
      offset = FE::dofs * elem_idx;
   }

   void Extract(const double glob_dof_data[], dof_data_type &dof_data) const
   {
      dof_data.Set(&glob_dof_data[offset]);
   }

   void Assemble(const dof_data_type &dof_data,
                 double glob_dof_data[]) const
   {
      dof_data.Assemble(&glob_dof_data[offset]);
   }

   template <typename IdxVectorizer>
   void VectorExtract(IdxVectorizer &iv,
                      const double glob_vdof_data[],
                      dof_data_type vdof_data[]) const
   {
      const int nc = iv.NumComponents();
      for (int j = 0; j < nc; j++)
      {
         for (int i = 0; i < FE::dofs; i++)
         {
            vdof_data[j].data[i] = glob_vdof_data[iv.VectorIndex(i+offset, j)];
         }
      }
   }

   template <typename IdxVectorizer>
   void VectorAssemble(IdxVectorizer &iv,
                       const dof_data_type vdof_data[],
                       double glob_vdof_data[]) const
   {
      const int nc = iv.NumComponents();
      for (int j = 0; j < nc; j++)
      {
         for (int i = 0; i < FE::dofs; i++)
         {
            glob_vdof_data[iv.VectorIndex(i+offset, j)] += vdof_data[j].data[i];
         }
      }
   }
};


// Mass assembler

template <typename meshFE, typename meshElemDof, typename meshVectorizer,
          typename spaceFE, typename spaceElemDof, typename IR>
class TMassAssembler : public Operator
{
protected:
   typedef ShapeEvaluator<meshFE, IR> meshShapeEval;
   typedef ShapeEvaluator<spaceFE, IR> spaceShapeEval;

   static const int dim = meshFE::dim;
   static const int qpts = IR::qpts;

   meshFE mesh_fe;
   spaceFE space_fe;

   meshShapeEval meshEval;
   spaceShapeEval spaceEval;

   const FiniteElementSpace &meshFES;
   GridFunction &meshNodes;

   mutable meshElemDof meshElDof;
   mutable spaceElemDof spaceElDof;

   meshVectorizer meshVec;

   IR int_rule;

public:
   TMassAssembler(const FiniteElementSpace &spaceFES)
      : Operator(spaceFES.GetNDofs()),
        mesh_fe(), space_fe(),
        meshEval(mesh_fe), spaceEval(space_fe),
        meshFES(*spaceFES.GetMesh()->GetNodalFESpace()),
        meshNodes(*spaceFES.GetMesh()->GetNodes()),
        meshElDof(mesh_fe, meshFES), spaceElDof(space_fe, spaceFES),
        meshVec(meshFES), int_rule()
   { }

   ~TMassAssembler()
   { }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      y = 0.0;

      typename meshFE::dof_data_type nodes_dof[dim];
      typename spaceFE::dof_data_type x_dof, y_dof;

      typename meshShapeEval::grad_qpt_data_type J_qpt[dim];
      typename spaceShapeEval::qpt_data_type x_qpt, y_qpt;

      TMatrix<dim,dim> J;

#if 1
      const int NE = meshFES.GetNE();
      for (int el = 0; el < NE; el++)
      {
         meshElDof.SetElement(el);
         spaceElDof.SetElement(el);

         meshElDof.VectorExtract(meshVec, meshNodes, nodes_dof);
         meshEval.CalcVecGrad(nodes_dof, J_qpt);

         spaceElDof.Extract(x, x_dof);
         spaceEval.Calc(x_dof, x_qpt);

         for (int j = 0; j < qpts; j++)
         {
            meshEval.GetPointVecGrad(j, J_qpt, J);

            y_qpt.data[j] = J.Det() * x_qpt.data[j];
         }

         int_rule.ApplyWeights(y_qpt);

         spaceEval.template CalcT<false>(y_qpt, y_dof);
         spaceElDof.Assemble(y_dof, y);
      }
#else
      // For better performance, create local copies of meshElDof, spaceElDof,
      // meshEval, spaceEval, and int_rule.
      // Is performance actually better with this implementation?
      meshElemDof meshElDof_l(meshElDof);
      spaceElemDof spaceElDof_l(spaceElDof);

      meshShapeEval meshEval_l(meshEval);
      spaceShapeEval spaceEval_l(spaceEval);

      IR int_rule_l(int_rule);

      const int NE = meshFES.GetNE();
      for (int el = 0; el < NE; el++)
      {
         meshElDof_l.SetElement(el);
         spaceElDof_l.SetElement(el);

         meshElDof_l.VectorExtract(meshVec, meshNodes, nodes_dof);
         meshEval_l.CalcVecGrad(nodes_dof, J_qpt);

         spaceElDof_l.Extract(x, x_dof);
         spaceEval_l.Calc(x_dof, x_qpt);

         for (int j = 0; j < qpts; j++)
         {
            meshEval_l.GetPointVecGrad(j, J_qpt, J);

            y_qpt.data[j] = J.Det() * x_qpt.data[j];
         }

         int_rule_l.ApplyWeights(y_qpt);

         spaceEval_l.template CalcT<false>(y_qpt, y_dof);
         spaceElDof_l.Assemble(y_dof, y);
      }
#endif
   }
};


} // namespace mfem

#endif // MFEM_TEMPLATE
