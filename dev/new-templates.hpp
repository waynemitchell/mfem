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

   void Random()
   {
      for (int i = 0; i < size; i++)
      {
         data[i] = std::rand() / (RAND_MAX + 1.0);
      }
   }
};

template <int N1, int N2>
struct TMatrix : public TVector<N1*N2>
{
   using TVector<N1*N2>::size;
   using TVector<N1*N2>::data;

   static const int ind(int i1, int i2) { return (i1+N1*i2); }

   double &operator()(int i, int j) { return data[ind(i,j)]; }
   const double &operator()(int i, int j) const { return data[ind(i,j)]; }

   // operator()(int) returns a column of the matrix
   TVector<N1> &operator()(int i2)
   { return (TVector<N1> &)(operator()(0,i2)); }
   const TVector<N1> &operator()(int i2) const
   { return (const TVector<N1> &)(operator()(0,i2)); }

   template <bool Add>
   void Mult(const TVector<N2> &x, TVector<N1> &y) const
   {
      if (!Add) { y.Set(0.0); }
      for (int i1 = 0; i1 < N1; i1++)
      {
         for (int i2 = 0; i2 < N2; i2++)
         {
            y[i1] += operator()(i1,i2) * x[i2];
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
            y[i2] += operator()(i1,i2) * x[i1];
         }
      }
   }
};

template <int N1, int N2, int N3>
struct TTensor : TVector<N1*N2*N3>
{
   using TVector<N1*N2*N3>::size;
   using TVector<N1*N2*N3>::data;

   static const int ind(int i1, int i2, int i3) { return (i1+N1*(i2+N2*i3)); }

   double &operator()(int i, int j, int k) { return data[ind(i,j,k)]; }
   const double &operator()(int i, int j, int k) const
   { return data[ind(i,j,k)]; }

   // operator()(int) returns a sub-matrix of the tensor by fixing the given
   // last index
   TMatrix<N1,N2> &operator()(int i3)
   { return (TMatrix<N1,N2> &)(operator()(0,0,i3)); }
   const TMatrix<N1,N2> &operator()(int i3) const
   { return (const TMatrix<N1,N2> &)(operator()(0,0,i3)); }
};

template <int N1, int N2, int N3, int N4>
struct TTensor4 : TVector<N1*N2*N3*N4>
{
   using TVector<N1*N2*N3*N4>::size;
   using TVector<N1*N2*N3*N4>::data;

   static const int ind(int i1, int i2, int i3, int i4)
   { return (i1+N1*(i2+N2*(i3+N3*i4))); }

   double &operator()(int i, int j, int k, int l)
   { return data[ind(i,j,k,l)]; }
   const double &operator()(int i, int j, int k, int l) const
   { return data[ind(i,j,k,l)]; }

   // operator()(int) returns a rank-3 sub-tensor of the tensor by fixing the
   // given last index
   TTensor<N1,N2,N3> &operator()(int i4)
   { return (TTensor<N1,N2,N3> &)(operator()(0,0,0,i4)); }
   const TTensor<N1,N2,N3> &operator()(int i4) const
   { return (const TTensor<N1,N2,N3> &)(operator()(0,0,0,i4)); }
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
TMatrix<N1,N2> &Reshape(TVector<N1*N2> &tensor)
{
   return (TMatrix<N1,N2> &)tensor;
}

template <int N1, int N2>
const TMatrix<N1,N2> &Reshape(const TVector<N1*N2> &tensor)
{
   return (const TMatrix<N1,N2> &)tensor;
}

// Reshape to (view as) TTensor
template <int N1, int N2, int N3>
TTensor<N1,N2,N3> &Reshape(TVector<N1*N2*N3> &tensor)
{
   return (TTensor<N1,N2,N3> &)tensor;
}

template <int N1, int N2, int N3>
const TTensor<N1,N2,N3> &Reshape(const TVector<N1*N2*N3> &tensor)
{
   return (const TTensor<N1,N2,N3> &)tensor;
}

// Reshape to (view as) TTensor4
template <int N1, int N2, int N3, int N4>
TTensor4<N1,N2,N3,N4> &Reshape(TVector<N1*N2*N3*N4> &tensor)
{
   return (TTensor4<N1,N2,N3,N4> &)tensor;
}

template <int N1, int N2, int N3, int N4>
const TTensor4<N1,N2,N3,N4> &Reshape(const TVector<N1*N2*N3*N4> &tensor)
{
   return (const TTensor4<N1,N2,N3,N4> &)tensor;
}


// Tensor multiplication

// C_{i,j,k,l}  {=|+=}  \sum_s A_{i,s,j} B_{k,s,l}
// The string '1234' in the name indicates the order of the ijkl indices
// in the output tensor C.
template <bool Add, int A1, int A2, int A3, int B1, int B3>
void Mult_1234(const TTensor<A1,A2,A3> &A, const TTensor<B1,A2,B3> &B,
               TTensor4<A1,A3,B1,B3> &C)
{
   if (!Add) { C.Set(0.0); }
   for (int l = 0; l < B3; l++)
   {
      for (int k = 0; k < B1; k++)
      {
         for (int j = 0; j < A3; j++)
         {
            for (int i = 0; i < A1; i++)
            {
               for (int s = 0; s < A2; s++)
               {
                  C(i,j,k,l) += A(i,s,j) * B(k,s,l);
               }
            }
         }
      }
   }
}

// C_{i,k,l,j}  {=|+=}  \sum_s A_{i,s,j} B_{k,s,l}
// The string '1342' in the name indicates the order of the ijkl indices
// in the output tensor C.
template <bool Add, int A1, int A2, int A3, int B1, int B3>
void Mult_1342(const TTensor<A1,A2,A3> &A, const TTensor<B1,A2,B3> &B,
               TTensor4<A1,B1,B3,A3> &C)
{
   if (!Add) { C.Set(0.0); }
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
                  C(i,k,l,j) += A(i,s,j) * B(k,s,l);
               }
            }
         }
      }
   }
}

// C  {=|+=}  A.B
template <bool Add, int A1, int A2, int B2>
void Mult_AB(const TMatrix<A1,A2> &A, const TMatrix<A2,B2> &B,
             TMatrix<A1,B2> &C)
{
   Mult_1234<Add>(Reshape<A1,A2,1>(A),
                  Reshape<1,A2,B2>(B),
                  Reshape<A1,1,1,B2>(C));
}

// C  {=|+=}  At.B
template <bool Add, int A1, int A2, int B2>
void Mult_AtB(const TMatrix<A1,A2> &A, const TMatrix<A1,B2> &B,
              TMatrix<A2,B2> &C)
{
   Mult_1234<Add>(Reshape<1,A1,A2>(A),
                  Reshape<1,A1,B2>(B),
                  Reshape<1,A2,1,B2>(C));
}

// C  {=|+=}  A.Bt
template <bool Add, int A1, int A2, int B1>
void Mult_ABt(const TMatrix<A1,A2> &A, const TMatrix<B1,A2> &B,
              TMatrix<A1,B1> &C)
{
   Mult_1234<Add>(Reshape<A1,A2,1>(A),
                  Reshape<B1,A2,1>(B),
                  Reshape<A1,1,B1,1>(C));
}

// C  {=|+=}  At.Bt
template <bool Add, int A1, int A2, int B1>
void Mult_AtBt(const TMatrix<A1,A2> &A, const TMatrix<B1,A1> &B,
               TMatrix<A2,B1> &C)
{
   Mult_1234<Add>(Reshape<1,A1,A2>(A),
                  Reshape<B1,A1,1>(B),
                  Reshape<1,A2,B1,1>(C));
}

// C_{i,j,k}  {=|+=}  \sum_s A_{s,i} B_{s,j,k}
template <bool Add, int A1, int A2, int B2, int B3>
void Mult_1_1(const TMatrix<A1,A2> &A, const TTensor<A1,B2,B3> &B,
              TTensor<A2,B2,B3> &C)
{
   Mult_AtB<Add>(Reshape<A1,A2>(A),
                 Reshape<A1,B2*B3>(B),
                 Reshape<A2,B2*B3>(C));
}

// C_{i,j,k}  {=|+=}  \sum_s A_{s,j} B_{i,s,k}
template <bool Add, int A1, int A2, int B1, int B3>
void Mult_1_2(const TMatrix<A1,A2> &A, const TTensor<B1,A1,B3> &B,
              TTensor<B1,A2,B3> &C)
{
   Mult_1342<Add>(Reshape<B1,A1,B3>(B),
                  Reshape<1,A1,A2>(A),
                  Reshape<B1,1,A2,B3>(C));
}

// C_{i,j,k}  {=|+=}  \sum_s A_{s,k} B_{i,j,s}
template <bool Add, int A1, int A2, int B1, int B2>
void Mult_1_3(const TMatrix<A1,A2> &A, const TTensor<B1,B2,A1> &B,
              TTensor<B1,B2,A2> &C)
{
   Mult_AB<Add>(Reshape<B1*B2,A1>(B),
                Reshape<A1,A2>(A),
                Reshape<B1*B2,A2>(C));
}

// C_{i,j,k}  {=|+=}  \sum_s A_{i,s} B_{s,j,k}
template <bool Add, int A1, int A2, int B2, int B3>
void Mult_2_1(const TMatrix<A1,A2> &A, const TTensor<A2,B2,B3> &B,
              TTensor<A1,B2,B3> &C)
{
   Mult_AB<Add>(Reshape<A1,A2>(A),
                Reshape<A2,B2*B3>(B),
                Reshape<A1,B2*B3>(C));
}

// C_{i,j,k}  {=|+=}  \sum_s A_{j,s} B_{i,s,k}
template <bool Add, int A1, int A2, int B1, int B3>
void Mult_2_2(const TMatrix<A1,A2> &A, const TTensor<B1,A2,B3> &B,
              TTensor<B1,A1,B3> &C)
{
   Mult_1342<Add>(Reshape<B1,A2,B3>(B),
                  Reshape<A1,A2,1>(A),
                  Reshape<B1,A1,1,B3>(C));
}

// C_{i,j,k}  {=|+=}  \sum_s A_{k,s} B_{i,j,s}
template <bool Add, int A1, int A2, int B1, int B2>
void Mult_2_3(const TMatrix<A1,A2> &A, const TTensor<B1,B2,A2> &B,
              TTensor<B1,B2,A1> &C)
{
   Mult_ABt<Add>(Reshape<B1*B2,A2>(B),
                 Reshape<A1,A2>(A),
                 Reshape<B1*B2,A1>(C));
}

// C_{k,i,l,j}  {=|+=}  A_{s,i} A_{s,j} B_{k,s,l}
template <bool Add, int A1, int A2, int B1, int B3>
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

template <Geometry::Type G, int P>
class H1_FiniteElement;

template <int P>
class H1_FiniteElement<Geometry::SEGMENT, P> : public H1_SegmentElement
{
public:
   static const int geom    = Geometry::SEGMENT;
   static const int dim     = 1;
   static const int degree  = P;
   static const int dofs_1d = P+1;
   static const int dofs    = P+1;

   static const bool tensor_prod = true;

   H1_FiniteElement() : H1_SegmentElement(P) { }

   void CalcShapeMatrix(const IntegrationRule &ir, double *B) const
   { Calc1DShapeMatrix(ir, B); }
   void CalcGradTensor(const IntegrationRule &ir, double *G) const
   { Calc1DGradMatrix(ir, G); }

   void Calc1DShapeMatrix(const IntegrationRule &ir_1d, double *B) const
   {
      MFEM_WARNING("TODO");
   }
   void Calc1DGradMatrix(const IntegrationRule &ir_1d, double *G) const
   {
      MFEM_WARNING("TODO");
   }
};

template <int P>
class H1_FiniteElement<Geometry::SQUARE, P> : public H1_QuadrilateralElement
{
public:
   static const int geom    = Geometry::SQUARE;
   static const int dim     = 2;
   static const int degree  = P;
   static const int dofs_1d = P+1;
   static const int dofs    = (P+1)*(P+1);

   static const bool tensor_prod = true;

   H1_FiniteElement() : H1_QuadrilateralElement(P) { }

   void CalcShapeMatrix(const IntegrationRule &ir, double *B) const
   {
      MFEM_WARNING("TODO");
   }
   void CalcGradTensor(const IntegrationRule &ir, double *G) const
   {
      MFEM_WARNING("TODO");
   }

   void Calc1DShapeMatrix(const IntegrationRule &ir_1d, double *B) const
   {
      MFEM_WARNING("TODO");
   }
   void Calc1DGradMatrix(const IntegrationRule &ir_1d, double *G) const
   {
      MFEM_WARNING("TODO");
   }
};

template <int P>
class H1_FiniteElement<Geometry::CUBE, P> : public H1_HexahedronElement
{
public:
   static const int geom    = Geometry::CUBE;
   static const int dim     = 3;
   static const int degree  = P;
   static const int dofs_1d = P+1;
   static const int dofs    = (P+1)*(P+1)*(P+1);

   static const bool tensor_prod = true;

   H1_FiniteElement() : H1_HexahedronElement(P) { }

   void CalcShapeMatrix(const IntegrationRule &ir, double *B) const
   {
      MFEM_WARNING("TODO");
   }
   void CalcGradTensor(const IntegrationRule &ir, double *G) const
   {
      MFEM_WARNING("TODO");
   }

   void Calc1DShapeMatrix(const IntegrationRule &ir_1d, double *B) const
   {
      MFEM_WARNING("TODO");
   }
   void Calc1DGradMatrix(const IntegrationRule &ir_1d, double *G) const
   {
      MFEM_WARNING("TODO");
   }
};


// Integration rules

template <Geometry::Type G, int Q, int Order>
class GenericIntegrationRule
{
public:
   static const int geom = G;
   static const int qpts = Q;
   static const int order = Order;

   static const bool tensor_prod = false;

   static const IntegrationRule &GetIntRule()
   {
      return IntRules.Get(geom, order);
   }
};

template <int Dim, int Q, int Order>
class TProductIntegrationRule
{
public:
   static const int geom = (Dim == 1) ? Geometry::SEGMENT :
                           ((Dim == 2) ? Geometry::SQUARE : Geometry::CUBE);
   static const int dim = Dim;
   static const int qpts_1d = Q;
   static const int qpts = (Dim == 1) ? Q : ((Dim == 2) ? (Q*Q) : (Q*Q*Q));
   static const int order = Order;

   static const bool tensor_prod = true;
};

template <int Dim, int Q>
class GaussIntegrationRule
   : public TProductIntegrationRule<Dim, Q, 2*Q-1>
{
public:
   using TProductIntegrationRule<Dim, Q, 2*Q-1>::geom;
   using TProductIntegrationRule<Dim, Q, 2*Q-1>::order;
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


// Shape evaluators

template <class FE, class IR, bool TP>
class ShapeEvaluator_base;

template <class FE, class IR>
class ShapeEvaluator_base<FE, IR, false>
{
protected:
   static const int DOF = FE::dofs;
   static const int NIP = IR::qpts;
   static const int DIM = FE::dim;

   TMatrix<NIP, DOF> B;
   TTensor<NIP, DIM, DOF> G;

public:
   typedef TVector<DOF> dof_data_type;
   typedef TVector<NIP> qpt_data_type;
   typedef TMatrix<NIP,DIM> grad_qpt_data_type;
   typedef TMatrix<DOF,DOF> asm_data_type;

   // TODO: add an option to compute the gradient tensor as well
   ShapeEvaluator_base()
   {
      FE fe;
      fe.CalcShapeMatrix(IR::GetIntRule(), B.data);
      fe.CalcGradTensor(IR::GetIntRule(), G.data);
      std::cout << '\n' << _MFEM_FUNC_NAME << std::endl;
   }

   // TODO: add an option to copy the gradient tensor as well
   ShapeEvaluator_base(const ShapeEvaluator_base &se)
   {
      B.Set(se.B);
      G.Set(se.G);
      std::cout << '\n' << _MFEM_FUNC_NAME << std::endl;
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
      Reshape<NIP*DIM,DOF>(G).template MultTranspose<false>(
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
      std::cout << '\n' << _MFEM_FUNC_NAME << std::endl;
   }

   void Calc(const TCellData<1, DOF> &dof_data,
             TCellData<1, NIP>       &qpt_data) const
   {
      B_1d.template Mult<false>(dof_data, qpt_data);
   }

   template <bool Add>
   void CalcT(const TCellData<1, NIP> &qpt_data,
              TCellData<1, DOF>       &dof_data) const
   {
      B_1d.template MultTranspose<Add>(qpt_data, dof_data);
   }

   void CalcGrad(const TCellData<1, DOF> &dof_data,
                 grad_qpt_data_type &grad_qpt_data) const
   {
      G_1d.template Mult<false>(dof_data, grad_qpt_data);
   }

   template <bool Add>
   void CalcGradT(const grad_qpt_data_type &grad_qpt_data,
                  TCellData<1, DOF> &dof_data) const
   {
      G_1d.template MultTranspose<Add>(grad_qpt_data, dof_data);
   }

   void Assemble(const TCellData<1, NIP> &qpt_data, TMatrix<TDOF,TDOF> &M) const
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
      std::cout << '\n' << _MFEM_FUNC_NAME << std::endl;
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
      std::cout << '\n' << _MFEM_FUNC_NAME << std::endl;
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
   typedef TCellData<FE::dim, FE::dofs_1d> dof_data_type;
   typedef TCellData<IR::dim, IR::qpts_1d> qpt_data_type;
   typedef TMatrix<TDOF, TDOF> asm_data_type;

   // TODO: add an option to compute the gradient matrix as well
   ShapeEvaluator_base()
   {
      FE fe;
      fe.Calc1DShapeMatrix(IR::Get1DIntRule(), B_1d.data);
      fe.Calc1DGradMatrix(IR::Get1DIntRule(), G_1d.data);
      std::cout << '\n' << _MFEM_FUNC_NAME << std::endl;
   }

   // TODO: add an option to copy the gradient matrix as well
   ShapeEvaluator_base(const ShapeEvaluator_base &se)
   {
      B_1d.Set(se.B_1d);
      G_1d.Set(se.G_1d);
      std::cout << '\n' << _MFEM_FUNC_NAME << std::endl;
   }
};

template <class FE, class IR>
class ShapeEvaluator
   : public ShapeEvaluator_base<FE, IR, FE::tensor_prod && IR::tensor_prod>
{
public:
   static const bool tensor_prod = FE::tensor_prod && IR::tensor_prod;

   ShapeEvaluator()
   {
      std::cout << '\n' << _MFEM_FUNC_NAME << std::endl;
   }

   ShapeEvaluator(const ShapeEvaluator &se)
      : ShapeEvaluator_base<FE, IR, tensor_prod>(se)
   {
      std::cout << '\n' << _MFEM_FUNC_NAME << std::endl;
   }
};


} // namespace mfem

#endif // MFEM_TEMPLATE
