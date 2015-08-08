
#include "tmatrix.hpp"
#include "mfem.hpp"

namespace mfem {

// Dim = spatial dimension
// P = polynomial degree,
// Q = number of quadrature points in one spatial direction
// N = number of entries (elements/fields/field-components) to process at a time
template <int Dim, int P, int Q, int N>
class TensorProductBasis
{
public:
   static const int dim         = Dim;
   static const int degree      = P;
   static const int dofs_1d     = P+1;
   static const int qpts_1d     = Q;
   static const int num_entries = N;

   static const int total_dofs =
      (Dim == 1) ? (P+1) : ((Dim == 2) ? ((P+1)*(P+1)) : ((P+1)*(P+1)*(P+1)));
   static const int total_qpts =
      (Dim == 1) ? Q : ((Dim == 2) ? (Q*Q) : (Q*Q*Q));


   TMatrix<Q, P+1> I; // 1D interpolation matrix
   TMatrix<Q, P+1> G; // 1D gradient interpolation matrix

   // Set the interpolation matrices I and G.
   // Instead of 'fe', uses poly1d.ClosedBasis(P) for now
   void SetInterp(const FiniteElement &fe, const IntegrationRule &ir);

   template <bool Dx, bool Dy>
   inline void Calc_2D(const TMatrix<P+1, P+1> &d, TMatrix<Q, P+1> &a,
                       TMatrix<Q, Q> &q) const;

   template <bool Dx, bool Dy, bool Add>
   inline void CalcT_2D(const TMatrix<Q, Q> &q, TMatrix<Q, P+1> &a,
                        TMatrix<P+1, P+1> &d) const;

   inline void GlobToLoc_2D(const int n, const int *entr_dof,
                            const double *glob_dof, TMatrix<P+1, P+1> &d) const;

   template <bool Add>
   inline void LocToGlob_2D(const int n, const TMatrix<P+1, P+1> &d,
                            const int *entr_dof, double *glob_dof) const;

   // Storage convention:
   // in an array of dimensions N1 x N2 x N3 x ..., the entry (i1, i2, i3, ...)
   // is located at offset i1 + N1*(i2 + N2*(i3 + ...)...)

   // In the 'Calc' and 'GradCalc' methods
   // 'loc_dof'  is (P+1)^Dim x N
   // 'entr_dof' is (P+1)^Dim x N
   // 'entr_dof' defines the offsets in 'glob_dof' defining 'loc_dof'

   // In the 'Calc' methods:
   // 'loc_qpt' is Q^Dim x N

   // Calc, direct and indirect version
   inline void Calc(const double *loc_dof, double *loc_qpt) const;
   inline void Calc(const int *entr_dof, const double *glob_dof,
                    double *loc_qpt) const;

   // CalcT, direct and indirect version
   template <bool Add>
   inline void CalcT(const double *loc_qpt, double *loc_dof) const;
   template <bool Add>
   inline void CalcT(const int *entr_dof, const double *loc_qpt,
                     double *glob_dof) const;

   // In the 'GradCalc' methods:
   // 'loc_qpt' is Q^Dim x N x Dim

   // GradCalc, direct and indirect version
   inline void GradCalc(const double *loc_dof, double *loc_qpt) const;
   inline void GradCalc(const int *entr_dof, const double *glob_dof,
                        double *loc_qpt) const;

   // GradCalcT, direct and indirect version
   template <bool Add>
   inline void GradCalcT(const double *loc_qpt, double *loc_dof) const;
   template <bool Add>
   inline void GradCalcT(const int *entr_dof, const double *loc_qpt,
                         double *glob_dof) const;
};


// Calc_2D
template <int Dim, int P, int Q, int N> template <bool Dx, bool Dy>
inline void TensorProductBasis<Dim, P, Q, N>::Calc_2D(
   const TMatrix<P+1, P+1> &d, TMatrix<Q, P+1> &a, TMatrix<Q, Q> &q) const
{
   // q = Ix.d.Iy^t = (Ix.d).Iy^t
   // where Ix and Iy are I or G
   if (!Dx)
      Mult(I, d, a);     // [q x (p+1)] . [(p+1) x (p+1)]
   else
      Mult(G, d, a);     // [q x (p+1)] . [(p+1) x (p+1)]

   if (!Dy)
      MultABt(a, I, q);  // [q x (p+1)] . [(p+1) x q]
   else
      MultABt(a, G, q);  // [q x (p+1)] . [(p+1) x q]
}

// CalcT_2D
template <int Dim, int P, int Q, int N> template <bool Dx, bool Dy, bool Add>
inline void TensorProductBasis<Dim, P, Q, N>::CalcT_2D(
   const TMatrix<Q, Q> &q, TMatrix<Q, P+1> &a, TMatrix<P+1, P+1> &d) const
{
   // d {=,+=} Ix^t.q.Iy = Ix^t.(q.Iy)
   // where Ix and Iy are I or G
   if (!Dy)
      Mult(q, I, a);        // [q x q] . [q x (p+1)]
   else
      Mult(q, G, a);        // [q x q] . [q x (p+1)]

   if (!Add)
      d.Set(0.0);

   if (!Dx)
      AddMultAtB(I, a, d);  // [(p+1) x q] . [q x (p+1)]
   else
      AddMultAtB(G, a, d);  // [(p+1) x q] . [q x (p+1)]
}

// GlobToLoc_2D
template <int Dim, int P, int Q, int N>
inline void TensorProductBasis<Dim, P, Q, N>::GlobToLoc_2D(
   const int n, const int *entr_dof, const double *glob_dof,
   TMatrix<P+1, P+1> &d) const
{
   for (int j = 0; j <= P; j++)
      for (int i = 0; i <= P; i++)
         d.data[j][i] = glob_dof[entr_dof[i+j*(P+1)+n*((P+1)*(P+1))]];
}

// LocToGlob_2D
template <int Dim, int P, int Q, int N> template <bool Add>
inline void TensorProductBasis<Dim, P, Q, N>::LocToGlob_2D(
   const int n, const TMatrix<P+1, P+1> &d, const int *entr_dof,
   double *glob_dof) const
{
   if (Add)
   {
      for (int j = 0; j <= P; j++)
         for (int i = 0; i <= P; i++)
            glob_dof[entr_dof[i+j*(P+1)+n*((P+1)*(P+1))]] += d.data[j][i];
   }
   else
   {
      for (int j = 0; j <= P; j++)
         for (int i = 0; i <= P; i++)
            glob_dof[entr_dof[i+j*(P+1)+n*((P+1)*(P+1))]] = d.data[j][i];
   }
}

// Calc direct
template <int Dim, int P, int Q, int N>
inline void TensorProductBasis<Dim, P, Q, N>::Calc(
   const double *loc_dof, double *loc_qpt) const
{
   if (Dim == 2)
   {
      const TMatrix<P+1, P+1> *d = (const TMatrix<P+1, P+1> *)loc_dof;
      TMatrix<Q, Q>           *q = (TMatrix<Q, Q>           *)loc_qpt;

      TMatrix<Q, P+1> a;

      for (int n = 0; n < N; n++)
         Calc_2D<false,false>(d[n], a, q[n]);
   }
   else
   {
      // TODO
   }
}

// Calc indirect
template <int Dim, int P, int Q, int N>
inline void TensorProductBasis<Dim, P, Q, N>::Calc(
   const int *entr_dof, const double *glob_dof, double *loc_qpt) const
{
   if (Dim == 2)
   {
      TMatrix<Q, Q> *q = (TMatrix<Q, Q> *)loc_qpt;

      TMatrix<P+1, P+1> d;
      TMatrix<Q, P+1> a;

      for (int n = 0; n < N; n++)
      {
         // copy data from 'glob_dof' to 'd'
         GlobToLoc_2D(n, entr_dof, glob_dof, d);

         Calc_2D<false,false>(d, a, q[n]);
      }
   }
   else
   {
      // TODO
   }
}

// CalcT direct
template <int Dim, int P, int Q, int N> template <bool Add>
inline void TensorProductBasis<Dim, P, Q, N>::CalcT(
   const double *loc_qpt, double *loc_dof) const
{
   if (Dim == 2)
   {
      const TMatrix<Q, Q> *q = (const TMatrix<Q, Q> *)loc_qpt;
      TMatrix<P+1, P+1>   *d = (TMatrix<P+1, P+1>   *)loc_dof;

      TMatrix<Q, P+1> a;

      for (int n = 0; n < N; n++)
         CalcT_2D<false,false,Add>(q[n], a, d[n]);
   }
   else
   {
      // TODO
   }
}

// CalcT indirect
template <int Dim, int P, int Q, int N> template <bool Add>
inline void TensorProductBasis<Dim, P, Q, N>::CalcT(
   const int *entr_dof, const double *loc_qpt, double *glob_dof) const
{
   if (Dim == 2)
   {
      const TMatrix<Q, Q> *q = (const TMatrix<Q, Q> *)loc_qpt;

      TMatrix<P+1, P+1> d;
      TMatrix<Q, P+1> a;

      for (int n = 0; n < N; n++)
      {
         CalcT_2D<false,false,false>(q[n], a, d);

         // add or copy 'd' to 'glob_dof'
         LocToGlob_2D<Add>(n, d, entr_dof, glob_dof);
      }
   }
   else
   {
      // TODO
   }
}

// GradCalc direct
template <int Dim, int P, int Q, int N>
inline void TensorProductBasis<Dim, P, Q, N>::GradCalc(
   const double *loc_dof, double *loc_qpt) const
{
   if (Dim == 2)
   {
      const TMatrix<P+1, P+1> *d = (const TMatrix<P+1, P+1> *)loc_dof;
      TMatrix<Q, Q>           *q = (TMatrix<Q, Q>           *)loc_qpt;

      TMatrix<Q, P+1> a;

      for (int n = 0; n < N; n++)
      {
         Calc_2D<true,false>(d[n], a, q[n]);
         Calc_2D<false,true>(d[n], a, q[n+N]);
      }
   }
   else
   {
      // TODO
   }
}

// GradCalc indirect
template <int Dim, int P, int Q, int N>
inline void TensorProductBasis<Dim, P, Q, N>::GradCalc(
   const int *entr_dof, const double *glob_dof, double *loc_qpt) const
{
   if (Dim == 2)
   {
      TMatrix<Q, Q> *q = (TMatrix<Q, Q> *)loc_qpt;

      TMatrix<P+1, P+1> d;
      TMatrix<Q, P+1> a;

      for (int n = 0; n < N; n++)
      {
         // copy data from 'glob_dof' to 'd'
         GlobToLoc_2D(n, entr_dof, glob_dof, d);

         Calc_2D<true,false>(d, a, q[n]);
         Calc_2D<false,true>(d, a, q[n+N]);
      }
   }
   else
   {
      // TODO
   }
}

// GradCalcT direct
template <int Dim, int P, int Q, int N> template <bool Add>
inline void TensorProductBasis<Dim, P, Q, N>::GradCalcT(
   const double *loc_qpt, double *loc_dof) const
{
   if (Dim == 2)
   {
      const TMatrix<Q, Q> *q = (const TMatrix<Q, Q> *)loc_qpt;
      TMatrix<P+1, P+1>   *d = (TMatrix<P+1, P+1>   *)loc_dof;

      TMatrix<Q, P+1> a;

      for (int n = 0; n < N; n++)
      {
         CalcT_2D<true,false,Add> (q[n],   a, d[n]);
         CalcT_2D<false,true,true>(q[n+N], a, d[n]);
      }
   }
   else
   {
      // TODO
   }
}

// GradCalcT indirect
template <int Dim, int P, int Q, int N> template <bool Add>
inline void TensorProductBasis<Dim, P, Q, N>::GradCalcT(
   const int *entr_dof, const double *loc_qpt, double *glob_dof) const
{
   if (Dim == 2)
   {
      const TMatrix<Q, Q> *q = (const TMatrix<Q, Q> *)loc_qpt;

      TMatrix<P+1, P+1> d;
      TMatrix<Q, P+1> a;

      for (int n = 0; n < N; n++)
      {
         CalcT_2D<true,false,false>(q[n],   a, d);
         CalcT_2D<false,true,true> (q[n+N], a, d);

         // add or copy 'd' to 'glob_dof'
         LocToGlob_2D<Add>(n, d, entr_dof, glob_dof);
      }
   }
   else
   {
      // TODO
   }
}

template <int Dim, int P, int Q, int N>
void TensorProductBasis<Dim, P, Q, N>::SetInterp(
   const FiniteElement &fe, const IntegrationRule &ir)
{
   if (fe.GetDim() != Dim)
      mfem_error("TensorProductBasis<>::SetInterp: invalid FE dimension");
   if (fe.GetOrder() != P)
      mfem_error("TensorProductBasis<>::SetInterp: invalid FE order");
   if (ir.GetNPoints() != Q)
      mfem_error("TensorProductBasis<>::SetInterp: invalid quadrature rule");

#if 0
   // implementation using a class ProductFiniteElement (TODO) that serves as
   // a base class for all tensor product FE and adds a method Calc1D:
   //    void Calc1D(const double x, double *u, double *d) const;

   const ProductFiniteElement *prod_fe =
      dynamic_cast<const ProductFiniteElement *>(&fe);

   if (prod_fe == NULL)
      mfem_error("TensorProductBasis<>::SetInterp: not a product FE");

   Vector val(P+1), der(P+1, 1);
   for (int i = 0; i < Q; i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      prod_fe->Calc1D(ip.x, val, der);

      for (int j = 0; j <= P; j++)
      {
         I.data[j][i] = val(j);
         G.data[j][i] = der(j);
      }
   }
#else
   // ignore 'fe' and use poly1d.ClosedBasis(P)

   Poly_1D::Basis &basis_1d = poly1d.ClosedBasis(P);

   Vector val(P+1), der(P+1);
   for (int i = 0; i < Q; i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      basis_1d.Eval(ip.x, val, der);

      for (int j = 0; j <= P; j++)
      {
         I.data[j][i] = val(j);
         G.data[j][i] = der(j);
      }
   }
#endif
}

}
