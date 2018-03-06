// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "vector.hpp"
#include "operator.hpp"
#include "../fem/pabilininteg.hpp"
#include "../fem/bilinearform.hpp"

#include <iostream>
#include <iomanip>

namespace mfem
{

void Operator::FormLinearSystem(const Array<int> &ess_tdof_list,
                                Vector &x, Vector &b,
                                Operator* &Aout, Vector &X, Vector &B,
                                int copy_interior)
{
   const Operator *P = this->GetProlongation();
   const Operator *R = this->GetRestriction();
   Operator *rap;

   if (P)
   {
      // Variational restriction with P
      B.SetSize(P->Width());
      P->MultTranspose(b, B);
      X.SetSize(R->Height());
      R->Mult(x, X);
      rap = new RAPOperator(*P, *this, *P);
   }
   else
   {
      // rap, X and B point to the same data as this, x and b
      X.NewDataAndSize(x.GetData(), x.Size());
      B.NewDataAndSize(b.GetData(), b.Size());
      rap = this;
   }

   if (!copy_interior) { X.SetSubVectorComplement(ess_tdof_list, 0.0); }

   // Impose the boundary conditions through a ConstrainedOperator, which owns
   // the rap operator when P and R are non-trivial
   ConstrainedOperator *A = new ConstrainedOperator(rap, ess_tdof_list,
                                                    rap != this);
   A->EliminateRHS(X, B);
   Aout = A;
}

void Operator::RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x)
{
   const Operator *P = this->GetProlongation();
   if (P)
   {
      // Apply conforming prolongation
      x.SetSize(P->Height());
      P->Mult(X, x);
   }
   else
   {
      // X and x point to the same data
   }
}

void Operator::PrintMatlab(std::ostream & out, int n, int m) const
{
   using namespace std;
   if (n == 0) { n = width; }
   if (m == 0) { m = height; }

   Vector x(n), y(m);
   x = 0.0;

   out << setiosflags(ios::scientific | ios::showpos);
   for (int i = 0; i < n; i++)
   {
      x(i) = 1.0;
      Mult(x, y);
      for (int j = 0; j < m; j++)
      {
         if (y(j))
         {
            out << j+1 << " " << i+1 << " " << y(j) << '\n';
         }
      }
      x(i) = 0.0;
   }
}


ConstrainedOperator::ConstrainedOperator(Operator *A, const Array<int> &list,
                                         bool _own_A)
   : Operator(A->Height(), A->Width()), A(A), own_A(_own_A)
{
   constraint_list.MakeRef(list);
   z.SetSize(height);
   w.SetSize(height);
}

void ConstrainedOperator::EliminateRHS(const Vector &x, Vector &b) const
{
   w = 0.0;

   for (int i = 0; i < constraint_list.Size(); i++)
   {
      w(constraint_list[i]) = x(constraint_list[i]);
   }

   A->Mult(w, z);

   b -= z;

   for (int i = 0; i < constraint_list.Size(); i++)
   {
      b(constraint_list[i]) = x(constraint_list[i]);
   }
}

void ConstrainedOperator::Mult(const Vector &x, Vector &y) const
{
   if (constraint_list.Size() == 0)
   {
      A->Mult(x, y);
      return;
   }

   z = x;

   for (int i = 0; i < constraint_list.Size(); i++)
   {
      z(constraint_list[i]) = 0.0;
   }

   A->Mult(z, y);

   for (int i = 0; i < constraint_list.Size(); i++)
   {
      y(constraint_list[i]) = x(constraint_list[i]);
   }
}


PAIOperator::PAIOperator(Array<BilinearFormIntegrator*> &PAI, int h, int w) : Operator(h,w)
{
   A.SetSize(PAI.Size());
   PAIntegrator *ainteg;
   for (int i = 0; i < A.Size(); ++i)
   {
      ainteg = dynamic_cast<PAIntegrator*>(PAI[i]);
      MFEM_ASSERT(ainteg != NULL, "PAI operators require partial assembly integrators.");
      A[i] = ainteg;
   }
}


void PAIOperator::Mult(const Vector &x, Vector &y) const
{
   //Scatter the x,y vectors into the element by element representation
   //with the degrees of freedom in lexographical order
   Vector exp_x, exp_y, temp;
   A[0]->GetFES()->ToLocalVector(x, exp_x);
   exp_y.SetSize(exp_x.Size());
   temp.SetSize(exp_x.Size());

   A[0]->PAMult(exp_x, exp_y);
   for (int i = 1; i < A.Size(); ++i)
   {
      temp = exp_y;
      A[i]->PAMult(temp,exp_y);
   }

   //Gather the expanded y vector into the compact assembled form
   A[0]->GetFES()->ToGlobalVector(exp_y, y);
}

}
