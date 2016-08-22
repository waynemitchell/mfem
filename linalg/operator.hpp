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

#ifndef MFEM_OPERATOR
#define MFEM_OPERATOR

#include "vector.hpp"

namespace mfem
{

/// Abstract operator
class Operator
{
protected:
   int height, width;

public:
   /// Construct a square Operator with given size s (default 0)
   explicit Operator(int s = 0) { height = width = s; }

   /** Construct an Operator with the given height (output size) and width
       (input size). */
   Operator(int h, int w) { height = h; width = w; }

   /// Get the height (size of output) of the Operator. Synonym with NumRows.
   inline int Height() const { return height; }
   /** Get the number of rows (size of output) of the Operator. Synonym with
       Height. */
   inline int NumRows() const { return height; }

   /// Get the width (size of input) of the Operator. Synonym with NumCols.
   inline int Width() const { return width; }
   /** Get the number of columns (size of input) of the Operator. Synonym with
       Width. */
   inline int NumCols() const { return width; }

   /// Operator application
   virtual void Mult(const Vector &x, Vector &y) const = 0;

   /// Action of the transpose operator
   virtual void MultTranspose(const Vector &x, Vector &y) const
   { mfem_error ("Operator::MultTranspose() is not overloaded!"); }

   /// Evaluate the gradient operator at the point x
   virtual Operator &GetGradient(const Vector &x) const
   {
      mfem_error("Operator::GetGradient() is not overloaded!");
      return const_cast<Operator &>(*this);
   }

   /// Prolongation operator from linear algebra (linear system) vectors, to
   /// input vectors for the operator. NULL means identity.
   virtual const Operator *GetProlongation() const { return NULL; }
   /// Restriction operator from input vectors for the operator to linear
   /// algebra (linear system) vectors. NULL means identity.
   virtual const Operator *GetRestriction() const  { return NULL; }

   /** Assuming square operator, form the operator linear system A(X) = B,
       corresponding to it and the right-hand side b, by applying any necessary
       transformations such as: parallel assembly, conforming constraints for
       non-conforming AMR and eliminating boundary conditions. Note that static
       condensation and hybridization are not supported for general operators
       (cf. the analogous FormLinearSystem method for bilinear forms).

       The constraints are specified through the prolongation and restriction
       operators above, P and R, which are e.g. available through the (parallel)
       finite element space of any (parallel) bilinear form operator. We assume
       that the operator is square, using the same input and output space, so we
       have: A(X) = [P^t (*this) P](X), B = P^t b, and x = P(X).

       The vector x must contain the essential boundary condition values. These
       are eliminated through the ConstrainedOperator class and the vector X is
       initialized by setting its essential entries to the boundry conditions
       and all other entries to zero (copy_interior == 0) or copied from x
       (copy_interior != 0).

       This method can be called multiple times (with the same ess_tdof_list
       array) to initialize different right-hand sides and boundary condition
       values.

       After solving the linear system A(X) = B, the (finite element) solution x
       can be recovered by calling RecoverFEMSolution (with the same vectors X,
       b, and x).

       NOTE: The caller is responsible for destroying the output operator A!
       NOTE: If there are no transformations, X simply reuses the data of x. */
   void FormLinearSystem(const Array<int> &ess_tdof_list,
                         Vector &x, Vector &b,
                         Operator* &A, Vector &X, Vector &B,
                         int copy_interior = 0);

   /** Call this method after solving a linear system constructed using the
       Operator::FormLinearSystem to recover the solution as an input vector, x,
       for the operator (presumably a finite element grid function). This method
       has identical signature to the analogous method for bilinear forms,
       though currently b is not being used for general operators. */
   virtual void RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x);

   /// Prints operator with input size n and output size m in matlab format.
   void PrintMatlab (std::ostream & out, int n = 0, int m = 0) const;

   virtual ~Operator() { }
};


/// Base abstract class for time dependent operators: (x,t) -> f(x,t)
class TimeDependentOperator : public Operator
{
protected:
   double t;

public:
   /** Construct a "square" time dependent Operator y = f(x,t), where x and y
       have the same dimension 'n'. */
   explicit TimeDependentOperator(int n = 0, double _t = 0.0)
      : Operator(n) { t = _t; }

   /** Construct a time dependent Operator y = f(x,t), where x and y have
       dimensions 'w' and 'h', respectively. */
   TimeDependentOperator(int h, int w, double _t = 0.0)
      : Operator(h, w) { t = _t; }

   virtual double GetTime() const { return t; }

   virtual void SetTime(const double _t) { t = _t; }

   /** Solve the equation: k = f(x + dt*k, t), for the unknown k.
       This method allows for the abstract implementation of some time
       integration methods, including diagonal implicit Runge-Kutta (DIRK)
       methods and the backward Euler method in particular. */
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k)
   {
      mfem_error("TimeDependentOperator::ImplicitSolve() is not overloaded!");
   }

   virtual ~TimeDependentOperator() { }
};


/// Base class for solvers
class Solver : public Operator
{
public:
   /// If true, use the second argument of Mult as an initial guess
   bool iterative_mode;

   /** Initialize a square Solver with size 's'.
       WARNING: use a boolean expression for the second parameter (not an int)
       to distinguish this call from the general rectangular constructor. */
   explicit Solver(int s = 0, bool iter_mode = false)
      : Operator(s) { iterative_mode = iter_mode; }

   /// Initialize a Solver with height 'h' and width 'w'
   Solver(int h, int w, bool iter_mode = false)
      : Operator(h, w) { iterative_mode = iter_mode; }

   /// Set/update the solver for the given operator
   virtual void SetOperator(const Operator &op) = 0;
};


/// Operator I: x -> x
class IdentityOperator : public Operator
{
public:
   /// Creates I_{nxn}
   explicit IdentityOperator(int n) : Operator(n) { }

   /// Operator application
   virtual void Mult(const Vector &x, Vector &y) const { y = x; }

   ~IdentityOperator() { }
};


/// The transpose of a given operator
class TransposeOperator : public Operator
{
private:
   const Operator &A;

public:
   /// Construct the transpose of a given operator
   TransposeOperator(const Operator *a)
      : Operator(a->Width(), a->Height()), A(*a) { }

   /// Construct the transpose of a given operator
   TransposeOperator(const Operator &a)
      : Operator(a.Width(), a.Height()), A(a) { }

   /// Operator application
   virtual void Mult(const Vector &x, Vector &y) const
   { A.MultTranspose(x, y); }

   virtual void MultTranspose(const Vector &x, Vector &y) const
   { A.Mult(x, y); }

   ~TransposeOperator() { }
};


/// The operator x -> R*A*P*x
class RAPOperator : public Operator
{
private:
   Operator & Rt;
   Operator & A;
   Operator & P;
   mutable Vector Px;
   mutable Vector APx;

public:
   /// Construct the RAP operator given R^T, A and P
   RAPOperator(Operator &Rt_, Operator &A_, Operator &P_)
      : Operator(Rt_.Width(), P_.Width()), Rt(Rt_), A(A_), P(P_),
        Px(P.Height()), APx(A.Height()) { }

   /// Operator application
   virtual void Mult(const Vector & x, Vector & y) const
   { P.Mult(x, Px); A.Mult(Px, APx); Rt.MultTranspose(APx, y); }

   virtual void MultTranspose(const Vector & x, Vector & y) const
   { Rt.Mult(x, APx); A.MultTranspose(APx, Px); P.MultTranspose(Px, y); }

   virtual ~RAPOperator() { }
};


/// General triple product operator x -> A*B*C*x, with ownership of the factors
class TripleProductOperator : public Operator
{
   Operator *A;
   Operator *B;
   Operator *C;
   bool ownA, ownB, ownC;
   mutable Vector t1, t2;

public:
   TripleProductOperator(Operator *A, Operator *B, Operator *C,
                         bool ownA, bool ownB, bool ownC)
      : Operator(A->Height(), C->Width())
      , A(A), B(B), C(C)
      , ownA(ownA), ownB(ownB), ownC(ownC)
      , t1(C->Height()), t2(B->Height())
   {}

   virtual void Mult(const Vector &x, Vector &y) const
   { C->Mult(x, t1); B->Mult(t1, t2); A->Mult(t2, y); }

   virtual void MultTranspose(const Vector &x, Vector &y) const
   { A->MultTranspose(x, t2); B->MultTranspose(t2, t1); C->MultTranspose(t1, y); }

   virtual ~TripleProductOperator()
   {
      if (ownA) { delete A; }
      if (ownB) { delete B; }
      if (ownC) { delete C; }
   }
};


/// Square operator constrained by fixing certain entries in the solution to
/// given "essential boundary condition" values, a generalization of
/// FormLinearSystem to abstract operators.
class ConstrainedOperator : public Operator
{
protected:
   Array<int> constraint_list;
   Operator *A;
   bool own_A;
   mutable Vector z, w;

public:
   /// Specify the unconstrained operator and a list of vector entries to
   /// constrain (i.e. list[i] is analogous to an essential-dof).
   explicit ConstrainedOperator(Operator *A, const Array<int> &list,
                                bool own_A=false);

   /// Eliminate "essential boundary condition" values specified in x from a
   /// given right-hand side b: z = A((0,xb)); bi -= zi, bb = xb.
   void EliminateRHS(const Vector &x, Vector &b) const;

   /// Constrained operator action: z = A((xi,0)); yi = zi, yb = xb.
   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~ConstrainedOperator() { if (own_A) { delete A; } }
};

}

#endif
