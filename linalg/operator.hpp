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
   /// Construct a square Operator with given size s (default 0).
   explicit Operator(int s = 0) { height = width = s; }

   /** @brief Construct an Operator with the given height (output size) and
       width (input size). */
   Operator(int h, int w) { height = h; width = w; }

   /// Get the height (size of output) of the Operator. Synonym with NumRows().
   inline int Height() const { return height; }
   /** @brief Get the number of rows (size of output) of the Operator. Synonym
       with Height(). */
   inline int NumRows() const { return height; }

   /// Get the width (size of input) of the Operator. Synonym with NumCols().
   inline int Width() const { return width; }
   /** @brief Get the number of columns (size of input) of the Operator. Synonym
       with Width(). */
   inline int NumCols() const { return width; }

   /// Operator application: `y=A(x)`.
   virtual void Mult(const Vector &x, Vector &y) const = 0;

   /** @brief Action of the transpose operator: `y=A^t(x)`. The default behavior
       in class Operator is to generate an error. */
   virtual void MultTranspose(const Vector &x, Vector &y) const
   { mfem_error("Operator::MultTranspose() is not overloaded!"); }

   /** @brief Evaluate the gradient operator at the point @a x. The default
       behavior in class Operator is to generate an error. */
   virtual Operator &GetGradient(const Vector &x) const
   {
      mfem_error("Operator::GetGradient() is not overloaded!");
      return const_cast<Operator &>(*this);
   }

   /** @brief Prolongation operator from linear algebra (linear system) vectors,
       to input vectors for the operator. `NULL` means identity. */
   virtual const Operator *GetProlongation() const { return NULL; }
   /** @brief Restriction operator from input vectors for the operator to linear
       algebra (linear system) vectors. `NULL` means identity. */
   virtual const Operator *GetRestriction() const  { return NULL; }

   /** @brief Form a constrained linear system using a matrix-free approach.

       Assuming square operator, form the operator linear system `A(X)=B`,
       corresponding to it and the right-hand side @a b, by applying any
       necessary transformations such as: parallel assembly, conforming
       constraints for non-conforming AMR and eliminating boundary conditions.
       @note Static condensation and hybridization are not supported for general
       operators (cf. the analogous methods BilinearForm::FormLinearSystem and
       ParBilinearForm::FormLinearSystem).

       The constraints are specified through the prolongation P from
       GetProlongation(), and restriction R from GetRestriction() methods, which
       are e.g. available through the (parallel) finite element space of any
       (parallel) bilinear form operator. We assume that the operator is square,
       using the same input and output space, so we have: `A(X)=[P^t (*this)
       P](X)`, `B=P^t(b)`, and `x=P(X)`.

       The vector @a x must contain the essential boundary condition values.
       These are eliminated through the ConstrainedOperator class and the vector
       @a X is initialized by setting its essential entries to the boundary
       conditions and all other entries to zero (@a copy_interior == 0) or
       copied from @a x (@a copy_interior != 0).

       After solving the system `A(X)=B`, the (finite element) solution @a x can
       be recovered by calling Operator::RecoverFEMSolution with the same
       vectors @a X, @a b, and @a x.

       @note The caller is responsible for destroying the output operator @a A!
       @note If there are no transformations, @a X simply reuses the data of @a
       x. */
   void FormLinearSystem(const Array<int> &ess_tdof_list,
                         Vector &x, Vector &b,
                         Operator* &A, Vector &X, Vector &B,
                         int copy_interior = 0);

   /** @brief Reconstruct a solution vector @a x (e.g. a GridFunction) from the
       solution @a X of a constrained linear system obtained from
       Operator::FormLinearSystem.

       Call this method after solving a linear system constructed using
       Operator::FormLinearSystem to recover the solution as an input vector, @a
       x, for this Operator (presumably a finite element grid function). This
       method has identical signature to the analogous method for bilinear
       forms, though currently @a b is not used in the implementation. */
   virtual void RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x);

   /// Prints operator with input size n and output size m in matlab format.
   void PrintMatlab(std::ostream & out, int n = 0, int m = 0) const;

   /// Virtual destructor.
   virtual ~Operator() { }
};


/// Base abstract class for time dependent operators:
/// a) (x,t) -> f(x,t) or b) F(x,xdot,t) = G(x,t)
/// In case a): f(x,t) is implemented with the Mult method of the base class
/// In case b): G(x,t) is implemented with the Mult method of the base class
///             and F(x,xdot,t) by TimeDependentOperator::Mult
class TimeDependentOperator : public Operator
{
protected:
   double t;
   bool has_lhs;
   bool has_rhs;

public:
   using Operator::Mult;

   /** @brief Construct a "square" time dependent Operator `y=f(x,t)`, where `x`
       and `y` have the same dimension @a n. */
   explicit TimeDependentOperator(int n = 0, double _t = 0.0, bool _lhs = false,
                                  bool _rhs = true)
      : Operator(n) { t = _t; has_lhs = _lhs; has_rhs = _rhs; }

   /** @brief Construct a "square" time dependent Operator `y=f(x,t)`, where `x`
       and `y` have the same dimension @a n. */
   explicit TimeDependentOperator(int n = 0, bool _lhs = false, bool _rhs = true)
      : Operator(n) { t = 0.0; has_lhs = _lhs; has_rhs = _rhs; }

   /** @brief Construct a time dependent Operator `y=f(x,t)`, where `x` and `y`
       have dimensions @a w and @a h, respectively. */
   TimeDependentOperator(int h, int w, double _t = 0.0, bool _lhs = false,
                         bool _rhs = true)
      : Operator(h, w) { t = _t; has_lhs = _lhs; has_rhs = _rhs; }

   /** Returns true if the Operator has a non-trivial left-hand side */
   bool HasLHS() const { return has_lhs; }

   /** Returns true if the Operator has a non-zero right-hand side */
   bool HasRHS() const { return has_rhs; }

   /// Read the currently set time.
   virtual double GetTime() const { return t; }

   /// Set the current time.
   virtual void SetTime(const double _t) { t = _t; }

   /** @brief Solve the equation: @a k = f(@a x + @a dt @a k, t), for the
       unknown @a k at the current time t.

       This method allows for the abstract implementation of some time
       integration methods, including diagonal implicit Runge-Kutta (DIRK)
       methods and the backward Euler method in particular.

       If not re-implemented, this method simply generates an error. */
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k)
   {
      mfem_error("TimeDependentOperator::ImplicitSolve() is not overloaded!");
   }

   /** implements F(x,dxdt,t) */
   virtual void Mult(const Vector &x, const Vector &dxdt, Vector &k) const
   {
      mfem_error("TimeDependentOperator::Mult(y,dydt,k) is not overloaded!");
   }

   /** Implements F_dxdt(y,dydt) * shift + F_x(y,dydt), with F_dxdt and F_x the jacobians of
       F with respect to dx/dt and x evaluated at y and dydt.
       For more details, see PETSc Manual */
   virtual Operator& GetGradient(const Vector &y, const Vector &dydt,
                                 double shift) const
   {
      mfem_error("TimeDependentOperator::GetGradient(y,dydt,shift) is not overloaded!");
      return const_cast<Operator &>(dynamic_cast<const Operator &>(*this));
   }

   virtual ~TimeDependentOperator() { }
};

/// Base class for solvers
class Solver : public Operator
{
public:
   /// If true, use the second argument of Mult() as an initial guess.
   bool iterative_mode;

   /** @brief Initialize a square Solver with size @a s.

       @warning Use a boolean expression for the second parameter (not an int)
       to distinguish this call from the general rectangular constructor. */
   explicit Solver(int s = 0, bool iter_mode = false)
      : Operator(s) { iterative_mode = iter_mode; }

   /// Initialize a Solver with height @a h and width @a w.
   Solver(int h, int w, bool iter_mode = false)
      : Operator(h, w) { iterative_mode = iter_mode; }

   /// Set/update the solver for the given operator.
   virtual void SetOperator(const Operator &op) = 0;
};


/// Identity Operator I: x -> x.
class IdentityOperator : public Operator
{
public:
   /// Create an identity operator of size @a n.
   explicit IdentityOperator(int n) : Operator(n) { }

   /// Operator application
   virtual void Mult(const Vector &x, Vector &y) const { y = x; }
};


/** @brief The transpose of a given operator. Switches the roles of the methods
    Mult() and MultTranspose(). */
class TransposeOperator : public Operator
{
private:
   const Operator &A;

public:
   /// Construct the transpose of a given operator @a *a.
   TransposeOperator(const Operator *a)
      : Operator(a->Width(), a->Height()), A(*a) { }

   /// Construct the transpose of a given operator @a a.
   TransposeOperator(const Operator &a)
      : Operator(a.Width(), a.Height()), A(a) { }

   /// Operator application. Apply the transpose of the original Operator.
   virtual void Mult(const Vector &x, Vector &y) const
   { A.MultTranspose(x, y); }

   /// Application of the transpose. Apply the original Operator.
   virtual void MultTranspose(const Vector &x, Vector &y) const
   { A.Mult(x, y); }
};


/// The operator x -> R*A*P*x.
class RAPOperator : public Operator
{
private:
   Operator & Rt;
   Operator & A;
   Operator & P;
   mutable Vector Px;
   mutable Vector APx;

public:
   /// Construct the RAP operator given R^T, A and P.
   RAPOperator(Operator &Rt_, Operator &A_, Operator &P_)
      : Operator(Rt_.Width(), P_.Width()), Rt(Rt_), A(A_), P(P_),
        Px(P.Height()), APx(A.Height()) { }

   /// Operator application.
   virtual void Mult(const Vector & x, Vector & y) const
   { P.Mult(x, Px); A.Mult(Px, APx); Rt.MultTranspose(APx, y); }

   /// Application of the transpose.
   virtual void MultTranspose(const Vector & x, Vector & y) const
   { Rt.Mult(x, APx); A.MultTranspose(APx, Px); P.MultTranspose(Px, y); }
};


/// General triple product operator x -> A*B*C*x, with ownership of the factors.
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


/** @brief Square Operator for imposing essential boundary conditions using only
    the action, Mult(), of a given unconstrained Operator.

    Square operator constrained by fixing certain entries in the solution to
    given "essential boundary condition" values. This class is used by the
    general, matrix-free system formulation of Operator::FormLinearSystem. */
class ConstrainedOperator : public Operator
{
protected:
   Array<int> constraint_list;  ///< List of constrained indices/dofs.
   Operator *A;                 ///< The unconstrained Operator.
   bool own_A;                  ///< Ownership flag for A.
   mutable Vector z, w;         ///< Auxiliary vectors.

public:
   /** @brief Constructor from a general Operator and a list of essential
       indices/dofs.

       Specify the unconstrained operator @a *A and a @a list of indices to
       constrain, i.e. each entry @a list[i] represents an essential-dof. If the
       ownership flag @a own_A is true, the operator @a *A will be destroyed
       when this object is destroyed. */
   ConstrainedOperator(Operator *A, const Array<int> &list, bool own_A = false);

   /** @brief Eliminate "essential boundary condition" values specified in @a x
       from the given right-hand side @a b.

       Performs the following steps:

           z = A((0,x_b));  b_i -= z_i;  b_b = x_b;

       where the "_b" subscripts denote the essential (boundary) indices/dofs of
       the vectors, and "_i" -- the rest of the entries. */
   void EliminateRHS(const Vector &x, Vector &b) const;

   /** @brief Constrained operator action.

       Performs the following steps:

           z = A((x_i,0));  y_i = z_i;  y_b = x_b;

       where the "_b" subscripts denote the essential (boundary) indices/dofs of
       the vectors, and "_i" -- the rest of the entries. */
   virtual void Mult(const Vector &x, Vector &y) const;

   /// Destructor: destroys the unconstrained Operator @a A if @a own_A is true.
   virtual ~ConstrainedOperator() { if (own_A) { delete A; } }
};

}

#endif
