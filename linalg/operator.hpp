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

#ifndef MFEM_OPERATOR
#define MFEM_OPERATOR

#include <iostream>
#include "vector.hpp"

namespace mfem
{

/// Abstract operator
class Operator
{
protected:
   int size;

public:
   /// Construct Operator with given size s (default 0)
   explicit Operator (int s = 0) { size = s; }

   /// Returns the size of the input
   inline int Size() const { return size; }

   /// Operator application
   virtual void Mult (const Vector & x, Vector & y) const = 0;

   /// Action of the transpose operator
   virtual void MultTranspose (const Vector & x, Vector & y) const
   { mfem_error ("Operator::MultTranspose() is not overloaded!"); }

   /// Evaluate the gradient operator at the point x
   virtual Operator &GetGradient(const Vector &x) const
   {
      mfem_error("Operator::GetGradient() is not overloaded!");
      return *((Operator *)this);
   }

   /// Prints operator with input size n and output size m in matlab format.
   void PrintMatlab (std::ostream & out, int n = 0, int m = 0);

   virtual ~Operator() { }
};


/// Base abstract class for time dependent operators: (x,t) -> f(x,t)
class TimeDependentOperator : public Operator
{
protected:
   double t;

public:
   explicit TimeDependentOperator(int n = 0, double _t = 0.0)
      : Operator(n) { t = _t; }

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
   // Use the second argument of Mult as an initial guess?
   bool iterative_mode;

   Solver(int s = 0, bool iter_mode = false)
      : Operator(s) { iterative_mode = iter_mode; }

   /// Set/update the solver for the given operator
   virtual void SetOperator(const Operator &op) = 0;
};


/// Operator I: x -> x
class IdentityOperator : public Operator
{
public:
   /// Creates I_{nxn}
   explicit IdentityOperator (int n) { size = n; }

   /// Operator application
   virtual void Mult (const Vector & x, Vector & y) const { y = x; }

   ~IdentityOperator() { }
};


/// The transpose of a given operator (square matrix)
class TransposeOperator : public Operator
{
private:
   const Operator * A;

public:
   /// Saves the operator
   TransposeOperator (const Operator * a, int s = -1) : A(a) { size = (s == -1) ? A -> Size() : s; }

   /// Operator application
   virtual void Mult (const Vector & x, Vector & y) const
   { A -> MultTranspose(x,y); }

   virtual void MultTranspose (const Vector & x, Vector & y) const
   { A -> Mult(x,y); }

   ~TransposeOperator() { }
};

}

#endif
