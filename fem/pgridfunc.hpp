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

#ifndef MFEM_PGRIDFUNC
#define MFEM_PGRIDFUNC

/// Class for parallel grid function
class ParGridFunction : public GridFunction
{
protected:
   ParFiniteElementSpace *pfes;

   Vector face_nbr_data;

public:
   ParGridFunction() { pfes = NULL; }

   ParGridFunction(ParFiniteElementSpace *pf) : GridFunction(pf), pfes(pf) { }

   /** Construct a ParGridFunction corresponding to *pf and the data from *gf
       which is a local GridFunction on each processor. */
   ParGridFunction(ParFiniteElementSpace *pf, GridFunction *gf);

   /** Creates grid function on (all) dofs from a given vector on the true dofs,
       i.e. P tv. */
   ParGridFunction(ParFiniteElementSpace *pf, HypreParVector *tv);

   /** Construct a ParGridFunction from the given serial GridFunction.
       The data from 'gf' is NOT copied. */
   ParGridFunction(ParMesh *pmesh, GridFunction *gf);

   ParGridFunction &operator=(double value)
   { GridFunction::operator=(value); return *this; }

   ParGridFunction &operator=(const Vector &v)
   { GridFunction::operator=(v); return *this; }

   ParFiniteElementSpace *ParFESpace() { return pfes; }

   void Update(ParFiniteElementSpace *f);

   void Update(ParFiniteElementSpace *f, Vector &v, int v_offset);

   /** Set the grid function on (all) dofs from a given vector on the
       true dofs, i.e. P tv. */
   void Distribute(const Vector *tv);

   /// Short semantic for Distribute
   ParGridFunction &operator=(const HypreParVector &tv)
   { Distribute(&tv); return (*this); }

   /// Returns the true dofs in a HypreParVector
   void GetTrueDofs(Vector &tv) const;

   /// Returns the true dofs in a new HypreParVector
   HypreParVector *GetTrueDofs() const;

   /// Returns the vector averaged on the true dofs.
   void ParallelAverage(HypreParVector &tv) const;

   /// Returns a new vector averaged on the true dofs.
   HypreParVector *ParallelAverage() const;

   /// Returns the vector assembled on the true dofs.
   void ParallelAssemble(HypreParVector &tv) const;

   /// Returns a new vector assembled on the true dofs.
   HypreParVector *ParallelAssemble() const;

   void ExchangeFaceNbrData();
   Vector &FaceNbrData() { return face_nbr_data; }
   const Vector &FaceNbrData() const { return face_nbr_data; }

   using GridFunction::GetValue;
   double GetValue(ElementTransformation &T);

   using GridFunction::ProjectCoefficient;
   void ProjectCoefficient(Coefficient &coeff);

   double ComputeL1Error(Coefficient *exsol[],
                         const IntegrationRule *irs[] = NULL) const;

   double ComputeL1Error(Coefficient &exsol,
                         const IntegrationRule *irs[] = NULL) const
   { Coefficient *exsol_p = &exsol; return ComputeL1Error(&exsol_p, irs); }

   double ComputeL1Error(VectorCoefficient &exsol,
                         const IntegrationRule *irs[] = NULL) const;

   double ComputeL2Error(Coefficient *exsol[],
                         const IntegrationRule *irs[] = NULL) const;

   double ComputeL2Error(Coefficient &exsol,
                         const IntegrationRule *irs[] = NULL) const
   { Coefficient *exsol_p = &exsol; return ComputeL2Error(&exsol_p, irs); }

   double ComputeL2Error(VectorCoefficient &exsol,
                         const IntegrationRule *irs[] = NULL,
                         Array<int> *elems = NULL) const;

   double ComputeMaxError(Coefficient *exsol[],
                          const IntegrationRule *irs[] = NULL) const;

   double ComputeMaxError(Coefficient &exsol,
                          const IntegrationRule *irs[] = NULL) const
   { Coefficient *exsol_p = &exsol; return ComputeMaxError(&exsol_p, irs); }

   double ComputeMaxError(VectorCoefficient &exsol,
                          const IntegrationRule *irs[] = NULL) const;

   /** Save the local portion of the ParGridFunction. It differs from the
       serial GridFunction::Save in that it takes into account the signs of
       the local dofs. */
   virtual void Save(ostream &out);

   /// Merge the local grid functions
   void SaveAsOne(ostream &out = cout);

   virtual ~ParGridFunction() { }
};

#endif
