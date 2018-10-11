#include "solvers.hpp"

namespace mfem
{

namespace hypre
{

AMGSolver::AMGSolver(ParMatrix *A_)
   : Solver(A_->InLayout()->As<Layout>(), A_->OutLayout()->As<Layout>()),
     A(NULL), x_vec(NULL), y_vec(NULL), sw(), setup_time(0), solve_time(0)
{
   HYPRE_BoomerAMGCreate(&solver);
   Setup(A_);
}

void AMGSolver::Setup(ParMatrix *A_)
{
   A = A_;
   if (x_vec != NULL) hypre_ParVectorDestroy(x_vec);
   if (y_vec != NULL) hypre_ParVectorDestroy(y_vec);
   x_vec = InitializeVector(A->InLayout()->As<Layout>());
   y_vec = InitializeVector(A->OutLayout()->As<Layout>());
   // HYPRE_BoomerAMGSetPrintLevel(solver, 3);
   // HYPRE_BoomerAMGSetRelaxType(solver, 18);
   HYPRE_BoomerAMGSetMaxIter(solver, 2);

   sw.Clear();
   sw.Start();
   HYPRE_BoomerAMGSetup(solver, A->HypreMatrix(), x_vec, y_vec);
   sw.Stop();
   setup_time = sw.UserTime();
}

AMGSolver::~AMGSolver()
{
   hypre_ParVectorDestroy(x_vec);
   hypre_ParVectorDestroy(y_vec);
   HYPRE_BoomerAMGDestroy(solver);
}

void AMGSolver::Mult(const mfem::Vector &x, mfem::Vector &y) const {
   sw.Clear();
   sw.Start();
   hypre_VectorData(hypre_ParVectorLocalVector(x_vec)) = (HYPRE_Complex *) x.Get_PVector()->GetData();
   hypre_VectorData(hypre_ParVectorLocalVector(y_vec)) = (HYPRE_Complex *) y.Get_PVector()->GetData();
   HYPRE_BoomerAMGSolve(solver, A->HypreMatrix(), x_vec, y_vec);
   sw.Stop();
   solve_time += sw.UserTime();
}

} // namespace mfem::hypre

} // namespace mfem
