//                                MFEM Example 8
//
// Compile with: make ex8
//
// Sample runs:  ex8 ../data/square-disc.mesh
//               ex8 ../data/star.mesh
//               ex8 ../data/escher.mesh
//               ex8 ../data/fichera.mesh
//
// Description:  This example code demonstrates the use of the Discontinuous
//               Petrov-Galerkin (DPG) method as a simple isoparametric finite
//               element discretization of the Laplace problem -Delta u = f with
//               homogeneous Dirichlet boundary conditions. Specifically, we
//               discretize with the FE space coming from the mesh (linear by
//               default, quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of interfacial (numerical trace)
//               finite elements, interior face boundary integrators and the
//               definition of block operators and preconditioners.
//
//               We recommend viewing examples 1-5 before viewing this example.

#include <fstream>
#include "mfem.hpp"

using namespace std;
using namespace mfem;


SparseMatrix *RAP(const SparseMatrix & Rt, const SparseMatrix & A,
                  const SparseMatrix & P)
{
   SparseMatrix * R = Transpose(Rt);
   SparseMatrix * RA = Mult(*R,A);
   delete R;
   SparseMatrix * out = Mult(*RA, P);
   delete RA;
   return out;
}

int main(int argc, char *argv[])
{
   Mesh *mesh;

   if (argc == 1)
   {
      cout << "\nUsage: ex8 <mesh_file>\n" << endl;
      return 1;
   }

   // 1. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral or hexahedral elements with the same code.
   ifstream imesh(argv[1]);
   if (!imesh)
   {
      cerr << "\nCan not open mesh file: " << argv[1] << '\n' << endl;
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();
   const int dim = mesh->Dimension();

   // 2. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(5000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
         mesh->UniformRefinement();
   }

   // 3. Define the trial, interfacial (numerical trace) and test DPG spaces:
   //    - The trial space, x0_space, contains the non-interfacial unknowns and
   //      has the essential BC.
   //    - The interfacial space, xhat_space, contains the interfacial unknowns
   //      and does not have essential BC.
   //    - The test space, test_space, is an enriched space where the enrichment
   //      degree depends on the dimension of the space.
   int trial_order = 1;
   int nt_order    = trial_order - 1;
   int test_order  = nt_order + dim;
   if (test_order < trial_order)
      cerr << "Warning, test space not enriched enough to handle primal trial space\n";

   FiniteElementCollection *x0_fec   = new H1_FECollection(trial_order, dim);
   FiniteElementCollection *xhat_fec = new NT_FECollection(nt_order, dim);
   FiniteElementCollection *test_fec = new L2_FECollection(test_order, dim);

   FiniteElementSpace *x0_space   = new FiniteElementSpace(mesh, x0_fec);
   FiniteElementSpace *xhat_space = new FiniteElementSpace(mesh, xhat_fec);
   FiniteElementSpace *test_space = new FiniteElementSpace(mesh, test_fec);

   // 4. Define the block structure of the problem, by creating the offset variable.
   // Also allocate two BlockVector objects to store the solution and rhs.

   enum {x0_var, xhat_var, test_var};

   int s0 = x0_space->GetVSize();
   int s1 = xhat_space->GetVSize();
   int s2 = test_space->GetVSize();

   Array<int> offsets(4);
   offsets[0] = 0;
   offsets[1] = s0;
   offsets[2] = s0+s1;
   offsets[3] = s0+s1+s2;

   BlockVector x(offsets), b(offsets);

   x = 0.;
   b = 0.;

   // 5. Set up the linear form F(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (f,phi_i) where f=1.0 and
   //    phi_i are the basis functions in the finite element fespace.

   ConstantCoefficient one(1.0);
   {
      LinearForm F;
      F.Update(test_space, b.GetBlock(test_var), 0);
      F.AddDomainIntegrator(new DomainLFIntegrator(one));
      F.Assemble();
   }


   // 6. Set up the mixed bilinear form for the non interfacial unknowns, B0,
   //    the mixed bilinear form for the interfacial unknowns, Bhat,
   //    the stiffness matrix and its inverse on the discontinuous test space, S and Sinv,
   //    the stiffness matrix on the continuous trial space, S0.

   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;

   MixedBilinearForm *B0 = new MixedBilinearForm(x0_space,test_space);
   B0->AddDomainIntegrator(new DiffusionIntegrator(one));
   B0->Assemble();
   B0->EliminateTrialDofs(ess_bdr, x.GetBlock(x0_var), b.GetBlock(test_var));
   B0->Finalize();


   MixedBilinearForm *Bhat = new MixedBilinearForm(xhat_space,test_space);
   Bhat->AddFaceIntegrator(new NTMassJumpIntegrator());
   Bhat->Assemble();
   Bhat->Finalize();

   BilinearForm *S = new BilinearForm(test_space);
   S->AddDomainIntegrator(new DiffusionIntegrator(one));
   S->AddDomainIntegrator(new MassIntegrator(one));
   S->Assemble();
   S->Finalize();

   BilinearForm *Sinv = new BilinearForm(test_space);
   Sinv->AddDomainIntegrator(new DiffusionIntegrator(one));
   Sinv->AddDomainIntegrator(new MassIntegrator(one));
   Sinv->AssembleDomainInverse();
   Sinv->Finalize();

   BilinearForm *S0 = new BilinearForm(x0_space);
   S0->AddDomainIntegrator(new DiffusionIntegrator(one));
   S0->AddDomainIntegrator(new MassIntegrator(one));
   S0->Assemble();
   S0->EliminateEssentialBC(ess_bdr);
   S0->Finalize();

   SparseMatrix &matB0   = B0->SpMat();
   SparseMatrix &matBhat = Bhat->SpMat();
   SparseMatrix &matS  = S->SpMat();
   SparseMatrix &matSinv = Sinv->SpMat();
   SparseMatrix &matS0 = S0->SpMat();

   // 7. Set up the 3x3 block DPG operator
   //
   //        [  0     0    B0^T  ]
   //        [  0     0   Bhat^T ]
   //        [ B0   Bhat    S    ]
   //
   //    corresponding to the primal (x0), interfacial (x1) and test (x2)
   //    unknowns. This operator is indefinite.

   Operator * B0t = new TransposeOperator(matB0);
   Operator * Bhatt = new TransposeOperator(matBhat);

   BlockOperator A(offsets);
   A.SetBlock(0,2,B0t);
   A.SetBlock(1,2,Bhatt);
   A.SetBlock(2,0,&matB0);
   A.SetBlock(2,1,&matBhat);
   A.SetBlock(2,2,&matS);


   // 8. Set up a block-diagonal preconditioner for the 3x3 block DPG operator of the form
   //
   //        [ S0^{-1}     0        0   ]
   //        [   0     Shat^{-1}    0   ]      Shat = (Bhat^T Sinv Bhat)
   //        [   0         0       Sinv ]
   //
   //    corresponding to the primal (x0), interfacial (x1) and test (x2) unknowns.

   SparseMatrix * Shat = RAP(matBhat, matSinv, matBhat);

#ifdef MFEM_USE_UMFPACK
   Operator * S0inv = new UMFPackSolver(matS0);
   Operator * Shatinv = new UMFPackSolver(*Shat);
#else
   CGSolver * S0inv = new CGSolver;
   S0inv->SetOperator(matS0);
   S0inv->SetPrintLevel(-1);
   S0inv->SetRelTol(1e-12);
   S0inv->SetMaxIter(300);
   CGSolver * Shatinv = new CGSolver;
   Shatinv->SetOperator(*Shat);
   Shatinv->SetPrintLevel(-1);
   Shatinv->SetRelTol(1e-12);
   Shatinv->SetMaxIter(300);
#endif

   BlockDiagonalPreconditioner P(offsets);
   P.SetDiagonalBlock(0, S0inv);
   P.SetDiagonalBlock(1, Shatinv);
   P.SetDiagonalBlock(2, &matSinv);

   // 9. Compute the solution using MINRES iterative solver.
   //    Check the weighted norm of residual for the DPG least square problem
   //    Wrap the primal variable in a GridFunction for visualization purposes.

   MINRES(A, P, b, x, 1, 300, 1e-16, 0.);

   double res2;
   res2 = matS.InnerProduct(x.GetBlock(test_var), x.GetBlock(test_var));
   std::cout << "|| Xtest||_S = || B0*x0 + Bhat*xhat - F ||_{S^-1} = " << sqrt(res2) << "\n";

   GridFunction x0;
   x0.Update(x0_space, x.GetBlock(x0_var), 0);

   // 10. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   {
      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x0.Save(sol_ofs);
   }

   // 11. (Optional) Send the solution by socket to a GLVis server.
   char vishost[] = "localhost";
   int  visport   = 19916;
   osockstream sol_sock(visport, vishost);
   sol_sock << "solution\n";
   sol_sock.precision(8);
   mesh->Print(sol_sock);
   x0.Save(sol_sock);
   sol_sock.send();

   // 12. Free the used memory.
   delete S0inv;
   delete Shatinv;
   delete B0t;
   delete Bhatt;
   delete Shat;
   delete Bhat;
   delete B0;
   delete S;
   delete S0;
   delete Sinv;
   delete test_space;
   delete test_fec;
   delete xhat_space;
   delete xhat_fec;
   delete x0_space;
   delete x0_fec;
   delete mesh;

   return 0;
}
