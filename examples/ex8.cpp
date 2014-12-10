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
using namespace std;

#include "mfem.hpp"
using namespace mfem;


// The 2x2 block DPG operator
//
//     [   B0^T Sinv B0       B0^T Sinv Bhat ]
//     [ Bhat^T Sinv B0     Bhat^T Sinv Bhat ]
//
// corresponding to the primal (x0) and the interfacial (x1) unkowns. This
// operator is symmetric and positive definite.
class DPG2x2Operator : public Operator
{
protected:
   SparseMatrix *B0, *Bhat, *Sinv;
   int s0, s1, s2;

public:
   DPG2x2Operator(SparseMatrix &A0, SparseMatrix &A1, SparseMatrix &A2)
      : B0(&A0), Bhat(&A1), Sinv(&A2)
   {
      s0 = B0->Width();
      s1 = Bhat->Width();
      s2 = Sinv->Width();
      size = s0+s1;
   }

   virtual void Mult (const Vector & x, Vector & y) const
   {
      Vector y0(y.GetData(),s0), x0(x.GetData(),s0); // primal
      Vector y1(y.GetData()+s0,s1), x1(x.GetData()+s0,s1); // interfacial
      Vector u(s2), v(s2); // test (temporary)

      B0->Mult(x0,u);
      Bhat->AddMult(x1,u);
      Sinv->Mult(u,v);
      B0->MultTranspose(v,y0);
      Bhat->MultTranspose(v,y1);
   }

   virtual ~DPG2x2Operator() { }
};

// The 3x3 block DPG operator
//
//     [  0     0    B0^T  ]
//     [  0     0   Bhat^T ]
//     [ B0   Bhat    S    ]
//
// corresponding to the primal (x0), interfacial (x1) and test (x2)
// unkowns. This operator is indefinite.
class DPG3x3Operator : public Operator
{
protected:
   SparseMatrix *B0;
   SparseMatrix *Bhat;
   SparseMatrix *S;
   int s0, s1, s2;

public:
   DPG3x3Operator(SparseMatrix & A0, SparseMatrix & A1, SparseMatrix & A2)
      : B0(&A0), Bhat(&A1), S(&A2)
   {
      s0 = B0->Width();
      s1 = Bhat->Width();
      s2 = S->Width();
      size = s0+s1+s2;
   }

   virtual void Mult (const Vector & x, Vector & y) const
   {
      Vector y0(y.GetData(),s0), x0(x.GetData(),s0); // primal
      Vector y1(y.GetData()+s0,s1), x1(x.GetData()+s0,s1); // interfacial
      Vector y2(y.GetData()+s0+s1,s2), x2(x.GetData()+s0+s1,s2); // test

      B0->MultTranspose(x2,y0);
      Bhat->MultTranspose(x2,y1);
      B0->Mult(x0,y2);
      Bhat->AddMult(x1,y2);
      S->AddMult(x2,y2);
   }

   virtual ~DPG3x3Operator() { }
};

// A block-diagonal preconditioner for the 3x3 block DPG operator of the form
//
//     [ S0^{-1}         0                  0   ]
//     [   0     (Bhat^T Sinv Bhat)^{-1}    0   ]
//     [   0             0                 Sinv ]
//
// corresponding to the primal (x0), interfacial (x1) and test (x2) unkowns.
class DPG3x3Preconditioner : public Solver
{
protected:
   SparseMatrix *X0pc;
   SparseMatrix *Xhatpc;
   SparseMatrix *Sinvpc;
   int s0, s1, s2;

public:
   DPG3x3Preconditioner(SparseMatrix & A0, SparseMatrix & A1, SparseMatrix & A2)
      : X0pc(&A0), Sinvpc(&A2)
   {
      SparseMatrix *BhatT = Transpose(A1);
      Xhatpc = RAP(*Sinvpc,*BhatT);
      delete BhatT;

      s0 = X0pc->Width();
      s1 = Xhatpc->Width();
      s2 = Sinvpc->Width();
      size = s0+s1+s2;
   }

   virtual void Mult (const Vector & x, Vector & y) const
   {
      Vector y0(y.GetData(),s0), x0(x.GetData(),s0); // primal
      Vector y1(y.GetData()+s0,s1), x1(x.GetData()+s0,s1); // interfacial
      Vector y2(y.GetData()+s0+s1,s2), x2(x.GetData()+s0+s1,s2); // test

      y = 0.0;
      CG(*X0pc, x0, y0, -1, 300, 1e-24, 0);
      CG(*Xhatpc, x1, y1, -1, 300, 1e-24, 0);
      Sinvpc->Mult(x2,y2);
   }

   virtual void SetOperator(const Operator &op) { }

   virtual ~DPG3x3Preconditioner() { delete Xhatpc; }
};


int main(int argc, char *argv[])
{
   Mesh *mesh;

   if (argc == 1)
   {
      cout << "\nUsage: ex8 <mesh_file>\n" << endl;
      return 1;
   }

   // Which DPG formulation to solve: the 2x2 one without preconditioning or the
   // 3x3 one with preconditioning?
   bool dpg2x2 = false;

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

   int s0 = x0_space->GetVSize();
   int s1 = xhat_space->GetVSize();
   int s2 = test_space->GetVSize();
   int size = s0+s1;
   if (dpg2x2 != 1)
      size += s2;

   // 4. Set up the linear form F(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (f,phi_i) where f=1.0 and
   //    phi_i are the basis functions in the finite element fespace.

   ConstantCoefficient one(1.0);
   LinearForm *F = new LinearForm(test_space);
   F->AddDomainIntegrator(new DomainLFIntegrator(one));
   F->Assemble();


   // 5. Set up the mixed bilinear form for the non interfacial unknowns, B0,
   //    the mixed bilinear form for the interfacial unknowns, Bhatm, and the
   //    inverse stiffness matrix on the discontinuous test space, Sinv.
   MixedBilinearForm *B0 = new MixedBilinearForm(x0_space,test_space);
   B0->AddDomainIntegrator(new DiffusionIntegrator(one));
   B0->Assemble();

   MixedBilinearForm *Bhat = new MixedBilinearForm(xhat_space,test_space);
   Bhat->AddFaceIntegrator(new NTMassJumpIntegrator());
   Bhat->Assemble();
   Bhat->Finalize();

   BilinearForm *Sinv = new BilinearForm(test_space);
   Sinv->AddDomainIntegrator(new DiffusionIntegrator(one));
   Sinv->AddDomainIntegrator(new MassIntegrator(one));
   Sinv->AssembleDomainInverse();
   Sinv->Finalize();

   // 6. Define the unknown vector, a grid function based on it, and apply the
   //    essential boundary conditions.
   Vector X(size);
   GridFunction x;
   x.Update(x0_space, X, 0);

   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   x = 0.0;
   B0->EliminateTrialDofs(ess_bdr, x, *F);
   B0->Finalize();

   // 7. Extract matrices and define the DPG right-hand side.
   SparseMatrix &matB0   = B0->SpMat();
   SparseMatrix &matBhat = Bhat->SpMat();
   SparseMatrix &matSinv = Sinv->SpMat();

   Vector b(size);
   if (dpg2x2)
   {
      Vector b0(b.GetData(),s0);
      Vector b1(b.GetData()+s0,s1);
      // right-hand side of the 2x2 operator is (B0^T Sinv F, Bhat^T Sinv)
      {
         Vector SinvF(F->Size());
         matSinv.Mult(*F,SinvF);
         matB0.MultTranspose(SinvF,b0);
         matBhat.MultTranspose(SinvF,b1);
      }
   }
   else
   {
      Vector b0(b.GetData(),s0);
      Vector b1(b.GetData()+s0,s1);
      Vector b2(b.GetData()+s0+s1,s2);
      b0=0.0;
      b1=0.0;
      b2=0.0;
      b2.Add(1.0,*F);
   }

   // 8. Define the DPG operator and the DPG preconditioner in the 3x3
   //    case. Solve the resulting linear system with CG or MINRES.
   Operator *A;
   if (dpg2x2)
   {
      Vector Xhat(X.GetData()+s0, s1);
      Xhat = 0.0;

      A = new DPG2x2Operator(matB0, matBhat, matSinv);

      CG(*A, b, X, 1, 300, 1e-16, 0);
   }
   else
   {
      Vector Xhat(X.GetData()+s0,s1);
      Vector Xtest(X.GetData()+s0+s1,s2);
      Xhat  = 0.0;
      Xtest = 0.0;

      // set up bilinear forms for the stiffness matrix S on the test and trial
      // spaces with v^T S v = ||v||^2_(H^1)
      BilinearForm *S = new BilinearForm(test_space);
      S->AddDomainIntegrator(new DiffusionIntegrator(one));
      S->AddDomainIntegrator(new MassIntegrator(one));
      S->Assemble();
      S->Finalize();

      BilinearForm *S0 = new BilinearForm(x0_space);
      S0->AddDomainIntegrator(new DiffusionIntegrator(one));
      S0->AddDomainIntegrator(new MassIntegrator(one));
      S0->Assemble();
      S0->EliminateEssentialBC(ess_bdr);
      S0->Finalize();

      SparseMatrix &matS  = S->SpMat();
      SparseMatrix &matS0 = S0->SpMat();

      A = new DPG3x3Operator(matB0, matBhat, matS);
      DPG3x3Preconditioner *B =
         new DPG3x3Preconditioner(matS0, matBhat, matSinv);

      MINRES(*A, *B, b, X, 1, 300, 1e-16, 0.);

      delete B;
      delete S0;
      delete S;
   }

   // 9. Save the refined mesh and the solution. This output can be viewed later
   //    using GLVis: "glvis -m refined.mesh -g sol.gf".
   {
      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 10. (Optional) Send the solution by socket to a GLVis server.
   char vishost[] = "localhost";
   int  visport   = 19916;
   osockstream sol_sock(visport, vishost);
   sol_sock << "solution\n";
   sol_sock.precision(8);
   mesh->Print(sol_sock);
   x.Save(sol_sock);
   sol_sock.send();

   // 11. Free the used memory.
   delete Bhat;
   delete B0;
   delete Sinv;
   delete F;
   delete test_space;
   delete test_fec;
   delete xhat_space;
   delete xhat_fec;
   delete x0_space;
   delete x0_fec;
   delete mesh;

   delete A;

   return 0;
}
