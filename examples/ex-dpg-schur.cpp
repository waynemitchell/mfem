//                                MFEM Example SchurSQUARE
//
// Compile with: make exSchurSQUARE
//
// Sample runs:  exSchurSQUARE ../data/square1.mesh
//               exSchurSQUARE ../data/square2.mesh
//
// Description:  This example code demonstrates the use of a preconditioner in
//               saddlepoint DPG in MFEM to solve the Laplace problem -Delta u
//               = f with homogeneous Dirichlet boundary conditions on a unit
//               square. It computes the error and reports the number of outer
//               MINRES iterations.  The preconditioner uses Bhat^T Sinv Bhat.


#include <fstream>
#include "mfem.hpp"

using namespace std;
using namespace mfem;

double exact_sol(Vector &);
double exact_rhs(Vector &);

/// Class for Adaptive Hybrid DPG operators:
class HybridDPGOperator : public Operator
{
protected:
   const SparseMatrix *A0;
   const SparseMatrix *Ahat;
   const SparseMatrix *TEST;

public:
   HybridDPGOperator(const SparseMatrix & A1,
                     const SparseMatrix & A2,
                     const SparseMatrix & A3)
   {
      size=((A1.Width())+(A2.Width())+(A3.Width()));
      A0=&A1;
      Ahat=&A2;
      TEST=&A3;
   }

   /// Returns the size of the input
   inline int Size() const { return size; }

   /// Operator application
   virtual void Mult (const Vector & x, Vector & y) const
   {
      int s0, s1, s2;
      s0=A0->Width();
      s1=Ahat->Width();
      s2=TEST->Width();
      Vector y0(y.GetData(),s0);
      Vector x0(x.GetData(),s0);
      Vector y1(y.GetData()+s0,s1);
      Vector x1(x.GetData()+s0,s1);
      Vector y2(y.GetData()+s0+s1,s2);
      Vector x2(x.GetData()+s0+s1,s2);
      A0->MultTranspose(x2,y0);
      Ahat->MultTranspose(x2,y1);
      A0->Mult(x0,y2);
      Ahat->AddMult(x1,y2);
      TEST->AddMult(x2,y2);
   }

   /// Action of the transpose operator
   virtual void MultTranspose (const Vector & x, Vector & y) const
   { mfem_error ("Operator::MultTranspose() is not overloaded!"); }

   /// Evaluate the gradient operator at the point x
   virtual Operator &GetGradient(const Vector &x) const
   {
      mfem_error("Operator::GetGradient() is not overloaded!");
      return *((Operator *)this);
   }

   virtual ~HybridDPGOperator() { }
};


/// Class for Adaptive Hybrid DPG preconditioners:
class HybridDPGPreconditioner : public Solver
{
protected:
   SparseMatrix *X0pc;
   SparseMatrix *Xhatpc;
   SparseMatrix *Sinvpc;

public:
   HybridDPGPreconditioner(SparseMatrix & A1,
                           SparseMatrix & A2,
                           SparseMatrix & A3)
   {
      size=((A1.Width())+(A2.Width())+(A3.Width()));
      X0pc=&A1;
      Sinvpc=&A3;
      SparseMatrix *BhatT=Transpose(A2);
      Xhatpc=RAP(*Sinvpc,*BhatT);
      delete BhatT;
   }

   /// Returns the size of the input
   inline int Size() const { return size; }

   /// Operator application
   virtual void Mult (const Vector & x, Vector & y) const
   {
      int s0, s1, s2;
      s0=X0pc->Width();
      s1=Xhatpc->Width();
      s2=Sinvpc->Width();
      y=0.0;
      Vector y0(y.GetData(),s0);
      Vector x0(x.GetData(),s0);
      Vector y1(y.GetData()+s0,s1);
      Vector x1(x.GetData()+s0,s1);
      Vector y2(y.GetData()+s0+s1,s2);
      Vector x2(x.GetData()+s0+s1,s2);
      CG(*X0pc, x0, y0,0,200000,1e-24,0); // only pos def if ess BC dof already removed
//     cout << "Inverted X0\n" << endl << flush;
      CG(*Xhatpc, x1, y1,0,200000,1e-24,0);
//     cout << "Inverted Xhat\n" << endl << flush;
      Sinvpc->Mult(x2,y2);
   }

   /// Action of the transpose operator
   virtual void MultTranspose (const Vector & x, Vector & y) const
   { mfem_error ("Operator::MultTranspose() is not overloaded!"); }

   /// Evaluate the gradient operator at the point x
   virtual Operator &GetGradient(const Vector &x) const
   {
      mfem_error("Operator::GetGradient() is not overloaded!");
      return *((Operator *)this);
   }

   virtual void SetOperator(const Operator &op) { }

   virtual ~HybridDPGPreconditioner() {
      delete Xhatpc;
   }
};


int main (int argc, char *argv[])
{
   Mesh *mesh;

   if (argc == 1)
   {
      cout << "\nUsage: exDPG <mesh_file>\n" << endl;
      return 1;
   }

   // 1. Read the 2D or 3D mesh from the given mesh file. In this example, we
   //    can handle triangular, quadrilateral, tetrahedral or hexahedral meshes
   //    with the same code.
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
   //    largest number that gives a final mesh with no more than 25,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(25000./mesh->GetNE())/log(2.)/dim);
      cout << "Enter ref levels=" ;
      cin >> ref_levels;
      for (int l = 0; l < ref_levels; l++)
         mesh->UniformRefinement();
   }
   int p = 1;
   cout << "Enter order of primal trial space=" << flush ;
   cin >> p;
   int n = p-1;
   cout << "Enter order of numerical trace trial space=" << flush ;
   cin >> n;
   int t = n+dim;
   cout << "Enter order of test space=" << flush ;
   cin >> t;
   if (t < p)
      cout << "Warning, test space not enriched enough to handle primal trial space " << endl << flush ;

   FunctionCoefficient e_sol(&exact_sol);

   // 3. Define the test finite element space on the mesh. This is an enriched space
   //    where the enrichment degree depends on the dimension of the space.
   FiniteElementCollection *testc = new L2_FECollection(t, mesh -> Dimension());
   FiniteElementSpace *testspace = new FiniteElementSpace(mesh, testc);
//   cout << "Number of unknowns: " << testspace->GetVSize() << endl;

   // 4. Set up the linear form F(.) which corresponds to the right-hand side
   //    of the FEM linear system, which in this case is (f,phi_i) where f=1.0 is
   //    given by the function f_exact and phi_i are the basis functions in the
   //    finite element fespace.
   LinearForm *F = new LinearForm(testspace);
   ConstantCoefficient one(1.0);
   FunctionCoefficient rhs_coeff(&exact_rhs);
   F->AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
   F->Assemble();


   // 5. Set up bilinear form for stiffness matrix S  i.e. v^T S v = ||v||^2_(H^1)
   BilinearForm *S = new BilinearForm(testspace);
   S->AddDomainIntegrator(new DiffusionIntegrator(one));
   S->AddDomainIntegrator(new MassIntegrator(one));
   S->Assemble();
   S->Finalize();
   const SparseMatrix &SMS = S->SpMat();

   // 6. Define the first hybrid part of the trial finite element space on the mesh.
   //    This space contains the non interfacial unknowns and has the essential BC.
   FiniteElementCollection *x0c = new H1_FECollection(p, mesh -> Dimension());
   FiniteElementSpace *x0space = new FiniteElementSpace(mesh, x0c);

   // 7. Define the vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction xbnd(x0space);
   xbnd = 0.0;

   // 8. Set up the mixed bilinear form for the non interfacial unknowns
   //    Apply the essential BC.
   MixedBilinearForm *B0 = new MixedBilinearForm(x0space,testspace);
   B0->AddDomainIntegrator(new DiffusionIntegrator(one));
//   B0->AddDomainIntegrator(new MassIntegrator(one));  //For reaction diffusion
   B0->Assemble();
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   B0->EliminateTrialDofs(ess_bdr, xbnd, *F);
   B0->Finalize();
   const SparseMatrix &SMB0 = B0->SpMat();

   // 9. Define the second hybrid part of the trial finite element space on the mesh.
   //    This space contains the interfacial unknowns and does not have essential BC.
   FiniteElementCollection *xhatc = new NT_FECollection(n, mesh -> Dimension());
   FiniteElementSpace *xhatspace = new FiniteElementSpace(mesh, xhatc);

   // 10. Set up the mixed bilinear form for the interfacial unknowns
   MixedBilinearForm *Bhat = new MixedBilinearForm(xhatspace,testspace);
   Bhat->AddFaceIntegrator(new NTMassJumpIntegrator());
   Bhat->Assemble();
   Bhat->Finalize();
//   cout << "NT Jump Integrator finalized\n" << endl << flush;
   const SparseMatrix &cSMBhat = Bhat->SpMat();

   // 11. Make Hybrid Operator
   HybridDPGOperator *A = new HybridDPGOperator(SMB0,cSMBhat,SMS);

   // 12. Make Hybrid vectors and RHS
   Vector bigX((SMS.Width())+(SMB0.Width())+(cSMBhat.Width()));
   bigX=0.0;
   Vector b((SMS.Width())+(SMB0.Width())+(cSMBhat.Width()));
   b=0.0;
   Vector bEPS(b.GetData()+SMB0.Width()+cSMBhat.Width(),SMS.Width());
//   bEPS=(F->GetData());
   bEPS.Add(1.0,*F);

   // 13. Use MINRES solver but first make a preconditioner
   SparseMatrix &SMBhat = Bhat->SpMat();
   BilinearForm *S0 = new BilinearForm(x0space);
   S0->AddDomainIntegrator(new DiffusionIntegrator(one));
   S0->AddDomainIntegrator(new MassIntegrator(one));
   S0->Assemble();
   S0->Finalize();
   SparseMatrix &SMS0 = S0->SpMat();
   BilinearForm *Sinv = new BilinearForm(testspace);
   Sinv->AddDomainIntegrator(new DiffusionIntegrator(one));
   Sinv->AddDomainIntegrator(new MassIntegrator(one));
   Sinv->AssembleDomainInverse();
   Sinv->Finalize();
   SparseMatrix &SMSinv = Sinv->SpMat();
   HybridDPGPreconditioner *B = new HybridDPGPreconditioner(SMS0,SMBhat,SMSinv);

   MINRES(*A,*B, b,bigX,1,10000, 1e-12, 1e-24);
   GridFunction x;
   x.Update(x0space, bigX, 0);

   cout << "\n|| u_h(SP Exact Schur PC) - u ||_{L^2} = " << x.ComputeL2Error(e_sol)
        << '\n' << endl;


   // 14. Save the refined mesh and the solution. This output can be viewed later
   //    using GLVis: "glvis -m refined.mesh -g sol.gf".
   {

      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 15. (Optional) Send the solution by socket to a GLVis server.
   char vishost[] = "localhost";
   int  visport   = 19916;
   osockstream sol_sock(visport, vishost);
   sol_sock << "solution\n";
   sol_sock.precision(8);
   mesh->Print(sol_sock);
   x.Save(sol_sock);
   sol_sock.send();

   // 16. Free the used memory.
   delete B;
   delete Sinv;
   delete A;
   delete Bhat;
   delete xhatspace;
   delete xhatc;
   delete S0;
   delete B0;
   delete x0space;
   delete x0c;
   delete S;
   delete F;
   delete testspace;
   delete testc;
   delete mesh;

   return 0;
}

const double kappa = M_PI;

double exact_sol(Vector &x)
{
   int dim = x.Size();
   if (dim == 2)
   {
      return sin(kappa*(x(0)))*sin(kappa*(x(1)));
   }
   else if (dim == 3)
   {
      return sin(kappa*(x(0)))*sin(kappa*(x(1)))*sin(kappa*(x(2)));
   }
   return 0.;
}

double exact_rhs(Vector &x)
{
   int dim = x.Size();
   if (dim == 2)
   {
      return 2*kappa*kappa*sin(kappa*(x(0)))*sin(kappa*(x(1)));
   }
   else if (dim == 3)
   {
      return (3*kappa*kappa*
              sin(kappa*(x(0)))*sin(kappa*(x(1)))*sin(kappa*(x(2))));
   }
   return 0.;
}

