//                                MFEM Example 5
//
// Compile with: make ex5
//
// Sample runs:  ex5 ../data/square1.mesh
//               ex5 ../data/square2.mesh
//
// Description:  This example code demonstrates the use of DPG MFEM to define
//               a simple isoparametric finite element discretization of the
//               Laplace problem -Delta u = f with homogeneous Dirichlet
//               boundary conditions. Specifically, we discretize with the
//               FE space coming from the mesh (linear by default, quadratic
//               for quadratic curvilinear mesh, NURBS for NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of boundary conditions on all boundary edges, and the optional
//               connection to the GLVis tool for visualization.

#include <fstream>
#include "mfem.hpp"

double zero_sol(Vector &);
double exact_sol(Vector &);
double exact_rhs(Vector &);
void exact_grad(const Vector &, Vector &);

/// Classes for Hybrid DPG driver operators
class DPGdlsOperator : public Operator
{
protected:
   SparseMatrix *B0;
   SparseMatrix *Bhat;
   SparseMatrix *Sinv;
   int s0,s1,s2;

public:
   DPGdlsOperator(SparseMatrix & A0,
                  SparseMatrix & A1,
                  SparseMatrix & A2)
   {
      B0=&A0;
      Bhat=&A1;
      Sinv=&A2;
      s0=B0->Width();
      s1=Bhat->Width();
      s2=Sinv->Width();
      size=s0+s1;
   }

   /// Returns the size of the input
   inline int Size() const { return size; }

   /// Operator application
   virtual void Mult (const Vector & x, Vector & y) const
   {
      Vector y0(y.GetData(),s0);
      Vector x0(x.GetData(),s0);
      Vector y1(y.GetData()+s0,s1);
      Vector x1(x.GetData()+s0,s1);
      Vector u(s2);
      B0->Mult(x0,u);
      Bhat->AddMult(x1,u);
      Vector v(s2);
      Sinv->Mult(u,v);
      B0->MultTranspose(v,y0);
      Bhat->MultTranspose(v,y1);
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

   virtual ~DPGdlsOperator() { }
};

class DPGspOperator : public Operator
{
protected:
   SparseMatrix *B0;
   SparseMatrix *Bhat;
   SparseMatrix *S;
   int s0,s1,s2;

public:
   DPGspOperator(SparseMatrix & A0,
                 SparseMatrix & A1,
                 SparseMatrix & A2)
   {
      B0=&A0;
      Bhat=&A1;
      S=&A2;
      s0=B0->Width();
      s1=Bhat->Width();
      s2=S->Width();
      size=s0+s1+s2;
   }

   /// Returns the size of the input
   inline int Size() const { return size; }

   /// Operator application
   virtual void Mult (const Vector & x, Vector & y) const
   {
      Vector y0(y.GetData(),s0);
      Vector x0(x.GetData(),s0);
      Vector y1(y.GetData()+s0,s1);
      Vector x1(x.GetData()+s0,s1);
      Vector y2(y.GetData()+s0+s1,s2);
      Vector x2(x.GetData()+s0+s1,s2);
      B0->MultTranspose(x2,y0);
//     cout << "MINRES Mult y0 complete " << endl << flush;
      Bhat->MultTranspose(x2,y1);
//     cout << "MINRES Mult y1 complete " << endl << flush;
      S->Mult(x2,y2);
      B0->AddMult(x0,y2);
      Bhat->AddMult(x1,y2);
//     cout << "MINRES Mult y2 complete " << endl << flush;
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

   virtual ~DPGspOperator() { }
};


/// Class for Adaptive Hybrid DPG preconditioners:
class HybridDPGPreconditioner : public Solver
{
protected:
   SparseMatrix *X0pc;
   SparseMatrix *Xhatpc;
   SparseMatrix *Sinvpc;
   int s0,s1,s2,Hdiv;


public:
   HybridDPGPreconditioner(SparseMatrix & A1,
                           SparseMatrix & A2,
                           SparseMatrix & A3,
                           int pHdiv)
   {
      X0pc=&A1;
      Sinvpc=&A3;
      Hdiv=pHdiv;
      if (1==Hdiv)
         Xhatpc=&A2;
      else {
//     To use Schur Complement norm, pass in SMBhat as second argument
         SparseMatrix *BhatT=Transpose(A2);
         Xhatpc=RAP(*Sinvpc,*BhatT);
         delete BhatT;
      }
      s0=X0pc->Width();
      s1=Xhatpc->Width();
      s2=Sinvpc->Width();
      size=s0+s1+s2;
   }

   /// Returns the size of the input
   inline int Size() const { return size; }

   /// Operator application
   virtual void Mult (const Vector & x, Vector & y) const
   {
      y=0.0;
      Vector y0(y.GetData(),s0);
      Vector x0(x.GetData(),s0);
      Vector y1(y.GetData()+s0,s1);
      Vector x1(x.GetData()+s0,s1);
      Vector y2(y.GetData()+s0+s1,s2);
      Vector x2(x.GetData()+s0+s1,s2);
      CG(*X0pc, x0, y0,0,100000,1e-24,0);
      CG(*Xhatpc, x1, y1,0,100000,1e-24,0);
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
//     To use Schur Complement norm, pass in SMBhat as second argument
      if(1!=Hdiv)
         delete Xhatpc;
   }
};


int main(int argc, char *argv[])
{
   Mesh *mesh;

   if (argc == 1)
   {
      cout << "\nUsage: exDPG3 <mesh_file>\n" << endl;
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
   char c = '1';
   // 2. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(50000./mesh->GetNE())/log(2.)/mesh->Dimension());
//      mesh->PrintCharacteristics();
      cout << "Enter ref. levels = " << flush;
      cin >> ref_levels;
      for (int l = 0; l < ref_levels; l++)
      {
//         cout << "refinement level " << l + 1 << " ... " << flush;
         mesh->UniformRefinement();
//         cout << "done." << endl;
//         cout << endl;
//         mesh->PrintCharacteristics();
      }
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
   int dls=1;
//   cout << "Enter 1 to solve DLS: " << flush ;  cin >> dls;
   int spnoPC;
   spnoPC=0;
//   spnoPC=1;
//   cout << "Enter 1 to solve SP no PC: " << flush ;  cin >> spnoPC;
   int Hdiv;
   Hdiv=0;
   if( 1!=spnoPC) {
      Hdiv=1;
//     cout << "Enter 1 to solve PC with Hdiv norm: " << flush ;  cin >> Hdiv;
      if ((0 < n)&&(1==Hdiv))
         cout << "Warning, Hdiv norm (for PC) currently does not support higher order numerical trace spaces. " << endl << flush ;
   }

   int Galerkin;
//   Galerkin=0;
   Galerkin=1;
//     cout << "Enter 1 to solve the Galerkin problem: " << flush ;  cin >> Galerkin;

   // print the boudary element orientations
   if (0 && mesh->Dimension() == 3)
   {
      cout << "\nBoundary element orientations:\n";
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         int f, o;
         mesh->GetBdrElementFace(i, &f, &o);
         cout << "i = " << setw(6) << i << " | f = " << setw(6) << f
              << " | o = " << setw(2) << o << '\n';
      }
      cout << endl;
   }


   // 4. Define the test finite element space on the mesh. This is an enriched space
   //    where the enrichment degree depends on the dimension of the space.
   FiniteElementCollection *testc = new L2_FECollection(t, mesh -> Dimension());
   FiniteElementSpace *testspace = new FiniteElementSpace(mesh, testc);
//   cout << "Number of unknowns: " << testspace->GetVSize() << endl;

   // 5. Define the first hybrid part of the trial finite element space on the mesh.
   //    This space contains the non interfacial unknowns and has the essential BC.
   FiniteElementCollection *x0c = new H1_FECollection(p, mesh -> Dimension());
   FiniteElementSpace *x0space = new FiniteElementSpace(mesh, x0c);

   // 6. Set up the linear form F(.) which corresponds to the right-hand side
   //    of the FEM linear system, which in this case is (f,phi_i) where f=1.0 is
   //    given by the function f_exact and phi_i are the basis functions in the
   //    finite element fespace.


   FunctionCoefficient z_sol(&zero_sol);
   FunctionCoefficient e_sol(&exact_sol);
   VectorFunctionCoefficient e_grad(dim,&exact_grad);
   ConstantCoefficient one(1.0);

   bool useL2proj = (c == '0') ? true : false;
   // bool useL2proj = true;

   GridFunction xp(x0space);
   if (!useL2proj)
   {
      xp.ProjectCoefficient(e_sol);
   }
   else
   {
      // L2 projection
      //     cout << "\nComputing the L2 projection of the exact solution:\n"
      //          << flush;
      LinearForm bp(x0space);
      bp.AddDomainIntegrator(new DomainLFIntegrator(e_sol));
//      cout << "rhs ... " << flush;
      bp.Assemble();

      BilinearForm ap(x0space);
      ap.AddDomainIntegrator(new MassIntegrator);
//      cout << "matrix ... " << flush;
      ap.Assemble();

      const SparseMatrix &Ap = ap.SpMat();

      GSSmoother Mp(Ap);
      xp = 0.0;
      cout << "solving Essential B.C. system ..." << endl;
      PCG(Ap, Mp, bp, xp, 1, 5000, 1e-30, 0.0);
      cout << " ... done. (L2 projection)" << endl;
   }

//   cout << "\nAssembling: " << flush;;

   FunctionCoefficient rhs_coeff(&exact_rhs);
   LinearForm *dlsF = new LinearForm(testspace);
   dlsF->AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
   dlsF->Assemble();
   LinearForm *spF = new LinearForm(testspace);
   spF->AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
   spF->Assemble();

   // 7. Set up bilinear form for stiffness matrix S  i.e. v^T S v = ||v||^2_(H^1)
   BilinearForm *S = new BilinearForm(testspace);
   S->AddDomainIntegrator(new DiffusionIntegrator(one));
   S->AddDomainIntegrator(new MassIntegrator(one));
   S->Assemble();
   S->Finalize();
   SparseMatrix &SMS = S->SpMat();
   BilinearForm *Sinv = new BilinearForm(testspace);
   Sinv->AddDomainIntegrator(new DiffusionIntegrator(one));
   Sinv->AddDomainIntegrator(new MassIntegrator(one));
   Sinv->AssembleDomainInverse();
   Sinv->Finalize();
   SparseMatrix &SMSinv = Sinv->SpMat();



   // 8. Define the vector x as a finite element grid function
   //    corresponding to the trial space.
   GridFunction xbnd(x0space);

   // 9. Set up the mixed bilinear form for the non interfacial unknowns
   //    Apply the essential BC.
   MixedBilinearForm *B0dls = new MixedBilinearForm(x0space,testspace);
   B0dls->AddDomainIntegrator(new DiffusionIntegrator(one));
   B0dls->Assemble();
   MixedBilinearForm *B0sp = new MixedBilinearForm(x0space,testspace);
   B0sp->AddDomainIntegrator(new DiffusionIntegrator(one));
   B0sp->Assemble();


   BilinearForm *S0 = new BilinearForm(x0space);
   S0->AddDomainIntegrator(new DiffusionIntegrator(one));
   S0->AddDomainIntegrator(new MassIntegrator(one));
   S0->Assemble();
   S0->Finalize();
   SparseMatrix &SMS0 = S0->SpMat();

   // 10. Define the second hybrid part of the trial finite element space on the mesh.
   //    This space contains the interfacial unknowns and does not have essential BC.
   FiniteElementCollection *xhatc;
   if(0==n) {
      xhatc = new RT_FECollection(n, mesh -> Dimension());
   }
   else {
      xhatc = new NT_FECollection(n, mesh -> Dimension());
   }
   FiniteElementSpace *xhatspace = new FiniteElementSpace(mesh, xhatc);

   // 11. Set up the mixed bilinear form for the interfacial unknowns
   MixedBilinearForm *Bhat = new MixedBilinearForm(xhatspace,testspace);
   Bhat->AddFaceIntegrator(new NTMassJumpIntegrator());
   Bhat->Assemble();
   Bhat->Finalize();
   SparseMatrix &SMBhat = Bhat->SpMat();
   BilinearForm *Shat;
   Shat=NULL;
   if(1==Hdiv) {
      Shat = new BilinearForm(xhatspace);
      Shat->AddDomainIntegrator(new DivDivIntegrator(one));
      Shat->AddDomainIntegrator(new VectorFEMassIntegrator(one));
      Shat->Assemble();
      Shat->Finalize();
   }

   // 12. Make Hybrid vectors and RHS
   int s0=SMS0.Width();
   int s1=SMBhat.Width();
   int s2=SMS.Width();

   Vector dlsX(s0+s1);
   Vector spX(s0+s1+s2);

   GridFunction dlsXa;
   dlsXa.Update(x0space, dlsX, 0);
   dlsXa=0.0;

   GridFunction spXa;
   spXa.Update(x0space, spX, 0);
   spXa=0.0;

   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   cout << "b.c. ... " << endl << flush;
   if (!useL2proj)
   {
      dlsXa.ProjectBdrCoefficient(e_sol, ess_bdr);
      spXa.ProjectBdrCoefficient(e_sol, ess_bdr);
   }
   else
   {
      // copy boundary dofs from the domain L2 projection
      Array<int> dofs;
      x0space->GetEssentialVDofs(ess_bdr, dofs);
      for (int i = 0; i < dofs.Size(); i++)
         if (dofs[i])
         {
            dlsXa(i) = xp(i);
            spXa(i) = xp(i);
         }
   }
   // xbnd.ProjectCoefficient(e_sol);
   B0dls->EliminateTrialDofs(ess_bdr, dlsXa, *dlsF);
   B0dls->Finalize();
   SparseMatrix &SMB0dls = B0dls->SpMat();

   B0sp->EliminateTrialDofs(ess_bdr, spXa, *spF);
   B0sp->Finalize();
   SparseMatrix &SMB0sp = B0sp->SpMat();

   // 13. Make Hybrid Operator
   DPGdlsOperator *DLS = new DPGdlsOperator(SMB0dls,SMBhat,SMSinv);
   DPGspOperator *SP = new DPGspOperator(SMB0sp,SMBhat,SMS);

   Vector dlsXb(dlsX.GetData()+s0,s1);
   dlsXb=0.0;
   Vector dlsB(s0+s1);
   Vector dlsB0(dlsB.GetData(),s0);
   Vector dlsB1(dlsB.GetData()+s0,s1);
   {
      Vector SinvF(dlsF->Size());
      SMSinv.Mult(*dlsF,SinvF);
      SMB0dls.MultTranspose(SinvF,dlsB0);
      SMBhat.MultTranspose(SinvF,dlsB1);
   }

   Vector spXb(spX.GetData()+s0,s1);
   spXb=0.0;
   Vector spXc(spX.GetData()+s0+s1,s2);
   spXc=0.0;
   Vector spB(s0+s1+s2);
   Vector spB0(spB.GetData(),s0);
   Vector spB1(spB.GetData()+s0,s1);
   Vector spB2(spB.GetData()+s0+s1,s2);
   spB0=0.0;
   spB1=0.0;
   spB2=0.0;
   spB2.Add(1.0,*spF);
//   spB2=(spF->GetData()); // ?


   // 14. Use CG solver
   GridFunction dlsx;
   if (dls) {
      CG(*DLS, dlsB, dlsX,1,10000,1e-16,0);
      dlsx.Update(x0space, dlsX, 0);
   }

   // 14. Use MINRES solver


   GridFunction spx;
   HybridDPGPreconditioner *PC=NULL;
   if (spnoPC) {
      MINRES(*SP,spB,spX,1, 10000, 1e-16, 0.); // 1e-16 is tol^2
   }
   else if(1==Hdiv) {
      SparseMatrix &SMShat = Shat->SpMat();
      PC = new HybridDPGPreconditioner(SMS0,SMShat,SMSinv,Hdiv);
      MINRES(*SP,*PC,spB,spX,1, 10000, 1e-16, 0.); // 1e-16 is tol^2
   }
   else {
      PC = new HybridDPGPreconditioner(SMS0,SMBhat,SMSinv,Hdiv);
      MINRES(*SP,*PC,spB,spX,1, 10000, 1e-16, 0.); // 1e-16 is tol^2
   }
   spx.Update(x0space, spX, 0);



   // 15. Save the refined mesh and the solution. This output can be viewed later
   //    using GLVis: "glvis -m refined.mesh -g sol.gf".
   if (dls) {

      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      dlsx.Save(sol_ofs);
   }

   // 16. (Optional) Send the solution by socket to a GLVis server.

   if (dls) {
      char vishost[] = "localhost";
      int  visport   = 19916;
      osockstream sol_sock(visport, vishost);
      sol_sock << "solution\n";
      sol_sock.precision(8);
      mesh->Print(sol_sock);
      dlsx.Save(sol_sock);
      sol_sock.send();
   }

   // 17. Compute Error


   double h1_error;
   if (dls) {
      h1_error = dlsx.ComputeH1Error(&e_sol, &e_grad, &one, 0., 1);
      cout << "\n|| u_h(DLS) - u ||_{H^1} = " << h1_error << '\n' << endl;
   }
   if (spnoPC) {
      h1_error = spx.ComputeH1Error(&e_sol, &e_grad, &one, 0., 1);
      cout << "\n|| u_h(SP no PC) - u ||_{H^1} = " << h1_error << '\n' << endl;
   }
   else if(1==Hdiv) {
      h1_error = spx.ComputeH1Error(&e_sol, &e_grad, &one, 0., 1);
      cout << "\n|| u_h(SP Exact Hdiv PC) - u ||_{H^1} = " << h1_error << '\n' << endl;
   }
   else {
      h1_error = spx.ComputeH1Error(&e_sol, &e_grad, &one, 0., 1);
      cout << "\n|| u_h(SP Exact Schur PC) - u ||_{H^1} = " << h1_error << '\n' << endl;
   }

   {
      cout << "projecton error :"
           << "\n|| u_h - u ||_{L^2} = "
           << xp.ComputeL2Error(e_sol) << '\n' << endl;
   }
   if (dls) {
      cout << "\n|| u_h(DLS) - u ||_{L^2} = " << dlsx.ComputeL2Error(e_sol)
           << '\n' << endl;
   }
   if (spnoPC) {
      cout << "\n|| u_h(SP no PC) - u ||_{L^2} = " << spx.ComputeL2Error(e_sol)
           << '\n' << endl;
      if (dls) {
         spx.Add(-1.0,dlsx);
         cout << "\n|| u_h(SP no PC) - u_h(DLS) ||_{L^2} = " << spx.ComputeL2Error(z_sol)
              << '\n' << endl;
      }
   }
   else if(1==Hdiv) {
      cout << "\n|| u_h(SP Exact Hdiv PC) - u ||_{L^2} = " << spx.ComputeL2Error(e_sol)
           << '\n' << endl;
      if (dls) {
         spx.Add(-1.0,dlsx);
         cout << "\n|| u_h(SP Exact Hdiv PC) - u_h(DLS) ||_{L^2} = " << spx.ComputeL2Error(z_sol)
              << '\n' << endl;
      }
   }
   else {
      cout << "\n|| u_h(SP Exact Schur PC) - u ||_{L^2} = " << spx.ComputeL2Error(e_sol)
           << '\n' << endl;
      if (dls) {
         spx.Add(-1.0,dlsx);
         cout << "\n|| u_h(SP Exact Schur PC) - u_h(DLS) ||_{L^2} = " << spx.ComputeL2Error(z_sol)
              << '\n' << endl;
      }
   }

   // 18.    // 17. Compute Galerkin solution
   if(1==Galerkin) {
      BilinearForm *B2 = new BilinearForm(x0space);
      B2->AddDomainIntegrator(new DiffusionIntegrator(one));
      //   cout << "matrix ... " << flush;
      B2->Assemble();
      const SparseMatrix &SMB2 = B2->SpMat();
      Vector bigX2(SMB2.Width());
      GridFunction bigXc;
      bigXc.Update(x0space, bigX2, 0);
      bigXc=0.0;

      cout << "galerkin b.c. ... " << flush;
      if (!useL2proj)
      {
         bigXc.ProjectBdrCoefficient(e_sol, ess_bdr);
      }
      else
      {
         // copy boundary dofs from the domain L2 projection
         Array<int> dofs;
         x0space->GetEssentialVDofs(ess_bdr, dofs);
         for (int i = 0; i < dofs.Size(); i++)
            if (dofs[i])
               bigXc(i) = xp(i);
      }
      // xbnd.ProjectCoefficient(e_sol);
      LinearForm *F2 = new LinearForm(x0space);
      F2->AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
      //   cout << "rhs ... " << flush;
      F2->Assemble();
      B2->EliminateEssentialBC(ess_bdr, bigXc, *F2);
      B2->Finalize();
      const SparseMatrix &SMB2b = B2->SpMat();
      GSSmoother M(SMB2b);

      cout << "galerkin solving ... " << flush;
      PCG(SMB2b, M, *F2, bigXc, 0, 10000, 1e-20, 0.0);
      cout << "\n|| u_h(Galerkin) - u ||_{L^2} = " << bigXc.ComputeL2Error(e_sol)
           << '\n' << endl;
      if (dls) {
         bigXc.Add(-1.0,dlsx);
         cout << "\n|| u_h(Galerkin) - u_h(DLS) ||_{L^2} = " << bigXc.ComputeL2Error(z_sol)
              << '\n' << endl;
      }
      delete F2;
      delete B2;
   }



   // 19. Free the used memory.
   if (NULL!=PC) delete PC;
   delete DLS;
   delete SP;
   if (NULL!=Shat) delete Shat;
   delete S0;
   delete Bhat;
   delete xhatspace;
   delete xhatc;
   delete B0dls;
   delete B0sp;
   delete x0space;
   delete x0c;
   delete Sinv;
   delete S;
   delete spF;
   delete dlsF;
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

double zero_sol(Vector &x)
{
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

void exact_grad(const Vector &x, Vector &grad)
{
   int dim = x.Size();
   double *graddata;
   if (dim == 2)
   {
      graddata=grad.GetData();
      graddata[0]=kappa*cos(kappa*(x(0)))*sin(kappa*(x(1)));
      graddata[1]=kappa*sin(kappa*(x(0)))*cos(kappa*(x(1)));
//      graddata[2]=sin(kappa*(x(0)))*sin(kappa*(x(1)));
   }
   else if (dim == 3)
   {
      graddata=grad.GetData();
      graddata[0]=kappa*cos(kappa*(x(0)))*sin(kappa*(x(1)))*sin(kappa*(x(2)));
      graddata[1]=kappa*sin(kappa*(x(0)))*cos(kappa*(x(1)))*sin(kappa*(x(2)));
      graddata[2]=kappa*sin(kappa*(x(0)))*sin(kappa*(x(1)))*cos(kappa*(x(2)));
   }
}

