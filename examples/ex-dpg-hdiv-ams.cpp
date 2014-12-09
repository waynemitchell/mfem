//                                MFEM Example HdivSQUARE - Parallel Version
//
// Compile with: make exHdivSQUAREp
//
// Sample runs:  exHdivSQUAREp ../data/square1.mesh
//               exHdivSQUAREp ../data/square2.mesh
//
// Description:  This example code demonstrates the use of a preconditioner in
//               saddlepoint DPG in MFEM to solve the Laplace problem -Delta u
//               = f with homogeneous Dirichlet boundary conditions on a unit
//               square. It computes the error and reports the number of outer
//               MINRES iterations.  The preconditioner uses the Hdiv norm.
//
//               The precondition uses BoomerAMG for fast preconditioning on
//               the primal unknowns, and AMS to precondition the Hdiv norm.
//

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
   SparseMatrix *Sinvpc;
   int part0[2];
   int part1[2];
   int epart[2];
   int s0, s1, s2;
   int dim;
   Mesh *mesh;
   HypreParVector *hx0;
   HypreParVector *hy0;
   HypreParMatrix *hX0pc;
   HypreSolver *amg0;
   int row1[2];
   int col1[2];
   int part2[2];
   HYPRE_Solver ams;
   HypreParMatrix *hXhatpc;
   FiniteElementSpace *edge_fespace;
   FiniteElementCollection *vert_fec;
   FiniteElementSpace *vert_fespace;
   int vpart[2];
   HypreParVector *hB;
   HypreParVector *hX;
   DiscreteLinearOperator *grad;
   SparseMatrix *Pi, *Pix, *Piy, *Piz;
   HypreParMatrix *hG;
   DiscreteLinearOperator *id_ND;
   HypreParMatrix *hPix;
   HypreParMatrix *hPiy;
   FiniteElementSpace *vert_fespace_d;


public:
   HybridDPGPreconditioner(SparseMatrix & X0pc,
                           SparseMatrix & Xhatpc,
                           SparseMatrix & A3,
                           int pdim, FiniteElementSpace *pfespace, Mesh *pmesh)
   {
//   Preconditioner for Test Space
      Sinvpc=&A3;
//   Member Data
      s0=X0pc.Width();
      s1=Xhatpc.Width();
      s2=Sinvpc->Width();
      size=s0+s1+s2;
      dim=pdim;
      mesh=pmesh;
      edge_fespace=pfespace;
      // int p = edge_fespace->GetOrder(0);
      int p = 1;
      part0[0] = 0; part0[1] = s0;
      part1[0] = 0; part1[1] = s1;
      epart[0] = 0; epart[1] = s1;
      row1[0] = 0; row1[1] = s2;
      col1[0] = 0; col1[1] = s1;
      part2[0] = 0; part2[1] = s2;
//   Recipe and Preconditioner for Primal Space
      hX0pc = new HypreParMatrix(s0, part0, &X0pc);
      amg0 = new HypreBoomerAMG(*hX0pc);
      hx0 = new HypreParVector(s0, part0);
      hy0 = new HypreParVector(s0, part0);
//   Recipe for Preconditioner for Numerical Trace Space
      hXhatpc = new HypreParMatrix(s1, part1, &Xhatpc);
      hB = new HypreParVector(s1, epart);
      hX = new HypreParVector(s1, epart);
//   Setup AMS
      int cycle_type       = 13;
      int rlx_type         = 2;
      int rlx_sweeps       = 1;
      double rlx_weight    = 1.0;
      double rlx_omega     = 1.0;
      int amg_coarsen_type = 10;
      int amg_agg_levels   = 0;
      int amg_rlx_type     = 8;
      double theta         = 0.25;
      int amg_interp_type  = 6;
      int amg_Pmax         = 4;
      HYPRE_AMSCreate(&ams);
      HYPRE_AMSSetDimension(ams, dim); // 2D H(div) and 3D H(curl) problems
      HYPRE_AMSSetTol(ams, 0.0);
      HYPRE_AMSSetMaxIter(ams, 1); // use as a preconditioner
      HYPRE_AMSSetCycleType(ams, cycle_type);
      HYPRE_AMSSetPrintLevel(ams, 1);
      HYPRE_AMSSetSmoothingOptions(ams, rlx_type, rlx_sweeps, rlx_weight, rlx_omega);
      HYPRE_AMSSetAlphaAMGOptions(ams, amg_coarsen_type, amg_agg_levels, amg_rlx_type,
                                  theta, amg_interp_type, amg_Pmax);
      HYPRE_AMSSetBetaAMGOptions(ams, amg_coarsen_type, amg_agg_levels, amg_rlx_type,
                                 theta, amg_interp_type, amg_Pmax);


      vert_fec = new H1_FECollection(p, dim);
      vert_fespace = new FiniteElementSpace(mesh, vert_fec);
      int vsize = vert_fespace->GetVSize();
      vpart[0] = 0; vpart[1] = vsize;
//   Generate, convert and set vertex coordinates
      GridFunction xc(vert_fespace), yc(vert_fespace);
      if (p == 1)
      {
         double *c;
         for (int i = 0; i < mesh->GetNV(); i++)
         {
            c = mesh -> GetVertex(i);
            xc(i) = -c[1];
            yc(i) = c[0];
         }
      }
      HypreParVector hXc(vsize, vpart), hYc(vsize, vpart);
      for (int i = 0; i < vsize; i++)
      {
         hXc(i) = xc(i);
         hYc(i) = yc(i);
      }
      if (p == 1)
         HYPRE_AMSSetCoordinateVectors(ams, (HYPRE_ParVector) hXc,
                                       (HYPRE_ParVector) hYc, NULL);
      else
         HYPRE_AMSSetCoordinateVectors(ams, NULL, NULL, NULL);
      // generate, convert and set discrete gradient
      grad = new DiscreteLinearOperator(vert_fespace, edge_fespace);
      grad->AddDomainInterpolator(new GradientInterpolator);
      grad->Assemble();
      grad->Finalize();
      SparseMatrix &G = grad->SpMat();
      hG = new HypreParMatrix(s1, vsize, epart, vpart, &G);
      HYPRE_AMSSetDiscreteGradient(ams, (HYPRE_ParCSRMatrix) *hG);
      // generate, convert and set Nedelec interpolation
      if (cycle_type < 10)
         vert_fespace_d = new FiniteElementSpace(mesh, vert_fec, dim,
                                                 Ordering::byVDIM);
      else
         vert_fespace_d = new FiniteElementSpace(mesh, vert_fec, dim,
                                                 Ordering::byNODES);
      id_ND = new DiscreteLinearOperator(vert_fespace_d, edge_fespace);
      id_ND->AddDomainInterpolator(new IdentityInterpolator);
      id_ND->Assemble();
      Piz=NULL;
      if (cycle_type < 10)
      {
         id_ND->Finalize();
         Pi = &id_ND->SpMat();
      }
      else
      {
         Array2D<SparseMatrix *> Pi_blocks;
         id_ND->GetBlocks(Pi_blocks);
         Pix = Pi_blocks(0,0);
         Piy = Pi_blocks(0,1);
      }
      hPix = new HypreParMatrix(s1, vsize, epart, vpart, Pix);
      hPiy = new HypreParMatrix(s1, vsize, epart, vpart, Piy);
      if (p > 1)
         HYPRE_AMSSetInterpolations(ams, NULL, (HYPRE_ParCSRMatrix) *hPix,
                                    (HYPRE_ParCSRMatrix) *hPiy, NULL);

/*      // define and apply a hypre-PCG solver with the above AMS preconditioner
        HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &pcg_solver);
        HYPRE_ParCSRPCGSetTol(pcg_solver, 1e-10);
        HYPRE_ParCSRPCGSetMaxIter(pcg_solver, 1);
        HYPRE_ParCSRPCGSetPrintLevel(pcg_solver, 2);
        HYPRE_ParCSRPCGSetPrecond(pcg_solver, HYPRE_AMSSolve, HYPRE_AMSSetup, ams);
//     HYPRE_ParCSRPCGSetup(pcg_solver, (HYPRE_ParCSRMatrix) (*hXhatpc),
//                          (HYPRE_ParVector) (*hB), (HYPRE_ParVector) (*hX));
*/
      HYPRE_AMSSetup(ams, (HYPRE_ParCSRMatrix) (*hXhatpc),
                     (HYPRE_ParVector) (*hB), (HYPRE_ParVector) (*hX));


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
//     Fast version of
//     CG(*X0pc, x0, y0,0,200000,1e-24,0); // only pos def if ess BC dof already removed
      for (int i = 0; i < s0; i++)
      {
         (*hx0)(i) = x0(i);
//      (*hy0)(i) = 0.0;
      }
      amg0->Mult(*hx0, *hy0);
      for (int i = 0; i < s0; i++)
         y0(i) = (*hy0)(i);//   Fast version of
//   CG(*Xhatpc, x1, y1,0,200000,1e-24,0);
      if (dim != 2)
         mfem_error("Currently working only in 2D!");

      for (int i = 0; i < s1; i++)
      {
         (*hB)(i) = x1(i);
         (*hX)(i) = 0.0;
      }
//      HYPRE_ParCSRPCGSetup(pcg_solver, (HYPRE_ParCSRMatrix) *hXhatpc,
//                           (HYPRE_ParVector) *hB, (HYPRE_ParVector) *hX);
      HYPRE_AMSSolve(ams, (HYPRE_ParCSRMatrix) *hXhatpc,
                     (HYPRE_ParVector) *hB, (HYPRE_ParVector) *hX);
      for (int i = 0; i < s1; i++)
         y1(i) = (*hX)(i);
//     cout << "Inverted Xhat\n" << endl << flush;

//Use already computed local inversion of test norm
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
      delete hX0pc;
      delete amg0;
      delete hx0;
      delete hy0;
      delete hXhatpc;
      delete edge_fespace;
      delete vert_fec;
      delete vert_fespace;
      delete hB;
      delete hX;
      delete grad;
      delete Pi;
      delete Pix;
      delete Piy;
      if (Piz!=NULL)
         delete Piz;
      delete hG;
      delete id_ND;
      delete hPix;
      delete hPiy;
      delete vert_fespace_d;

   }
};

int main (int argc, char *argv[])
{
   int num_procs, myid;

   // 0. Initialize MPI
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);


   // 1. Read the 2D or 3D mesh from the given mesh file. In this example, we
   //    can handle triangular, quadrilateral, tetrahedral or hexahedral meshes
   //    with the same code.

   Mesh *mesh;

   if (argc == 1)
   {
      cout << "\nUsage: exHdivSQUAREp <mesh_file>\n" << endl;
      return 1;
   }


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
   if (0 < n) {
      cout << "Warning, AMS Preconditioner currently does not support higher order numerical trace spaces. " << endl << flush ;
   }
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
   FiniteElementCollection *xhatc = new RT_FECollection(n, mesh -> Dimension());
   FiniteElementSpace *xhatspace = new FiniteElementSpace(mesh, xhatc);

   // 10. Set up the mixed bilinear form for the interfacial unknowns
   MixedBilinearForm *Bhat = new MixedBilinearForm(xhatspace,testspace);
   Bhat->AddFaceIntegrator(new NTMassJumpIntegrator());
   Bhat->Assemble();
   Bhat->Finalize();
//   cout << "NT Jump Integrator finalized\n" << endl << flush;
   const SparseMatrix &SMBhat = Bhat->SpMat();

   // 11. Make Hybrid Operator
   HybridDPGOperator *A = new HybridDPGOperator(SMB0,SMBhat,SMS);

   // 12. Make Hybrid vectors and RHS
   Vector bigX((SMS.Width())+(SMB0.Width())+(SMBhat.Width()));
   bigX=0.0;
   Vector b((SMS.Width())+(SMB0.Width())+(SMBhat.Width()));
   b=0.0;
   Vector bEPS(b.GetData()+SMB0.Width()+SMBhat.Width(),SMS.Width());
//   bEPS=(F->GetData());
   bEPS.Add(1.0,*F);

   // 13. Use MINRES solver but first make a preconditioner
   BilinearForm *Shat = new BilinearForm(xhatspace);
   Shat->AddDomainIntegrator(new DivDivIntegrator(one));
   Shat->AddDomainIntegrator(new VectorFEMassIntegrator(one));
   Shat->Assemble();
   Shat->Finalize();
   SparseMatrix &SMShat = Shat->SpMat();

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

   HybridDPGPreconditioner *B = new HybridDPGPreconditioner(SMS0,SMShat,SMSinv,dim,xhatspace,mesh);

   MINRES(*A,*B, b,bigX,1,10000, 1e-12, 1e-24);
   GridFunction x;
   x.Update(x0space, bigX, 0);

   cout << "\n|| u_h(SP AMG AMS Hdiv PC) - u ||_{L^2} = " << x.ComputeL2Error(e_sol)
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
   delete Shat;
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
   cout << "\n Deleting Preconditioner ... \n" << flush ;
   delete B;

   MPI_Finalize();

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

