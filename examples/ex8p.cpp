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


class RAPOperator : public Operator
{
public:
	RAPOperator(Operator * Rt_, Operator *A_, Operator *P_, int size):
		Operator(size),
		Rt(Rt_),
		A(A_),
		P(P_),
		Px(P->Size() ),
		APx(A->Size() )
	{

	}

	void Mult(const Vector & x, Vector & y) const
	{
		P->Mult(x,Px);
		A->Mult(Px,APx);
		Rt->MultTranspose(APx, y);
	}

	void MultTranspose(const Vector & x, Vector & y) const
	{
		Rt->Mult(x, APx);
		A->MultTranspose(APx,Px);
		P->MultTranspose(Px, y);
	}
private:
	Operator * Rt;
	Operator * A;
	Operator * P;
	mutable Vector Px;
	mutable Vector APx;
};

int main(int argc, char *argv[])
{
	int num_procs, myid;

	// 1. Initialize MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

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

   // 4. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = 0;
      for (int l = 0; l < par_ref_levels; l++)
         pmesh->UniformRefinement();
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

   ParFiniteElementSpace *x0_space   = new ParFiniteElementSpace(pmesh, x0_fec);
   ParFiniteElementSpace *xhat_space = new ParFiniteElementSpace(pmesh, xhat_fec);
   ParFiniteElementSpace *test_space = new ParFiniteElementSpace(pmesh, test_fec);

   // 5. Set up the linear form F(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (f,phi_i) where f=1.0 and
   //    phi_i are the basis functions in the finite element fespace.

   ConstantCoefficient one(1.0);
   ParLinearForm * F = new ParLinearForm(test_space);
   F->AddDomainIntegrator(new DomainLFIntegrator(one));
   F->Assemble();

   ParGridFunction * x0 = new ParGridFunction(x0_space);
   *x0 = 0.;

   // 6. Set up the mixed bilinear form for the non interfacial unknowns, B0,
   //    the mixed bilinear form for the interfacial unknowns, Bhat,
   //    the stiffness matrix and its inverse on the discontinuous test space, S and Sinv,
   //    the stiffness matrix on the continuous trial space, S0.

   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   Array<int> ess_dof;
   x0_space->GetEssentialVDofs(ess_bdr, ess_dof);

   ParMixedBilinearForm *B0 = new ParMixedBilinearForm(x0_space,test_space);
   B0->AddDomainIntegrator(new DiffusionIntegrator(one));
   B0->Assemble();
   B0->EliminateEssentialBCFromTrialDofs(ess_dof, *x0, *F);
   B0->Finalize();

   ParMixedBilinearForm *Bhat = new ParMixedBilinearForm(xhat_space,test_space);
   Bhat->AddFaceIntegrator(new NTMassJumpIntegrator());
   Bhat->Assemble();
   Bhat->Finalize();

   ParBilinearForm *Sinv = new ParBilinearForm(test_space);
   Sinv->AddDomainIntegrator(new DiffusionIntegrator(one));
   Sinv->AddDomainIntegrator(new MassIntegrator(one));
   Sinv->AssembleDomainInverse();
   Sinv->Finalize();

   ParBilinearForm *S0 = new ParBilinearForm(x0_space);
   S0->AddDomainIntegrator(new DiffusionIntegrator(one));
   S0->AddDomainIntegrator(new MassIntegrator(one));
   S0->Assemble();
   S0->EliminateEssentialBC(ess_bdr);
   S0->Finalize();

   HypreParMatrix * matB0   = B0->ParallelAssemble();
   HypreParMatrix * matBhat = Bhat->ParallelAssemble();
   HypreParMatrix * matSinv = Sinv->ParallelAssemble();
   HypreParMatrix * matS0   = S0->ParallelAssemble();

   // 4. Define the block structure of the problem, by creating the offset variable.
   // Also allocate two BlockVector objects to store the solution and rhs.

   enum {x0_var, xhat_var, NVAR};

   int true_s0 = x0_space->TrueVSize();
   int true_s1 = xhat_space->TrueVSize();
   int true_s_test = test_space->TrueVSize();

   Array<int> true_offsets(NVAR+1);
   true_offsets[0] = 0;
   true_offsets[1] = true_s0;
   true_offsets[2] = true_s0+true_s1;

   Array<int> true_offsets_test(2);
   true_offsets_test[0] = 0;
   true_offsets_test[1] = true_s_test;

   BlockVector x(true_offsets), b(true_offsets);
   x = 0.;
   b = 0.;


   // 7. Set up the 1x2 block Least Squares DPG operator B = [ B0   Bhat ]
   //    and the normal equation operator A = B^t Sinv B

   BlockOperator B(true_offsets_test, true_offsets);
   B.SetBlock(0,0,matB0);
   B.SetBlock(0,1,matBhat);

   RAPOperator A(&B, matSinv, &B, true_offsets.Last() );

   // 8. Set up a block-diagonal preconditioner for the 2x2 normal equation
   //
   //        [ S0^{-1}     0     ]
   //        [   0     Shat^{-1} ]      Shat = (Bhat^T Sinv Bhat)
   //
   //    corresponding to the primal (x0), interfacial (x1) unknowns.

   HypreParMatrix * Shat = RAP(matSinv, matBhat);

   HypreBoomerAMG * S0inv = new HypreBoomerAMG(*matS0);
   // This is not a good preconditioner, should we try ADS instead?
   FiniteElementCollection *rt_fec = new RT_FECollection(nt_order, dim);
   ParFiniteElementSpace * rt_space = new ParFiniteElementSpace(pmesh, rt_fec);
   Solver * Shatinv;
   if(dim==2)
	   Shatinv = new HypreAMS(*Shat, rt_space);
   else
	   Shatinv = new HypreADS(*Shat, rt_space);

   BlockDiagonalPreconditioner P(true_offsets);
   P.SetDiagonalBlock(0, S0inv);
   P.SetDiagonalBlock(1, Shatinv);

   // 9. Compute the reduced rhs for the Normal Equation problem and
   //    compute the solution using PCG iterative solver.
   //    Check the weighted norm of residual for the DPG least square problem
   //    Wrap the primal variable in a GridFunction for visualization purposes.
   Vector SinvF(true_s_test);
   HypreParVector * trueF = F->ParallelAssemble();
   matSinv->Mult(*trueF,SinvF);
   B.MultTranspose(SinvF, b);

   CGSolver pcg(MPI_COMM_WORLD);
   pcg.SetOperator(A);
   pcg.SetPreconditioner(P);
   pcg.SetMaxIter(300);
   pcg.SetPrintLevel(myid == 0);
   pcg.SetRelTol(1e-6);
   pcg.Mult(b,x);

   HypreParVector * LSres = test_space->NewTrueDofVector();
   HypreParVector * tmp = test_space->NewTrueDofVector();
   B.Mult(x, *LSres);
   LSres->Add(-1., *trueF);
   matSinv->Mult(*LSres, *tmp);
   double res2 = InnerProduct(LSres, tmp);
   if(myid == 0)
	   std::cout << " || B0*x0 + Bhat*xhat - F ||_{S^-1} = " << sqrt(res2) << "\n";
   delete tmp;
   delete LSres;

   x0->Distribute( &(x.GetBlock(x0_var)) );
   // 10. Save the refined pmesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   {
	  ostringstream mesh_name, sol_name;
	  mesh_name << "mesh." << setfill('0') << setw(6) << myid;
	  sol_name << "sol." << setfill('0') << setw(6) << myid;

	  ofstream mesh_ofs(mesh_name.str().c_str());
	  mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x0->Save(sol_ofs);
   }

   // 11. (Optional) Send the solution by socket to a GLVis server.
   char vishost[] = "localhost";
   int  visport   = 19916;
   osockstream sol_sock(visport, vishost);
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock << "solution\n";
   sol_sock.precision(8);
   pmesh->Print(sol_sock);
   x0->Save(sol_sock);
   sol_sock.send();

   // 12. Free the used memory.
   delete trueF;
   delete Shatinv;
   delete rt_fec;
   delete rt_space;
   delete S0inv;
   delete Shat;
   delete matB0;
   delete matBhat;
   delete matSinv;
   delete matS0;
   delete Bhat;
   delete B0;
   delete Sinv;
   delete S0;
   delete x0;
   delete F;
   delete test_space;
   delete test_fec;
   delete xhat_space;
   delete xhat_fec;
   delete x0_space;
   delete x0_fec;
   delete pmesh;

   MPI_Finalize();

   return 0;
}
