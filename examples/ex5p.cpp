//                                MFEM Example 5
//
// Compile with: make ex5p
//
// Sample runs:  mpirun -np 4 ex5p ../data/square-disc.mesh
//               mpirun -np 4 ex5p ../data/star.mesh
//               mpirun -np 4 ex5p ../data/beam-tet.mesh
//               mpirun -np 4 ex5p ../data/beam-hex.mesh
//               mpirun -np 4 ex5p ../data/escher.mesh
//               mpirun -np 4 ex5p ../data/fichera.mesh
//
// Description:  This example code solves a simple 2D/3D mixed Darcy problem
//               corresponding to the saddle point system
//               k*u + grad p = f
//               div u        = g
//               with boundary condition -p = <given pressure>.
//               Here, we use a given exact solution (u,p)
//               and compute the corresponding r.h.s. (f,g).  We discretize with
//               Raviart-Thomas finite elements (velocity u)
//               and piecewise discontinuous polynomials (pressure p).
//
//               The example demonstrates the use of the BlockOperator class.
//
//               We recommend viewing examples 1-4 before viewing this example.

#include <fstream>
#include "mfem.hpp"

int main (int argc, char *argv[])
{
	// 1. Initialize MPI
	int num_procs, myid;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	bool verbose(0 == myid );
	StopWatch chrono;

	   Mesh *mesh;

	   if (argc == 1)
	   {
	      if (myid == 0)
	         cout << "\nUsage: mpirun -np <np> ex1p <mesh_file>\n" << endl;
	      MPI_Finalize();
	      return 1;
	   }

	   // 2. Read the (serial) mesh from the given mesh file on all processors.
	   //    We can handle triangular, quadrilateral, tetrahedral or hexahedral
	   //    elements with the same code.
	   ifstream imesh(argv[1]);
	   if (!imesh)
	   {
	      if (myid == 0)
	         cerr << "\nCan not open mesh file: " << argv[1] << '\n' << endl;
	      MPI_Finalize();
	      return 2;
	   }
	   mesh = new Mesh(imesh, 1, 1);
	   imesh.close();

	   // 3. Refine the serial mesh on all processors to increase the resolution. In
	   //    this example we do 'ref_levels' of uniform refinement. We choose
	   //    'ref_levels' to be the largest number that gives a final mesh with no
	   //    more than 10,000 elements.
	   {
	      int ref_levels =
	         (int)floor(log(10000./mesh->GetNE())/log(2.)/mesh->Dimension());
	      for (int l = 0; l < ref_levels; l++)
	         mesh->UniformRefinement();
	   }

	   // 4. Define a parallel mesh by a partitioning of the serial mesh. Refine
	   //    this mesh further in parallel to increase the resolution. Once the
	   //    parallel mesh is defined, the serial mesh can be deleted.
	   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
	   delete mesh;
	   {
	      int par_ref_levels = 2;
	      for (int l = 0; l < par_ref_levels; l++)
	         pmesh->UniformRefinement();
	   }


   // 5. Define a parallel finite element space on the parallel mesh. Here we
   //    use the lowest order Raviart-Thomas finite elements, but we can easily
   //    swich to higher-order spaces by changing the value of p.
   int order(0);
   FiniteElementCollection * hdiv_coll(new RT_FECollection(order,3));
   FiniteElementCollection * l2_coll(new L2_FECollection(order,3));

   ParFiniteElementSpace * R_space = new ParFiniteElementSpace(pmesh,  hdiv_coll);
   ParFiniteElementSpace * W_space = new ParFiniteElementSpace(pmesh,    l2_coll);

   int dimR(R_space->GlobalTrueVSize());
   int dimW(W_space->GlobalTrueVSize());

	if(verbose)
	{
		std::cout << "***********************************************************\n";
		std::cout << "dim(R) = " << dimR << std::endl;
		std::cout << "dim(W) = " << dimW << std::endl;
		std::cout << "dim(R+W) = " << dimR + dimW << std::endl;
		std::cout << "***********************************************************\n";
	}

	ConstantCoefficient k(1);

	// Darcy augmented operator and preconditioner
	/*
	 * \D = [ M_k        B^T ]
	 *      [  B         0   ]
	 *
	 * \P = [ M_k         0            ]
	 *      [  0   (D diag(M_{k})^-1 D^T)^-1 ]
	 *
	 *  M_k = \int_\Omega k u_h \cdot v_h d\Omega   \u_h, v_h \in R_h
	 *  W   = \int_\Omega p_h q_h d\Omega            p_h, q_h \in W_h
	 *
	 *  B   = \int_\Omega \div u_h q_h d\Omega       u_h \in R_h, q_h \in W_h
	 *  D   : R_h --> W_h s.t. p_h = D u_h --> p_h = \div u_h aka B = WD
	 */

	ParBilinearForm * mVarf( new ParBilinearForm(R_space));
	ParBilinearForm * wVarf( new ParBilinearForm(W_space) );
	ParDiscreteLinearOperator * discreteDiv(
			new ParDiscreteLinearOperator(R_space, W_space)
	                                           );

	HypreParMatrix *M, *W, *D;

	mVarf->AddDomainIntegrator(new VectorFEMassIntegrator(k));
	mVarf->Assemble();
	mVarf->Finalize();
	M = mVarf->ParallelAssemble();

	wVarf->AddDomainIntegrator(new MassIntegrator);
	wVarf->Assemble();
	wVarf->Finalize();
	W = wVarf->ParallelAssemble();

	discreteDiv->AddDomainInterpolator( new DivergenceInterpolator);
	discreteDiv->Assemble();
	discreteDiv->Finalize();
	D = discreteDiv->ParallelAssemble();

	HypreParMatrix * B = ParMult(W,D);
	HypreParMatrix * BT = B->Transpose();
	HypreParMatrix * MkinvBt = B->Transpose();
	HypreParVector * Md = new HypreParVector(MPI_COMM_WORLD, M->GetGlobalNumRows(), M->GetRowStarts());
	M->GetDiag(*Md);

	MkinvBt->InvLeftScaling(*Md);
	HypreParMatrix * S = ParMult(B, MkinvBt );

	HypreParMatrix * A = M;
	HypreSolver * invA, *invS;
    invA = new HypreDiagScale(*A);
	invS = new HypreBoomerAMG(*S);

	invA->iterative_mode = false;
	invS->iterative_mode = false;

	BlockOperator * darcyOp = new BlockOperator(2,2);
	darcyOp->SetBlock(0,0, A, A->GetNumRows(), A->GetNumRows() );
	darcyOp->SetBlock(0,1, BT, BT->GetNumRows(), B->GetNumRows() );
	darcyOp->SetBlock(1,0, B, B->GetNumRows(), BT->GetNumRows() );
	darcyOp->Finalize();

    BlockDiagonalPreconditioner * darcyPr = new BlockDiagonalPreconditioner(2);
	darcyPr->SetDiagonalBlock(0, invA, A->GetNumRows() );
	darcyPr->SetDiagonalBlock(1, invS, S->GetNumRows() );
	darcyPr->Finalize();

	HypreParVector f_u(MPI_COMM_WORLD, A->GetGlobalNumRows(), A->RowPart());
	HypreParVector f_p(MPI_COMM_WORLD, B->GetGlobalNumRows(), B->RowPart());
	f_u = 0.0;
	f_p = 0.0;

	Array<int> row_starts_monolithic;
	BlockHypreParVector * rhs( stride(f_u, f_p, row_starts_monolithic) );
	BlockHypreParVector * xtrue( new BlockHypreParVector(*rhs) );
	BlockHypreParVector * x( new BlockHypreParVector(*rhs) );
	xtrue->Randomize( 0 );
	double factor=xtrue->Norml2();
	(*xtrue) *= (1./factor);
	darcyOp->Mult(*xtrue, *rhs);
	(*x) = 0.0;

	int maxIter(500);
	double rtol(1.e-6);
	double atol(1.e-10);

	chrono.Clear();
    chrono.Start();
    MINRESSolver solver(MPI_COMM_WORLD);
	solver.SetAbsTol(atol);
	solver.SetRelTol(rtol);
	solver.SetMaxIter(maxIter);
	solver.SetOperator(*darcyOp);
	solver.SetPreconditioner(*darcyPr);
	solver.SetPrintLevel(myid == 0);
	solver.Mult(*rhs, *x);
    chrono.Stop();

    if(myid == 0)
    {
    	if( solver.GetConverged() )
    		std::cout << "MINRES converged in " << solver.GetNumIterations() << " with a residual norm of " << solver.GetFinalNorm() << ".\n";
    	else
    		std::cout << "MINRES did not converge in " << solver.GetNumIterations() << ". Residual norm is " << solver.GetFinalNorm() << ".\n";
            std::cout << "MINRES solver took " << chrono.RealTime() << " s. \n";
    }

	BlockHypreParVector * r( new BlockHypreParVector(*rhs)  );
	BlockHypreParVector * Pr( new BlockHypreParVector(*rhs) );
	darcyOp->Mult(*x, *r);
	subtract(*rhs, *r, *r);
	darcyPr->Mult(*r, *Pr);

	double residual_norm(r->Norml2());
	double presidual_norm(Pr->Norml2());
	double rhs_norm(rhs->Norml2());

	subtract(*x, *xtrue, *x);
	double error = x->Norml2();

	if(verbose)
	{
		std::cout<<"|| Ax_n - b ||_2 = "<<residual_norm<<"\n";
		std::cout<<"|| Ax_n - b ||_2/||b||_2 = "<<residual_norm/rhs_norm<<"\n";
		std::cout<<"|| P^-1(Ax_n - b) ||_2 = "<<presidual_norm<<"\n";
		std::cout<<"|| x_true - x_n ||_2/||x_true||_2 = "<<error<<"\n";
	}

   // 15. Free the used memory.;
	delete darcyOp;
	delete darcyPr;
	delete x;
	delete xtrue;
	delete rhs;
	delete r;
	delete Pr;
	delete invA;
	delete invS;
	delete S;
	delete Md;
	delete MkinvBt;
	delete BT;
	delete B;
	delete M;
	delete W;
	delete D;
	delete mVarf;
	delete wVarf;
	delete discreteDiv;
	delete W_space;
	delete R_space;
	delete l2_coll;
	delete hdiv_coll;
	delete pmesh;


   MPI_Finalize();

   return 0;
}

