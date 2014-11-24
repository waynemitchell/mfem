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
//               - div u      = g
//               with natural boundary condition -p = <given pressure>.
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
using namespace std;

using namespace mfem;

// Define the analytical solution and forcing terms / bc
void uFun_ex(const Vector & x, Vector & u);
double pFun_ex(Vector & x);
void fFun(const Vector & x, Vector & f);
double gFun(Vector & x);
double f_natural(Vector & x);

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
			cout << "\nUsage: mpirun -np <np> ex5p <mesh_file>\n" << endl;
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
	//    switch to higher-order spaces by changing the value of *order*.
	int order(0);
	FiniteElementCollection * hdiv_coll(new RT_FECollection(order,pmesh->Dimension()));
	FiniteElementCollection * l2_coll(new L2_FECollection(order,pmesh->Dimension()));

	ParFiniteElementSpace * R_space = new ParFiniteElementSpace(pmesh,  hdiv_coll);
	ParFiniteElementSpace * W_space = new ParFiniteElementSpace(pmesh,    l2_coll);

	int dimR(R_space->GlobalTrueVSize());
	int dimW(W_space->GlobalTrueVSize());

	if(verbose)
	{
		std::cout << "***********************************************************\n";
		std::cout << "dim(R) = " << dimR << "\n";
		std::cout << "dim(W) = " << dimW << "\n";
		std::cout << "dim(R+W) = " << dimR + dimW << "\n";
		std::cout << "***********************************************************\n";
	}

	// 6. Define the two BlockStructure of the problem. block_offsets is used for Vector based
	// on dof (like ParGridFunction or ParLinearForm), block_trueOffstes is used for Vector based
	// on trueDof (HypreParVector for the rhs and solution of the linear system).
	// The offsets computed here are local to the processor.

	Array<int> block_offsets(3); //Number of variables + 1
	block_offsets[0] = 0;
	block_offsets[1] = R_space->GetVSize();
	block_offsets[2] = W_space->GetVSize();
	block_offsets.PartialSum();

	Array<int> block_trueOffsets(3); //Number of variables + 1
	block_trueOffsets[0] = 0;
	block_trueOffsets[1] = R_space->TrueVSize();
	block_trueOffsets[2] = W_space->TrueVSize();
	block_trueOffsets.PartialSum();


	// 7. Define the coefficients, analytical solution, and rhs of the PDE
	ConstantCoefficient k( 1. );

	VectorFunctionCoefficient fcoeff(pmesh->Dimension(), fFun);
	FunctionCoefficient fnatcoeff(f_natural);
	FunctionCoefficient gcoeff(gFun);

	VectorFunctionCoefficient ucoeff(pmesh->Dimension(), uFun_ex);
	FunctionCoefficient pcoeff(pFun_ex);

	// 8. Define the parallel grid function and parallel linear forms, solution vector and rhs
	BlockVector x(block_offsets), rhs(block_offsets);
	BlockVector trueX(block_trueOffsets), trueRhs(block_trueOffsets);

	ParLinearForm * fform( new ParLinearForm );
	fform->Update(R_space, rhs.GetBlock(0), 0);
	fform->AddDomainIntegrator( new VectorFEDomainLFIntegrator(fcoeff));
	fform->AddBoundaryIntegrator( new VectorFEBoundaryFluxLFIntegrator(fnatcoeff));
	fform->Assemble();
	fform->ParallelAssemble(trueRhs.GetBlock(0));

	ParLinearForm * gform( new ParLinearForm );
	gform->Update(W_space, rhs.GetBlock(1), 0);
	gform->AddDomainIntegrator( new DomainLFIntegrator(gcoeff));
	gform->Assemble();
	gform->ParallelAssemble(trueRhs.GetBlock(1));

	// 9. Assemble the finite element matrices for the Darcy operator
	/*
	 * \D = [ M        B^T ]
	 *      [ B         0   ]
	 *
	 *  where:
	 *
	 * 	M = \int_\Omega k u_h \cdot v_h d\Omega   u_h, v_h \in R_h
	 *  B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
	 *
	 */

	ParBilinearForm * mVarf( new ParBilinearForm(R_space));
	ParMixedBilinearForm * bVarf(new ParMixedBilinearForm(R_space, W_space));

	HypreParMatrix *M, *B;

	mVarf->AddDomainIntegrator(new VectorFEMassIntegrator(k));
	mVarf->Assemble();
	mVarf->Finalize();
	M = mVarf->ParallelAssemble();

	bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
	bVarf->Assemble();
	bVarf->Finalize();
	B = bVarf->ParallelAssemble();
	(*B) *= -1;

	HypreParMatrix * BT = B->Transpose();
	
	BlockOperator * darcyOp = new BlockOperator(block_trueOffsets);
	darcyOp->SetBlock(0,0, M);
	darcyOp->SetBlock(0,1, BT);
	darcyOp->SetBlock(1,0, B );

	// 10. Construct the operators for preconditioner
	/*
	 *  \P = [ diag(M)         0         ]
	 *       [  0       B diag(M)^-1 B^T ]
	 *
	 *  Here we use HypreBoomerAMG to approximate the inverse of the pressure Schur Complement
	 */
	 
	HypreParMatrix * MinvBt = B->Transpose();
	HypreParVector * Md = new HypreParVector(MPI_COMM_WORLD, M->GetGlobalNumRows(), M->GetRowStarts());
	M->GetDiag(*Md);

	MinvBt->InvScaleRows(*Md);
	HypreParMatrix * S = ParMult(B, MinvBt );

	HypreSolver * invM, *invS;
	invM = new HypreDiagScale(*M);
	invS = new HypreBoomerAMG(*S);

	invM->iterative_mode = false;
	invS->iterative_mode = false;
	
	BlockDiagonalPreconditioner * darcyPr = new BlockDiagonalPreconditioner(block_trueOffsets);
	darcyPr->SetDiagonalBlock(0, invM );
	darcyPr->SetDiagonalBlock(1, invS );

	// 11. Solve the linear system with MINRES.
	//     Check the norm of the unpreconditioned residual.

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
	solver.SetPrintLevel(verbose);
	trueX = 0.;
	solver.Mult(trueRhs, trueX);
	chrono.Stop();

	if(verbose)
	{
		if( solver.GetConverged() )
			std::cout << "MINRES converged in " << solver.GetNumIterations() << " iterations with a residual norm of " << solver.GetFinalNorm() << ".\n";
		else
			std::cout << "MINRES did not converge in " << solver.GetNumIterations() << " iterations. Residual norm is " << solver.GetFinalNorm() << ".\n";
		std::cout << "MINRES solver took " << chrono.RealTime() << " s. \n";
	}

	// 12. Extract the parallel grid function corresponding to the finite element
	//     approximation X. This is the local solution on each processor. Compute L2 error norms.
	ParGridFunction *u(new ParGridFunction);
	ParGridFunction *p(new ParGridFunction);
	u->Update(R_space, x.GetBlock(0), 0);
	p->Update(W_space, x.GetBlock(1), 0);
	u->Distribute( &(trueX.GetBlock(0)) );
	p->Distribute( &(trueX.GetBlock(1)) );

	int order_quad = max(2, 2*order+1);
	const IntegrationRule * irs[Geometry::NumGeom];
	for(int i(0); i < Geometry::NumGeom; ++i)
		irs[i] = &(IntRules.Get(i, order_quad));

	double err_u  = u->ComputeL2Error(ucoeff, irs);
	double norm_u = ComputeGlobalLpNorm(2, ucoeff, *pmesh, irs);
	double err_p = p->ComputeL2Error(pcoeff, irs);
	double norm_p = ComputeGlobalLpNorm(2, pcoeff, *pmesh, irs);

	if(verbose)
	{
		std::cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
		std::cout << "|| p_h - p_ex || / || p_ex || = " << err_p / norm_p << "\n";
	}

	// 13. Save the refined mesh and the solution in parallel. This output can
	//     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol_*".
	{
		ostringstream mesh_name, u_name, p_name;
		mesh_name << "mesh." << setfill('0') << setw(6) << myid;
		u_name << "sol_u." << setfill('0') << setw(6) << myid;
		p_name << "sol_p." << setfill('0') << setw(6) << myid;

		ofstream mesh_ofs(mesh_name.str().c_str());
		mesh_ofs.precision(8);
		pmesh->Print(mesh_ofs);

		ofstream u_ofs(u_name.str().c_str());
		u_ofs.precision(8);
		u->Save(u_ofs);

		ofstream p_ofs(p_name.str().c_str());
		p_ofs.precision(8);
		p->Save(p_ofs);
	}

	// 14. (Optional) Send the solution by socket to a GLVis server.
	{
		char vishost[] = "localhost";
		int  visport   = 19916;
		socketstream u_sock(vishost, visport);
		u_sock << "parallel " << num_procs << " " << myid << "\n";
		u_sock << "solution\n";
		u_sock.precision(8);
		pmesh->Print(u_sock);
		u->Save(u_sock);
		socketstream p_sock(vishost, visport);
		p_sock << "parallel " << num_procs << " " << myid << "\n";
		p_sock << "solution\n";
		p_sock.precision(8);
		pmesh->Print(p_sock);
		p->Save(p_sock);
	}

	// 15. Free the used memory.;

	delete fform;
	delete gform;
	delete u;
	delete p;
	delete darcyOp;
	delete darcyPr;
	delete invM;
	delete invS;
	delete S;
	delete Md;
	delete MinvBt;
	delete BT;
	delete B;
	delete M;
	delete mVarf;
	delete bVarf;
	delete W_space;
	delete R_space;
	delete l2_coll;
	delete hdiv_coll;
	delete pmesh;


	MPI_Finalize();

	return 0;
}

void uFun_ex(const Vector & x, Vector & u)
{
	double xi(x(0));
	double yi(x(1));
	double zi( 0. );
	if( x.Size() == 3)
		zi = x(2);

	u(0) = - exp(xi)*sin(yi)*cos(zi);
	u(1) = - exp(xi)*cos(yi)*cos(zi);

	if(x.Size() == 3 )
		u(2) = exp(xi)*sin(yi)*sin(zi);
}


//Change me if you need
double pFun_ex( Vector & x)
{
	double xi(x(0));
	double yi(x(1));
	double zi( 0. );

	if( x.Size() == 3)
		zi = x(2);

	return exp(xi)*sin(yi)*cos(zi);
}


void fFun(const Vector & x, Vector & f)
{
	f = 0.;
}

double gFun( Vector & x)
{
	if( x.Size() == 3)
		return -pFun_ex(x);
	else
		return 0;
}

double f_natural(Vector & x)
{
	return (-pFun_ex(x));
}

