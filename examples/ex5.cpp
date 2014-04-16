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

void uFun_ex(const Vector & x, Vector & u);
double pFun_ex(Vector & x);
void fFun(const Vector & x, Vector & f);
double gFun(Vector & x);
double f_natural(Vector & x);

int main (int argc, char *argv[])
{
	StopWatch chrono;
	Mesh *mesh;

	if (argc == 1)
	{
		cout << "\nUsage: mpirun -np <np> ex1p <mesh_file>\n" << endl;
		return 1;
	}

	// 2. Read the (serial) mesh from the given mesh file on all processors.
	//    We can handle triangular, quadrilateral, tetrahedral or hexahedral
	//    elements with the same code.
	ifstream imesh(argv[1]);
	if (!imesh)
	{
		cerr << "\nCan not open mesh file: " << argv[1] << '\n' << endl;
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

	// 5. Define a finite element space on the mesh. Here we
	//    use the lowest order Raviart-Thomas finite elements, but we can easily
	//    swich to higher-order spaces by changing the value of p.
	int order(0);
	FiniteElementCollection * hdiv_coll(new RT_FECollection(order,mesh->Dimension()));
	FiniteElementCollection * l2_coll(new L2_FECollection(order,mesh->Dimension()));

	FiniteElementSpace * R_space = new FiniteElementSpace(mesh,  hdiv_coll);
	FiniteElementSpace * W_space = new FiniteElementSpace(mesh,    l2_coll);

	int dimR(R_space->GetVSize());
	int dimW(W_space->GetVSize());
        int dimRW(dimR + dimW );

	std::cout << "***********************************************************\n";
	std::cout << "dim(R) = " << dimR << std::endl;
	std::cout << "dim(W) = " << dimW << std::endl;
	std::cout << "dim(R+W) = " << dimRW << std::endl;
	std::cout << "***********************************************************\n";

	// 6. Coefficients
	ConstantCoefficient k( 1. );

	VectorFunctionCoefficient fcoeff(mesh->Dimension(), fFun);
	FunctionCoefficient fnatcoeff(f_natural);
	FunctionCoefficient gcoeff(gFun);

	VectorFunctionCoefficient ucoeff(mesh->Dimension(), uFun_ex);
	FunctionCoefficient pcoeff(pFun_ex);

	// 7. Define the grid function and linear forms
	Vector x( dimRW ), rhs( dimRW );
        GridFunction u, p;
        u.Update(R_space, x, 0);
        p.Update(W_space, x, dimR );

	LinearForm * fform( new LinearForm );
        fform->Update(R_space, rhs, 0);
	fform->AddDomainIntegrator( new VectorFEDomainLFIntegrator(fcoeff));
	fform->AddBoundaryIntegrator( new VectorFEBoundaryFluxLFIntegrator(fnatcoeff));
	fform->Assemble();

	LinearForm * gform( new LinearForm );
        gform->Update(W_space, rhs, dimR );
	gform->AddDomainIntegrator( new DomainLFIntegrator(gcoeff));
	gform->Assemble();

	// 8. Assemble the finite element matrices for the Darcy operator
	// Darcy augmented operator and preconditioner
	/*
	 * \D = [ M        B^T ]
	 *      [ B         0   ]
	 *
	 * 	M = \int_\Omega k u_h \cdot v_h d\Omega   \u_h, v_h \in R_h
	 *  W = \int_\Omega p_h q_h d\Omega            p_h, q_h \in W_h
	 *
	 *  B   = -\int_\Omega \div u_h q_h d\Omega       u_h \in R_h, q_h \in W_h
	 *  D   : R_h --> W_h s.t. p_h = D u_h --> p_h = \div u_h aka B = -WD
	 */

	BilinearForm * mVarf( new BilinearForm(R_space));
	BilinearForm * wVarf( new BilinearForm(W_space) );
	DiscreteLinearOperator * discreteDiv(
			new DiscreteLinearOperator(R_space, W_space)
	);

	mVarf->AddDomainIntegrator(new VectorFEMassIntegrator(k));
	mVarf->Assemble();
	mVarf->Finalize();
        SparseMatrix & M( mVarf->SpMat() );

	wVarf->AddDomainIntegrator(new MassIntegrator);
	wVarf->Assemble();
	wVarf->Finalize();
        SparseMatrix & W( wVarf->SpMat() );

	discreteDiv->AddDomainInterpolator( new DivergenceInterpolator);
	discreteDiv->Assemble();
	discreteDiv->Finalize();
        SparseMatrix & D(discreteDiv->SpMat());

	SparseMatrix * B = Mult( W, D );
	(*B) *= -1;
	SparseMatrix * BT = Transpose(*B);


	// 8. Create the operators for preconditioner
	/*
	 *  \P = [ diag(M)         0         ]
	 *       [  0       B diag(M)^-1 B^T ]
	 *
	 *  We use HypreBoomerAMG to approximate the inverse of the pressure Schur Complement
	 */
	SparseMatrix * MinvBt = Transpose(*B);
	Vector Md(M.Size());
	M.GetDiag(Md);

        for(int i = 0; i < Md.Size(); ++i)
	    MinvBt->ScaleRow(i, 1./Md(i));
	SparseMatrix * S = Mult(*B, *MinvBt );

	Solver * invM, *invS;
	invM = new DSmoother(M);
	invS = new GSSmoother(*S);

	invM->iterative_mode = false;
	invS->iterative_mode = false;

	// 8. Setup the BlockOperators and solve the linear system with MINRES
	BlockMatrix darcyMatrix(2,2);
	darcyMatrix.SetBlock(0,0, M );
	darcyMatrix.SetBlock(0,1, *BT );
	darcyMatrix.SetBlock(1,0, *B);
	darcyMatrix.Finalize();

	BlockDiagonalPreconditioner darcyPrec(2);
	darcyPrec.SetDiagonalBlock(0, invM, M.Size() );
	darcyPrec.SetDiagonalBlock(1, invS, S->Size() );
	darcyPrec.Finalize();

	x = 0.0;

	int maxIter(500);
	double rtol(1.e-6);
	double atol(1.e-10);

	chrono.Clear();
	chrono.Start();
	MINRESSolver solver;
	solver.SetAbsTol(atol);
	solver.SetRelTol(rtol);
	solver.SetMaxIter(maxIter);
	solver.SetOperator(darcyMatrix);
	solver.SetPreconditioner(darcyPrec);
	solver.SetPrintLevel(0);
	solver.Mult(rhs, x);
	chrono.Stop();

	if( solver.GetConverged() )
		std::cout << "MINRES converged in " << solver.GetNumIterations() << " with a residual norm of " << solver.GetFinalNorm() << ".\n";
	else
		std::cout << "MINRES did not converge in " << solver.GetNumIterations() << ". Residual norm is " << solver.GetFinalNorm() << ".\n";
		std::cout << "MINRES solver took " << chrono.RealTime() << " s. \n";

	Vector r( rhs.Size()  );
	darcyMatrix.Mult(x, r);
	subtract(rhs, r, r);


	double residual_norm(r.Norml2());
	double rhs_norm(rhs.Norml2());

	std::cout<<"|| Ax_n - b ||_2 = "<<residual_norm<<"\n";
	std::cout<<"|| Ax_n - b ||_2/||b||_2 = "<<residual_norm/rhs_norm<<"\n";

	// Update the grid functions and compute L2 error norms.
	u.Update(R_space, x, 0);
	p.Update(W_space, x, dimR );

	int order_quad = 2;
	Array<const IntegrationRule *> irs;
	for(int i(0); i < Geometry::NumGeom; ++i)
		irs.Append(&(IntRules.Get(i, order_quad)));

	double err_u  = u.ComputeL2Error(ucoeff, irs.GetData());
	double norm_u = ComputeLpNorm(2., ucoeff, *mesh, irs.GetData());
	double err_p = p.ComputeL2Error(pcoeff, irs.GetData());
	double norm_p = ComputeLpNorm(2., pcoeff, *mesh, irs.GetData());

	std::cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
	std::cout << "|| p_h - p_ex || / || p_ex || = " << err_p / norm_p << "\n";

	// 12. Save the refined mesh and the solution in parallel. This output can
	//     be viewed later using GLVis: "glvis -m mesh -g sol".
	{
		ofstream mesh_ofs("ex5.mesh");
		mesh_ofs.precision(8);
		mesh->Print(mesh_ofs);

		ofstream u_ofs("sol_u.gf");
		u_ofs.precision(8);
		u.Save(u_ofs);

		ofstream p_ofs("sol_p.gf");
		p_ofs.precision(8);
		p.Save(p_ofs);
	}

	// 13. (Optional) Send the solution by socket to a GLVis server.
	{
		char vishost[] = "localhost";
		int  visport   = 19916;
		socketstream u_sock(vishost, visport);
		u_sock << "solution\n";
		u_sock.precision(8);
		mesh->Print(u_sock);
		u.Save(u_sock);
		socketstream p_sock(vishost, visport);
		p_sock << "solution\n";
		p_sock.precision(8);
		mesh->Print(p_sock);
		p.Save(p_sock);
	}

	// 14. Free the used memory.;
	delete fform;
	delete gform;
	delete invM;
	delete invS;
	delete S;
	delete MinvBt;
	delete BT;
	delete B;
	delete mVarf;
	delete wVarf;
	delete discreteDiv;
	delete W_space;
	delete R_space;
	delete l2_coll;
	delete hdiv_coll;
	delete mesh;

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

