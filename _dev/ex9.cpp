//                                MFEM Example 9
//
// Compile with: make ex9
//
// Description: This example code implements a advection-based DG remap
//              scheme using explicit flux correction.

#include <fstream>
#include "mfem.hpp"

int    pcg_max_iter = 20;
double pcg_rel_tol  = 1e-30;
int    vtk_subdiv;
const double eps = 1e-12;

int rho_exact_option = -1;

double rho_exact(Vector &x);
void smooth_displacement(const Vector &, Vector &);

Mesh *Extrude1D(Mesh *mesh, const int ny, const bool closed = false);
GridFunction *Extrude1DGridFunction(Mesh *mesh, Mesh *mesh2d,
                                    GridFunction *sol, const int ny);

void SocketSend(socketstream &sol_sock, Mesh *mesh, GridFunction *sol);


// Class for computing a forward Euler update using FCT

class FCT_Update
{
private:
   Mesh *mesh;
   FiniteElementSpace *fes; // for rho
   GridFunction *u;

   mutable double old_tau;
   VectorGridFunctionCoefficient vc_u;
   mutable BilinearForm M, A;
   FunctionCoefficient rho_out;
   mutable LinearForm b;
   mutable int num_calls;

public:
   FCT_Update(Mesh *_mesh, GridFunction *_u, GridFunction *rho) :
        fes(rho->FESpace()), vc_u(_u), M(fes), A(fes),
        rho_out(rho_exact), b(fes)
   {
      mesh = _mesh;
      u = _u;

      old_tau = 0.0;
      num_calls = 0;

      M.AddDomainIntegrator(new MassIntegrator);
      M.Assemble(0);
      M.Finalize(0);

      const double bt = 0.5;

      // A is -(A^T+2S) from the notes
      A.AddDomainIntegrator(new ConvectionIntegrator(vc_u));
      A.AddInteriorFaceIntegrator(
	 new TransposeIntegrator(
	    new DGTraceIntegrator(vc_u, -1.0, -bt)));


      A.Assemble(0);
      A.Finalize(0);

      b.AddBdrFaceIntegrator(
         new BoundaryFlowIntegrator(rho_out, vc_u, 1.0, -bt));
      b.Assemble();
   }

   virtual ~FCT_Update() {};

   void Assemble(double tau) const
   {
      if (tau == old_tau) return;

      GridFunction &nodes(*mesh->GetNodes());
      Vector nodes0(nodes);

      nodes.Add(tau, *u);

      A = 0.0;
      A.Assemble();

      M = 0.0;
      M.Assemble();

      b.Assemble();

      old_tau = tau;

      // Move nodes back to original locations
      nodes = nodes0;
   }

   void Step(Vector &, double, double, Table&) const;

   void Reset()
   {
      num_calls = 0;

      Assemble(0.0);
   }

   void GetNeighborhood(Table &nbrs) const
   {
      SparseMatrix &a = A.SpMat();
      nbrs.SetIJ(a.GetI(), a.GetJ(), a.Size());
   }

};

void FCT_Update::Step(Vector &x, double tau, double dtau, Table& nbrs) const
{
   cout << "FCT_Update::Step : tau = " << tau << " --> tau = "
        << tau +dtau << endl;

   num_calls++;

   Assemble(tau);

   SparseMatrix &Mmat = M.SpMat();
   GSSmoother GS(Mmat);

   SparseMatrix Minv(Mmat.Size());
   int* I = Mmat.GetI();
   int* J = Mmat.GetJ();

   // For each element, invert the block diagonal submatrix

   int ne = fes->GetNE();
   int block_size = x.Size() / ne;

   for (int nb = 0; nb < ne; nb++) {
      DenseMatrix block(block_size);

      Array<int> rv(block_size);
      Array<int> cv(block_size);
      for (int r = 0; r < block_size; r++) {
	 for (int c = 0; c < block_size; c++) {

	    int i = r +nb*block_size;
	    int j = c +nb*block_size;

	    rv[r] = i;
	    cv[c] = j;
	    block(r,c) = M(i,j);
	 }
      }

      block.Invert();

      Minv.SetSubMatrix(rv,cv,block);
   }
   Minv.Finalize();

   // Create the lumped mass matrix
   SparseMatrix ML(Mmat.Size());
   SparseMatrix MLinv(Mmat.Size());
   I = Mmat.GetI(); J = Mmat.GetJ();
   for (int i = 0; i < x.Size(); i++) {

      double sum = 0;

      for (int n = I[i]; n < I[i+1]; n++) {

	 int j = J[n];

	 sum += Mmat(i,j);
      }
      ML.Set(i,i,sum);
      MLinv.Set(i,i,1.0/sum);
   }
   ML.Finalize();
   MLinv.Finalize();

   SparseMatrix& Amat = A.SpMat();

   // Compute the matrix K =  Minv A
   SparseMatrix* Kp = ::Mult(Minv, Amat);
   SparseMatrix& K = *Kp;

   Vector dx_dtau(x.Size());

   // direct high order increment
   K.Mult(x, dx_dtau);

   Vector dxH(x.Size());
   dxH = 0.0;
   dxH.Add(dtau, dx_dtau);

   int size = x.Size();

   // Create the "diffusion matrix" D for K
   SparseMatrix D(A.Size());

   I = Amat.GetI();
   J = Amat.GetJ();

   for (int i = 0; i < size; i++) {

      for (int n = I[i]; n < I[i+1]; n++) {

	 int j = J[n];

	 double dij = 0;
	 if (i != j) {
  	    dij = max(0.0, max(-Amat(i,j), -Amat(j,i)));
	 }

	 D.Set(i, j, dij);
      }
   }
   bool skip_zeros = false;
   D.Finalize(skip_zeros);

   // Set diagonals to -sum of rows
   for (int i = 0; i < size; i++) {

      double sum = 0.0;

      for (int n = I[i]; n < I[i+1]; n++) {

	 int j = J[n];

	 if (i != j) {
	    sum += D(i,j);
	 };

      }
      D.Set(i,i,-sum);
   }

   // Compute A* = A + D
   SparseMatrix Astar(Amat.Size());
   I = Amat.GetI(); J = Amat.GetJ();
   for (int i = 0; i < size; i++) {

      for (int n = I[i]; n < I[i+1]; n++) {

	 int j = J[n];

	 Astar.Set(i,j,Amat(i,j));
      }
   }
   Astar.Finalize(false);

   Astar.Add(+1.0, D);

   // G = MLinv*A* - the complete low order operator
   SparseMatrix* Gp = ::Mult(MLinv,Astar);
   SparseMatrix& G = *Gp;

   G.Mult(x, dx_dtau);
   dx_dtau += b;

   // Low order increment
   Vector dxL(x.Size());
   dxL = 0.0;
   dxL.Add(dtau, dx_dtau);

   // advance low-order solution
   Vector xLnew(x.Size());

   xLnew = x;
   xLnew += dxL;

   // Compute the allowable bounds on changes from anti-diffusion
   Vector dx_maxp(x.Size());
   Vector dx_maxm(x.Size());

   Vector dx_maxp_old(x.Size());
   Vector dx_maxm_old(x.Size());

   for (int i = 0; i < x.Size(); i++)
   {
      int *inbrs = nbrs.GetRow(i);
      int n = nbrs.RowSize(i);
      double xmin = numeric_limits<double>::infinity();
      double xmax = -xmin;
      for (int j = 0; j < n; j++)
      {
         if (x(inbrs[j]) > xmax)
            xmax = x(inbrs[j]);
         else if (x(inbrs[j]) < xmin)
            xmin = x(inbrs[j]);
      }

      dx_maxp(i) = xmax -xLnew(i);
      dx_maxm(i) = xLnew(i) -xmin;

      dx_maxp_old(i) = xmax -x(i);
      dx_maxm_old(i) = x(i) -xmin;
   }

   // Compute the matrix of anti-diffusive fluxes
   SparseMatrix F(Amat.Size());

   I = Mmat.GetI(); J = Mmat.GetJ();
   for (int i = 0; i < size; i++) {

      for (int n = I[i]; n < I[i+1]; n++) {

	 int j = J[n];

	 double fij = M(i,j)*(dxH(i)-dxH(j)) +dtau*D(i,j)*(x(i)-x(j));

	 F.Set(i,j,fij);
      }
   }
   F.Finalize(false);


   // Compute increments from low-order soln using + and - antidiffusion

   // rowsums of F+/mi and F-/mi

   Vector dx_P(x.Size());
   Vector dx_M(x.Size());

   I = F.GetI(); J = F.GetJ();
   for (int i = 0; i < size; i++) {

      double rowsum_p = 0.0;
      double rowsum_m = 0.0;

      for (int k = I[i]; k < I[i+1]; k++) {

	 int j = J[k];

	 F(i,j) > 0.0 ? rowsum_p += F(i,j) : rowsum_m += F(i,j);
      }

      dx_P(i) = rowsum_p*MLinv(i,i);
      dx_M(i) = rowsum_m*MLinv(i,i);

   }

   // Compute per-DOF scaling factors necessary for + and - AD fluxes
   Vector alpha_p(x.Size());
   Vector alpha_m(x.Size());
   for (int i = 0; i < x.Size(); i++) {
      alpha_p(i) = 1.0;
      alpha_m(i) = 1.0;

      if (dx_P(i) > 0.0) alpha_p(i) = min(dx_maxp(i)/dx_P(i), 1.0);
      if (dx_M(i) < 0.0) alpha_m(i) = min(-dx_maxm(i)/dx_M(i), 1.0);
   }

   // Compute symmetrix flux correction scaling factors
   SparseMatrix alpha_ij(Amat.Size());
   I = F.GetI(); J = F.GetJ();
   for (int i = 0; i < size; i++) {

      for (int k = I[i]; k < I[i+1]; k++) {

	 int j = J[k];

	 double aij = 1.0;
	 if (F(i,j) >= 0) {
	    aij = min(alpha_p(i), alpha_m(j));
	 }
	 else {
	    aij = min(alpha_p(j), alpha_m(i));
	 }
    	 alpha_ij.Set(i, j, aij);
      }
   }
   alpha_ij.Finalize(false);

   // Compute the limited anti-diffusive increment

   Vector dxAL(x.Size());
   I = F.GetI(); J = F.GetJ();
   for (int i = 0; i < size; i++) {

      double rowsum = 0.0;

      for (int k = I[i]; k < I[i+1]; k++) {

	 int j = J[k];

	 rowsum += alpha_ij(i,j)*F(i,j);

      }

      dxAL(i) = rowsum*MLinv(i,i);

   }

   // Update soln with low-order and limited antidiffusive updates

   x += dxL;

   x += dxAL;

   return;
}

// Remap x_in defined on the given mesh to x_out defined on the mesh displaced
// with the given grid function u. If type=1, x corresponds to the original grid
// function (rho), otherwise it corresponds to the function moments (m).
void DG_remap(Mesh &mesh, GridFunction &u, GridFunction &x_in,
              GridFunction &x_out, int nsteps)
{
   FCT_Update fct_update(&mesh, &u, &x_in);

   double tau = 0.0;
   GridFunction x(x_in.FESpace()), rho;
   x = x_in;
   rho.Update(x.FESpace(), x, 0);

   char vishost[] = "localhost";
   int  visport   = 19916;

   GridFunction &nodes(*mesh.GetNodes());
   Vector nodes0(nodes);

   socketstream *rho_sock = NULL;

   Table nbrs;

   rho_sock = new socketstream(vishost, visport);
   rho_sock->precision(8);

   std::string str;

   fct_update.Reset();

   fct_update.GetNeighborhood(nbrs);

   x = x_in;
   tau = 0.0;

   double dtau = 1.0/nsteps;

   SocketSend(*rho_sock, &mesh, &x);
   *rho_sock  << "window_title 'FCT Solution'" << endl;
   *rho_sock  << "keys cRjlmR \n";
   *rho_sock  << "window_size 600 600" << endl;
   *rho_sock  << "viewcenter 0.25 0.0" << endl;
   *rho_sock  << "zoom 1.18" << endl;

//    printf("press enter to take step\n");
//    getline(cin, str);

   for (int i = 0; i < nsteps; i++) {

      fct_update.Step(x, tau, dtau, nbrs);

      tau += dtau;

//      ode_solver.Step(x, tau, dtau);

      nodes.Add(tau, u);
      SocketSend(*rho_sock, &mesh, &x);

//       printf("mesh\n");
//       mesh.Print();

//       printf("soln\n");
//       x.Print();

// //       for (int k = 0; k < x.Size(); k++) {
// // 	 printf("%f %f\n",nodes(k),x(k));
// //       }
      nodes = nodes0;

//      printf("press enter to move to step %d\n",i+1);
//      getline(cin, str);
   }

   nbrs.LoseData();

   x_out = x;
}

int main (int argc, char *argv[])
{
   Mesh *mesh;

   if (argc == 1)
   {
      int nx, ny;
      double xL, xR;

      cout <<
         "1) Generate 1D mesh with nx elements for the interval [xL,xR]\n"
         "2) Generate nx-by-ny mesh for the rectangle [xL,xR]x[0,1]\n"
         " --> "  << flush;
      cin >> ny;

      cout << "enter xL --> " << flush;
      cin >> xL;
      cout << "enter xR --> " << flush;
      cin >> xR;

      cout << "enter nx --> " << flush;
      cin >> nx;

      if (ny != 1)
      {
         cout << "enter ny --> " << flush;
         cin >> ny;

         mesh = new Mesh(nx, ny, Element::QUADRILATERAL, 1, xR - xL, 1.0);

         for (int i = 0; i < mesh->GetNV(); i++)
         {
            double *v = mesh->GetVertex(i);
            v[0] += xL;
         }
      }
      else
      {
         mesh = new Mesh(nx);

         for (int i = 0; i < mesh->GetNV(); i++)
         {
            double *v = mesh->GetVertex(i);
            v[0] = (1.0-v[0])*xL + v[0]*xR;
         }
      }

   }
   else
   {
      // Read the mesh from the given mesh file. We can handle triangular,
      // quadrilateral, tetrahedral or hexahedral elements with the same code.
      ifstream imesh(argv[1]);
      if (!imesh)
      {
         cerr << "can not open mesh file: " << argv[1] << endl;
         return 2;
      }
      mesh = new Mesh(imesh, 1, 1);
   }

   int dim = mesh->Dimension();

   // Refine the mesh to increase the resolution. In this example we do
   // 'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   // largest number that gives a final mesh with no more than 1000 elements.
   {
      int ref_levels =
         (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
      cout << "enter ref. levels [" << ref_levels << "] --> " << flush;
      cin >> ref_levels;
      for (int l = 0; l < ref_levels; l++)
         mesh->UniformRefinement();
   }

   // Define a finite element space on the mesh. Here we use vector finite
   // elements which are tensor products of quadratic finite elements. The
   // dimensionality of the vector finite element space is specified by the last
   // parameter of the FiniteElementSpace constructor.
  cout << "Mesh curvature: ";
   if (mesh->GetNodes())
      cout << mesh->GetNodes()->OwnFEC()->Name();
   else
      cout << "(NONE)";
   cout << endl;

   int mesh_poly_deg = 0;
   if (mesh->Dimension() == 1) {
      mesh_poly_deg = 1;
   }

   cout <<
      "Enter polynomial degree of mesh finite element space:\n"
      "0) QuadraticPos (quads only)\n"
      "p) Degree p >= 1\n"
      " --> " << flush;
   cin >> mesh_poly_deg;
   FiniteElementCollection *mesh_fec;
   if (mesh_poly_deg <= 0)
   {
      mesh_fec = new QuadraticPosFECollection;
      mesh_poly_deg = 2;
   }
   else
      mesh_fec = new H1_FECollection(mesh_poly_deg, dim);
   FiniteElementSpace *mesh_fespace = new FiniteElementSpace(mesh, mesh_fec, dim);
   mesh_fespace->BuildElementToDofTable();

   // Make the mesh curved based on the above finite element space. This means
   // that we define the mesh elements through a fespace-based transformation of
   // the reference element.
   mesh->SetNodalFESpace(mesh_fespace);

   // Choose the displacement field
   int disp_field_opt = 2;
   cout <<
      "Choose displacement field:\n"
      "1) Random\n"
      "2) Smooth\n"
      " --> " << flush;
   cin >> disp_field_opt;

   GridFunction u(mesh_fespace);
   if (disp_field_opt == 1)
   {
      // Define a vector representing the minimal local mesh size in the nodes
      Vector h0(mesh_fespace->GetNDofs());
      h0 = numeric_limits<double>::infinity();
      {
         Array<int> dofs;
         // loop over the mesh elements
         for (int i = 0; i < mesh_fespace->GetNE(); i++)
         {
            // get the local scalar element degrees of freedom in dofs
            mesh_fespace->GetElementDofs(i, dofs);
            // adjust the value of h0 in dofs based on the local mesh size
            for (int j = 0; j < dofs.Size(); j++)
               h0(dofs[j]) = min(h0(dofs[j]), mesh->GetElementSize(i));
         }
      }
      // Define a grid function corresponding to random perturbation of the
      // nodes which is zero on the boundary.
      double jitter = 0.25; // perturbation scaling factor
      cout << "Enter jitter --> " << flush;
      cin >> jitter;
      u.Randomize(729);
      u -= 0.5; // shift to random values in [-0.5,0.5]
      u *= jitter;
      {
         // scale the random values to be of order of the local mesh size
         for (int i = 0; i < mesh_fespace->GetNDofs(); i++)
            for (int d = 0; d < dim; d++)
               u(mesh_fespace->DofToVDof(i,d)) *= h0(i);

         Array<int> vdofs;
         // loop over the boundary elements
         for (int i = 0; i < mesh_fespace->GetNBE(); i++)
         {
            // get the vector degrees of freedom in the boundary element
            mesh_fespace->GetBdrElementVDofs(i, vdofs);
            // set the boundary values to zero
            for (int j = 0; j < vdofs.Size(); j++)
               u(vdofs[j]) = 0.0;
         }
      }
   }
   else if (disp_field_opt == 2)
   {
      VectorFunctionCoefficient smooth_displ_coeff(dim, smooth_displacement);
      u.ProjectCoefficient(smooth_displ_coeff);
   }
   else
      cerr << "Option not implemented" << endl;

   // Choose the discontinuous basis
   int rho_poly_deg = 2;
   cout <<
      "Enter polynomial degree for the function to remap:\n"
      " p) Discontinuous function of degree p >= 0\n"
      " --> " << flush;
   cin >> rho_poly_deg;
   int l2_fec_type = 0;
   cout <<
      "Enter type of discontinuous basis:\n"
      "0) Nodal basis, using Gauss-Legengre points\n"
      "1) Nodal basis, using Gauss-Lobatto points\n"
      "2) Bernstein (positive) basis\n"
      " --> " << flush;
   cin >> l2_fec_type;

   if (l2_fec_type == 2)
      pcg_max_iter = 100;

   FiniteElementCollection *rho_fec;
   rho_fec = new L2_FECollection(abs(rho_poly_deg), dim, l2_fec_type);
   FiniteElementSpace *rho_fespace = new FiniteElementSpace(mesh, rho_fec);
   rho_fespace->BuildElementToDofTable();

   vtk_subdiv = (rho_poly_deg == 0) ? 1 : rho_poly_deg*2;

   GridFunction rho(rho_fespace);
   int rho_opt = 1;
   cout <<
      "Choose function to remap:\n"
      "0) constant field\n"
      "1) linear field\n"
      "2) smooth sinusoidal field\n"
      "3) steep atan field\n"
      "4) step function\n"
      " --> " << flush;
   cin >> rho_opt;

   rho_exact_option = rho_opt;

   Coefficient *c_rho = new FunctionCoefficient(rho_exact);

   // use Gauss-Legendre (all interior) points to project
   // this handles mesh-aligned discontinuities
   L2_FECollection gl_fec(abs(rho_poly_deg), dim, 0);
   FiniteElementSpace gl_fes(mesh, &gl_fec);
   GridFunction gl_rho(&gl_fes);
   gl_rho.ProjectCoefficient(*c_rho);
   GridFunctionCoefficient gl_rho_coeff(&gl_rho);

   DenseMatrix I1;
   gl_fes.GetFE(0)->Project(*rho_fespace->GetFE(0),
			    *mesh->GetElementTransformation(0),
			    I1);
   // I1.Invert();
   I1.TestInversion();
   Vector v1(gl_rho, I1.Size()), v2(rho, I1.Size());
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      v1.SetData(&gl_rho(i*I1.Size()));
      v2.SetData(&rho(i*I1.Size()));
      I1.Mult(v1, v2);
   }

   // Save initial function on the original mesh
   {
      ofstream vtk_mesh("initial.vtk");
      vtk_mesh.precision(8);
      if (mesh->Dimension() == 1) {
	 Mesh* mesh2d = Extrude1D(mesh, 1);
	 mesh2d->PrintVTK(vtk_mesh, vtk_subdiv);
	 GridFunction *sol2d = Extrude1DGridFunction(mesh, mesh2d, &rho, 1);
	 sol2d->SaveVTK(vtk_mesh,"rho",vtk_subdiv);
	 delete sol2d;
	 delete mesh2d;
      }
      else {
	 mesh->PrintVTK(vtk_mesh, vtk_subdiv);
	 rho.SaveVTK(vtk_mesh, "rho", vtk_subdiv);
      }
   }

   int ode_solver_steps = 50;
   cout << "Enter number of pseudotime steps --> " << flush;
   cin >> ode_solver_steps;
   cin.ignore(numeric_limits<streamsize>::max(), '\n');

   GridFunction &x = *mesh->GetNodes();
   Vector x0(x);

   if (1)
   {
      ofstream mesh_file("ex9_initial.mesh");
      mesh_file.precision(8);
      mesh->Print(mesh_file);

      ofstream sol_file("ex9_initial.sol");
      sol_file.precision(8);
      rho.Save(sol_file);
   }

   if (1)
   {
      x += u;
      ofstream mesh_file("ex9_final.mesh");
      mesh_file.precision(8);
      mesh->Print(mesh_file);
      x = x0;
   }

   BilinearForm M(rho_fespace);
   M.AddDomainIntegrator(new MassIntegrator);
   M.Assemble(0);
   M.Finalize(0);

   GridFunction m(rho_fespace), m_new(rho_fespace);
   M.Mult(rho, m);

   Array<const IntegrationRule *> irs(Geometry::NumGeom);
   irs = NULL;
   int geom = mesh->GetElementBaseGeometry(0);
   irs[geom] = &(IntRules.Get(geom, 2*abs(rho_poly_deg) + 3));
   cout << "\nError quadrature rule: " << irs[geom]->GetNPoints() << " pts"
        << endl;

   // Compute initial mass (integral of the function)
   GridFunction one(rho_fespace);
   one = 1.0;
   double mass0 = M.InnerProduct(one, rho);
   double l2norm0 = sqrt(M.InnerProduct(rho, rho));
   cout.precision(18);
   cout << endl << "Total mass before remap: " << mass0 << endl
        << "L2 norm of the density:  " << l2norm0 << endl
        << "Proj. errors: L1   = " << rho.ComputeL1Error(*c_rho, irs) << endl
        << "              L2   = " << rho.ComputeL2Error(*c_rho, irs) << endl
        << "              Linf = " << rho.ComputeMaxError(*c_rho, irs) << endl;

   // Perform rho-based DG remap
   GridFunction rho_new(rho_fespace);
   DG_remap(*mesh, u, rho, rho_new, ode_solver_steps);

   // Move mesh to final position
   x += u;

   // rho-based DG scheme
   M = 0.0;
   M.Assemble();
   double mass1_rho = M.InnerProduct(one, rho_new);
   double l2norm1_rho = sqrt(M.InnerProduct(rho_new, rho_new));
   cout << endl << "[ rho-based DG scheme ]" << endl
        << "Total mass after remap:  " << mass1_rho << endl
        << "Remap mass loss: " << mass0 - mass1_rho << endl
        << "L2 norm loss:    " << l2norm0 - l2norm1_rho << endl
        << "Remap errors: L1   = "
        << rho_new.ComputeL1Error(*c_rho, irs) << endl
        << "              L2   = "
        << rho_new.ComputeL2Error(*c_rho, irs) << endl
        << "              Linf = "
        << rho_new.ComputeMaxError(*c_rho, irs) << endl;

   // Save final function on the final mesh
   {
      ofstream vtk_mesh("final.vtk");
      vtk_mesh.precision(8);
      mesh->PrintVTK(vtk_mesh, vtk_subdiv);
      rho_new.SaveVTK(vtk_mesh,"density",vtk_subdiv);
   }

   // Free the used memory
   delete c_rho;
   delete rho_fespace;
   delete rho_fec;
   delete mesh_fespace;
   delete mesh_fec;
   delete mesh;
}


class NodeExtrudeCoefficient : public VectorCoefficient
{
private:
   int layer, ny;
   double p[1];
   Vector tip;
public:
   NodeExtrudeCoefficient(const int _ny)
      : VectorCoefficient(2), ny(_ny), tip(p, 1) { }
   void SetLayer(const int l) { layer = l; }
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);
   virtual ~NodeExtrudeCoefficient() { }
};

void NodeExtrudeCoefficient::Eval(Vector &V, ElementTransformation &T,
                                  const IntegrationPoint &ip)
{
   // T is 1D transformation
   V.SetSize(2);
   T.Transform(ip, tip);
   V(0) = p[0];
   V(1) = (ip.y + layer) / ny;
}

class ExtrudeCoefficient : public Coefficient
{
private:
   int ny;
   Mesh *mesh1d;
   Coefficient &sol1d;
   double p[1];
   Vector tip;
public:
   ExtrudeCoefficient(Mesh *m, Coefficient &s, int _ny)
      : ny(_ny), mesh1d(m), sol1d(s), tip(p, 1) { }
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      ElementTransformation *T1d =
         mesh1d->GetElementTransformation(T.ElementNo / ny);
      T1d->SetIntPoint(&ip);
      return sol1d.Eval(*T1d, ip);
   }
   virtual void Read(istream &in) { }
   virtual ~ExtrudeCoefficient() { }
};

Mesh *Extrude1D(Mesh *mesh, const int ny, const bool closed)
{
   if (mesh->Dimension() != 1)
   {
      cerr << "Extrude1D : Not a 1D mesh!" << endl;
      return NULL;
   }

   int nvy = (closed) ? (ny) : (ny + 1);
   int nvt = mesh->GetNV() * nvy;

   Mesh *mesh2d;

   if (closed)
      mesh2d = new Mesh(2, nvt, mesh->GetNE()*ny, mesh->GetNBE()*ny);
   else
      mesh2d = new Mesh(2, nvt, mesh->GetNE()*ny,
                        mesh->GetNBE()*ny+2*mesh->GetNE());

   // vertices
   for (int i = 0; i < mesh->GetNV(); i++)
   {
      double *v = mesh->GetVertex(i);
      for (int j = 0; j < nvy; j++)
      {
         v[1] = double(j) / ny;
         mesh2d->AddVertex(v);
      }
   }
   // elements
   Array<int> vert;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      const Element *elem = mesh->GetElement(i);
      elem->GetVertices(vert);
      const int attr = elem->GetAttribute();
      for (int j = 0; j < ny; j++)
      {
         int qv[4];
         qv[0] = vert[0] * nvy + j;
         qv[1] = vert[1] * nvy + j;
         qv[2] = vert[1] * nvy + (j + 1) % nvy;
         qv[3] = vert[0] * nvy + (j + 1) % nvy;

         mesh2d->AddQuad(qv, attr);
      }
   }
   // 2D boundary from the 1D boundary
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      const Element *elem = mesh->GetBdrElement(i);
      elem->GetVertices(vert);
      const int attr = elem->GetAttribute();
      for (int j = 0; j < ny; j++)
      {
         int sv[2];
         sv[0] = vert[0] * nvy + j;
         sv[1] = vert[0] * nvy + (j + 1) % nvy;

         if (attr%2)
            Swap<int>(sv[0], sv[1]);

         mesh2d->AddBdrSegment(sv, attr);
      }
   }

   if (!closed)
   {
      // 2D boundary from the 1D elements (bottom + top)
      int nba = mesh->bdr_attributes.Max();
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         const Element *elem = mesh->GetElement(i);
         elem->GetVertices(vert);
         const int attr = nba + elem->GetAttribute();
         int sv[2];
         sv[0] = vert[0] * nvy;
         sv[1] = vert[1] * nvy;

         mesh2d->AddBdrSegment(sv, attr);

         sv[0] = vert[1] * nvy + ny;
         sv[1] = vert[0] * nvy + ny;

         mesh2d->AddBdrSegment(sv, attr);
      }
   }

   mesh2d->FinalizeQuadMesh(1, 0);

   GridFunction *nodes = mesh->GetNodes();
   if (nodes)
   {
      // duplicate the fec of the 1D mesh so that it can be deleted safely
      // along with its nodes, fes and fec
      FiniteElementCollection *fec2d;
      FiniteElementSpace *fes2d;
      const char *name = nodes->FESpace()->FEColl()->Name();
      string cname = name;
      if (cname == "Linear")
         fec2d = new LinearFECollection;
      else if (cname == "Quadratic")
         fec2d = new QuadraticFECollection;
      else if (cname == "Cubic")
         fec2d = new CubicFECollection;
      else if (!strncmp(name, "H1_", 3))
         fec2d = new H1_FECollection(atoi(name + 7), 2);
      else
      {
         cerr << "Extrude1D : The mesh uses unknown FE collection :"
              << cname << endl;
         delete mesh2d;
         return NULL;
      }
      fes2d = new FiniteElementSpace(mesh2d, fec2d, 2);
      mesh2d->SetNodalFESpace(fes2d);
      GridFunction *nodes2d = mesh2d->GetNodes();
      nodes2d->MakeOwner(fec2d);

      NodeExtrudeCoefficient ecoeff(ny);
      Vector lnodes;
      Array<int> vdofs2d;
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         ElementTransformation &T = *mesh->GetElementTransformation(i);
         for (int j = ny-1; j >= 0; j--)
         {
            fes2d->GetElementVDofs(i*ny+j, vdofs2d);
            lnodes.SetSize(vdofs2d.Size());
            ecoeff.SetLayer(j);
            fes2d->GetFE(i*ny+j)->Project(ecoeff, T, lnodes);
            nodes2d->SetSubVector(vdofs2d, lnodes);
         }
      }
   }
   return mesh2d;
}

GridFunction *Extrude1DGridFunction(Mesh *mesh, Mesh *mesh2d,
                                    GridFunction *sol, const int ny)
{
   GridFunction *sol2d;

   FiniteElementCollection *solfec2d;
   const char *name = sol->FESpace()->FEColl()->Name();
   string cname = name;
   if (cname == "Linear")
      solfec2d = new LinearFECollection;
   else if (cname == "Quadratic")
      solfec2d = new QuadraticFECollection;
   else if (cname == "Cubic")
      solfec2d = new CubicFECollection;
   else if (!strncmp(name, "H1_", 3))
      solfec2d = new H1_FECollection(atoi(name + 7), 2);
   else if (!strncmp(name, "L2_T", 4))
      solfec2d = new L2_FECollection(atoi(name + 10), 2);
   else if (!strncmp(name, "L2_", 3))
      solfec2d = new L2_FECollection(atoi(name + 7), 2);
   else
   {
      cerr << "Extrude1DGridFunction : unknown FE collection : "
           << cname << endl;
      return NULL;
   }
   FiniteElementSpace *solfes2d;
   // assuming sol is scalar
   solfes2d = new FiniteElementSpace(mesh2d, solfec2d);
   sol2d = new GridFunction(solfes2d);
   sol2d->MakeOwner(solfec2d);
   {
      GridFunctionCoefficient csol(sol);
      ExtrudeCoefficient c2d(mesh, csol, ny);
      sol2d->ProjectCoefficient(c2d);
   }
   return sol2d;
}

void SocketSend(socketstream &sol_sock, Mesh *mesh, GridFunction *sol)
{
   sol_sock << "solution\n";

   if (mesh->Dimension() > 1)
   {
      mesh->Print(sol_sock);
      sol->Save(sol_sock);
   }
   else
   {
      Mesh *mesh2d = Extrude1D(mesh, 1);
      mesh2d->Print(sol_sock);
      GridFunction *sol2d = Extrude1DGridFunction(mesh, mesh2d, sol, 1);
      sol2d->Save(sol_sock);
      delete sol2d;
      delete mesh2d;
   }

   sol_sock << flush;
}


double rho_exact(Vector &X)
{
   const int dim = X.Size();
   double x, y, z;
   x = X(0);
   y = (dim > 1) ? X(1) : 0.0;
   z = (dim > 2) ? X(2) : 0.0;

   switch (rho_exact_option) {

   case 0: // constant
      return 1.0;

   case 1: // linear
      if (1 == dim) return x;
      return x+y;

   case 2: // smooth
      if (1 == dim) return sin(M_PI*x);
      return sin(M_PI*x)*sin(M_PI*y);

   case 3: // smooth but steep
      return M_PI_2+atan(20*(y-0.5));

   case 4: // step
      return x < 0.5 ? 2.0 : 1.0;
   default:
      cout << "unknown function option." << endl;
      exit(1);
   }

   // return 1.0;
   // return x;
   // return x+y;
   // return x+y-2*x*y;
   // return x*x+4*y*y;
   // return sin(M_PI*x)*sin(M_PI*y);
   // return 1+cos(2*M_PI*x)*sin(M_PI*y);
   // return M_PI_2+atan(20*(x-0.5));
   // return M_PI_2+atan(50*((x-0.5)+sqrt(2.)*(y-0.5)));
   // return 1.0+erf(6*((x-0.5)+sqrt(2.)*(y-0.5)));
   // return erfc(80*(sqrt((x-0.5)*(x-0.5)+sqrt(2.)*(y-0.5)*(y-0.5))-0.3));
   // return exp(-640*pow(sqrt(pow(x-0.4,2.)+sqrt(2.)*pow(y-0.5,2.))-0.25, 2.));
   // return M_PI_2+atan(20*(y-0.5));
   // return sqrt(fmax(0.,1./9-(pow(x-0.5,2)+pow(y-0.5,2)))); // hemisphere
   // return sqrt(fmax(0.,1-(pow(x-0.5,2)+pow(y-0.5,2)))); // hemisphere
   // return (x < 0.5) ? 0.5 : 1.0;
   // return (x < 0.5 - eps) ? 0.5 : ((x > 0.5 + eps) ? 1.0: 0.75);
   // return (y < 0.5) ? 0.5 : 1.0;
   // return (x < 0.5) ? M_PI_2 : M_PI_2+atan(5*(x-0.5));
   // return pow(fmax(0.0, fmin(x, 1. - x)), 4.);
   // return (pow(fmax(0.0, fmin(x, 1. - x)), 4.)*
   //         pow(fmax(0.0, fmin(y, 1. - y)), 4.));
   // return sin(2*M_PI*x);
   // return sin(0.5*M_PI*x);
   // return ((sin(M_PI*x)+sin(M_PI*y))*sin(M_PI*z) +
   //         (sin(M_PI*x)+sin(M_PI*z))*sin(M_PI*y) +
   //         (sin(M_PI*y)+sin(M_PI*z))*sin(M_PI*x));
   // return x+y+z;
   // return 1+cos(2*M_PI*x)*cos(M_PI*y)*cos(3*M_PI*z);
   // return exp(-32*((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)));
   // return exp(-512*((x-0.25)*(x-0.25)+(y-0.25)*(y-0.25)));
}

void smooth_displacement(const Vector &X, Vector &u)
{
   const int dim = X.Size();
   double x, y, z, d[3];
   x = X(0);
   y = (dim > 1) ? X(1) : 0.0;
   z = (dim > 2) ? X(2) : 0.0;

   // rotation
   const double xc = 0.5, yc = 0.5;
   double fix_bdr = 16*x*(1-x)*y*(1-y);
   //double fix_bdr = 1; // do not fix the boundary
   double angle = 30.*(M_PI/180)*fix_bdr;
   double s = sin(angle), c = cos(angle);
   d[0] = xc + c*(x-xc) - s*(y-yc) - x;
   d[1] = yc + s*(x-xc) + c*(y-yc) - y;

   // // rotation, with tangential displacement at the boundary
   // double angle = 30.*(M_PI/180);
   // double s = sin(angle), c = cos(angle);
   // d[0] = 0.5 + c*(x-0.5) - s*(y-0.5) - x;
   // d[1] = 0.5 + s*(x-0.5) + c*(y-0.5) - y;
   // d[0] *= 4*x*(1-x);
   // d[1] *= 4*y*(1-y);

   // // waves 1
   // const double dx = sqrt(2./3.), dy = sqrt(1./3.), ampl = 0.025, freq = 8;
   // double dd = ampl*sin(freq*M_PI*(dx*x + dy*y));
   // d[0] = dd*dx;
   // d[1] = dd*dy;
   // // d[0] *= 4*x*(1-x);
   // // d[1] *= 4*y*(1-y);
   // d[0] *= 1. - pow(fabs(2*x - 1), 4.);
   // d[1] *= 1. - pow(fabs(2*y - 1), 4.);

   // // waves 2
   // const double dx = sqrt(3./3.), dy = sqrt(0./3.), ampl = 0.02, freq = 9;
   // double dd = ampl*sin(freq*M_PI*(dx*y - dy*x));
   // d[0] = dd*dx;
   // d[1] = dd*dy;
   // // d[0] *= 4*x*(1-x);
   // // d[1] *= 4*y*(1-y);
   // d[0] *= 1. - pow(fabs(2*x - 1), 4.);
   // d[1] *= 1. - pow(fabs(2*y - 1), 4.);

//    // 1D compression in x-direction
//    const double xc = 6./7; // 1/2 --> xc \in (0,1)
//    d[0] = x/(1.0 + (1./xc - 2.)*(1.0 - x)) - x;
//    d[1] = 0.0;

   // // 1D translation in x-direction
   // d[0] = 4./1;
   // d[1] = 0.0;

   // // 1D translation and compression: [-4,1] --> [0,3]
   // d[0] = (3./5)*(x - 1.0) + 3.0 - x;
   // d[1] = 0.0;

   // // 1D translation and expansion: [0,1] --> [2,4]
   // d[0] = 2.0*x + 2.0 - x;
   // d[1] = 0.0;

   // // 3D rotations, with tangential displacement at the boundary
   // double angle = 20.*(M_PI/180);
   // double s = sin(angle), c = cos(angle);
   // double x0, y0, z0, x1, y1, z1;
   // // transform x,y,z in [0,1]^3
   // x0 = (x - (-1.))/(1. - (-1.));
   // y0 = (y - (-1.))/(1. - (-1.));
   // z0 = (z - (-1.))/(1. - (-1.));
   // x1 = 4*x0*(1-x0)*(0.5 + c*(x0-0.5) - s*(y0-0.5) - x0) + x0;
   // y1 = 4*y0*(1-y0)*(0.5 + s*(x0-0.5) + c*(y0-0.5) - y0) + y0;
   // z1 = z0;
   // angle = -20.*(M_PI/180);
   // s = sin(angle); c = cos(angle);
   // d[0] = 4*x1*(1-x1)*(0.5 + c*(x1-0.5) - s*(z1-0.5) - x1) + x1;
   // d[1] = y1;
   // d[2] = 4*z1*(1-z1)*(0.5 + s*(x1-0.5) + c*(z1-0.5) - z1) + z1;
   // d[0] = -1. + d[0]*(1. - (-1.)) - x;
   // d[1] = -1. + d[1]*(1. - (-1.)) - y;
   // d[2] = -1. + d[2]*(1. - (-1.)) - z;

   for (int i = 0; i < dim; i++)
      u(i) = d[i];
}
