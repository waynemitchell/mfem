//                                MFEM Example 7
//
// Compile with: make ex7
//
// Sample runs:  ex7 square-disc.mesh2d
//               ex7 star.mesh2d
//
// Description: This example code implements a simple advection-based DG remap
//              scheme.

#include <fstream>
#include <limits>
#include <time.h>     // nanosleep
#include <cstdio>     // remove
#include "mfem.hpp"

// lumping and upwinding for the function-based DG method
bool lump_mass  = false;
bool lump_jump  = false;   // only has effect when lump_mass is false
bool use_mono   = true;    // use partial lumping and upwinding based on
                           // monotonicity measure (limiter must be -1)
int  limiter    = -1;
// limiter can be:
// -1 - no upwinding or limiting
//  0 - no limiter, full upwind
//  1 - minmod
//  2 - Van Leer
//  3 - MC
//  4 - superbee

int    pcg_max_iter = 20;
double pcg_rel_tol  = 1e-30;
int    vtk_subdiv;

double rho_exact(Vector &x);
void smooth_displacement(const Vector &, Vector &);

double smooth_transition(double x);
void LimitMult(const SparseMatrix &M, const Vector &x, Vector &y, int lim);
void Lump(SparseMatrix &M, const Vector &mu);
void Upwind(SparseMatrix &M, const Vector &mu);

Mesh *Extrude1D(Mesh *mesh, const int ny, const bool closed = false);
GridFunction *Extrude1DGridFunction(Mesh *mesh, Mesh *mesh2d,
                                    GridFunction *sol, const int ny);

void SocketSend(socketstream &sol_sock, Mesh *mesh, GridFunction *sol);


// Class for computing the righ-hand side of the DG remap ODE scheme
class DG_remap_ode : public TimeDependentOperator
{
private:
   Mesh *mesh;
   FiniteElementSpace *fes; // for rho
   GridFunction *u, *old_rho, *mono;
   int type; // 1=rho-based DG scheme, 2=moment-based DG scheme
   bool m_based_conserve_const;

   mutable double old_tau;
   VectorGridFunctionCoefficient vc_u;
   mutable BilinearForm M, A, AJ, APS;
   FunctionCoefficient rho_out;
   mutable LinearForm b;
   mutable int num_calls;

public:
   DG_remap_ode(Mesh *_mesh, GridFunction *_u, GridFunction *rho, int _type,
                bool rk2_const_pres)
      : TimeDependentOperator(rho->Size(), 0.0),
        fes(rho->FESpace()), vc_u(_u), M(fes), A(fes), AJ(fes), APS(fes),
        rho_out(rho_exact), b(fes)
   {
      mesh = _mesh;
      u = _u;
      old_rho = rho;
      type = _type;
      m_based_conserve_const = rk2_const_pres;

      if (use_mono && type == 1)
      {
         mono = new GridFunction(fes);
         (*mono) = 0.0;
      }
      else
         mono = NULL;

      old_tau = 0.0;
      num_calls = 0;

      if (lump_mass && type == 1)
         M.AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
      else
         M.AddDomainIntegrator(new MassIntegrator);
      M.Assemble(0);
      M.Finalize(0);

      const double bt = 0.5;

      if (type == 1)
      {
         // A is -(A^T+2S) from the notes
         A.AddDomainIntegrator(new ConvectionIntegrator(vc_u));
         BilinearForm *pAJ = lump_jump ? &AJ : &A;
         pAJ->AddInteriorFaceIntegrator(
            new TransposeIntegrator(
               new DGTraceIntegrator(vc_u, -1.0, -bt)));
         pAJ->AddBdrFaceIntegrator(
            new TransposeIntegrator(
               new DGTraceIntegrator(vc_u, -1.0, -bt)));
      }
      else
      {
         // A is the advection matrix from the notes
         A.AddDomainIntegrator(new TransposeIntegrator(
                                  new ConvectionIntegrator(vc_u, -1.0)));
         A.AddInteriorFaceIntegrator(new DGTraceIntegrator(vc_u, 1.0, -bt));
         A.AddBdrFaceIntegrator(new DGTraceIntegrator(vc_u, 1.0, -bt));

         // APS is (A+S) from the notes
         APS.AddDomainIntegrator(new TransposeIntegrator(
                                    new ConvectionIntegrator(vc_u, -1.0)));
         APS.AddInteriorFaceIntegrator(new DGTraceIntegrator(vc_u, 1.0, 0.));
         APS.AddBdrFaceIntegrator(new DGTraceIntegrator(vc_u, 1.0, 0.));
         APS.Assemble(0);
         APS.Finalize(0);
#if 0
         Vector ones(APS.Size()), ans(APS.Size());
         ones = 1.0;
         APS.SpMat().MultTranspose(ones, ans);
         cout << "|(A+S)^t 1| = " << ans.Norml2() << endl;
#endif
      }
      A.Assemble(0);
      A.Finalize(0);
      if (lump_jump && type == 1)
      {
         AJ.Assemble(0);
         AJ.Finalize(0);
      }

      b.AddBdrFaceIntegrator(
         new BoundaryFlowIntegrator(rho_out, vc_u, 1.0, -bt));
      b.Assemble();

#if 0
      // save the matrices M and A, for tau = 0
      {
         const char *M_filename[] = {"M0-matrix-rho.txt", "M0-matrix-mom.txt"};
         ofstream M_file(M_filename[type-1]);
         M_file.precision(14);
         M.SpMat().PrintMatlab(M_file);

         const char *A_filename[] = {"A0-matrix-rho.txt", "A0-matrix-mom.txt"};
         ofstream A_file(A_filename[type-1]);
         A_file.precision(14);
         A.SpMat().PrintMatlab(A_file);

         if (type == 1)
         {
            const char AJ_filename[] = "AJ0-matrix-rho.txt";
            if (lump_jump)
            {
               ofstream AJ_file(AJ_filename);
               AJ_file.precision(14);
               AJ.SpMat().PrintMatlab(AJ_file);
            }
            else
               remove(AJ_filename);
         }
      }
#endif
   }

   void Assemble(double tau, bool mom_m_assemble) const
   {
      if ((use_mono && type == 1) || tau != old_tau)
      {
         GridFunction &nodes(*mesh->GetNodes());
         Vector nodes0(nodes);

         nodes.Add(tau, *u);

         A = 0.0;
         A.Assemble();
         if (use_mono && type == 1)
            Upwind(A.SpMat(), *mono);
         if (lump_jump && type == 1)
         {
            AJ = 0.0;
            AJ.Assemble();
         }

         if (type == 1 || !m_based_conserve_const)
         {
            M = 0.0;
            M.Assemble();
            if (use_mono && type == 1)
               Lump(M.SpMat(), *mono);
         }
         else
         {
            if (mom_m_assemble)
            {
               M = 0.0;
               M.Assemble();
               APS = 0.0;
               APS.Assemble();
            }
            else
            {
               // M += (APS + APS^t)/2
               APS.SpMat().Symmetrize();
               M.SpMat().Add(2*(tau-old_tau), APS.SpMat());
            }
         }

         b.Assemble();

         old_tau = tau;

         // Move nodes back to original locations
         nodes = nodes0;
      }
   }

   virtual void Mult(const Vector &, Vector &) const;

   void Reset()
   {
      num_calls = 0;

      Assemble(0.0, true);
   }

   GridFunction &GetMono() { return *mono; }

   void GetRho(double tau, const Vector &mom, Vector &rho)
   {
      Assemble(tau, true);

      SparseMatrix &Mmat = M.SpMat();
      GSSmoother GS(Mmat);

      rho = 0.0;
      PCG(Mmat, GS, mom, rho, 0, pcg_max_iter, pcg_rel_tol, 0.0);
   }

   void GetNeighborhood(Table &nbrs)
   {
      SparseMatrix &a = A.SpMat();
      nbrs.SetIJ(a.GetI(), a.GetJ(), a.Size());
   }

   virtual ~DG_remap_ode()
   {
#if 0
      // save the matrices M and A, for tau = 1
      {
         Assemble(1.0, true);

         const char *M_filename[] = {"M1-matrix-rho.txt", "M1-matrix-mom.txt"};
         ofstream M_file(M_filename[type-1]);
         M_file.precision(14);
         M.SpMat().PrintMatlab(M_file);

         const char *A_filename[] = {"A1-matrix-rho.txt", "A1-matrix-mom.txt"};
         ofstream A_file(A_filename[type-1]);
         A_file.precision(14);
         A.SpMat().PrintMatlab(A_file);

         if (type == 1)
         {
            const char AJ_filename[] = "AJ1-matrix-rho.txt";
            if (lump_jump)
            {
               ofstream AJ_file(AJ_filename);
               AJ_file.precision(14);
               AJ.SpMat().PrintMatlab(AJ_file);
            }
            else
               remove(AJ_filename);
         }
      }
#endif
      delete mono;
   }
};

void DG_remap_ode::Mult(const Vector &x, Vector &dx_dtau) const
{
   double tau = GetTime();

#if 0
   cout << "DG_remap_ode::Mult : tau = " << old_tau << " --> tau = "
        << tau << endl;
#endif

   num_calls++;

   Assemble(tau, num_calls%2);

   SparseMatrix &Mmat = M.SpMat();
   GSSmoother GS(Mmat);

   if (type == 1)
   {
      Vector Arho(x.Size());
      if (limiter < 0)
         A.Mult(x, Arho);
      else
         LimitMult(A.SpMat(), x, Arho, limiter);
      if (!lump_jump)
         Arho += b;
      dx_dtau = 0.0;
      PCG(Mmat, GS, Arho, dx_dtau, 0, pcg_max_iter, pcg_rel_tol, 0.0);
      if (lump_jump)
      {
         Vector ML(Mmat.Size());
         Mmat.GetRowSums(ML);
         AJ.Mult(x, Arho);
         Arho += b;
         for (int i = 0; i < ML.Size(); i++)
            dx_dtau(i) += Arho(i)/ML(i);
      }
   }
   else
   {
      Vector rho(x.Size());
      rho = 0.0;
      PCG(Mmat, GS, x, rho, 0, pcg_max_iter, pcg_rel_tol, 0.0);
      A.Mult(rho, dx_dtau);
      dx_dtau += b;
   }
}


void CalcMonotonicityMeasure(GridFunction &xold, GridFunction &xnew,
                             GridFunction &mono, Table &nbrs)
{
   for (int i = 0; i < xold.Size(); i++)
   {
      int *inbrs = nbrs.GetRow(i);
      int n = nbrs.RowSize(i);
      double xmin = numeric_limits<double>::infinity();
      double xmax = -xmin;
      for (int j = 0; j < n; j++)
      {
         if (xold(inbrs[j]) > xmax)
            xmax = xold(inbrs[j]);
         else if (xold(inbrs[j]) < xmin)
            xmin = xold(inbrs[j]);
      }
      double delta = fabs(xmax - xmin);
      if (delta > 0)
      {
         delta = 1;
         if (xnew(i) > xmax)
            mono(i) = smooth_transition(fabs((xnew(i) - xmax))/delta);
         else if (xnew(i) < xmin)
            mono(i) = smooth_transition(fabs((xnew(i) - xmin))/delta);
         else
            mono(i) = 0.;
      }
      else
      {
         mono(i) = 0.;
      }
   }
   if (0)
   {
      FiniteElementSpace *fes = mono.FESpace();
      Mesh *mesh = fes->GetMesh();
      Array<int> dofs;
      Vector mono_local;
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         fes->GetElementDofs(i,dofs);
         mono.GetSubVector(dofs,mono_local);
         mono_local = mono_local.Max();
         mono.SetSubVector(dofs,mono_local);
      }
   }
}

// Remap x_in defined on the given mesh to x_out defined on the mesh displaced
// with the given grid function u. If type=1, x corresponds to the original grid
// function (rho), otherwise it corresponds to the function moments (m).
void DG_remap(Mesh &mesh, GridFunction &u, GridFunction &x_in,
              GridFunction &x_out, ODESolver &ode_solver, int nsteps,
              bool rk2_const_pres, int type)
{
   DG_remap_ode remap_ode(&mesh, &u, &x_in, type, rk2_const_pres);

   ode_solver.Init(remap_ode);

   double tau = 0.0, tau_old;
   GridFunction x(x_in.FESpace()), rho;
   x = x_in;
   if (type == 1)
      rho.Update(x.FESpace(), x, 0);
   else
      rho.Update(x.FESpace());

   char vishost[] = "localhost";
   int  visport   = 19916;

   GridFunction &nodes(*mesh.GetNodes());
   Vector nodes0(nodes);

   GridFunction *mono = NULL, *mono_save = NULL;

   struct timespec req;
   req.tv_sec  = 0;
   req.tv_nsec = 50 * 1000000; // sleep for 0.05 seconds

   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);

   if (type == 2)
      remap_ode.GetRho(tau, x, rho);
   SocketSend(sol_sock, &mesh, &rho);

   sol_sock << "window_title 'rho'" << endl;
   //sol_sock << "palette 25" << endl;
   sol_sock << "keys cRjlm \n";
   sol_sock << "window_size 600 600" << endl;
   sol_sock << "viewcenter 0.25 0.0" << endl;
   sol_sock << "zoom 1.18" << endl;

   socketstream *mu_sock = NULL, *muold_sock = NULL;
   Table nbrs;

   if (use_mono && type == 1)
   {
      mono = &remap_ode.GetMono();
      mono_save = new GridFunction(mono->FESpace());
      (*mono_save) = 0.0;

      mu_sock = new socketstream(vishost, visport);
      mu_sock->precision(8);
      muold_sock = new socketstream(vishost, visport);
      muold_sock->precision(8);

      remap_ode.GetNeighborhood(nbrs);

      SocketSend(*mu_sock, &mesh, mono);
      SocketSend(*muold_sock, &mesh, mono_save);

      *mu_sock  << "window_title 'current monotonicity'" << endl;
      //*mu_sock  << "palette 25" << endl;
      *mu_sock  << "keys cRjlm \n";
      *mu_sock  << "window_size 600 600" << endl;
      *mu_sock  << "viewcenter 0.25 0.0" << endl;
      *mu_sock  << "zoom 1.18" << endl;

      *muold_sock  << "window_title 'accumulated monotonicity'" << endl;
      //*muold_sock  << "palette 25" << endl;
      *muold_sock  << "keys cRjlm \n";
      *muold_sock  << "window_size 600 600" << endl;
      *muold_sock  << "viewcenter 0.25 0.0" << endl;
      *muold_sock  << "zoom 1.18" << endl;
   }

   ConstantCoefficient zero(0.0);

   bool have_final_x = false;
   char mode = 'q';
   std::string str;
   do
   {
      cout <<
         "\nChoose visualization mode:\n"
         "a) animate\n"
         "s) step-by-step\n"
         "q) quit to final state\n"
         " [" << mode << "] --> " << flush;
      getline(cin, str);
      mode = str.empty() ? mode : str[0];
      if (mode == 'q' && have_final_x)
         break;

      remap_ode.Reset();
      x = x_in;
      tau = 0.0;

      double dtau = 1.0/nsteps;

      double mu_cut  = 1e-10;
      double mu_norm = 0.0;
      int counter    = 0;
      int nfix       = 20;

      for (int i = 0; i <= nsteps; i++)
      {
         if (i > 0)
         {
            if (!use_mono || type != 1)
            {
               ode_solver.Step(x, tau, dtau);
            }
            else
            {
               x_out = x;
               tau_old = tau;
               (*mono_save) = (*mono) = 0.0;
               counter = 0;

            redo_ode_solve:
               ode_solver.Step(x, tau, dtau);
               // Begin correction loop
               (*mono_save) = (*mono);
               CalcMonotonicityMeasure(x_out, x, *mono, nbrs);
               SocketSend(*mu_sock, &mesh, mono);
               mu_norm = mono->ComputeL2Error(zero);
               cout << "(" << counter
                    << ") Updating monotonicity measure, mu_norm = "
                    << mu_norm << endl;
               if (mu_norm > mu_cut && counter < nfix)
               {
                  counter++;
                  for (int j = 0; j < mono->Size(); j++)
                  {
                     (*mono)(j) = fmax((*mono)(j),(*mono_save)(j));
                     // (*mono)(j) = fmin((*mono_save)(j) + (*mono)(j),1.0);
                  }
                  SocketSend(*muold_sock, &mesh, mono);
                  // Reset the solution and try again
                  x = x_out;
                  tau = tau_old;
                  goto redo_ode_solve;
               }
            }
         }

         if (mode != 'q' || i == nsteps)
         {
            if (type == 2)
               remap_ode.GetRho(tau, x, rho);
            nodes.Add(tau, u);
            SocketSend(sol_sock, &mesh, &rho);
            nodes = nodes0;
            // if (use_mono && type == 1)
            //    SocketSend(*mu_sock, &mesh, mono);
         }

         if (mode == 's')
         {
            if (0) //(i == 1)
            {
               stringstream ss;
               ss << setfill('0') << setw(4) << i;
               string cyc = ss.str();

               sol_sock << "screenshot rho_" << cyc << ".png \n";
               *mu_sock <<  "screenshot mon_" << cyc << ".png \n";
            }
            cout << "(a)nimate, (s)tep, or (q)uit [" << mode << "] --> "
                 << flush;
            getline(cin, str);
            mode = str.empty() ? mode : str[0];
         }
         else if (mode == 'a')
            nanosleep(&req, NULL);
      }
      have_final_x = true;
   }
   while (mode != 'q');

   x_out = x;

   if (use_mono && type == 1)
   {
      delete muold_sock;
      delete mu_sock;
      delete mono_save;
      nbrs.LoseData();
   }

   // Save the final monotonicty measure on the final mesh
   if (use_mono && type == 1)
   {
      nodes += u;
      ofstream vtk_mesh("mono.vtk");
      vtk_mesh.precision(8);
      mesh.PrintVTK(vtk_mesh, vtk_subdiv);
      (*mono).SaveVTK(vtk_mesh, "monotonicity_measure", vtk_subdiv);
      nodes = nodes0;
   }
}

int main (int argc, char *argv[])
{
   Mesh *mesh;
   char vishost[] = "localhost";
   int  visport   = 19916;
   int  ans;

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

      // cout << "Usage: ex7 <mesh_file>" << endl;
      // return 1;
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

   int mesh_poly_deg = 1;
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
   int disp_field_opt = 1;
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

#if 0
   // Plot the displacement field
   if (dim > 1)
   {
      cout << "\nSending the displacement field to GLVis ... " << flush;
      osockstream sol_sock(visport, vishost);
      SocketSend(sol_sock, mesh, &u); // does not work for vector fields in 1D
      cout << "done.\n" << endl;
   }
#endif

   // Choose the function to remap
   int rho_poly_deg = 1;
   cout <<
      "Enter polynomial degree for the function to remap:\n"
      "-p) Discontinuous function of degree p >= 0\n"
      " p) Continuous function of degree p >= 1\n"
      " --> " << flush;
   cin >> rho_poly_deg;
   int l2_fec_type = 0;
   if (rho_poly_deg <= 0)
   {
      cout <<
         "Enter type of discontinuous basis:\n"
         "0) Nodal basis, using Gauss-Legengre points\n"
         "1) Nodal basis, using Gauss-Lobatto points\n"
         "2) Bernstein (positive) basis\n"
         " --> " << flush;
      cin >> l2_fec_type;
      if (l2_fec_type == 2)
         pcg_max_iter = 100;
   }
   FiniteElementCollection *rho_fec;
   if (rho_poly_deg <= 0)
      rho_fec = new L2_FECollection(abs(rho_poly_deg), dim, l2_fec_type);
   else
      rho_fec = new H1_FECollection(rho_poly_deg, dim);
   FiniteElementSpace *rho_fespace = new FiniteElementSpace(mesh, rho_fec);
   rho_fespace->BuildElementToDofTable();

   vtk_subdiv = (rho_poly_deg == 0) ? 1 : abs(rho_poly_deg)*2;

   GridFunction rho(rho_fespace);
   int rho_opt = 1;
   cout <<
      "Choose function to remap:\n"
      "1) rho_exact()\n"
      " --> " << flush;
   cin >> rho_opt;
   Coefficient *c_rho = new FunctionCoefficient(rho_exact);
   if (rho_opt == 1)
   {
      if (rho_poly_deg <= 0 && l2_fec_type != 0)
      {
         // use Gauss-Legendre (all interior) points to project
         // this handles mesh-aligned discontinuities
         L2_FECollection gl_fec(abs(rho_poly_deg), dim, 0);
         FiniteElementSpace gl_fes(mesh, &gl_fec);
         GridFunction gl_rho(&gl_fes);
         gl_rho.ProjectCoefficient(*c_rho);
         GridFunctionCoefficient gl_rho_coeff(&gl_rho);
         if (0)
         {
            rho.ProjectCoefficient(gl_rho_coeff);
         }
         else
         {
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
         }
      }
      else
         rho.ProjectCoefficient(*c_rho);
   }
   else
      cerr << "Option not implemented" << endl;

   // Save initial function on the original mesh
   {
      ofstream vtk_mesh("initial.vtk");
      vtk_mesh.precision(8);
      mesh->PrintVTK(vtk_mesh, vtk_subdiv);
      rho.SaveVTK(vtk_mesh,"density",vtk_subdiv);
   }

   ODESolver *ode_solver;
   cout <<
      "Choose an ODE solver:\n"
      "1) Forward Euler\n"
      "2) RK2 (midpoint)\n"
      // "2) RK2 (Heun)\n"
      "3) RK2, constant preserving (midpoint)\n"
      "4) RK3, SSP\n"
      "5) RK4\n"
      "6) RK6\n"
      "7) RK8\n"
      " --> " << flush;
   int ode_solver_opt, ode_solver_steps;
   cin >> ode_solver_opt;
   switch (ode_solver_opt)
   {
   case 1:  ode_solver = new ForwardEulerSolver; break;
   case 2:  ode_solver = new RK2Solver(0.5);     break; // midpoint
      // case 2:  ode_solver = new RK2Solver(1.0);     break; // Heun
   case 3:  ode_solver = new RK2Solver(0.5);     break;
   case 4:  ode_solver = new RK3SSPSolver;       break;
   case 5:  ode_solver = new RK4Solver;          break;
   case 6:  ode_solver = new RK6Solver;          break;
   case 7:  ode_solver = new RK8Solver;          break;
   default: ode_solver = new RK2Solver(0.5);
   }
   cout << "Enter number of ODE solver steps --> " << flush;
   cin >> ode_solver_steps;
   cin.ignore(numeric_limits<streamsize>::max(), '\n');

   GridFunction &x = *mesh->GetNodes();
   Vector x0(x);

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
   DG_remap(*mesh, u, rho, rho_new, *ode_solver, ode_solver_steps,
            ode_solver_opt == 3, 1);

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

#if 0
   // Plot the error of the rho-DG remapped function on the displaced mesh
   {
      osockstream sol_sock(visport, vishost);
      if (1)
      {
         // project the error in higher-order space
         // L2_FECollection ho_fec(abs(rho_poly_deg) + 2, dim, 1);
         L2_FECollection ho_fec(abs(rho_poly_deg) + 2, dim, 0);
         FiniteElementSpace ho_fes(mesh, &ho_fec);
         GridFunction ho_exact_rho(&ho_fes), ho_interp_rho(&ho_fes);
         ho_exact_rho.ProjectCoefficient(*c_rho);
         GridFunctionCoefficient gfc_rho(&rho_new);
         ho_interp_rho.ProjectCoefficient(gfc_rho);
         ho_interp_rho -= ho_exact_rho;
         SocketSend(sol_sock, mesh, &ho_interp_rho);
      }
      else
      {
         rho.ProjectCoefficient(*c_rho);
         subtract(rho_new, rho, rho);
         SocketSend(sol_sock, mesh, &rho);
      }
   }
#endif
   // Save rho-DG remapped function on the displaced mesh
   {
      ofstream vtk_mesh("final_rho.vtk");
      vtk_mesh.precision(8);
      mesh->PrintVTK(vtk_mesh, vtk_subdiv);
      rho_new.SaveVTK(vtk_mesh,"density",vtk_subdiv);
   }

   // Move the mesh back to the original position
   x = x0;

   // Perform moment-based DG remap
   DG_remap(*mesh, u, m, m_new, *ode_solver, ode_solver_steps,
            ode_solver_opt == 3, 2);

   delete ode_solver;

   // Moment-based DG scheme
   SparseMatrix &Mmat = M.SpMat();
   GSSmoother GS(Mmat);
   rho_new = 0.0;
   PCG(Mmat, GS, m_new, rho_new, 0, pcg_max_iter, pcg_rel_tol, 0.0);

   // Move mesh to final position
   x += u;

   double mass1_m = M.InnerProduct(one, rho_new);
   double l2norm1_m = sqrt(M.InnerProduct(rho_new, rho_new));
   cout << endl << "[ moment-based DG scheme ]" << endl
        << "Total mass after remap:  " << mass1_m << endl
        << "Remap mass loss: " << mass0 - mass1_m << endl
        << "L2 norm loss:    " << l2norm0 - l2norm1_m << endl
        << "Remap errors: L1   = "
        << rho_new.ComputeL1Error(*c_rho, irs) << endl
        << "              L2   = "
        << rho_new.ComputeL2Error(*c_rho, irs) << endl
        << "              Linf = "
        << rho_new.ComputeMaxError(*c_rho, irs) << endl << endl;

#if 0
   // Plot the error of the moment-DG remapped function on the displaced mesh
   {
      osockstream sol_sock(visport, vishost);
      if (1)
      {
         // project the error in higher-order space
         // L2_FECollection ho_fec(abs(rho_poly_deg) + 2, dim, 1);
         L2_FECollection ho_fec(abs(rho_poly_deg) + 2, dim, 0);
         FiniteElementSpace ho_fes(mesh, &ho_fec);
         GridFunction ho_exact_rho(&ho_fes), ho_interp_rho(&ho_fes);
         ho_exact_rho.ProjectCoefficient(*c_rho);
         GridFunctionCoefficient gfc_rho(&rho_new);
         ho_interp_rho.ProjectCoefficient(gfc_rho);
         ho_interp_rho -= ho_exact_rho;
         SocketSend(sol_sock, mesh, &ho_interp_rho);
      }
      else
      {
         rho.ProjectCoefficient(*c_rho);
         subtract(rho_new, rho, rho);
         SocketSend(sol_sock, mesh, &rho);
      }
   }
#endif
   // Save moment-DG remapped function on the displaced mesh
   {
      ofstream vtk_mesh("final_mom.vtk");
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


inline double Limit(const double r, int lim)
{
   if (r <= 0.0)
      return 0.0;
   switch (lim)
   {
   case 1: return fmin(1.0, r); // minmod
   case 2: return 2.0*r/(1.0 + r); // Van Leer
   case 3: return fmin(0.5*(1.0 + r), 2.0*fmin(1.0, r)); // MC
   case 4: return fmax(fmin(2.0, r), fmin(1.0, 2*r)); // superbee
   }
   return 0.0; // no limiter, pure upwind
}

void LimitMult(const SparseMatrix &M, const Vector &x, Vector &y, int lim)
{
   // M must be finalized

   int i, j, k, size = M.Size();
   int *I = M.GetI(), *J = M.GetJ();
   double xi, a, b, c, fi, fji;
   double *A = M.GetData();
   Vector rp(size), rm(size);

   for (i = 0; i < size; i++)
   {
      double pp = 0.0, pm = 0.0, qp = 0.0, qm = 0.0;
      xi = x(i);
      for (k = I[i]; k < I[i+1]; k++)
      {
         j = J[k];
         a = A[k];
         b = x(j) - xi;
         if (a >= 0.0)
         {
            if (b >= 0.0)
               qp += a*b;
            else
               qm += a*b;
         }
         else
         {
            if (b >= 0.0)
               pm += a*b;
            else
               pp += a*b;
         }
      }
      if (pp > 0.0)
         rp(i) = Limit(qp/pp, lim);
      if (pm < 0.0)
         rm(i) = Limit(qm/pm, lim);
   }

   y = 0.0;
   for (i = 0; i < size; i++)
   {
      xi = x(i);
      fi = 0.0;
      for (k = I[i]; k < I[i+1]; k++)
      {
         j = J[k];
         a = A[k];
         b = x(j);
         fi += a*b;
         b -= xi;
         if (b != 0.0 && j < i)
         {
            c = M(j,i); // <-----
            if (c >= a)
            {
               if (a < 0.0)
               {
                  // dij = -a
                  if (b < 0.0)
                     fji = fmin(a*(1.0-rp(i)), c)*b;
                  else
                     fji = fmin(a*(1.0-rm(i)), c)*b;
                  fi -= fji;
                  y(j) += fji;
               }
            }
            else
            {
               if (c < 0.0)
               {
                  // dij = -c
                  if (b > 0.0)
                     fji = fmin(c*(1.0-rp(j)), a)*b;
                  else
                     fji = fmin(c*(1.0-rm(j)), a)*b;
                  fi -= fji;
                  y(j) += fji;
               }
            }
         }
      }
      y(i) += fi;
   }
}

void Lump(SparseMatrix &M, const Vector &mu)
{
   // M must be finalized

   int i, j, k, d, size = M.Size();
   int *I = M.GetI(), *J = M.GetJ();
   double *A = M.GetData();

   for (i = k = 0; i < size; i++)
   {
      double Dii = 0.0;
      d = -1;
      for (int end = I[i+1]; k < end; k++)
      {
         if ((j = J[k]) == i)
         {
            d = k;
         }
         else
         {
            double Dij = -A[k];
            if (Dij != 0.0)
            {
               Dij *= fmax(mu(i), mu(j));
               // Dij *= mu(i)*mu(j);
               // Dij *= (mu(i)+mu(j))/2.0;

               A[k] += Dij;
               Dii  -= Dij;
            }
         }
      }
      if (d >= 0)
         A[d] += Dii;
      else
         mfem_error("Lump : no diagonal entry!");
   }
}

void Upwind(SparseMatrix &M, const Vector &mu)
{
   // M must be finalized

   int i, j, k, size = M.Size();
   int *I = M.GetI(), *J = M.GetJ();
   double *A = M.GetData();

   for (i = 1; i < size; i++)
      for (k = I[i]; k < I[i+1]; k++)
         if ((j = J[k]) < i)
         {
            double &Aji = M(j,i);
            double Dij = fmax(fmax(0.0, -A[k]), -Aji);
            if (Dij != 0.0)
            {
               Dij *= fmax(mu(i), mu(j));
               // Dij *= mu(i)*mu(j);
               // Dij *= (mu(i)+mu(j))/2.0;

               A[k] += Dij;
               Aji  += Dij;

               M(i,i) -= Dij;
               M(j,j) -= Dij;
            }
         }
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


const double eps = 1e-12;

double rho_exact(Vector &X)
{
   const int dim = X.Size();
   double x, y, z;
   x = X(0);
   y = (dim > 1) ? X(1) : 0.0;
   z = (dim > 2) ? X(2) : 0.0;

   // return 1.0;
   // return x+y;
   // return x*x+4*y*y;
   // return sin(M_PI*x)*sin(M_PI*y);
   // return 1+cos(2*M_PI*x)*sin(M_PI*y);
   // return M_PI_2+atan(20*(x-0.5));
   // return M_PI_2+atan(20*(y-0.5));
   return (x < 0.5) ? 0.5 : 1.0;
   // return (x < 0.5 - eps) ? 0.5 : ((x > 0.5 + eps) ? 1.0: 0.75);
   // return (y < 0.5) ? 0.5 : 1.0;
   // return (x < 0.5) ? M_PI_2 : M_PI_2+atan(5*(x-0.5));
   // return pow(fmax(0.0, fmin(x, 1. - x)), 4.);
   // return sin(2*M_PI*x);
   // return sin(0.5*M_PI*x);
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
   // double fix_bdr = 1; // do not fix the boundary
   double angle = 30.*(M_PI/180)*fix_bdr;
   double s = sin(angle), c = cos(angle);
   d[0] = xc + c*(x-xc) - s*(y-yc) - x;
   d[1] = yc + s*(x-xc) + c*(y-yc) - y;

   // // rotation, with tangential displacement at the boundary
   // double angle = 15.*(M_PI/180);
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

   // // 1D compression in x-direction
   // const double xc = 6./7; // 1/2 --> xc \in (0,1)
   // d[0] = x/(1.0 + (1./xc - 2.)*(1.0 - x)) - x;
   // d[1] = 0.0;

   // // 1D translation in x-direction
   // d[0] = 4./1;
   // d[1] = 0.0;

   for (int i = 0; i < dim; i++)
      u(i) = d[i];
}

double smooth_transition(double x)
{
   //return 1.0 - exp(-1e5*pow(x,2.0));
   // return fmin(1e4*pow(x,2.0),1);
   if (x > 1e-12)
      return 1.0;
   else
      return 0.0;
}
