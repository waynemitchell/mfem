//                                MFEM Example 3
//
// Compile with: make ex3
//
// Sample runs:  ex3 ../data/beam-tet.mesh
//               ex3 ../data/beam-hex.mesh
//               ex3 ../data/escher.mesh
//               ex3 ../data/fichera.mesh
//               ex3 ../data/fichera-q2.vtk
//               ex3 ../data/fichera-q3.mesh
//               ex3 ../data/beam-hex-nurbs.mesh
//
// Description:  This example code solves a simple 3D electromagnetic diffusion
//               problem corresponding to the second order definite Maxwell
//               equation curl curl E + E = f with boundary condition
//               E x n = <given tangential field>. Here, we use a given exact
//               solution E and compute the corresponding r.h.s. f.
//               We discretize with the lowest order Nedelec finite elements.
//
//               The example demonstrates the use of H(curl) finite element
//               spaces with the curl-curl and the (vector finite element) mass
//               bilinear form, the projection of grid functions between finite
//               element spaces and the computation of discretization error when
//               the exact solution is known.
//
//               We recommend viewing examples 1-2 before viewing this example.

#include <fstream>
#include "mfem.hpp"

// Exact solution, E, and r.h.s., f. See below for implementation.
void E_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);

int main(int argc, char *argv[])
{
   Mesh *mesh;

   if (argc == 1)
   {
      cout << "\nUsage: ex3x <mesh_file>\n" << endl;
      return 1;
   }

   // 1. Read the mesh from the given mesh file. In this 3D example, we can
   //    handle tetrahedral or hexahedral meshes with the same code.
   ifstream imesh(argv[1]);
   if (!imesh)
   {
      cerr << "\nCan not open mesh file: " << argv[1] << '\n' << endl;
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();
   int dim = mesh->Dimension();
   if (dim != 3)
   {
      cerr << "\nThis example requires a 3D mesh\n" << endl;
      return 3;
   }

   // 2. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels;
      mesh->PrintCharacteristics();
      cout << "enter ref. levels --> " << flush;
      cin >> ref_levels;
      for (int l = 0; l < ref_levels; l++)
      {
         cout << "refinement level " << l + 1 << " ... " << flush;
         mesh->UniformRefinement();
         cout << "done." << endl;
      }
      cout << endl;

      if (mesh->MeshGenerator() & 1)
      {
         cout << "Reorienting tetrahedral mesh ... " << flush;
         mesh->ReorientTetMesh();
         cout << "done.\n" << endl;
      }

      mesh->PrintCharacteristics();
   }

   // print the boundary element orientations
   if (0)
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

   // 3. Define a finite element space on the mesh. Here we use the lowest order
   //    Nedelec finite elements.
   int p;
   cout << "enter Nedelec space order --> " << flush;
   cin >> p;
   FiniteElementCollection *fec = new ND_FECollection(p, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

   cout << "\nNumber of dofs: " << fespace->GetVSize() << endl;

   // 5. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary edges will be used
   //    when eliminating the non-homogenious boundary condition to modify the
   //    r.h.s. vector b.
   GridFunction x(fespace);
   VectorFunctionCoefficient E(dim, E_exact);

   GridFunction xp(fespace);
   xp.ProjectCoefficient(E);

   cout << "\nprojection error:\n|| E_h - E ||_{L^2} = "
        << xp.ComputeL2Error(E) << endl;

   // 9. In order to visualize the solution, we first represent it in the space
   //    of linear discontinuous vector finite elements. The representation in
   //    this space is obtained by (exact) projection with ProjectVectorFieldOn.
   FiniteElementCollection *dfec;
   if (1)
      dfec = new L2_FECollection(p, dim);
   else
      dfec = new L2_FECollection(p + 1, dim); // curved meshes?
   FiniteElementSpace *dfespace = new FiniteElementSpace(mesh, dfec, dim);
   GridFunction dx(dfespace);

   // view the projection
   if (1)
   {
      xp.ProjectVectorFieldOn(dx);

      char vishost[] = "localhost";
      int  visport   = 19916;
      osockstream sol_sock(visport, vishost);
      sol_sock << "solution\n";
      sol_sock.precision(8);
      mesh->Print(sol_sock);
      dx.Save(sol_sock);
      sol_sock.send();
   }

   cout << "\nAssebling: ";

   // 4. Set up the linear form b(.) which corresponds to the right-hand side
   //    of the FEM linear system, which in this case is (f,phi_i) where f is
   //    given by the function f_exact and phi_i are the basis functions in the
   //    finite element fespace.
   VectorFunctionCoefficient f(dim, f_exact);
   LinearForm *b = new LinearForm(fespace);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   cout << "rhs ... " << flush;
   b->Assemble();

   // 6. Set up the bilinear form corresponding to the EM diffusion operator
   //    curl muinv curl + sigma I, by adding the curl-curl and the mass domain
   //    integrators and finally imposing the non-homogeneous Dirichlet boundary
   //    conditions. The boundary conditions are implemented by marking all the
   //    boundary attributes from the mesh as essential (Dirichlet). After
   //    assembly and finalizing we extract the corresponding sparse matrix A.
   Coefficient *muinv = new ConstantCoefficient(1.0);
   Coefficient *sigma = new ConstantCoefficient(1.0);
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   a->AddDomainIntegrator(new VectorFEMassIntegrator(sigma));
   cout << "matrix ... " << flush;
   a->Assemble();
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   cout << "b.c. ... " << flush;

   x = 0.0;
   x.ProjectBdrCoefficientTangent(E, ess_bdr);
   // x = xp;

   a->EliminateEssentialBC(ess_bdr, x, *b);
   a->Finalize();
   const SparseMatrix &A = a->SpMat();
   cout << "done." << endl;

   if (1)
   {
      cout << "\nMatrix stats: size = " << A.Size() << ", nnz = "
           << A.NumNonZeroElems() << ", nnz/size = "
           << double(A.NumNonZeroElems())/A.Size() << '\n' << endl;
   }

   // 7. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //    solve the system Ax=b with PCG.
   GSSmoother M(A);
   // x = 0.0; // warning: resets the b.c. to 0
   PCG(A, M, *b, x, 1, 5000, 1e-32, 0.0);

   // 8. Compute and print the L^2 norm of the error.
   cout << "\n|| E_h - E ||_{L^2} = " << x.ComputeL2Error(E) << '\n' << endl;

   cout << "projection error:\n|| E_h - E ||_{L^2} = "
        << xp.ComputeL2Error(E) << '\n' << endl;

   // x -= xp;

   x.ProjectVectorFieldOn(dx);

   // 10. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -m refined.mesh -g sol.gf".
   {
      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      dx.Save(sol_ofs);
   }

   // 11. (Optional) Send the solution by socket to a GLVis server.
   char vishost[] = "localhost";
   int  visport   = 19916;
   osockstream sol_sock(visport, vishost);
   sol_sock << "solution\n";
   sol_sock.precision(8);
   mesh->Print(sol_sock);
   dx.Save(sol_sock);
   sol_sock.send();

   // 12. Free the used memory.
   delete dfespace;
   delete dfec;
   delete a;
   delete sigma;
   delete muinv;
   delete b;
   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}

// A parameter for the exact solution.
const double kappa = M_PI/5;

const double sx = 1./3.;
const double sy = 5./7.;
const double sz = 7./23.;

void E_exact(const Vector &x, Vector &E)
{
   double Xx, Yy, Zz;
   Xx = kappa*(x(0) - sx);
   Yy = kappa*(x(1) - sy);
   Zz = kappa*(x(2) - sz);
   E(0) = sin(Xx)*sin(Yy)*cos(Zz);
   E(1) = cos(Xx)*cos(Yy)*cos(Zz);
   E(2) = cos(Xx)*cos(Yy)*sin(Zz);
}

void f_exact(const Vector &x, Vector &f)
{
   double Xx, Yy, Zz, c;
   Xx = kappa*(x(0) - sx);
   Yy = kappa*(x(1) - sy);
   Zz = kappa*(x(2) - sz);
   c = kappa*kappa;
   f(0) = cos(Zz)*sin(Xx)*(-c*cos(Yy) + (1 + 3*c)*sin(Yy));
   f(1) = cos(Xx)*cos(Zz)*((1 + 3*c)*cos(Yy) - c*sin(Yy));
   f(2) = (1 + 2*c)*cos(Xx)*cos(Yy)*sin(Zz);
}
