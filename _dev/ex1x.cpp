//                                MFEM Example 1
//
// Compile with: make ex1
//
// Sample runs:  ex1 ../data/square-disc.mesh
//               ex1 ../data/star.mesh
//               ex1 ../data/escher.mesh
//               ex1 ../data/fichera.mesh
//               ex1 ../data/square-disc-p2.vtk
//               ex1 ../data/square-disc-p3.mesh
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple linear finite element discretization of the Laplace
//               problem -Delta u = 1 with homogeneous Dirichlet boundary
//               conditions.
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of boundary conditions on all boundary edges, and the optional
//               connection to the GLVis tool for visualization.

#include <fstream>
#include "mfem.hpp"

double exact_sol(Vector &);
double exact_rhs(Vector &);

int main(int argc, char *argv[])
{
   Mesh *mesh;

   if (argc == 1)
   {
      cout << "\nUsage: ex1 <mesh_file>\n" << endl;
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

   // 2. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(50000./mesh->GetNE())/log(2.)/mesh->Dimension());
      mesh->PrintCharacteristics();
      cout << "Enter ref. levels = " << flush;
      cin >> ref_levels;
      for (int l = 0; l < ref_levels; l++)
         mesh->UniformRefinement();
      cout << endl;
      mesh->PrintCharacteristics();
   }

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

   // 3. Define a finite element space on the mesh. Here we use linear finite
   //    elements.
   int p;
   do
   {
      cout << "Enter p = " << flush;
      cin >> p;
   }
   while (p < 1 || p > 32);
   FiniteElementCollection *fec = new H1_FECollection(p, mesh->Dimension());
   // p = 3; FiniteElementCollection *fec = new CubicFECollection;
   // p = 2; FiniteElementCollection *fec = new QuadraticFECollection;
   // p = 1; FiniteElementCollection *fec = new LinearFECollection;
   {
      // testing
      L2_SegmentElement     d_seg_fe(p);
      L2_TriangleElement    d_tri_fe(p);
      L2_TetrahedronElement d_tet_fe(p);
   }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

   // 4. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm *b = new LinearForm(fespace);
   FunctionCoefficient rhs_coeff(&exact_rhs);
   b->AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
   b->Assemble();

   // 5. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   FunctionCoefficient e_sol(&exact_sol);
   x = 0.0;

   // 6. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator and imposing homogeneous Dirichlet boundary
   //    conditions. The boundary conditions are implemented by marking all the
   //    boundary attributes from the mesh as essential (Dirichlet). After
   //    assembly and finalizing we extract the corresponding sparse matrix A.
   BilinearForm *a = new BilinearForm(fespace);
   ConstantCoefficient one(1.0);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->Assemble();
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   x.ProjectBdrCoefficient(e_sol, ess_bdr);
   // x.ProjectCoefficient(e_sol);
   a->EliminateEssentialBC(ess_bdr, x, *b);
   // a->Finalize();
   const SparseMatrix &A = a->SpMat();

   if (1)
   {
      cout << "\nMatrix stats: size = " << A.Size() << " , nnz = "
           << A.NumNonZeroElems() << " , nnz/size = "
           << double(A.NumNonZeroElems())/A.Size() << '\n' << endl;
   }

   // 7. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //    solve the system Ax=b with PCG.
   GSSmoother M(A);
   PCG(A, M, *b, x, 1, 5000, 1e-30, 0.0);
   // PCG(A, M, *b, x, 1, 2, 1e-12, 0.0);

   cout << "\n|| u_h - u ||_{L^2} = " << x.ComputeL2Error(e_sol)
        << '\n' << endl;
   {
      GridFunction xp(fespace);
      xp.ProjectCoefficient(e_sol);
      cout << "projecton error :"
           << "\n|| u_h - u ||_{L^2} = "
           << xp.ComputeL2Error(e_sol) << '\n' << endl;
   }

   // 8. Save the refined mesh and the solution. This output can be viewed later
   //    using GLVis: "glvis -m refined.mesh -g sol.gf".
   {
      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   if (0)
   {
      ofstream sol_vtk("solution.vtk");
      sol_vtk.precision(8);
      GlobGeometryRefiner.SetType(1);
      mesh->PrintVTK(sol_vtk, p);
      x.SaveVTK(sol_vtk, "solution", p);
   }

   // 9. (Optional) Send the solution by socket to a GLVis server.
   char vishost[] = "localhost";
   int  visport   = 19916;
   osockstream sol_sock(visport, vishost);
   sol_sock << "solution\n";
   sol_sock.precision(8);
   mesh->Print(sol_sock);
   x.Save(sol_sock);
   sol_sock.send();

   if (0)
   {
      L2_FECollection dfec(p, mesh->Dimension());
      FiniteElementSpace dfes(mesh, &dfec);
      GridFunction dx(&dfes);
      GridFunctionCoefficient cx(&x);
      dx.ProjectCoefficient(cx);

      osockstream dsol_sock(visport, vishost);
      dsol_sock << "solution\n";
      dsol_sock.precision(8);
      mesh->Print(dsol_sock);
      dx.Save(dsol_sock);
      dsol_sock.send();
   }

   // 10. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}

const double sx = 1./3.;
const double sy = 11./23.;
const double sz = 3./7.;

double exact_sol(Vector &x)
{
   int dim = x.Size();
   if (dim == 2)
   {
      return sin(M_PI*(x(0)-sx))*sin(M_PI*(x(1)-sy));
   }
   else if (dim == 3)
   {
      return sin(M_PI*(x(0)-sx))*sin(M_PI*(x(1)-sy))*sin(M_PI*(x(2)-sz));
   }
   return 0.;
}

double exact_rhs(Vector &x)
{
   int dim = x.Size();
   if (dim == 2)
   {
      return 2*M_PI*M_PI*sin(M_PI*(x(0)-sx))*sin(M_PI*(x(1)-sy));
   }
   else if (dim == 3)
   {
      return (3*M_PI*M_PI*
              sin(M_PI*(x(0)-sx))*sin(M_PI*(x(1)-sy))*sin(M_PI*(x(2)-sz)));
   }
   return 0.;
}
