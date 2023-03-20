//                       MFEM Example 1 - Parallel Version
//
// Compile with: make ex1p
//
// Sample runs:  mpirun -np 4 ex1p -m ../data/square-disc.mesh
//               mpirun -np 4 ex1p -m ../data/star.mesh
//               mpirun -np 4 ex1p -m ../data/star-mixed.mesh
//               mpirun -np 4 ex1p -m ../data/escher.mesh
//               mpirun -np 4 ex1p -m ../data/fichera.mesh
//               mpirun -np 4 ex1p -m ../data/fichera-mixed.mesh
//               mpirun -np 4 ex1p -m ../data/toroid-wedge.mesh
//               mpirun -np 4 ex1p -m ../data/octahedron.mesh -o 1
//               mpirun -np 4 ex1p -m ../data/periodic-annulus-sector.msh
//               mpirun -np 4 ex1p -m ../data/periodic-torus-sector.msh
//               mpirun -np 4 ex1p -m ../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 ex1p -m ../data/square-disc-p3.mesh -o 3
//               mpirun -np 4 ex1p -m ../data/square-disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/star-mixed-p2.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/pipe-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/ball-nurbs.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/fichera-mixed-p2.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/star-surf.mesh
//               mpirun -np 4 ex1p -m ../data/square-disc-surf.mesh
//               mpirun -np 4 ex1p -m ../data/inline-segment.mesh
//               mpirun -np 4 ex1p -m ../data/amr-quad.mesh
//               mpirun -np 4 ex1p -m ../data/amr-hex.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh -o -1 -sc
//
// Device sample runs:
//               mpirun -np 4 ex1p -pa -d cuda
//               mpirun -np 4 ex1p -fa -d cuda
//               mpirun -np 4 ex1p -pa -d occa-cuda
//               mpirun -np 4 ex1p -pa -d raja-omp
//               mpirun -np 4 ex1p -pa -d ceed-cpu
//               mpirun -np 4 ex1p -pa -d ceed-cpu -o 4 -a
//               mpirun -np 4 ex1p -pa -d ceed-cpu -m ../data/square-mixed.mesh
//               mpirun -np 4 ex1p -pa -d ceed-cpu -m ../data/fichera-mixed.mesh
//             * mpirun -np 4 ex1p -pa -d ceed-cuda
//             * mpirun -np 4 ex1p -pa -d ceed-hip
//               mpirun -np 4 ex1p -pa -d ceed-cuda:/gpu/cuda/shared
//               mpirun -np 4 ex1p -pa -d ceed-cuda:/gpu/cuda/shared -m ../data/square-mixed.mesh
//               mpirun -np 4 ex1p -pa -d ceed-cuda:/gpu/cuda/shared -m ../data/fichera-mixed.mesh
//               mpirun -np 4 ex1p -m ../data/beam-tet.mesh -pa -d ceed-cpu
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
#include "HYPRE_parcsr_ls.h" 
#include "_hypre_parcsr_ls.h" 

/* *************************************** */
/* Anisotropic diffusion coefficients */
/* *************************************** */

// Isotropic
void alphaFunc0(const Vector &x, DenseMatrix &K)
{
   K.SetSize(2);
   double theta = 0.0;
   double eps = 1.0;
   double c = cos(theta);
   double s = sin(theta);
   K(0,0) = c * c + eps * s * s;
   K(0,1) = c * s - eps * c * s;
   K(1,0) = c * s - eps * c * s;
   K(1,1) = s * s + eps * c * c;
}

// Grid-aligned anisotropic
void alphaFunc1(const Vector &x, DenseMatrix &K)
{
   K.SetSize(x.Size());
   double theta = 0.0;
   double eps = 0.0001;
   double c = cos(theta);
   double s = sin(theta);
   K(0,0) = c * c + eps * s * s;
   K(0,1) = c * s - eps * c * s;
   K(1,0) = c * s - eps * c * s;
   K(1,1) = s * s + eps * c * c;
   if (K.Size() == 3)
   {
      K(0,2) = 0.0;
      K(1,2) = 0.0;
      K(2,0) = 0.0;
      K(2,1) = 0.0;
      K(2,2) = 1.0;
   }
}

// Non-grid-aligned anisotropic
void alphaFunc2(const Vector &x, DenseMatrix &K)
{
   K.SetSize(2);
   double theta = 0.59; // about 3pi/16
   double eps = 0.0001;
   double c = cos(theta);
   double s = sin(theta);
   K(0,0) = c * c + eps * s * s;
   K(0,1) = c * s - eps * c * s;
   K(1,0) = c * s - eps * c * s;
   K(1,1) = s * s + eps * c * c;
}

// Variable strength anisotropic
void alphaFunc3(const Vector &x, DenseMatrix &K)
{
   K.SetSize(2);
   double theta = 0.0;
   double eps = 0.000001 + pow(x(0), 4);
   double c = cos(theta);
   double s = sin(theta);
   K(0,0) = c * c + eps * s * s;
   K(0,1) = c * s - eps * c * s;
   K(1,0) = c * s - eps * c * s;
   K(1,1) = s * s + eps * c * c;
}

// Variable strength and direction anisotropic
void alphaFunc4(const Vector &x, DenseMatrix &K)
{
   K.SetSize(2);
   double theta = 0.59 * pow(x(1), 4);
   double eps = 0.000001 + pow(x(0), 4);
   double c = cos(theta);
   double s = sin(theta);
   K(0,0) = c * c + eps * s * s;
   K(0,1) = c * s - eps * c * s;
   K(1,0) = c * s - eps * c * s;
   K(1,1) = s * s + eps * c * c;
}

/* *************************************** */
/* main */
/* *************************************** */

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   char filename[256];
   char mesh_path[1024] = "../data/";
   const char *mesh_file = "star";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   bool fa = false;
   const char *device_config = "cpu";
   bool visualization = false;
   bool plot_soln = false;
   bool print_cf = false;
   bool print_matrices = false;
   bool algebraic_ceed = false;
   int cycle_type = 1;
   int aux_S = -1;
   int relax_type = 18;
   int coarsen_type = 8;
   int interp_type = 6;
   int max_coarse_size = 6;
   double strong_thresh = 0.25;
   int ref_levels = 2;
   int par_ref_levels = 0;
   int problem = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&cycle_type, "-c", "--cycle-type",
                  "BoomerAMG cycle type.");
   args.AddOption(&aux_S, "-s", "--aux-s",
                  "BoomerAMG aux S option.");
   args.AddOption(&relax_type, "-r", "--relax-type",
                  "BoomerAMG relax type.");
   args.AddOption(&coarsen_type, "-c", "--coarsen-type",
                  "BoomerAMG coarsening type.");
   args.AddOption(&interp_type, "-i", "--interp-type",
                  "BoomerAMG interpolation type.");
   args.AddOption(&max_coarse_size, "-C", "--max-coarse-size",
                  "BoomerAMG maximum coarse grid size.");
   args.AddOption(&strong_thresh, "-t", "--strong-thresh",
                  "Strength of connection threshold.");
   args.AddOption(&ref_levels, "-r", "--ref-levels",
                  "Number of uniform mesh refinement levels on the serial mesh.");
   args.AddOption(&par_ref_levels, "-pr", "--par-ref-levels",
                  "Number of uniform mesh refinement levels on the parallel mesh.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly", "Enable Full Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&problem, "-p", "--problem",
                  "Choose the problem (changes type of anisotropy).");
#ifdef MFEM_USE_CEED
   args.AddOption(&algebraic_ceed, "-a", "--algebraic",
                  "-no-a", "--no-algebraic",
                  "Use algebraic Ceed solver");
#endif
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&plot_soln, "-ps", "--plot-soln", "-no-ps",
                  "--no-plot-soln",
                  "Enable or disable saving a plot of the solution to file.");
   args.AddOption(&print_matrices, "-pm", "--print-matrices", "-no-pm",
                  "--no-print-matrices",
                  "Enable or disable saving of the matrices of the AMG hierarchy.");
   args.AddOption(&print_cf, "-pcf", "--print-cf", "-no-pcf",
                  "--no-print-cf",
                  "Enable or disable saving of the CF splitting.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh mesh;
   if ( std::strcmp(mesh_file, "square_quad") == 0 )
   {
      mesh = Mesh::MakeCartesian2D(16, 16, Element::QUADRILATERAL);
   }
   else if ( std::strcmp(mesh_file, "square_tri") == 0 )
   {
      mesh = Mesh::MakeCartesian2D(16, 16, Element::TRIANGLE);
   }
   else if ( std::strcmp(mesh_file, "cube_quad") == 0 )
   {
      mesh = Mesh::MakeCartesian3D(8, 8, 8, Element::QUADRILATERAL);
   }
   else if ( std::strcmp(mesh_file, "cube_tri") == 0 )
   {
      mesh = Mesh::MakeCartesian3D(8, 8, 8, Element::TRIANGLE);
   }
   else
   {
      strcat(mesh_path, mesh_file);
      strcat(mesh_path, ".mesh");
      mesh = Mesh(mesh_path, 1, 1);
   }
   int dim = mesh.Dimension();

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh.UniformRefinement();
      }
   }

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
   }
   else if (pmesh.GetNodes())
   {
      fec = pmesh.GetNodes()->OwnFEC();
      delete_fec = false;
      if (myid == 0)
      {
         cout << "Using isoparametric FEs: " << fec->Name() << endl;
      }
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }
   ParFiniteElementSpace fespace(&pmesh, fec);
   HYPRE_BigInt size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   /* WM: print out the coordinates for the dofs */
   ofstream outfile;
   if (print_cf)
   {
      sprintf(filename, "outputs/coords_%s_ref%d_prob%d", mesh_file, ref_levels, problem);
      outfile.open(filename);
      ParFiniteElementSpace coords_fespaces(&pmesh, fec, dim);
      ParGridFunction nodes(&coords_fespaces);
      pmesh.GetNodes(nodes);
      const int nNodes = nodes.Size() / dim;
      for (int i = 0; i < nNodes; ++i)
      {
         for (int j = 0; j < dim; ++j)
         {
            outfile << nodes(j * nNodes + i) << " "; 
         }   
         outfile << endl;
      }
   }
   outfile.close();

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // 10. Define the solution vector x as a parallel finite element grid
   //     function corresponding to fespace. Initialize x with initial guess of
   //     zero, which satisfies the boundary conditions.
   ParGridFunction x(&fespace);
   x = 0.0;

   // 11. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the
   //     Diffusion domain integrator.
   ParBilinearForm a(&fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   if (fa)
   {
      a.SetAssemblyLevel(AssemblyLevel::FULL);
      // Sort the matrix column indices when running on GPU or with OpenMP (i.e.
      // when Device::IsEnabled() returns true). This makes the results
      // bit-for-bit deterministic at the cost of somewhat longer run time.
      a.EnableSparseMatrixSorting(Device::IsEnabled());
   }
   MatrixFunctionCoefficient *alpha;
   if (problem == 1)
   {
      alpha = new MatrixFunctionCoefficient(dim, alphaFunc1);
   }
   else if (problem == 2)
   {
      alpha = new MatrixFunctionCoefficient(dim, alphaFunc2);
   }
   else if (problem == 3)
   {
      alpha = new MatrixFunctionCoefficient(dim, alphaFunc3);
   }
   else if (problem == 4)
   {
      alpha = new MatrixFunctionCoefficient(dim, alphaFunc4);
   }
   else
   {
      alpha = new MatrixFunctionCoefficient(dim, alphaFunc0);
   }
   a.AddDomainIntegrator(new DiffusionIntegrator(*alpha));

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   /* WM: randomize RHS */
   B.Randomize();

   // 13. Solve the linear system A X = B.
   //     * With full assembly, use the BoomerAMG preconditioner from hypre.
   //     * With partial assembly, use Jacobi smoothing, for now.
   // WM: only using the BoomerAMG preconditioner
   HypreBoomerAMG *prec = NULL;
   prec = new HypreBoomerAMG;
   prec->SetStrengthThresh(strong_thresh);
   prec->SetAggressiveCoarsening(0);

   HYPRE_Solver hypre_solver = *prec;
   HYPRE_BoomerAMGSetCycleType(hypre_solver, cycle_type);
   HYPRE_BoomerAMGSetUseAuxStrengthMatrix(hypre_solver, aux_S);
   if (relax_type >= 0) HYPRE_BoomerAMGSetRelaxType(hypre_solver, relax_type);
   if (print_matrices) HYPRE_BoomerAMGSetPrintLevel(hypre_solver, 3);
   HYPRE_BoomerAMGSetMaxRowSum(hypre_solver, 1.0);
   HYPRE_BoomerAMGSetCoarsenType(hypre_solver, coarsen_type);
   HYPRE_BoomerAMGSetInterpType(hypre_solver, interp_type);
   HYPRE_BoomerAMGSetTruncFactor(hypre_solver, 0.0);
   HYPRE_BoomerAMGSetMaxCoarseSize(hypre_solver, max_coarse_size);

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-8);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   if (prec) { cg.SetPreconditioner(*prec); }
   cg.SetOperator(*A);
   cg.Mult(B, X);

   /* WM: print out CF splitting */
   if (print_cf)
   {
      hypre_ParAMGData *amg_solver = (hypre_ParAMGData*) hypre_solver;
      for (int level = 0; level < hypre_ParAMGDataNumLevels(amg_solver) - 1; level++)
      {
         sprintf(filename, "outputs/cf_marker_%s_ref%d_prob%d_str%f_auxs%d_lvl%d", mesh_file, ref_levels, problem, strong_thresh, aux_S, level);
         outfile.open(filename);
         hypre_IntArray *CF_marker = hypre_ParAMGDataCFMarkerArray(amg_solver)[level];
         for (int i = 0; i < hypre_IntArraySize(CF_marker); i++)
         {
            outfile << hypre_IntArrayData(CF_marker)[i] << "\n";
         }
         outfile.close();
      }
   }

   delete prec;

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, x);

   // 15. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   if (plot_soln)
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh" << problem << "." << setfill('0') << setw(6) << myid;
      sol_name << "sol" << problem << "." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x << flush;
   }

   cout << "Number of finite element unknowns: " << size << endl;

   // 17. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}
