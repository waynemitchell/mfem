//                       J O U L E
//
// Usage:
//    srun -n 8 -p pdebug joule -m ../../data/cylinderHex.mesh -p test
//    srun -n 8 -p pdebug joule -m rod2eb3sshex27.gen -s 22 -dt 0.1 -tf 240.0 -p test
//
// Options:
// -m [string]   the mesh file name
// -o [int]      the order of the basis
// -rs [int]     number of times to serially refine the mesh
// -rp [int]     number of times to refine the mesh in parallel
// -s [int]      time integrator 1=backward Euler, 2=SDIRK2, 3=SDIRK3,
//               22=Midpoint, 23=SDIRK23, 34=SDIRK34
// -tf [double]  the final time
// -dt [double]  time step
// -mu [double]  the magnetic permeability
// -cnd [double] the electrical conductivity
// -f [double]   the frequency of the applied EM BC
// -vis [int]    GLVis -vis = true -no-vis = false
// -vs [int]     visualization step
// -k [string]   base file name for output file
// -print [int]  print solution (gridfunctions) to disk  0 = no, 1 = yes
// -amr [int]    0 = no amr, 1 = amr
// -sc [int]     0 = no static condensation, 1 = use static condensation
// -p [string]   specify the problem to run, "rod", "coil", or "test"
//
// Description:  This examples solves a time dependent eddy current
//               problem, resulting in Joule heating.
//
//               This version has electrostatic potential, Phi, which is a source
//               term in the EM diffusion equation. The potenatial itself is
//               driven by essential BC's
//
//               Div sigma Grad Phi = 0
//               sigma E  =  Curl B/mu - sigma grad Phi
//               dB/dt = - Curl E
//               F = -k Grad T
//               c dT/dt = -Div(F) + sigma E.E,
//
//               where B is the magnetic flux, E is the electric field,
//               T is the temperature, F is the thermal flux,
//               sigma is electrical conductivity, mu is the magnetic
//               permeability, and alpha is the thermal diffusivity.
//               The geometry of the domain is assumed to be as follows:
//
//
//                                   boundary attribute 3
//                                 +---------------------+
//                    boundary --->|                     | boundary
//                    attribute 1  |                     | atribute 2
//                    (driven)     +---------------------+
//
//               The voltage BC condition is essential BC on atribute 1 (front)
//               and 2 (rear) given by function p_bc() at bottom of this file.
//
//               The E-field boundary condition specifies the essential BC (n
//               cross E) on atribute 1 (front) and 2 (rear) given by function
//               edot_bc at bottom of this file. The E-field can be set on
//               attribute 3 also.
//
//               The thermal boundary condition for the flux F is the natural BC
//               on  atribute 1 (front) and 2 (rear). This means that dT/dt = 0
//               on the boundaries, and the initial T = 0.
//
//               See section 2.5 for how the material propertied are assigned to
//               mesh attribiutes, this needs to be changed for different
//               applications.
//
//               See section 8.0 for how the boundary conditions are assigned to
//               mesh attributes, this needs to be changed for different
//               applications.
//
//               This code supports a simple version of AMR, all elements
//               containing material attribute 1 are (optionally) refined.
//
// NOTE:         If the option "-p test" is provided, the code will compute the
//               analytical solution of the electric and magnetic fields and
//               compute the L2 error. This solution is only valid for the
//               particular problem of a right circular cylindrical rod of
//               length 1 and radius 1, and with particular boundary conditions.
//               Example meshes for this test are cylinderHex.mesh,
//               cylinderTet.mesh, rod2eb3sshex27.gen, rod2eb3sstet10.gen.
//               Note that the meshes with the "gen" extension require MFEM to
//               be built with NetCDF.
//
// NOTE:         This code is set up to solve two example problems, 1) a
//               straight metal rod surrounded by air, 2) a metal rod surrounded
//               by a metal coil all surrounded by air. To specify problem (1)
//               use the command line options "-p rod -m rod2eb3sshex27.gen", to
//               specify problem (2) use the command line options "-p coil -m
//               coil_centered_tet10.gen". Problem (1) has two materials and
//               problem (2) has three materials, and the BC's are different.
//
// NOTE:         We write out, optionally, grid functions for P, E, B, W, F, and
//               T. These can be visualized using glvis -np 4 -m mesh.mesh -g E,
//               assuming we used 4 processors.
//


#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>
#include "joule_solver.hpp"
#include "joule_globals.hpp"
#include "../common/pfem_extras.hpp"

#ifdef MFEM_USE_GSL
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_cblas.h>
#include <complex>
#endif

using namespace std;
using namespace mfem;


void display_banner(ostream & os);
void print_banner();

static double aj_ = 0.0;
static double mj_ = 0.0;
static double sj_ = 0.0;
static double wj_ = 0.0;
static double kj_ = 0.0;
static double hj_ = 0.0;
static double dtj_ = 0.0;
static double rj_ = 0.0;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   MPI_Session mpi(argc, argv);
   int myid = mpi.WorldRank();

   // print the cool banner
   if (mpi.Root()) { display_banner(cout); }
   // if (mpi.Root()) { print_banner(); }

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/cylinderHex.mesh";
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int order = 2;
   int ode_solver_type = 1;
   double t_final = 100.0;
   double dt = 0.5;
   double amp = 2.0;
   double mu = 1.0;
   double sigma = 2.0*M_PI*10;
   double Tcapacity = 1.0;
   double Tconductivity = 0.01;
   double alpha = Tconductivity/Tcapacity;
   double freq = 1.0/60.0;
   bool visualization = true;
   bool visit = true;
   int vis_steps = 1;
   int gfprint = 0;
   const char *basename = "Joule";
   int amr = 0;
   int debug = 0;
   const char *problem = "rod";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3\n\t."
                  "\t   22 - Mid-Point, 23 - SDIRK23, 34 - SDIRK34.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&mu, "-mu", "--permeability",
                  "Magnetic permeability coefficient.");
   args.AddOption(&sigma, "-cnd", "--sigma",
                  "Conductivity coefficient.");
   args.AddOption(&freq, "-f", "--frequency",
                  "Frequency of oscillation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&basename, "-k", "--outputfilename",
                  "Name of the visit dump files");
   args.AddOption(&gfprint, "-print", "--print",
                  "Print results (gridfunctions) to disk.");
   args.AddOption(&amr, "-amr", "--amr",
                  "Enable AMR");
   args.AddOption(&STATIC_COND, "-sc", "--static-condensation",
                  "Enable static condesnsation");
   args.AddOption(&debug, "-debug", "--debug",
                  "Print matrices and vectors to disk");
   args.AddOption(&SOLVERPRINTLEVEL, "-hl", "--hypre-print-level",
                  "Hypre print level");
   args.AddOption(&problem, "-p", "--problem",
                  "Name of problem to run");

   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (mpi.Root())
   {
      args.PrintOptions(cout);
   }

   aj_  = amp;
   mj_  = mu;
   sj_  = sigma;
   wj_  = 2.0*M_PI*freq;
   kj_  = sqrt(0.5*wj_*mj_*sj_);
   hj_  = alpha;
   dtj_ = dt;
   rj_  = 1.0;

   if (mpi.Root())
   {
      printf("\n");
      printf("Skin depth sqrt(2.0/(wj*mj*sj)) = %g\n",sqrt(2.0/(wj_*mj_*sj_)));
      printf("Skin depth sqrt(2.0*dt/(mj*sj)) = %g\n",sqrt(2.0*dt/(mj_*sj_)));
   }

   // 3.0
   //
   // Here I assign material properties to mesh attributes.
   // This code is not general, I assume the mesh has 3 regions
   // each with a different integer attribiute 1, 2 or 3.
   //
   // the coil problem has three regions 1) coil, 2) air, 3) the rod
   //
   // the rod problem has two regions 1) rod, 2) air
   //
   // turns out for the rod and coil problem we can use the same material maps

   std::map<int, double> sigmaMap, InvTcondMap, TcapMap, InvTcapMap;
   double sigmaAir;
   double TcondAir;
   double TcapAir;
   if (strcmp(problem,"rod")==0 || strcmp(problem,"coil")==0)
   {
      sigmaAir     = 1.0e-6 * sigma;
      TcondAir     = 1.0e6  * Tconductivity;
      TcapAir      = 1.0    * Tcapacity;
   }
   else if (strcmp(problem,"test")==0)
   {

      if (mpi.Root()) { cout << "Running test problem" << endl; }

      sigmaAir     = 1.0 * sigma;
      TcondAir     = 1.0 * Tconductivity;
      TcapAir      = 1.0 * Tcapacity;

#ifndef MFEM_USE_GSL
      cout << "You selected to run the test prpoblem, but did not"
           " build with GSL.\n"
           "The analytical solution requires GSL." << endl;
#endif

   }
   else
   {
      cerr << "Problem " << problem << " not recognized\n";
      mfem_error();
   }

   if (strcmp(problem,"rod")==0 || strcmp(problem,"coil")==0 ||
       strcmp(problem,"test")==0)
   {

      sigmaMap.insert(pair<int, double>(1, sigma));
      sigmaMap.insert(pair<int, double>(2, sigmaAir));
      sigmaMap.insert(pair<int, double>(3, sigmaAir));

      InvTcondMap.insert(pair<int, double>(1, 1.0/Tconductivity));
      InvTcondMap.insert(pair<int, double>(2, 1.0/TcondAir));
      InvTcondMap.insert(pair<int, double>(3, 1.0/TcondAir));

      TcapMap.insert(pair<int, double>(1, Tcapacity));
      TcapMap.insert(pair<int, double>(2, TcapAir));
      TcapMap.insert(pair<int, double>(3, TcapAir));

      InvTcapMap.insert(pair<int, double>(1, 1.0/Tcapacity));
      InvTcapMap.insert(pair<int, double>(2, 1.0/TcapAir));
      InvTcapMap.insert(pair<int, double>(3, 1.0/TcapAir));
   }
   else
   {
      cerr << "Problem " << problem << " not recognized\n";
      mfem_error();
   }



   // 4. Read the serial mesh from the given mesh file on all processors. We can
   //    handle triangular, quadrilateral, tetrahedral and hexahedral meshes
   //    with the same code.
   Mesh *mesh;
   mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   //
   // 5. Assign the boundary conditions
   //
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   Array<int> thermal_ess_bdr(mesh->bdr_attributes.Max());
   Array<int> poisson_ess_bdr(mesh->bdr_attributes.Max());
   if (strcmp(problem,"coil")==0)
   {

      // BEGIN CODE FOR THE COIL PROBLEM
      // For the coil in a box problem we have surfaces 1) coil end (+),
      // 2) coil end (-), 3) five sides of box, 4) side of box with coil BC

      ess_bdr = 0;
      ess_bdr[0] = 1; // boundary attribute 4 (index 3) is fixed
      ess_bdr[1] = 1; // boundary attribute 4 (index 3) is fixed
      ess_bdr[2] = 1; // boundary attribute 4 (index 3) is fixed
      ess_bdr[3] = 1; // boundary attribute 4 (index 3) is fixed

      // Same as above, but this is for the thermal operator
      // for HDiv formulation the essetial BC is the flux

      thermal_ess_bdr = 0;
      thermal_ess_bdr[2] = 1; // boundary attribute 4 (index 3) is fixed

      // Same as above, but this is for the poisson eq
      // for H1 formulation the essetial BC is the value of Phi

      poisson_ess_bdr = 0;
      poisson_ess_bdr[0] = 1; // boundary attribute 1 (index 0) is fixed
      poisson_ess_bdr[1] = 1; // boundary attribute 2 (index 1) is fixed
      // END CODE FOR THE COIL PROBLEM
   }
   else if (strcmp(problem,"rod")==0 || strcmp(problem,"test")==0)
   {

      // BEGIN CODE FOR THE STRAIGHT ROD PROBLEM
      // the boundary conditions below are for the straight rod problem

      ess_bdr = 0;
      ess_bdr[0] = 1; // boundary attribute 1 (index 0) is fixed (front)
      ess_bdr[1] = 1; // boundary attribute 2 (index 1) is fixed (rear)
      ess_bdr[2] = 1; // boundary attribute 3 (index 3) is fixed (outer)

      // Same as above, but this is for the thermal operator.
      // For HDiv formulation the essetial BC is the flux, which is zero on the
      // front and sides. Note the Natural BC is T = 0 on the outer surface.

      thermal_ess_bdr = 0;
      thermal_ess_bdr[0] = 1; // boundary attribute 1 (index 0) is fixed (front)
      thermal_ess_bdr[1] = 1; // boundary attribute 2 (index 1) is fixed (rear)

      // Same as above, but this is for the poisson eq
      // for H1 formulation the essetial BC is the value of Phi

      poisson_ess_bdr = 0;
      poisson_ess_bdr[0] = 1; // boundary attribute 1 (index 0) is fixed (front)
      poisson_ess_bdr[1] = 1; // boundary attribute 2 (index 1) is fixed (back)
      // END CODE FOR THE STRAIGHT ROD PROBLEM
   }
   else
   {
      cerr << "Problem " << problem << " not recognized\n";
      mfem_error();
   }


   // The following is required for mesh refinement
   mesh->EnsureNCMesh();

   // 6. Define the ODE solver used for time integration. Several implicit
   //    methods are available, including singly diagonal implicit
   //    Runge-Kutta (SDIRK).
   ODESolver *ode_solver;
   switch (ode_solver_type)
   {
      // Implicit L-stable methods
      case 1:  ode_solver = new BackwardEulerSolver; break;
      case 2:  ode_solver = new SDIRK23Solver(2); break;
      case 3:  ode_solver = new SDIRK33Solver; break;
      // Implicit A-stable methods (not L-stable)
      case 22: ode_solver = new ImplicitMidpointSolver; break;
      case 23: ode_solver = new SDIRK23Solver; break;
      case 34: ode_solver = new SDIRK34Solver; break;
      default:
         if (mpi.Root())
         {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         }
         delete mesh;
         return 3;
   }

   // 7. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 8. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }


   // 9. Apply non-uniform non-conforming mesh refinement to the mesh.
   //    The whole metal region is refined, i.e. this is not based on any error
   //    estimator.

   if (amr == 1)
   {
      Array<int> ref_list;
      int numElems = pmesh->GetNE();
      for (int ielem = 0; ielem < numElems; ielem++)
      {
         int thisAtt = pmesh->GetAttribute(ielem);
         if (thisAtt == 1)
         {
            ref_list.Append(ielem);
         }
      }

      pmesh->GeneralRefinement(ref_list);

      ref_list.DeleteAll();
   }

   //
   // 10. Reorient the mesh.
   //
   // Must be done after refinement but before definition
   // of higher order Nedelec spaces

   pmesh->ReorientTetMesh();

   // 11. Rebalance the mesh
   //
   // Since the mesh was adaptively refined in a non-uniform way it will be
   // computationally unbalanced.
   //

   if (pmesh->Nonconforming())
   {
      pmesh->Rebalance();
   }

   // 12. Define the parallel finite element spaces representing.
   //     We'll use H(curl) for electric field
   //     and H(div) for magnetic flux
   //     and H(div) for thermal flux
   //     and H(grad) for electrostatic potential
   //     and L2 for temperature

   // L2 is discontinous "cell-center" bases
   // type 2 is "positive"
   L2_FECollection L2FEC(order-1, dim);

   // ND stands for Nedelec
   ND_FECollection HCurlFEC(order, dim);

   // RT stands for Raviart-Thomas
   RT_FECollection HDivFEC(order-1, dim);

   // H1 is nodal continous "Lagrange" interpolatory bases
   H1_FECollection HGradFEC(order, dim);

   ParFiniteElementSpace    L2FESpace(pmesh, &L2FEC);
   ParFiniteElementSpace HCurlFESpace(pmesh, &HCurlFEC);
   ParFiniteElementSpace  HDivFESpace(pmesh, &HDivFEC);
   ParFiniteElementSpace  HGradFESpace(pmesh, &HGradFEC);

   // The terminology is TrueVSize is the unique (non-redundant) number of dofs
   HYPRE_Int glob_size_l2 =    L2FESpace.GlobalTrueVSize();
   HYPRE_Int glob_size_nd =    HCurlFESpace.GlobalTrueVSize();
   HYPRE_Int glob_size_rt =    HDivFESpace.GlobalTrueVSize();
   HYPRE_Int glob_size_h1 =    HGradFESpace.GlobalTrueVSize();

   if (mpi.Root())
   {
      cout << "Number of Temperature Flux unknowns:  " << glob_size_rt << endl;
      cout << "Number of Temperature unknowns:       " << glob_size_l2 << endl;
      cout << "Number of Electric Field unknowns:    " << glob_size_nd << endl;
      cout << "Number of Magnetic Field unknowns:    " << glob_size_rt << endl;
      cout << "Number of Electrostatic unknowns:     " << glob_size_h1 << endl;
   }

   int Vsize_l2 = L2FESpace.GetVSize();
   int Vsize_nd = HCurlFESpace.GetVSize();
   int Vsize_rt = HDivFESpace.GetVSize();
   int Vsize_h1 = HGradFESpace.GetVSize();

   /* the big BlockVector stores the fields as
   0 Temperature
   1 Temperature Flux
   2 P field
   3 E field
   4 B field
   5 Joule Heating
   */

   Array<int> true_offset(7);
   true_offset[0] = 0;
   true_offset[1] = true_offset[0] + Vsize_l2;
   true_offset[2] = true_offset[1] + Vsize_rt;
   true_offset[3] = true_offset[2] + Vsize_h1;
   true_offset[4] = true_offset[3] + Vsize_nd;
   true_offset[5] = true_offset[4] + Vsize_rt;
   true_offset[6] = true_offset[5] + Vsize_l2;


   // The BlockVector is a large contiguous chunck of memory for storing
   // the required data for the hyprevectors, in this case the temperature L2,
   // the T-flux HDiv, the E-field HCurl, and the B-field HDiv,
   // and scalar potential P
   BlockVector F(true_offset);

   // grid functions E, B, T, F, P, and w which is the Joule heating
   ParGridFunction E_gf, B_gf, T_gf, F_gf, w_gf, P_gf;
   T_gf.MakeRef(&L2FESpace,F,   true_offset[0]);
   F_gf.MakeRef(&HDivFESpace,F, true_offset[1]);
   P_gf.MakeRef(&HGradFESpace,F,true_offset[2]);
   E_gf.MakeRef(&HCurlFESpace,F,true_offset[3]);
   B_gf.MakeRef(&HDivFESpace,F, true_offset[4]);
   w_gf.MakeRef(&L2FESpace,F,   true_offset[5]);

   // This is for visit visualization of exact solution
   ParGridFunction Eexact_gf(&HCurlFESpace);
   ParGridFunction Bexact_gf(&HDivFESpace);
   ParGridFunction Texact_gf(&L2FESpace);

   // 13. Get the boundary conditions, set up the exact solution grid functions
   //
   // These VectorCoefficients have an Eval function.
   // Note that e_exact anf b_exact in this case are exact analytical
   // solutions, taking a 3-vector point as input and returning a 3-vector field
   VectorFunctionCoefficient E_exact(3, e_exact);
   VectorFunctionCoefficient B_exact(3, b_exact);
   FunctionCoefficient T_exact(t_exact);

   E_exact.SetTime(0.0);
   B_exact.SetTime(0.0);

   Eexact_gf.ProjectCoefficient(E_exact);
   Eexact_gf.ProjectCoefficient(B_exact);
   Texact_gf.ProjectCoefficient(T_exact);




   // 14. Initialize the Diffusion operator, the GLVis visualization and print
   //     the initial energies.

   MagneticDiffusionEOperator oper(true_offset[6], L2FESpace, HCurlFESpace,
                                   HDivFESpace, HGradFESpace,
                                   ess_bdr, thermal_ess_bdr, poisson_ess_bdr,
                                   mu, sigmaMap, TcapMap, InvTcapMap,
                                   InvTcondMap);

   // This function initializes all the fields to zero or some provided IC
   oper.Init(F);

   socketstream vis_T, vis_E, vis_B, vis_w, vis_P;
   char vishost[] = "localhost";
   int  visport   = 19916;
   if (visualization)
   {
      // Make sure all ranks have sent their 'v' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());

      vis_T.precision(8);
      vis_E.precision(8);
      vis_B.precision(8);
      vis_P.precision(8);
      vis_w.precision(8);

      int Wx = 0, Wy = 0; // window position
      int Ww = 350, Wh = 350; // window size
      int offx = Ww+10, offy = Wh+45; // window offsets

      miniapps::VisualizeField(vis_P, vishost, visport,
                               P_gf, "Electric Potential (Phi)", Wx, Wy, Ww, Wh);
      Wx += offx;

      miniapps::VisualizeField(vis_E, vishost, visport,
                               E_gf, "Electric Field (E)", Wx, Wy, Ww, Wh);
      Wx += offx;

      miniapps::VisualizeField(vis_B, vishost, visport,
                               B_gf, "Magnetic Field (B)", Wx, Wy, Ww, Wh);
      Wx = 0;
      Wy += offy;

      miniapps::VisualizeField(vis_w, vishost, visport,
                               w_gf, "Joule Heating", Wx, Wy, Ww, Wh);

      Wx += offx;

      miniapps::VisualizeField(vis_T, vishost, visport,
                               T_gf, "Temperature", Wx, Wy, Ww, Wh);

   }

   // visit visualization
   VisItDataCollection visit_dc(basename, pmesh);
   if ( visit )
   {
      visit_dc.RegisterField("E", &E_gf);
      visit_dc.RegisterField("B", &B_gf);
      visit_dc.RegisterField("T", &T_gf);
      visit_dc.RegisterField("w", &w_gf);
      visit_dc.RegisterField("Phi", &P_gf);
      visit_dc.RegisterField("F", &F_gf);
      if (strcmp(problem,"test")==0) { visit_dc.RegisterField("Eexact", &Eexact_gf); }
      if (strcmp(problem,"test")==0) { visit_dc.RegisterField("Bexact", &Bexact_gf); }

      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();
   }

   Vector zero_vec(3); zero_vec = 0.0;
   VectorConstantCoefficient Zero_vec(zero_vec);
   ConstantCoefficient Zero(0.0);
   double eng_E0 = E_gf.ComputeL2Error(Zero_vec);
   double eng_B0 = B_gf.ComputeL2Error(Zero_vec);

   E_exact.SetTime(0.0);
   B_exact.SetTime(0.0);

   double err_E0 = E_gf.ComputeL2Error(E_exact);
   double err_B0 = B_gf.ComputeL2Error(B_exact);

   //double me0 = oper.MagneticEnergy(B_gf);
   double el0 = oper.ElectricLosses(E_gf);

   if (mpi.Root() && (strcmp(problem,"test")==0))
   {
      cout << scientific << setprecision(3) << "initial electric L2 error    = "
           << err_E0/(eng_E0+1.0e-20) << endl;
      cout << scientific << setprecision(3) << "initial magnetic L2 error    = "
           << err_B0/(eng_B0+1.0e-20) << endl;
      cout << scientific << setprecision(3) << "initial electric losses (EL) = "
           << el0 << endl;
   }

   // 10. Perform time-integration (looping over the time iterations, ti, with a
   //     time-step dt).
   //
   // oper is the MagneticDiffusionOperator which has a Mult() method and an
   // ImplicitSolve() method which are used by the time integrators.
   ode_solver->Init(oper);
   double t = 0.0;


   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final - dt/2)
      {
         last_step = true;
      }

      // F is the vector of dofs, t is the current time, and dt is the
      // time step to advance.
      ode_solver->Step(F, t, dt);

      // update the exact solution GF
      if (strcmp(problem,"test")==0)
      {
         E_exact.SetTime(t);
         Eexact_gf.ProjectCoefficient(E_exact);
         B_exact.SetTime(t);
         Bexact_gf.ProjectCoefficient(B_exact);
         Texact_gf.ProjectCoefficient(T_exact);
      }

      if (debug == 1)
      {
         oper.Debug(basename,t);
      }

      if (gfprint == 1)
      {

         ostringstream T_name, E_name, B_name, F_name, w_name, P_name, mesh_name;
         T_name << basename << "_" << setfill('0') << setw(6) << t << "_"
                << "T." << setfill('0') << setw(6) << myid;
         E_name << basename << "_" << setfill('0') << setw(6) << t << "_"
                << "E." << setfill('0') << setw(6) << myid;
         B_name << basename << "_" << setfill('0') << setw(6) << t << "_"
                << "B." << setfill('0') << setw(6) << myid;
         F_name << basename << "_" << setfill('0') << setw(6) << t << "_"
                << "F." << setfill('0') << setw(6) << myid;
         w_name << basename << "_" << setfill('0') << setw(6) << t << "_"
                << "w." << setfill('0') << setw(6) << myid;
         P_name << basename << "_" << setfill('0') << setw(6) << t << "_"
                << "P." << setfill('0') << setw(6) << myid;
         mesh_name << basename << "_" << setfill('0') << setw(6) << t << "_"
                   << "mesh." << setfill('0') << setw(6) << myid;

         ofstream mesh_ofs(mesh_name.str().c_str());
         mesh_ofs.precision(8);
         pmesh->Print(mesh_ofs);
         mesh_ofs.close();

         ofstream T_ofs(T_name.str().c_str());
         T_ofs.precision(8);
         T_gf.Save(T_ofs);
         T_ofs.close();

         ofstream E_ofs(E_name.str().c_str());
         E_ofs.precision(8);
         E_gf.Save(E_ofs);
         E_ofs.close();

         ofstream B_ofs(B_name.str().c_str());
         B_ofs.precision(8);
         B_gf.Save(B_ofs);
         B_ofs.close();

         ofstream F_ofs(F_name.str().c_str());
         F_ofs.precision(8);
         F_gf.Save(B_ofs);
         F_ofs.close();

         ofstream P_ofs(P_name.str().c_str());
         P_ofs.precision(8);
         P_gf.Save(P_ofs);
         P_ofs.close();

         ofstream w_ofs(w_name.str().c_str());
         w_ofs.precision(8);
         w_gf.Save(w_ofs);
         w_ofs.close();
      }

      if (last_step || (ti % vis_steps) == 0)
      {
         if (mpi.Root())
         {
            cout << fixed;
            cout << "step " << setw(6) << ti << " t = " << setw(6)
                 << setprecision(3) << t;
         }

         if (strcmp(problem,"test")==0)
         {
            double eng_E = E_gf.ComputeL2Error(Zero_vec);
            double eng_B = B_gf.ComputeL2Error(Zero_vec);
            double eng_T = T_gf.ComputeL2Error(Zero);

            double err_E = E_gf.ComputeL2Error(E_exact);
            double err_B = B_gf.ComputeL2Error(B_exact);
            double err_T = T_gf.ComputeL2Error(T_exact);

            if (mpi.Root())
            {
               cout << " relative errors "  << scientific
                    << setprecision(3) << err_E/(eng_E+1.0e-20) << " "
                    << setprecision(3) << err_B/(eng_B+1.0e-20) << " "
                    << setprecision(3) << err_T/(eng_T+1.0e-20);
            }
         }

         if (mpi.Root()) { cout << endl; }


         // Make sure all ranks have sent their 'v' solution before initiating
         // another set of GLVis connections (one from each rank):
         MPI_Barrier(pmesh->GetComm());

         if (visualization)
         {

            int Wx = 0, Wy = 0; // window position
            int Ww = 350, Wh = 350; // window size
            int offx = Ww+10, offy = Wh+45; // window offsets

            miniapps::VisualizeField(vis_P, vishost, visport,
                                     P_gf, "Electric Potential (Phi)", Wx, Wy, Ww, Wh);
            Wx += offx;

            miniapps::VisualizeField(vis_E, vishost, visport,
                                     E_gf, "Electric Field (E)", Wx, Wy, Ww, Wh);
            Wx += offx;

            miniapps::VisualizeField(vis_B, vishost, visport,
                                     B_gf, "Magnetic Field (B)", Wx, Wy, Ww, Wh);

            Wx = 0;
            Wy += offy;

            miniapps::VisualizeField(vis_w, vishost, visport,
                                     w_gf, "Joule Heating", Wx, Wy, Ww, Wh);

            Wx += offx;

            miniapps::VisualizeField(vis_T, vishost, visport,
                                     T_gf, "Temperature", Wx, Wy, Ww, Wh);
         }

         if (visit)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();
         }
      }
   }
   if (visualization)
   {
      vis_T.close();
      vis_E.close();
      vis_B.close();
      vis_w.close();
      vis_P.close();
   }


   // 15. Free the used memory.
   delete ode_solver;
   delete pmesh;

   return 0;
}

void edot_bc(const Vector &x, Vector &E)
{
   E = 0.0;
}

void e_exact(const Vector &x, double t, Vector &E)
{

   E[0] = 0.0;
   E[1] = 0.0;
   E[2] = 0.0;

#ifdef MFEM_USE_GSL

   // formula for current in the z-direction is function of (r,t)
   // requires bessels functions with complex arugments
   // and these are approximated by a finite series of bessel
   // functions with real arguments

   double r      = sqrt(x[0]*x[0] + x[1]*x[1]);
   double k_real = 1.0/sqrt(2.0) * sqrt(wj_*mj_*sj_);
   double k_imag = -k_real;
   int sign      = -1;

   double besselJ0kr_real = 0.0;
   double besselJ0kr_imag = 0.0;
   double besselJ0kR_real = 0.0;
   double besselJ0kR_imag = 0.0;

   for (int m = -10; m <= 10; m++ )
   {
      double a = gsl_sf_bessel_Jn(sign*m,k_real*r);
      double b = gsl_sf_bessel_In(m,k_imag*r);
      besselJ0kr_real += a*b*cos(m*1.57079632);
      besselJ0kr_imag += a*b*sin(m*1.57079632);
      a = gsl_sf_bessel_Jn(sign*m,k_real*rj_);
      b = gsl_sf_bessel_In(m,k_imag*rj_);
      besselJ0kR_real += a*b*cos(m*1.57079632);
      besselJ0kR_imag += a*b*sin(m*1.57079632);

   }

   complex<double>  besselJ0kr(besselJ0kr_real,besselJ0kr_imag);
   complex<double>  besselJ0kR(besselJ0kR_real,besselJ0kR_imag);
   complex<double>  sinc(cos(wj_*t),sin(wj_*t));
   complex<double>  Jcmplx = sinc * besselJ0kr / besselJ0kR;
   double Ereal = real(Jcmplx);

   E[0] = 0.0;
   E[1] = 0.0;
   E[2] = aj_*Ereal;
#endif

}

void b_exact(const Vector &x, double t, Vector &B)
{

   B[0] = 0.0;
   B[1] = 0.0;
   B[2] = 0.0;

#ifdef MFEM_USE_GSL

   // formula for B-field in the theta-direction is function of (r,t)
   // requires bessels functions with complex arugments
   // and these are approximated by a finite series of bessel
   // functions with real arguments

   double r      = sqrt(x[0]*x[0] + x[1]*x[1]);
   double k_real = 1.0/sqrt(2.0) * sqrt(wj_*mj_*sj_);
   double k_imag = -k_real;
   int sign      = -1;

   //  d/dr J0[k r] = -k J1[k r]

   double besselJ1kr_real = 0.0;
   double besselJ1kr_imag = 0.0;
   double besselJ0kR_real = 0.0;
   double besselJ0kR_imag = 0.0;

   // J0[a + b] = sum Jk[a]*Jk[b],     k = -inf, inf
   // J1[a + b] = sum J(1+k)[a]*Jk[b], k = -inf, inf

   // Jm[i x] = i^m Im[x]

   for (int m = -10; m <= 10; m++ )
   {
      double a = gsl_sf_bessel_Jn(1+sign*m,k_real*r);
      double b = gsl_sf_bessel_In(m,k_imag*r);
      besselJ1kr_real += a*b*cos(m*1.57079632);
      besselJ1kr_imag += a*b*sin(m*1.57079632);
      a = gsl_sf_bessel_Jn(sign*m,k_real*rj_);
      b = gsl_sf_bessel_In(m,k_imag*rj_);
      besselJ0kR_real += a*b*cos(m*1.57079632);
      besselJ0kR_imag += a*b*sin(m*1.57079632);

   }

   complex<double> besselJ1kr(besselJ1kr_real,besselJ1kr_imag);
   complex<double> besselJ0kR(besselJ0kR_real,besselJ0kR_imag);
   complex<double> sinc(cos(wj_*t),sin(wj_*t));
   complex<double> kcmplx(k_real,k_imag);
   complex<double> Bcmplx =
      kcmplx / (complex<double>(0,1)*wj_) * sinc * besselJ1kr / besselJ0kR;
   double Breal = -1.0*real(Bcmplx);

   B[0] = -x[1]/r*aj_*Breal;
   B[1] =  x[0]/r*aj_*Breal;
   B[2] =  0.0;

#endif

}


double t_exact(Vector &x)
{
   double T = 0.0;
   return T;
}

double p_bc(const Vector &x, double t)
{

   // the value
   double T;
   if (x[0] < 0.0)
   {
      T = 1.0;
   }
   else
   {
      T = -1.0;
   }

   return T*cos(wj_ * t);
}

void display_banner(ostream & os)
{
   os << "     ____.            .__          " << endl
      << "    |    | ____  __ __|  |   ____  " << endl
      << "    |    |/  _ \\|  |  \\  | _/ __ \\ " << endl
      << "/\\__|    (  <_> )  |  /  |_\\  ___/ " << endl
      << "\\________|\\____/|____/|____/\\___  >" << endl
      << "                                \\/ " << endl << flush;
}

void print_banner()
{

   char banner[219] =
   {
      32,
      32,
      32,
      32,
      32,
      95,
      95,
      95,
      95,
      46,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      46,
      95,
      95,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      10,
      32,
      32,
      32,
      32,
      124,
      32,
      32,
      32,
      32,
      124,
      32,
      95,
      95,
      95,
      95,
      32,
      32,
      95,
      95,
      32,
      95,
      95,
      124,
      32,
      32,
      124,
      32,
      32,
      32,
      95,
      95,
      95,
      95,
      32,
      32,
      10,
      32,
      32,
      32,
      32,
      124,
      32,
      32,
      32,
      32,
      124,
      47,
      32,
      32,
      95,
      32,
      92,
      124,
      32,
      32,
      124,
      32,
      32,
      92,
      32,
      32,
      124,
      32,
      95,
      47,
      32,
      95,
      95,
      32,
      92,
      32,
      10,
      47,
      92,
      95,
      95,
      124,
      32,
      32,
      32,
      32,
      40,
      32,
      32,
      60,
      95,
      62,
      32,
      41,
      32,
      32,
      124,
      32,
      32,
      47,
      32,
      32,
      124,
      95,
      92,
      32,
      32,
      95,
      95,
      95,
      47,
      32,
      10,
      92,
      95,
      95,
      95,
      95,
      95,
      95,
      95,
      95,
      124,
      92,
      95,
      95,
      95,
      95,
      47,
      124,
      95,
      95,
      95,
      95,
      47,
      124,
      95,
      95,
      95,
      95,
      47,
      92,
      95,
      95,
      95,
      32,
      32,
      62,
      10,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      32,
      92,
      47,
      32,
      10,
      10,
      10,
      0
   };

   printf("%s",banner);

}
