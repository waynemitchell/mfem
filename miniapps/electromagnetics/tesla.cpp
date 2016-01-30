//               MFEM Tesla Mini App
//               Simple Magnetostatics Simulation Code
//
// Compile with: make tesla
//
// Sample runs:
//
//   By default the sources and fields are all zero
//     mpirun -np 4 tesla
//
//   A cylindrical bar magnet in a metal sphere
//     mpirun -np 4 tesla -bm '0 -0.5 0 0 0.5 0 0.2 1'
//
//   A spherical shell of paramagnetic material in a uniform B field
//     mpirun -np 4 tesla -ubbc '0 0 1' -ms '0 0 0 0.2 0.4 10'
//
//   A ring of current in a metal sphere
//     mpirun -np 4 tesla -cr '0 0 -0.2 0 0 0.2 0.2 0.4 1'
//
//   An example demonstrating the use of surface currents
//     mpirun -np 4 tesla -m ./square-angled-pipe.mesh
//                        -kbcs '3' -vbcs '1 2' -vbcv '-0.5 0.5'
//
// Description:
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "pfem_extras.hpp"

using namespace std;
using namespace mfem;

// Permeability Function
static Vector ms_params_(0);  // Center, Inner and Outer Radii, and
//                               Permeability of magnetic shell
double magnetic_shell(const Vector &);
double muInv(const Vector & x) { return 1.0/magnetic_shell(x); }

// Current Density Function
static Vector cr_params_(0);  // Axis Start, Axis End, Inner Ring Radius,
//                               Outer Ring Radius, and Total Current
//                               of current ring (annulus)
void current_ring(const Vector &, Vector &);

// Magnetization
static Vector bm_params_(0);  // Axis Start, Axis End, Bar Radius,
//                               and Magnetic Field Magnitude
void bar_magnet(const Vector &, Vector &);

// A Field Boundary Condition for B = (Bx,By,Bz)
static Vector b_uniform_(0);
void a_bc_uniform(const Vector &, Vector&);

// Phi_M Boundary Condition for H = (0,0,1)
double phi_m_bc_uniform(const Vector &x);

// Physical Constants
// Permeability of Free Space (units H/m)
static double mu0_ = 4.0e-7*M_PI;

class SurfaceCurrent
{
public:
  SurfaceCurrent(ParFiniteElementSpace & H1FESpace,
		 ParFiniteElementSpace & HCurlFESpace,
		 ParDiscreteGradOperator & Grad,
		 Array<int> & kbcs, Array<int> & vbcs, Vector & vbcv);
  ~SurfaceCurrent();

  void ComputeSurfaceCurrent(ParGridFunction & k);

  void Update();

  // const ParGridFunction & GetPsi() { return *psi_; }
  ParGridFunction * GetPsi() { return psi_; }

private:
  ParFiniteElementSpace   * H1FESpace_;
  ParFiniteElementSpace   * HCurlFESpace_;
  ParDiscreteGradOperator * Grad_;
  Array<int>              * kbcs_;
  Array<int>              * vbcs_;
  Vector                  * vbcv_;

  ParBilinearForm * s0_;
  ParGridFunction * psi_;

  HypreBoomerAMG  * amg_;
  HyprePCG        * pcg_;
  HypreParMatrix  * S0_;
  HypreParVector  * PSI_;
  HypreParVector  * RHS_;
  HypreParVector  * K_;

  Array<int> ess_bdr_;
  Array<int> non_k_bdr_;
};

int main(int argc, char *argv[])
{
   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Parse command-line options.
   const char *mesh_file = "./butterfly_3d.mesh";
   int order = 1;
   int sr = 0, pr = 0;
   bool visualization = true;
   bool visit = true;

   Array<int> kbcs;
   Array<int> vbcs;

   Vector vbcv;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&sr, "-rs", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&pr, "-rp", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   args.AddOption(&b_uniform_, "-ubbc", "--uniform-b-bc",
                  "Specify if the three components of the constant magnetic flux density");
   args.AddOption(&ms_params_, "-ms", "--magnetic-shell-params",
                  "Center, Inner Radius, Outer Radius, and Permeability of Magnetic Shell");
   args.AddOption(&cr_params_, "-cr", "--current-ring-params",
                  "Axis End Points, Inner Radius, Outer Radius and Total Current of Annulus");
   args.AddOption(&bm_params_, "-bm", "--bar-magnet-params",
                  "Axis End Points, Radius, and Magnetic Field of Cylindrical Magnet");
   args.AddOption(&kbcs, "-kbcs", "--surface-current-bc",
                  "Surfaces for the Surface Current (K) Boundary Condition");
   args.AddOption(&vbcs, "-vbcs", "--voltage-bc-surf",
                  "Voltage Boundary Condition Surfaces (to drive K)");
   args.AddOption(&vbcv, "-vbcv", "--voltage-bc-vals",
                  "Voltage Boundary Condition Values (to drive K)");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit",
                  "--no-visualization",
                  "Enable or disable VisIt visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.
   Mesh *mesh;
   ifstream imesh(mesh_file);
   if (!imesh)
   {
      if (myid == 0)
      {
         cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
      }
      MPI_Finalize();
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();

   int sdim = mesh->SpaceDimension();
   int dim = mesh->Dimension();

   if ( b_uniform_.Size() != 3 )
   {
     // Set the default boundary condition to B = (0,0,0)
     b_uniform_.SetSize(3);
     b_uniform_ = 0.0;
   }

   if ( ms_params_.Size() != sdim + 3 )
   {
      // The magnetic shell parameters have not been set.
      // We will set them to default values.
      ms_params_.SetSize(sdim+3);
      ms_params_ = 0.0;
      ms_params_(sdim+2) = 1.0;
   }
   if ( cr_params_.Size() != 2*sdim + 3 )
   {
      // The current ring parameters have not been set.
      // We will set them to default values.
      cr_params_.SetSize(2*sdim+3);
      cr_params_ = 0.0;
   }
   if ( bm_params_.Size() != 2*sdim + 2 )
   {
      // The bar magnet parameters have not been set.
      // We will set them to default values.
      bm_params_.SetSize(2*sdim+2);
      bm_params_ = 0.0;
   }

   // If values for Voltage BCs were not set issue a warning and exit
   if ( ( vbcs.Size() > 0 && kbcs.Size() == 0 ) ||
	( kbcs.Size() > 0 && vbcs.Size() == 0 ) ||
	( vbcv.Size() < vbcs.Size() ) )
   {
     if ( myid == 0 ) {
       cout << "The surface current (K) boundary condition requires "
	    << "surface current boundary condition surfaces (with -kbcs), "
	    << "voltage boundary condition surface (with -vbcs), "
	    << "and voltage boundary condition values (with -vbcv)."
	    << endl;
     }
      MPI_Finalize();
      return 3;
   }

   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement.
   {
      int ref_levels = sr;
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }
   mesh->EnsureNCMesh();

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Refine this mesh in parallel to increase the resolution.
   int par_ref_levels = pr;
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }

   socketstream a_sock, b_sock, h_sock, j_sock, k_sock, m_sock, p_sock;
   char vishost[] = "localhost";
   int  visport   = 19916;
   if (visualization)
   {
     if ( myid == 0 )
     {
       cout << "Initializing GLVis sockets" << endl;
     }

      a_sock.open(vishost, visport);
      a_sock.precision(8);

      MPI_Barrier(MPI_COMM_WORLD);

      b_sock.open(vishost, visport);
      b_sock.precision(8);

      MPI_Barrier(MPI_COMM_WORLD);

      h_sock.open(vishost, visport);
      h_sock.precision(8);

      MPI_Barrier(MPI_COMM_WORLD);

      j_sock.open(vishost, visport);
      j_sock.precision(8);

      MPI_Barrier(MPI_COMM_WORLD);

      k_sock.open(vishost, visport);
      k_sock.precision(8);

      MPI_Barrier(MPI_COMM_WORLD);

      m_sock.open(vishost, visport);
      m_sock.precision(8);

      MPI_Barrier(MPI_COMM_WORLD);

      p_sock.open(vishost, visport);
      p_sock.precision(8);
   }

   // Define compatible parallel finite element spaces on the parallel
   // mesh. Here we use arbitrary order H1 and Nedelec finite
   // elements.
   H1_ParFESpace H1FESpace(&pmesh,order,dim);
   ND_ParFESpace HCurlFESpace(&pmesh,order,dim);
   RT_ParFESpace HDivFESpace(&pmesh,order,dim);

   // Select DoFs on the requested surfaces as Dirichlet BCs
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   Array<int> non_k_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;
   non_k_bdr = 1;

   for (int i=0; i<kbcs.Size(); i++)
   {
     non_k_bdr[kbcs[i]-1] = 0;
   }

   // Set up the parallel bilinear form corresponding to the
   // electrostatic operator div eps grad, by adding the diffusion
   // domain integrator and finally imposing Dirichlet boundary
   // conditions. The boundary conditions are implemented by marking
   // all the boundary attributes from the mesh as essential
   // (Dirichlet). After serial and parallel assembly we extract the
   // parallel matrix A.
   FunctionCoefficient muInv_coef(muInv);
   VectorFunctionCoefficient j_coef(sdim,current_ring);
   VectorFunctionCoefficient m_coef(sdim,bar_magnet);

   ParBilinearForm curlMuInvCurl(&HCurlFESpace);
   curlMuInvCurl.AddDomainIntegrator(new CurlCurlIntegrator(muInv_coef));

   ParBilinearForm mass(&HCurlFESpace);
   mass.AddDomainIntegrator(new VectorFEMassIntegrator);

   ParBilinearForm massMuInv(&HDivFESpace);
   massMuInv.AddDomainIntegrator(new VectorFEMassIntegrator);

   // The gradient operator needed to compute H from PhiM
   ParDiscreteGradOperator Grad(&H1FESpace, &HCurlFESpace);

   // The curl operator needed to compute B from A
   ParDiscreteCurlOperator Curl(&HCurlFESpace, &HDivFESpace);

   // The projector needed to coerce J into the range of the CurlCurl operator
   DivergenceFreeProjector DivFreeProj(HCurlFESpace, H1FESpace);

   // Object to solve the subproblem of computing surface currents
   SurfaceCurrent SurfCur(H1FESpace, HCurlFESpace, Grad,
			  kbcs, vbcs, vbcv);

   // Create various grid functions
   // ParGridFunction phi_m(&H1FESpace);  // Magnetic Scalar Potential
   ParGridFunction a(&HCurlFESpace);   // Magnetic Potential
   ParGridFunction j(&HCurlFESpace);   // Volumetric Current Density
   ParGridFunction k(&HCurlFESpace);   // Surface Current Density
   ParGridFunction h(&HCurlFESpace);   // Magnetic Field
   ParGridFunction b(&HDivFESpace);    // Magnetic Flux Density
   ParGridFunction m(&HDivFESpace);    // Magnetization

   // Create coefficient for optional applied field
   VectorFunctionCoefficient a_bc(sdim,a_bc_uniform);

   // Setup VisIt visualization class
   VisItDataCollection visit_dc("Tesla-AMR-Parallel", &pmesh);

   if ( visit )
   {
     // visit_dc.RegisterField("Phi_M", &phi_m);
      visit_dc.RegisterField("Psi", SurfCur.GetPsi());
      visit_dc.RegisterField("A", &a);
      visit_dc.RegisterField("J", &j);
      visit_dc.RegisterField("K", &k);
      visit_dc.RegisterField("H", &h);
      visit_dc.RegisterField("B", &b);
      visit_dc.RegisterField("M", &m);
   }

   // The main AMR loop. In each iteration we solve the problem on the
   // current mesh, visualize the solution, estimate the error on all
   // elements, refine the worst elements and update all objects to work
   // with the new mesh.
   const int max_dofs = 200000;
   for (int it = 1; it <= 100; it++)
   {
      HYPRE_Int size_h1 = H1FESpace.GlobalTrueVSize();
      HYPRE_Int size_nd = HCurlFESpace.GlobalTrueVSize();
      HYPRE_Int size_rt = HDivFESpace.GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "\nIteration " << it << endl;
         cout << "Number of H1      unknowns: " << size_h1 << endl;
         cout << "Number of H(Curl) unknowns: " << size_nd << endl;
         cout << "Number of H(Div)  unknowns: " << size_rt << endl;
      }

      // Assemble Matrices
      curlMuInvCurl.Assemble();
      curlMuInvCurl.Finalize();

      mass.Assemble();
      mass.Finalize();

      massMuInv.Assemble();
      massMuInv.Finalize();

      // Initialize the magnetic vector potential with its boundary conditions
      a = 0.0;

      if ( kbcs.Size() > 0 )
      {
	SurfCur.ComputeSurfaceCurrent(k);
	a = k;
      }

      // Apply uniform B boundary condition on remaining surfaces
      a.ProjectBdrCoefficientTangent(a_bc, non_k_bdr);

      // Initialize the volumetric current density
      j.ProjectCoefficient(j_coef);

      m.ProjectCoefficient(m_coef);

      HypreParMatrix *Mass      = mass.ParallelAssemble();
      HypreParMatrix *MassMuInv = massMuInv.ParallelAssemble();

      HypreParVector *J    = j.ParallelProject();
      HypreParVector *M    = m.ParallelProject();
      HypreParVector *JD   = new HypreParVector(&HCurlFESpace);
      HypreParVector *Tmp  = new HypreParVector(&HDivFESpace);

      Mass->Mult(*J,*JD);

      cout << "Norm of J:  " << JD->Norml2() << endl;

      MassMuInv->Mult(*M,*Tmp);
      *Tmp *= mu0_;
      Curl.MultTranspose(*Tmp,*JD,1.0,1.0);

      delete M;

      cout << "Norm of J+Curl M:  " << JD->Norml2() << endl;

      {
         HyprePCG *pcgm = new HyprePCG(*Mass);
         pcgm->SetTol(1e-12);
         pcgm->SetMaxIter(500);
         pcgm->SetPrintLevel(0);
         pcgm->Mult(*JD, *J);
         j = *J;
         delete pcgm;
      }
      delete J;
      delete Mass;
      delete MassMuInv;
      delete Tmp;

      // Apply Dirichlet BCs to matrix and right hand side
      HypreParMatrix *CurlMuInvCurl = curlMuInvCurl.ParallelAssemble();
      HypreParVector *A             = a.ParallelProject();

      HypreParVector *RHS  = new HypreParVector(&HCurlFESpace);
      DivFreeProj.Mult(*JD, *RHS);
      delete JD;

      cout << "Norm of Div Free J+Curl M:  " << RHS->Norml2() << endl;
      cout << "Norm of A:  " << A->Norml2() << endl;

      // Apply the boundary conditions to the assembled matrix and vectors
      curlMuInvCurl.ParallelEliminateEssentialBC(ess_bdr,
						 *CurlMuInvCurl,
						 *A, *RHS);

      // Define and apply a parallel PCG solver for AX=B with the AMS
      // preconditioner from hypre.
      HypreAMS *ams = new HypreAMS(*CurlMuInvCurl, &HCurlFESpace);
      ams->SetSingularProblem();

      HyprePCG *pcg = new HyprePCG(*CurlMuInvCurl);
      pcg->SetTol(1e-12);
      pcg->SetMaxIter(500);
      pcg->SetPrintLevel(2);
      pcg->SetPreconditioner(*ams);
      pcg->Mult(*RHS, *A);

      delete ams;
      delete pcg;
      delete CurlMuInvCurl;
      delete RHS;

      // Extract the parallel grid function corresponding to the finite
      // element approximation Phi. This is the local solution on each
      // processor.
      a = *A;

      // Compute the negative Gradient of the solution vector.  This is
      // the magnetic field corresponding to the scalar potential
      // represented by phi.
      HypreParVector *B = new HypreParVector(&HDivFESpace);
      Curl.Mult(*A,*B);
      b = *B;

      delete A;
      delete B;

      if ( visit )
      {
         visit_dc.SetCycle(it+1);
         visit_dc.SetTime(size_nd);
         visit_dc.Save();
      }

      // Send the solution by socket to a GLVis server.
      if (visualization)
      {

         p_sock << "parallel " << num_procs << " " << myid << "\n";
         p_sock << "solution\n" << pmesh << *SurfCur.GetPsi()
                << "window_title 'Surface Current Potential (Psi)'\n" << flush;

         MPI_Barrier(pmesh.GetComm());

         a_sock << "parallel " << num_procs << " " << myid << "\n";
         a_sock << "solution\n" << pmesh << a
                << "window_title 'Vector Potential (A)'\n"
                << flush;

         MPI_Barrier(pmesh.GetComm());

         b_sock << "parallel " << num_procs << " " << myid << "\n";
         b_sock << "solution\n" << pmesh << b
                << "window_title 'Magnetic Flux Density (B)'\n" << flush;

         MPI_Barrier(pmesh.GetComm());

         h_sock << "parallel " << num_procs << " " << myid << "\n";
         h_sock << "solution\n" << pmesh << h
                << "window_title 'Magnetic Field (H)'\n" << flush;

         MPI_Barrier(pmesh.GetComm());

         j_sock << "parallel " << num_procs << " " << myid << "\n";
         j_sock << "solution\n" << pmesh << j
                << "window_title 'Current Density (J)'\n" << flush;

         MPI_Barrier(pmesh.GetComm());

         k_sock << "parallel " << num_procs << " " << myid << "\n";
         k_sock << "solution\n" << pmesh << k
                << "window_title 'Surface Current Density (K)'\n" << flush;

         MPI_Barrier(pmesh.GetComm());

         m_sock << "parallel " << num_procs << " " << myid << "\n";
         m_sock << "solution\n" << pmesh << m
                << "window_title 'Magnetization (M)'\n" << flush;
      }

      if (size_nd > max_dofs)
      {
         break;
      }

      // Estimate element errors using the Zienkiewicz-Zhu error estimator.
      // The bilinear form integrator must have the 'ComputeElementFlux'
      // method defined.
      Vector errors(pmesh.GetNE());
      {
         //errors.Randomize();
         // Space for the discontinuous (original) flux
         CurlCurlIntegrator flux_integrator(muInv_coef);
         RT_FECollection flux_fec(order-1, sdim);
         ParFiniteElementSpace flux_fes(&pmesh, &flux_fec);

         // Space for the smoothed (conforming) flux
         double norm_p = 1;
         ND_FECollection smooth_flux_fec(order, dim);
         ParFiniteElementSpace smooth_flux_fes(&pmesh, &smooth_flux_fec);

         // Another possible set of options for the smoothed flux space:
         // norm_p = 1;
         // H1_FECollection smooth_flux_fec(order, dim);
         // ParFiniteElementSpace smooth_flux_fes(&pmesh, &smooth_flux_fec, dim);

         L2ZZErrorEstimator(flux_integrator, a,
                            smooth_flux_fes, flux_fes, errors, norm_p);
      }
      double local_max_err = errors.Max();
      double global_max_err;
      MPI_Allreduce(&local_max_err, &global_max_err, 1,
                    MPI_DOUBLE, MPI_MAX, pmesh.GetComm());

      // Make a list of elements whose error is larger than a fraction
      // of the maximum element error. These elements will be refined.
      Array<int> ref_list;
      const double frac = 0.5;
      double threshold = frac * global_max_err;
      for (int i = 0; i < errors.Size(); i++)
      {
         if (errors[i] >= threshold) { ref_list.Append(i); }
      }

      // Refine the selected elements. Since we are going to transfer the
      // grid function x from the coarse mesh to the new fine mesh in the
      // next step, we need to request the "two-level state" of the mesh.
      pmesh.GeneralRefinement(ref_list);

      // Update the space to reflect the new state of the mesh. Also,
      // interpolate the solution x so that it lies in the new space but
      // represents the same function. This saves solver iterations since
      // we'll have a good initial guess of x in the next step.
      // The interpolation algorithm needs the mesh to hold some information
      // about the previous state, which is why the call UseTwoLevelState
      // above is required.
      //fespace.UpdateAndInterpolate(&x);

      // Note: If interpolation was not needed, we could just use the following
      //     six calls to update the space and the grid function. (No need to
      //     call UseTwoLevelState in this case.)
      H1FESpace.Update();
      HCurlFESpace.Update();
      HDivFESpace.Update();
      Grad.Update();
      Curl.Update();
      DivFreeProj.Update();
      SurfCur.Update();
      a.Update();
      j.Update();
      k.Update();
      h.Update();
      b.Update();
      m.Update();

      // Inform the bilinear forms that the space has changed.
      mass.Update();
      massMuInv.Update();
      curlMuInvCurl.Update();

      char c;
      if (myid == 0)
      {
         cout << "press (q)uit or (c)ontinue --> " << flush;
         cin >> c;
      }
      MPI_Bcast(&c, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

      if (c != 'c')
      {
         break;
      }
   }

   MPI_Finalize();

   return 0;
}

SurfaceCurrent::SurfaceCurrent(ParFiniteElementSpace & H1FESpace,
			       ParFiniteElementSpace & HCurlFESpace,
			       ParDiscreteGradOperator & Grad,
			       Array<int> & kbcs,
			       Array<int> & vbcs, Vector & vbcv)
  : H1FESpace_(&H1FESpace),
    HCurlFESpace_(&HCurlFESpace),
    Grad_(&Grad),
    kbcs_(&kbcs),
    vbcs_(&vbcs),
    vbcv_(&vbcv)
{
  s0_ = new ParBilinearForm(H1FESpace_);
  s0_->AddBoundaryIntegrator(new DiffusionIntegrator);
  s0_->Assemble();
  s0_->Finalize();
  S0_ = s0_->ParallelAssemble();

  amg_ = new HypreBoomerAMG(*S0_);
  amg_->SetPrintLevel(0);

  pcg_ = new HyprePCG(*S0_);
  pcg_->SetTol(1e-12);
  pcg_->SetMaxIter(200);
  pcg_->SetPrintLevel(0);
  pcg_->SetPreconditioner(*amg_);

  ess_bdr_.SetSize(H1FESpace_->GetParMesh()->bdr_attributes.Max());
  ess_bdr_ = 0;
  for (int i=0; i<vbcs_->Size(); i++)
  {
    ess_bdr_[(*vbcs_)[i]-1] = 1;
  }

  non_k_bdr_.SetSize(H1FESpace_->GetParMesh()->bdr_attributes.Max());
  non_k_bdr_ = 1;
  for (int i=0; i<kbcs_->Size(); i++)
  {
    non_k_bdr_[(*kbcs_)[i]-1] = 0;
  }

  psi_ = new ParGridFunction(H1FESpace_);
  *psi_ = 0.0;

  // Apply piecewise constant voltage boundary condition
  Array<int> vbc_bdr_attr(H1FESpace_->GetParMesh()->bdr_attributes.Max());
  for (int i=0; i<vbcs_->Size(); i++)
  {
    ConstantCoefficient voltage((*vbcv_)[i]);
    vbc_bdr_attr = 0;
    vbc_bdr_attr[(*vbcs_)[i]-1] = 1;
    psi_->ProjectBdrCoefficient(voltage, vbc_bdr_attr);
  }

  PSI_ = psi_->ParallelProject();
  RHS_ = new HypreParVector(H1FESpace_);
  K_   = new HypreParVector(HCurlFESpace_);

  s0_->ParallelEliminateEssentialBC(ess_bdr_,
				    *S0_,
				    *PSI_, *RHS_);  
}

SurfaceCurrent::~SurfaceCurrent()
{
  delete pcg_;
  delete amg_;
  delete S0_;
  delete PSI_;
  delete RHS_;
  delete K_;
  delete s0_;
  delete psi_;
}

void
SurfaceCurrent::ComputeSurfaceCurrent(ParGridFunction & k)
{
  k = 0.0;
  pcg_->Mult(*RHS_, *PSI_);
  PSI_->Print("PSI.vec");
  S0_->Print("S0.mat");
  *psi_ = *PSI_;
  Grad_->Mult(*PSI_,*K_);
  k = *K_;

  Vector vZero(3); vZero = 0.0;
  VectorConstantCoefficient Zero(vZero);
  k.ProjectBdrCoefficientTangent(Zero,non_k_bdr_);
}

void
SurfaceCurrent::Update()
{
  delete pcg_;
  delete amg_;
  delete S0_;
  delete PSI_;
  delete RHS_;
  delete K_;

  psi_->Update();
  *psi_ = 0.0;

  s0_->Update();
  s0_->Assemble();
  s0_->Finalize();
  S0_ = s0_->ParallelAssemble();

  amg_ = new HypreBoomerAMG(*S0_);
  amg_->SetPrintLevel(0);

  pcg_ = new HyprePCG(*S0_);
  pcg_->SetTol(1e-12);
  pcg_->SetMaxIter(200);
  pcg_->SetPrintLevel(0);
  pcg_->SetPreconditioner(*amg_);

  // Apply piecewise constant voltage boundary condition
  Array<int> vbc_bdr_attr(H1FESpace_->GetParMesh()->bdr_attributes.Max());
  for (int i=0; i<vbcs_->Size(); i++)
  {
    ConstantCoefficient voltage((*vbcv_)[i]);
    vbc_bdr_attr = 0;
    vbc_bdr_attr[(*vbcs_)[i]-1] = 1;
    psi_->ProjectBdrCoefficient(voltage, vbc_bdr_attr);
  }

  PSI_ = psi_->ParallelProject();
  RHS_ = new HypreParVector(H1FESpace_);
  K_   = new HypreParVector(HCurlFESpace_);

  s0_->ParallelEliminateEssentialBC(ess_bdr_,
				    *S0_,
				    *PSI_, *RHS_);
}

// A spherical shell with constant permeability.  The sphere has inner
// and outer radii, center, and relative permeability specified on the
// command line and stored in ms_params_.
double magnetic_shell(const Vector &x)
{
   double r2 = 0.0;

   for (int i=0; i<x.Size(); i++)
   {
      r2 += (x(i)-ms_params_(i))*(x(i)-ms_params_(i));
   }

   if ( sqrt(r2) >= ms_params_(x.Size()) &&
        sqrt(r2) <= ms_params_(x.Size()+1) )
   {
      return mu0_*ms_params_(x.Size()+2);
   }
   return mu0_;
}

// A annular ring of current density.  The ring has two axis end
// points, inner and outer radii, and a constant current in Amperes.
void current_ring(const Vector &x, Vector &j)
{
   MFEM_ASSERT(x.Size() == 3, "current_ring source requires 3D space.");

   j.SetSize(x.Size());
   j = 0.0;

   Vector  a(x.Size());  // Normalized Axis vector
   Vector xu(x.Size());  // x vector relative to the axis end-point
   Vector ju(x.Size());  // Unit vector in direction of current

   xu = x;

   for (int i=0; i<x.Size(); i++)
   {
      xu[i] -= cr_params_[i];
      a[i]   = cr_params_[x.Size()+i] - cr_params_[i];
   }

   double h = a.Norml2();

   if ( h == 0.0 )
   {
      return;
   }

   double ra = cr_params_[2*x.Size()+0];
   double rb = cr_params_[2*x.Size()+1];
   if ( ra > rb )
   {
      double rc = ra;
      ra = rb;
      rb = rc;
   }
   double xa = xu*a;

   if ( h > 0.0 )
   {
     xu.Add(-xa/(h*h),a);
   }

   double xp = xu.Norml2();

   if ( xa >= 0.0 && xa <= h*h && xp >= ra && xp <= rb )
   {
      ju(0) = a(1) * xu(2) - a(2) * xu(1);
      ju(1) = a(2) * xu(0) - a(0) * xu(2);
      ju(2) = a(0) * xu(1) - a(1) * xu(0);
      ju /= h;

      j.Add(cr_params_[2*x.Size()+2]/(h*(rb-ra)),ju);
   }
}

// A Cylindrical Rod of constant magnetization.  The cylinder has two
// axis end points, a radius, and a constant magnetic field oriented
// along the axis.
void bar_magnet(const Vector &x, Vector &m)
{
   m.SetSize(x.Size());
   m = 0.0;

   Vector  a(x.Size());  // Normalized Axis vector
   Vector xu(x.Size());  // x vector relative to the axis end-point

   xu = x;

   for (int i=0; i<x.Size(); i++)
   {
      xu[i] -= bm_params_[i];
      a[i]   = bm_params_[x.Size()+i] - bm_params_[i];
   }

   double h = a.Norml2();

   if ( h == 0.0 )
   {
      return;
   }

   double  r = bm_params_[2*x.Size()];
   double xa = xu*a;

   if ( h > 0.0 )
   {
      xu.Add(-xa/h,a);
   }

   double xp = xu.Norml2();

   if ( xa >= 0.0 && xa <= h && xp <= r )
   {
      m.Add(bm_params_[2*x.Size()+1]/h,a);
   }
}

// To produce a uniform magnetic flux the vector potential can be set
// to (-y,0,0).
void a_bc_uniform(const Vector & x, Vector & a)
{
   a.SetSize(3);
   a(0) = b_uniform_(1) * x(2);
   a(1) = b_uniform_(2) * x(0);
   a(2) = b_uniform_(0) * x(1);
}

// To produce a uniform magnetic field the scalar potential can be set
// to -z (or -y in 2D).
double phi_m_bc_uniform(const Vector &x)
{
   return -x(x.Size()-1);
}
