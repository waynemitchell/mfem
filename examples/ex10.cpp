//                                MFEM Example 10
//
// Compile with: make ex10
//
// Sample runs:  ex10 -m ../data/beam-quad.mesh -r 2 -o 2 -dt 0.03
//               ex10 -m ../data/beam-hex.mesh -r 1 -o 2 -dt 0.05
//
// Description:  Time dependent nonlinear elasticity

#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;


/// TODO: short description
class HyperelasticOperator : public TimeDependentOperator
{
private:
   FiniteElementSpace &fespace;

   BilinearForm M;
   DSmoother M_prec;
   CGSolver M_solver;

   double viscosity;
   BilinearForm S; // viscosity matrix

   HyperelasticModel *model;
   NonlinearForm H;

   mutable Vector z; // auxiliary vector

public:
   HyperelasticOperator(FiniteElementSpace &f, Array<int> &ess_bdr,
                        double visc);

   virtual void Mult(const Vector &vx, Vector &dvx_dt) const;

   double ElasticEnergy(Vector &x) const;
   double KineticEnergy(Vector &v) const;
   void GetElasticEnergyDensity(GridFunction &x, GridFunction &w) const;

   virtual ~HyperelasticOperator();
};

/// TODO: short description
class ElasticEnergyCoefficient : public Coefficient
{
private:
   HyperelasticModel &model;
   GridFunction      &x;

   DenseMatrix J;

public:
   ElasticEnergyCoefficient(HyperelasticModel &m, GridFunction &_x)
      : model(m), x(_x) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
   virtual ~ElasticEnergyCoefficient() { }
};

/// TODO: short description
void InitialDeformation(const Vector &x, Vector &y);

/// TODO: short description
void InitialVelocity(const Vector &x, Vector &v);


void visualize(ostream &out, Mesh *mesh, GridFunction *deformed_nodes,
               GridFunction *field, const char *field_name = NULL,
               bool init_vis = false);


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/beam-quad.mesh";
   int ref_levels = 2;
   int order = 2;
   int ode_solver_type = 4;
   double t_final = 300.0;
   double dt = 0.03;
   double visc = 1e-2;
   bool visualization = true;
   int vis_steps = 20;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forw. Euler, 2 - RK2, 3 - RK3 SSP,"
                  " 4 - RK4, 6 - RK6, 8 - RK8.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&visc, "-v", "--viscosity",
                  "Viscosity coefficient.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh *mesh;
   {
      ifstream imesh(mesh_file);
      if (!imesh)
      {
         cout << "Can not open mesh: " << mesh_file << endl;
         return 2;
      }
      mesh = new Mesh(imesh, 1, 1);
   }
   int dim = mesh->Dimension();

   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   ODESolver *ode_solver;
   switch (ode_solver_type)
   {
   case 1: ode_solver = new ForwardEulerSolver; break;
   case 2: ode_solver = new RK2Solver(0.5); break; // midpoint method
   case 3: ode_solver = new RK3SSPSolver; break;
   case 4: ode_solver = new RK4Solver; break;
   case 6: ode_solver = new RK6Solver; break;
   case 8:
   default: ode_solver = new RK8Solver; break;
   }

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ref_levels; lev++)
      mesh->UniformRefinement();


   H1_FECollection fe_coll(order, dim);
   FiniteElementSpace fespace(mesh, &fe_coll, dim);

   GridFunction x_ref(&fespace);
   mesh->GetNodes(x_ref);

   int fe_size = fespace.GetVSize();
   Vector vx(2*fe_size);
   GridFunction v, x;
   v.Update(&fespace, vx, 0);
   x.Update(&fespace, vx, fe_size);

   // Initial conditions for v and x
   VectorFunctionCoefficient velo(dim, InitialVelocity);
   v.ProjectCoefficient(velo);
   VectorFunctionCoefficient deform(dim, InitialDeformation);
   x.ProjectCoefficient(deform);

   // Boundary conditions based on the boundary attributes
   Array<int> ess_bdr(fespace.GetMesh()->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1; // boundary attribute 1 (index 0) is fixed

   HyperelasticOperator oper(fespace, ess_bdr, visc);

   ode_solver->Init(oper);

   double t = 0.0;

   L2_FECollection w_fec(order + 1, dim);
   FiniteElementSpace w_fespace(mesh, &w_fec);
   GridFunction w(&w_fespace);

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream vis_v(vishost, visport);
   socketstream vis_w(vishost, visport);
   vis_v.precision(8);
   visualize(vis_v, mesh, &x, &v, "Velocity", true);
   if (vis_w)
   {
      oper.GetElasticEnergyDensity(x, w);
      vis_w.precision(8);
      visualize(vis_w, mesh, &x, &w, "Elastic energy density", true);
   }

   double ee0 = oper.ElasticEnergy(x);
   double ke0 = oper.KineticEnergy(v);
   cout << "initial elastic energy (EE) = " << ee0 << endl;
   cout << "initial kinetic energy (KE) = " << ke0 << endl;
   cout << "initial   total energy (TE) = " << (ee0 + ke0) << endl;

   bool last_step = false;
   for (int i = 1; !last_step; i++)
   {
      if (t + dt >= t_final - dt/2)
         last_step = true;

      ode_solver->Step(vx, t, dt);

      if (last_step || (i % vis_steps) == 0)
      {
         double ee = oper.ElasticEnergy(x);
         double ke = oper.KineticEnergy(v);

         cout << "step " << i << ", t = " << t << ", EE = " << ee << ", KE = "
              << ke << ", ΔTE = " << (ee+ke)-(ee0+ke0) << endl;

         visualize(vis_v, mesh, &x, &v);
         if (vis_w)
         {
            oper.GetElasticEnergyDensity(x, w);
            visualize(vis_w, mesh, &x, &w);
         }
      }
   }

   // 11. Save the displaced mesh, the velocity and elastic energy.
   {
      GridFunction *nodes = &x;
      int owns_nodes = 0;
      mesh->SwapNodes(nodes, owns_nodes);
      ofstream mesh_ofs("deformed.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      mesh->SwapNodes(nodes, owns_nodes);
      ofstream velo_ofs("velocity.sol");
      velo_ofs.precision(8);
      v.Save(velo_ofs);
      ofstream ee_ofs("elastic_energy.sol");
      ee_ofs.precision(8);
      oper.GetElasticEnergyDensity(x, w);
      w.Save(ee_ofs);
   }

   // 10. Free the used memory.
   delete ode_solver;
   delete mesh;

   return 0;
}

void visualize(ostream &out, Mesh *mesh, GridFunction *deformed_nodes,
               GridFunction *field, const char *field_name, bool init_vis)
{
   if (!out)
      return;

   GridFunction *nodes = deformed_nodes;
   int owns_nodes = 0;

   mesh->SwapNodes(nodes, owns_nodes);

   out << "solution\n" << *mesh << *field;

   mesh->SwapNodes(nodes, owns_nodes);

   if (init_vis)
   {
      out << "window_size 800 800\n";
      out << "window_title '" << field_name << "'\n";
      if (mesh->SpaceDimension() == 2)
      {
         out << "view 0 0\n"; // view from top
         out << "keys jl\n";  // turn off perspective and light
      }
      out << "keys cm\n";         // show colorbar and mesh
      out << "autoscale value\n"; // update value-range; keep mesh-extents fixed
      out << "pause\n";
   }
   out << flush;
}


HyperelasticOperator::HyperelasticOperator(FiniteElementSpace &f,
                                           Array<int> &ess_bdr, double visc)
   : TimeDependentOperator(2*f.GetVSize(), 0.0), fespace(f),
     M(&fespace), S(&fespace), H(&fespace), z(height/2)
{
   const double ref_density = 1.0; // density in the reference configuration
   ConstantCoefficient rho0(ref_density);
   M.AddDomainIntegrator(new VectorMassIntegrator(rho0));
   M.Assemble();
   M.EliminateEssentialBC(ess_bdr);
   M.Finalize();

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-8);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(30);
   M_solver.SetPrintLevel(0);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M.SpMat());

   double mu = 0.25; // shear modulus
   double K  = 5.0;  // bulk modulus
   model = new NeoHookeanModel(mu, K);
   H.AddDomainIntegrator(new HyperelasticNLFIntegrator(model));
   H.SetEssentialBC(ess_bdr);

   viscosity = visc;
   if (viscosity != 0.0)
   {
      ConstantCoefficient visc_coeff(viscosity);
      S.AddDomainIntegrator(new VectorDiffusionIntegrator(visc_coeff));
      S.Assemble();
      S.EliminateEssentialBC(ess_bdr);
      S.Finalize();
   }
}

void HyperelasticOperator::Mult(const Vector &vx, Vector &dvx_dt) const
{
   int sc = height/2;
   Vector v(vx.GetData() +  0, sc);
   Vector x(vx.GetData() + sc, sc);
   Vector dv_dt(dvx_dt.GetData() +  0, sc);
   Vector dx_dt(dvx_dt.GetData() + sc, sc);

   H.Mult(x, z);
   if (viscosity != 0.0)
      S.AddMult(v, z);
   z.Neg(); // z = -z
   M_solver.Mult(z, dv_dt);

   dx_dt = v;
}

double HyperelasticOperator::ElasticEnergy(Vector &x) const
{
   return H.GetEnergy(x);
}

double HyperelasticOperator::KineticEnergy(Vector &v) const
{
   return 0.5*M.InnerProduct(v, v);
}

void HyperelasticOperator::GetElasticEnergyDensity(
   GridFunction &x, GridFunction &w) const
{
   ElasticEnergyCoefficient w_coeff(*model, x);

   w.ProjectCoefficient(w_coeff);
}

HyperelasticOperator::~HyperelasticOperator()
{
   delete model;
}


double ElasticEnergyCoefficient::Eval(ElementTransformation &T,
                                      const IntegrationPoint &ip)
{
   model.SetTransformation(T);
   x.GetVectorGradient(T, J);
   // return model.EvalW(J);  // in reference configuration
   return model.EvalW(J)/J.Det(); // in deformed configuration
}


void InitialDeformation(const Vector &x, Vector &y)
{
   // set the initial configuration to be the same as the reference, stress
   // free, configuration
   y = x;
}

void InitialVelocity(const Vector &x, Vector &v)
{
   const int dim = x.Size();
   const double s = 0.1/64.;

   v = 0.0;
   v(dim-1) = s*x(0)*x(0)*(8.0-x(0));
   v(0) = -s*x(0)*x(0);
}
