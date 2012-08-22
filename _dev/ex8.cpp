//                                MFEM Example 8
//
// Compile with: make ex8
//
// Sample runs:  ex8 ../data/beam-tri.mesh
//               ex8 ../data/beam-quad.mesh
//               ex8 ../data/beam-tet.mesh
//               ex8 ../data/beam-hex.mesh
//               ex8 ../data/beam-quad-nurbs.mesh
//               ex8 ../data/beam-hex-nurbs.mesh
//
// Description:  This example code computes the high order Laplacian of a scalar
//               field s, which is defined in terms of a specified vector field v.
//               Specifically, given a vector field v, we compute:
//
//                 (1) s = || 1/2(Grad(v) + Grad(v)^T) ||
//
//                 (2) x = (L^k)s
//
//               where the integer, k, denotes the number of times the
//               Laplacian operator, L, is applied to the field s.
//               These equations are used in defining "hyperviscosity" methods.
//               We discretize the equations using continuous scalar and
//               vector finite elements.
//
//               This example demonstrates the use of both scalar and vector H1
//               finite element spaces with the scalar finite mass and diffusion
//               bilinear forms, as well as the computation
//               of discretization error when the exact solution is known and
//               visualizing multiple solutions via GLvis.
//
//               We recommend viewing examples 1-4 before viewing this example.

#include <fstream>
#include "mfem.hpp"


// Choose a functional form for the velocity field
const int funcID = 5;

// Initial velocity field, v
void v_exact(const Vector &p , Vector &F );

// The "s" field, defined as the FNorm of the symmetrized velocity gradient
double s_exact(Vector &p);

// Exact solution for first Laplacian of s
double Ls_exact(Vector &p);

// Exact solution for second Laplacian of s
double LLs__exact(Vector &p);

// Build the "s" field from a specified velocity field
void Make_sfield(FiniteElementSpace *fespace, GridFunction& v,  GridFunction& S);

// Take absolute value of grid functions
void absGridFunction(FiniteElementSpace *fespace, GridFunction& v);

// Smooth a grid function
void smoothing(FiniteElementSpace *fespace, GridFunction& h0s );

// For boundary integral
class BoundaryGradIntegrator: public BilinearFormIntegrator
{
private:
   Vector shape1, dshape_dn, nor;
   DenseMatrix dshape, dshapedxt, invdfdx;

public:
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

int main (int argc, char *argv[])
{
   Mesh *mesh;

   if (argc == 1)
   {
      cout << "\nUsage: ex8 <mesh_file>\n" << endl;
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

   int dim = mesh->Dimension();

   // 2. Select the order of the finite element discretization space.
   int p;
   cout << "Enter finite element space order --> " << flush;
   cin >> p;


   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 5,000
   //    elements.
   int ref_levels;
   cout << "Enter number of mesh refinement levels --> " << flush;
   cin >> ref_levels;
   {
      for (int l = 0; l < ref_levels; l++)
         mesh->UniformRefinement();
   }

   // 4a. Define a finite element space on the mesh. Here we use scalar finite
   //     elements. The scalar FEM space is for the s field (FNorm of
   //     symmetrized velocity gradient) constructor.
   FiniteElementCollection *fec;
   FiniteElementSpace *fespace;

   fec = new H1_FECollection(p);
   fespace = new FiniteElementSpace(mesh, fec);

   // 4b. Define a finite element space on the mesh. Here we use vector finite
   //     elements, i.e. dim copies of a scalar finite element space. The vector
   //     dimension is specified by the last argument of the FiniteElementSpace
   //     constructor.
   FiniteElementCollection *vfec;
   FiniteElementSpace *vfespace;

   vfec = new H1_FECollection(p,dim);
   vfespace = new FiniteElementSpace(mesh, fec,dim);

   cout << "Number of unknowns for s: " << fespace->GetVSize() << endl
        << "Assembling: " << flush;

   cout << "Number of unknowns for v: " << vfespace->GetVSize() << endl
        << "Assembling: " << flush;

   // 5. Project the velocity function, v, onto the vector finite element space
   GridFunction v_proj(vfespace);
   VectorFunctionCoefficient v_coeff(dim, v_exact);
   v_proj.ProjectCoefficient(v_coeff);

   // 5a. Compute symmetrized velocity gradient and take Frobenius norm
   GridFunction s_proj(fespace);
   Make_sfield (vfespace, v_proj, s_proj);

   // Make exact solutions for comparison (both scalar fields)

   // Exact solution for s
   GridFunction f_exact(fespace);
   FunctionCoefficient f_proj(s_exact);
   f_exact.ProjectCoefficient(f_proj);

   // Calculate L2 norm of error in s
   cout << "\n s field: || s_h - s ||_{L^2} = " << s_proj.ComputeL2Error(f_proj) << '\n' << endl;
   outputFile<<mesh->GetElementSize(0)<<" "<< s_proj.ComputeL2Error(f_proj)<<" ";

   // Exact Laplacian of s
   GridFunction lap_exact(fespace);
   FunctionCoefficient lap_proj(Ls_exact);
   lap_exact.ProjectCoefficient(lap_proj);

   // Exact 2nd Laplacian of s
   GridFunction lap2_exact(fespace);
   FunctionCoefficient lap2_proj(LLs__exact);
   lap2_exact.ProjectCoefficient(lap2_proj);

   // 6. Set up the bilinear forms
   Coefficient *mu = new ConstantCoefficient(1.0);
   BilinearForm *S = new BilinearForm(fespace);
   S->AddDomainIntegrator(new DiffusionIntegrator(*mu));
   S->AddBdrFaceIntegrator(new BoundaryGradIntegrator);
   S->Assemble();
   S->Finalize();

   const SparseMatrix &Ssp = S->SpMat();

   // Make a solution vector for Ls
   GridFunction x(fespace);
   x = 0.0;

   // Make a solution vector for LLs
   GridFunction x2(fespace);
   x2 = 0.0;

   // Lumped Mass matrix
   BilinearForm *M = new BilinearForm(fespace);
   M->AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator(*mu)));
   M->Assemble();
   M->Finalize();

   const SparseMatrix &Msp = M->SpMat();

   // 7. Solve for Laplacian of s

   // Loop for number of Laplacian operations
   const int num_Laps = 2;

   for (unsigned int j=0; j<num_Laps; j++)
   {
      GridFunction b(fespace); //Right hand side vector
      b = 0.0;
      if (j==0)
      {
         Ssp.Mult(s_proj , b); // S*s

         // 7. Define a simple symmetric Gauss-Seidel preconditioner
         GSSmoother GSS(Msp);
         x = 0.0;
         PCG(Msp, GSS, b, x, 1, 500, 1e-35, 0.0);    // Solve the system Mx=Ss with PCG.

         // Calculate L2 norm
         cout << "\n L(s):  || x_h - x ||_{L^2} = " << x.ComputeL2Error(lap_proj) << '\n' << endl;
         outputFile<< x.ComputeL2Error(lap_proj)<<" ";

      }
      else
      {
         Ssp.Mult(x, b); // S*x

         // 7. Define a simple symmetric Gauss-Seidel preconditioner
         GSSmoother GSS(Msp);
         x2 = 0.0;
         PCG(Msp, GSS, b, x2, 1, 500, 1e-35, 1e-20); // Solve the system Mx=S(Ss) with PCG.

         // Calculate L2 norm
         cout << "\n LL(s):  || x_h - x ||_{L^2} = " << x2.ComputeL2Error(lap2_proj) << '\n' << endl;
         outputFile<< x2.ComputeL2Error(lap2_proj)<<" ";

      }
   }


   // 8. (Optional) Send the above data by socket to a GLVis server.

   char vishost[] = "localhost";
   int  visport   = 19916;

   // Projected v field
   osockstream v_sock(visport, vishost);
   v_sock << "solution\n";
   v_sock.precision(8);
   mesh->Print(v_sock);
   v_proj.Save(v_sock);
   v_sock <<  "window_title 'Projected v field'\n";
   v_sock <<  "shading cool \n";
   v_sock <<  "subdivisions 4 1 \n";
   v_sock <<  "keys m \n";
   v_sock.send();
   v_sock.send();

   // Computed s field
   osockstream s_sock(visport, vishost);
   s_sock << "solution\n";
   s_sock.precision(8);
   mesh->Print(s_sock);
   s_proj.Save(s_sock);
   s_sock <<  "window_title 'Computed s field'\n";
   s_sock <<  "shading cool \n";
   s_sock <<  "subdivisions 4 1 \n";
   s_sock <<  "keys m \n";
   s_sock.send();

   // Computed first Laplacian of s
   osockstream L_sock(visport, vishost);
   L_sock << "solution\n";
   L_sock.precision(8);
   mesh->Print(L_sock);
   x.Save(L_sock);
   L_sock <<  "window_title 'First Laplacian of s'\n";
   L_sock <<  "shading cool \n";
   L_sock <<  "subdivisions 4 1 \n";
   L_sock <<  "keys m \n";
   L_sock.send();

   // Computed second Laplacian of S
   osockstream LL_sock(visport, vishost);
   LL_sock << "solution\n";
   LL_sock.precision(8);
   mesh->Print(LL_sock);
   x2.Save(LL_sock);
   LL_sock <<  "window_title 'Second Laplacian of s'\n";
   LL_sock <<  "shading cool \n";
   LL_sock <<  "subdivisions 4 1 \n";
   LL_sock <<  "keys m \n";
   LL_sock.send();
   LL_sock.send();

   // Smoothed absolute value of second Laplacian
   absGridFunction(fespace, x2);
   smoothing(fespace, x2);
   osockstream LLsm_sock(visport, vishost);
   LLsm_sock << "solution\n";
   LLsm_sock.precision(8);
   mesh->Print(LLsm_sock);
   x2.Save(LLsm_sock);
   LLsm_sock <<  "window_title 'Smoothed AbsVal of Second Laplacian of s'\n";
   LLsm_sock <<  "shading cool \n";
   LLsm_sock <<  "subdivisions 4 1 \n";
   LLsm_sock <<  "keys m \n";
   LLsm_sock.send();
   LLsm_sock.send();
   LLsm_sock.send();

   // 8. Free the used memory.
   if (fec)
   {
      delete fespace;
      delete fec;
      delete vfespace;
      delete vfec;
   }
   delete mesh;

   outputFile<<endl;
   outputFile.close();
   return 0;
}


//----------------------------------------------------------------------
// Functions
//----------------------------------------------------------------------

// The velocity field, v
void v_exact(const Vector &p, Vector &v)
{
   double s;

   double x,y,z;

   int dim = p.Size();

   x = p(0);
   y = p(1);
   if(dim == 3)
      z = p(2);

   switch(funcID)
   {
   case 0:
      v(0) = sin(M_PI*x)*sin(M_PI*y);
      v(1) = 0.0;
      if(dim==3)
         v(2) = 0.0;
      break;

   case 1:
      v(0) = 1+cos(2*M_PI*x)*sin(M_PI*y);
      v(1) = 0.0;
      if(dim==3)
         v(2) = 0.0;
      break;

   case 2:
      v(0) = M_PI_2+atan(20*(x-0.5));
      v(1) = 0.0;
      if(dim==3)
         v(2) = 0.0;
      break;

   case 3:
      v(0) = (x < 0.5) ? 0.5 : 1.0;
      v(1) = 0.0;
      if(dim==3)
         v(2) = 0.0;
      break;

   case 4:
      v(0) = cos(M_PI*x);
      v(1) = 0.0;
      if(dim==3)
         v(2) = 0.0;
      break;

   case 5:
      v(0) = pow(x , 7);
      v(1) = 0.0;
      if(dim==3)
         v(2) = 0.0;
      break;

   case 6:
      v(0) = pow(x , 4);
      v(1) = 0.0;
      if(dim==3)
         v(2) = 0.0;
      break;

   case 7:
      v(0) = sin(M_PI*x);
      v(1) = 0.0;
      if(dim==3)
         v(2) = 0.0;
      break;
   }
}

// The s field
double s_exact(Vector &p)
{
   double x,y,z;

   int dim = p.Size();

   x = p(0);
   y = p(1);
   if(dim == 3)
      z = p(2);

   double s;

   switch(funcID)
   {
   case 0:

      break;

   case 1:

      break;

   case 2:

      break;

   case 3:
      s= 0.0;
      break;

   case 4:
      s = M_PI* sqrt(pow( sin(M_PI*x) , 2));
      break;

   case 5:
      s = 7.0*pow(x , 6);
      break;

   case 6:
      s = 4.0*pow(x , 3);
      break;
   }

   return s;
}

// The first Laplacian of the s field
double Ls_exact(Vector &p)
{
   double x,y,z;

   int dim = p.Size();

   x = p(0);
   y = p(1);
   if(dim == 3)
      z = p(2);

   double s;

   switch(funcID)
   {
   case 0:

      break;

   case 1:

      break;

   case 2:

      break;

   case 3:
      s = 0.0;
      break;

   case 4:
      s = M_PI*M_PI*M_PI* sqrt(pow( sin(M_PI*x) , 2));
      break;

   case 5:
      s = -210.0*pow(x , 4);
      break;

   case 6:
      s = -24.0*x;
      break;
   }

   return s;

}


// Second Laplacian of the s field
double LLs__exact(Vector &p)
{
   double x,y,z;

   int dim = p.Size();

   x = p(0);
   y = p(1);
   if(dim == 3)
      z = p(2);

   double s;

   switch(funcID)
   {
   case 0:

      break;

   case 1:

      break;

   case 2:

      break;

   case 3:
      s = 0.0;
      break;

   case 4:
      s =pow( M_PI , 5)* sqrt(pow( sin(M_PI*x) , 2));
      break;

   case 5:
      s = 2520.0*pow(x , 2);

      break;

   case 6:
      s = 0.0;
      break;
   }

   return s;

}

void Make_sfield(FiniteElementSpace *fespace, GridFunction& v,  GridFunction& S)
{
   int dim = fespace->GetVDim();
   int ndof = (fespace->GetFE(0))->GetDof();

   Array<int> dofs;
   Array<int> nodeCount(fespace->GetNDofs());
   DenseMatrix grad(ndof, dim);

   ElementTransformation *tr;
   nodeCount = 0;
   S = 0.0;

   for (int z = 0; z < fespace->GetNE(); z++)
   {
      const IntegrationRule *ir = &((fespace->GetFE(z))->GetNodes()) ;
      fespace->GetElementDofs(z, dofs);
      tr = fespace->GetElementTransformation(z);

      for (int p = 0; p < ir->GetNPoints(); p++)
      {
         const IntegrationPoint &ip = ir->IntPoint(p);
         IntegrationPoint eip1;
         tr->SetIntPoint(&ip);

         v.GetVectorGradient(*tr, grad);
         grad.Symmetrize();

         //cout<<"current node: "<<dofs[p]<<endl;

         S(dofs[p]) += grad.FNorm();
         nodeCount[dofs[p]]++;
      }

   }


   //loop over S and get average at element boundaries
   for (int i= 0; i < nodeCount.Size() ; i++)
   {
      //cout<<"Node Counter: "<<nodeCount[i]<<endl;
      S(i) = S(i) / nodeCount[i];
   }


}

void BoundaryGradIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   int i, j, ndof1;
   int dim, order;

   dim = el1.GetDim();
   ndof1 = el1.GetDof();

   //set to this for now, integration includes rational terms
   order = 2*el1.GetOrder() + 1;

   nor.SetSize(dim);
   shape1.SetSize(ndof1);
   dshape_dn.SetSize(ndof1);
   dshape.SetSize(ndof1,dim);
   dshapedxt.SetSize(ndof1,dim);
   invdfdx.SetSize(dim);

   elmat.SetSize(ndof1);
   elmat = 0.0;

   const IntegrationRule *ir = &IntRules.Get(Trans.FaceGeom, order);
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip1;
      Trans.Loc1.Transform(ip, eip1);
      el1.CalcShape(eip1, shape1);
      //d of shape function, evaluated at eip1
      el1.CalcDShape(eip1, dshape);

      Trans.Elem1->SetIntPoint(&eip1);

      CalcInverse(Trans.Elem1->Jacobian(), invdfdx); //inverse Jacobian
      //invdfdx.Transpose();
      Mult(dshape, invdfdx, dshapedxt); // dshapedxt = grad phi* J^-1

      //get normal vector
      Trans.Face->SetIntPoint(&ip);
      const DenseMatrix &J = Trans.Face->Jacobian(); //is this J^{-1} or J^{T}?
      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else if (dim == 2)
      {
         nor(0) =  J(1,0);
         nor(1) = -J(0,0);
      }
      else if (dim == 3)
      {
         nor(0) = J(1,0)*J(2,1) - J(2,0)*J(1,1);
         nor(1) = J(2,0)*J(0,1) - J(0,0)*J(2,1);
         nor(2) = J(0,0)*J(1,1) - J(1,0)*J(0,1);
      }

      // multiply weight into normal, make answer negative
      // (boundary integral is subtracted)
      nor *= -ip.weight;

      dshapedxt.Mult(nor, dshape_dn);

      for (i = 0; i < ndof1; i++)
         for (j = 0; j < ndof1; j++)
            elmat(i, j) += shape1(i)*dshape_dn(j);
   }
}

void absGridFunction(FiniteElementSpace *fespace, GridFunction& v)
{

   int dim = fespace->GetVDim();
   int ndof = (fespace->GetFE(0))->GetDof();

   Array<int> dofs;

   for (int z = 0; z < fespace->GetNE(); z++)
   {
      const IntegrationRule *ir = &((fespace->GetFE(z))->GetNodes()) ;
      fespace->GetElementDofs(z, dofs);

      for (int p = 0; p < ir->GetNPoints(); p++)
         v(dofs[p]) = fabs(v(dofs[p]));
   }
}


void smoothing(FiniteElementSpace *fespace, GridFunction& h0s )
{

   BilinearForm laplace(fespace);
   ConstantCoefficient one(1.0);
   laplace.AddDomainIntegrator(new DiffusionIntegrator(one));
   laplace.Assemble();
   laplace.Finalize();

   GridFunction u(fespace);
   u = 0.;
   int  h0_smoothing_type = 0;
   int  num_h0_smoothing_steps = 2;
   if (h0_smoothing_type == 0)
   {
      // symmetric Gauss-Seidel smoothing
      for (int i = 0; i < num_h0_smoothing_steps; i++)
      {
         laplace.SpMat().Gauss_Seidel_forw(u, h0s);
         laplace.SpMat().Gauss_Seidel_back(u, h0s);
      }
   }
   else if (h0_smoothing_type == 1)
   {
      // scaled Jacobi smoothing
      Vector h0a(h0s.Size());
      const double a = laplace.SpMat().GetJacobiScaling();
      cout << "Jacobi scaling factor = " << a << endl;
      for (int i = 0; i < num_h0_smoothing_steps; i++)
      {
         laplace.SpMat().Jacobi(u, h0s, h0a, a);
         laplace.SpMat().Jacobi(u, h0a, h0s, a);
      }
   }
   else
   {
      // Jacobi2 smoothing
      Vector h0a(h0s.Size());
      const double a = 1.8;
      for (int i = 0; i < num_h0_smoothing_steps; i++)
      {
         laplace.SpMat().Jacobi2(u, h0s, h0a, a);
         laplace.SpMat().Jacobi2(u, h0a, h0s, a);
      }
   }
}

