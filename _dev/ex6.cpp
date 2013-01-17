//                                MFEM Example 6
//
// Compile with: make ex6
//
// Sample runs:  ex6 square-disc.mesh2d
//               ex6 star.mesh2d
//
// Description: This example code performs a simple mesh smoothing based on a
//              topologically defined "mesh Laplacian" matrix.
//
//              The example highlights meshes with curved elements, the
//              assembling of a custom finite element matrix, the use of vector
//              finite element spaces, the definition of different spaces and
//              grid functions on the same mesh, and the setting of values by
//              iterating over the interior and the boundary elements.

#include <fstream>
#include <limits>
#include "mfem.hpp"

// 1. Define the bilinear form corresponding to a mesh Laplacian operator. This
//    will be used to assemble the global mesh Laplacian matrix based on the
//    local matrix provided in the AssembleElementMatrix method. More examples
//    of bilinear integrators can be found in ../fem/bilininteg.hpp.
class VectorMeshLaplacianIntegrator : public BilinearFormIntegrator
{
private:
   int geom, type;
   LinearFECollection lfec;
   IsoparametricTransformation T;
   VectorDiffusionIntegrator vdiff;

public:
   VectorMeshLaplacianIntegrator(int type_) { type = type_; geom = -1; }
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   virtual ~VectorMeshLaplacianIntegrator() { }
};

// 2. Implement the local stiffness matrix of the mesh Laplacian. This is a
//    block-diagonal matrix with each block having a unit diagonal and constant
//    negative off-diagonal entries, such that the row sums are zero.
void VectorMeshLaplacianIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   if (type == 0)
   {
      int dim = el.GetDim(); // space dimension
      int dof = el.GetDof(); // number of element degrees of freedom

      elmat.SetSize(dim*dof); // block-diagonal element matrix

      for (int d = 0; d < dim; d++)
         for (int k = 0; k < dof; k++)
            for (int l = 0; l < dof; l++)
               if (k==l)
                  elmat (dof*d+k, dof*d+l) = 1.0;
               else
                  elmat (dof*d+k, dof*d+l) = -1.0/(dof-1);
   }
   else
   {
      if (el.GetGeomType() != geom)
      {
         geom = el.GetGeomType();
         T.SetFE(lfec.FiniteElementForGeometry(geom));
         Geometries.GetPerfPointMat(geom, T.GetPointMat());
      }
      T.Attribute = Trans.Attribute;
      T.ElementNo = Trans.ElementNo;
      vdiff.AssembleElementMatrix(el, T, elmat);
   }
}

class HyperelasticModel
{
private:
   bool has_W, has_H;

public:
   HyperelasticModel(bool h_w, bool h_h) : has_W(h_w), has_H(h_h) { }

   // Evaluate the strain energy density function, W=W(J).
   virtual double EvalW(const DenseMatrix &J) const = 0;

   // Evaluate the 1st Piola-Kirchhoff stress tensor, P=P(J).
   virtual void EvalP(const DenseMatrix &J, DenseMatrix &P) const = 0;

   // Evaluate the derivative of the 1st Piola-Kirchhoff stress tensor
   // and assemble its contribution to the local gradient matrix A.
   // 'DS' is the gradient of the basis matrix (dof x dim), and 'weight'
   // is the quadrature weight.
   virtual void AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const = 0;

   inline bool HasW() const { return has_W; }
   inline bool HasH() const { return has_H; }
};

class HarmonicModel : public HyperelasticModel
{
public:
   HarmonicModel() : HyperelasticModel(true, true) { }

   virtual double EvalW(const DenseMatrix &J) const
   {
      return 0.5*(J*J);
   }

   virtual void EvalP(const DenseMatrix &J, DenseMatrix &P) const
   {
      P = J;
   }

   virtual void AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const
   {
      int dof = DS.Height(), dim = DS.Width();

      for (int i = 0; i < dof; i++)
         for (int j = 0; j <= i; j++)
         {
            double a = 0.0;
            for (int d = 0; d < dim; d++)
               a += DS(i,d)*DS(j,d);
            a *= weight;
            for (int d = 0; d < dim; d++)
            {
               A(i+d*dof,j+d*dof) += a;
               if (i != j)
                  A(j+d*dof,i+d*dof) += a;
            }
         }
   }
};

class InverseHarmonicModel : public HyperelasticModel
{
protected:
   mutable DenseMatrix adjJt, sigma; // dim x dim
   mutable DenseMatrix G, C;         // dof x dim

public:
   InverseHarmonicModel() : HyperelasticModel(true, true) { }

   virtual double EvalW(const DenseMatrix &J) const
   {
      adjJt.SetSize(J.Size());
      CalcAdjugateTranspose(J, adjJt);
      return 0.5*(adjJt*adjJt)/J.Det();
   }

   virtual void EvalP(const DenseMatrix &J, DenseMatrix &P) const
   {
      int dim = J.Size();
      double t;

      adjJt.SetSize(dim);
      sigma.SetSize(dim);
      CalcAdjugateTranspose(J, adjJt);
      MultAAt(adjJt, sigma);
      t = 0.5*sigma.Trace();
      for (int i = 0; i < dim; i++)
         sigma(i,i) -= t;
      t = J.Det();
      sigma *= -1.0/(t*t);
      Mult(sigma, adjJt, P);
   }

   virtual void AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const
   {
      int dof = DS.Height(), dim = DS.Width();
      double t;

      adjJt.SetSize(dim);
      sigma.SetSize(dim);
      G.SetSize(dof, dim);
      C.SetSize(dof, dim);

      CalcAdjugateTranspose(J, adjJt);
      MultAAt(adjJt, sigma);

      t = 1.0/J.Det();
      adjJt *= t;  // adjJt = J^{-t}
      sigma *= t;  // sigma = |J| (J.J^t)^{-1}
      t = 0.5*sigma.Trace();

      MultABt(DS, adjJt, G);  // G = DS.J^{-1}
      Mult(G, sigma, C);

      // 1.
      for (int i = 0; i < dof; i++)
         for (int j = 0; j <= i; j++)
         {
            double a = 0.0;
            for (int d = 0; d < dim; d++)
               a += G(i,d)*G(j,d);
            a *= weight;
            for (int k = 0; k < dim; k++)
               for (int l = 0; l <= k; l++)
               {
                  double b = a*sigma(k,l);
                  A(i+k*dof,j+l*dof) += b;
                  if (i != j)
                     A(j+k*dof,i+l*dof) += b;
                  if (k != l)
                  {
                     A(i+l*dof,j+k*dof) += b;
                     if (i != j)
                        A(j+l*dof,i+k*dof) += b;
                  }
               }
         }

      // 2.
#if 0
      for (int i = 0; i < dof; i++)
         for (int j = 0; j < dof; j++)
         {
            for (int k = 0; k < dim; k++)
               for (int l = 0; l < dim; l++)
               {
                  A(i+k*dof,j+l*dof) +=
                     weight*(C(i,l)*G(j,k) - C(i,k)*G(j,l) +
                             C(j,k)*G(i,l) - C(j,l)*G(i,k) +
                             t*(G(i,k)*G(j,l) - G(i,l)*G(j,k)));
               }
         }
#elif 0
      // skip
#else
      for (int i = 0; i < dof; i++)
         for (int j = 0; j < i; j++)
         {
            for (int k = 0; k < dim; k++)
               for (int l = 0; l < k; l++)
               {
                  double a =
                     weight*(C(i,l)*G(j,k) - C(i,k)*G(j,l) +
                             C(j,k)*G(i,l) - C(j,l)*G(i,k) +
                             t*(G(i,k)*G(j,l) - G(i,l)*G(j,k)));

                  A(i+k*dof,j+l*dof) += a;
                  A(j+l*dof,i+k*dof) += a;

                  A(i+l*dof,j+k*dof) -= a;
                  A(j+k*dof,i+l*dof) -= a;
               }
         }
#endif
   }
};

class TaubinSmoother : public Solver
{
protected:
   const SparseMatrix *K;

   int N;
   double lambda;
   double mu;

public:
   TaubinSmoother(int N_, double lambda_, double mu_)
      : N(N_), lambda(lambda_), mu(mu_)
   {
      K = NULL;
   }

   virtual void SetOperator(const Operator &op)
   {
      K = dynamic_cast<const SparseMatrix*>(&op);
      if (K == NULL)
         mfem_error("TaubinSolver::SetOperator : not a SparseMatrix!");
      size = K->Size();
   }

   virtual void Mult(const Vector &x0, Vector &x) const
   {
      x = x0;

      int n = K->Size();

      int dim = x.Size()/K->Size();

      Vector xdim(NULL, n);
      for (int d = 0; d < dim; d++)
      {
         xdim.SetData(&x(d*n));

         Vector dx(xdim.Size());

         for (int i = 0; i < N; i++) {

            K->Mult(xdim,dx);

            if ( 0 == (i % 2)) {
               dx *= lambda;
            }
            else {
               dx *= mu;
            }
            xdim += dx;
         }
      }

   }
};

class FIRSmoother : public Solver
{
protected:
   const SparseMatrix *K;

   int N;
   Vector* wn; // window coeffcients
   Vector* fn; // chebyshev coefficients

public:
   FIRSmoother(int N_, Vector& wn_, Vector& fn_)
      : N(N_)
   {
      K = NULL;
      wn = new Vector(wn_);
      fn = new Vector(fn_);
   }

   virtual void SetOperator(const Operator &op)
   {
      K = dynamic_cast<const SparseMatrix*>(&op);
      if (K == NULL)
         mfem_error("FIRSolver::SetOperator : not a SparseMatrix!");
      size = K->Size();
   }

   virtual void Mult(const Vector &x_in, Vector &x) const {

      // Algorithm:
      // x0 = x
      // x1 = Kx0
      // x1 = x0 - 0.5*x1
      // x3 = f0*x0 +f1*x1
      // for 2 to N
      //  x2 = Kx1
      //  x2 = (x1-x0) +(x1-x2)
      //  x3 = x3 +fn*x2
      //  x0 = x1
      //  x1 = x2
      // end
      // x = x3

      x = x_in;

      int n = K->Size();
      int dim = x.Size()/K->Size();

      Vector& w = *wn;
      Vector& f = *fn;

      Vector xdim(NULL, n);
      for (int d = 0; d < dim; d++) {

         xdim.SetData(&x(d*n));

         Vector x0(xdim);

         Vector x1(xdim.Size());
         K->Mult(x0, x1);
         x1 *= -1;

         x1 *= -0.5;
         x1 += x0;

         Vector x3(x0);
         x3 *= w[0]*f[0];

         Vector b(x1);
         b *= w[1]*f[1];
         x3 += b;

         Vector x2(xdim.Size());

         for (int i = 2; i <= N; i++) {

            K->Mult(x1, x2);

            x2 += x1;
            x2 -= x0;
            x2 += x1;

            Vector z(x2);
            double c = w[i]*f[i];
            z *= c;
            x3 += z;

            x0 = x1;
            x1 = x2;
         }

         xdim = x3;

      } // foreach dim
   }
};

class HyperelasticNLFIntegrator : public NonlinearFormIntegrator
{
private:
   HyperelasticModel *model;

   DenseMatrix dshape, J, P, PMatI, PMatO;

public:
   HyperelasticNLFIntegrator(HyperelasticModel *m)
      : model(m) { }

   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &Tr,
                                   const Vector &elfun)
   {
      int dof = el.GetDof(), dim = el.GetDim();
      double energy;

      dshape.SetSize(dof, dim);
      J.SetSize(dim);
      PMatI.UseExternalData(elfun.GetData(), dof, dim);

      int intorder = 2*el.GetOrder() + 3; // <---
      const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);

      energy = 0.0;
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);

         el.CalcDShape(ip, dshape);

         MultAtB(PMatI, dshape, J);

         energy += ip.weight*model->EvalW(J);
      }

      return energy;
   }

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      const Vector &elfun, Vector &elvect)
   {
      int dof = el.GetDof(), dim = el.GetDim();

      dshape.SetSize(dof, dim);
      J.SetSize(dim);
      P.SetSize(dim);
      PMatI.UseExternalData(elfun.GetData(), dof, dim);
      elvect.SetSize(dof*dim);
      PMatO.UseExternalData(elvect.GetData(), dof, dim);

      int intorder = 2*el.GetOrder() + 3; // <---
      const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);

      elvect = 0.0;
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);

         el.CalcDShape(ip, dshape);

         MultAtB(PMatI, dshape, J);

         model->EvalP(J, P);

         P *= ip.weight;
         AddMultABt(dshape, P, PMatO);
      }
   }

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &Tr,
                                    const Vector &elfun, DenseMatrix &elmat)
   {
      int dof = el.GetDof(), dim = el.GetDim();

      dshape.SetSize(dof, dim);
      J.SetSize(dim);
      PMatI.UseExternalData(elfun.GetData(), dof, dim);
      elmat.SetSize(dof*dim);

      int intorder = 2*el.GetOrder() + 3; // <---
      const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);

      elmat = 0.0;
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);

         el.CalcDShape(ip, dshape);

         MultAtB(PMatI, dshape, J);

         model->AssembleH(J, dshape, ip.weight, elmat);
      }
   }

   virtual ~HyperelasticNLFIntegrator()
   {
      PMatI.ClearExternalData();
      PMatO.ClearExternalData();
   }
};

void smooth_taubin(
   int dim,
   FiniteElementSpace *fespace,
   Mesh* mesh,
   FiniteElementCollection *fec,
   int mesh_poly_deg)
{
   GridFunction *x;
   x = mesh->GetNodes();

   FiniteElementCollection *grad_fec;
   FiniteElementSpace *scal_fes;
   FiniteElementSpace *grad_fes;
   DiscreteLinearOperator *grad;
   scal_fes = new FiniteElementSpace(mesh, fec);

   if (dim == 2)
   {
      grad_fec = new ND_FECollection(mesh_poly_deg, dim);
   }
   else
      grad_fec = new ND_FECollection(mesh_poly_deg, dim);
   grad_fes = new FiniteElementSpace(mesh, grad_fec);
   grad = new DiscreteLinearOperator(scal_fes, grad_fes);
   grad->AddDomainInterpolator(new GradientInterpolator);
   grad->Assemble();
   grad->Finalize();
   SparseMatrix &G = grad->SpMat();
   SparseMatrix *Gt = Transpose(G);
   SparseMatrix *GtG = Mult(*Gt, G);
   delete Gt;
   delete grad;
   delete grad_fes;
   delete grad_fec;

   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   Array<int> ess_dofs;
   scal_fes->GetEssentialVDofs(ess_bdr, ess_dofs);
   delete scal_fes;

   int n = GtG->Size();

   // Create K iteration matrix from GtG.
   SparseMatrix K(*GtG);

   // Rescale
   int nrows = GtG->Size();
   for (int i = 0; i < nrows; i++) {
      double diag = K.Elem(i,i);
      K.ScaleRow(i, -1.0/diag);
   }

   // Zero rows of stationary points
   for (int i = 0; i < ess_dofs.Size(); i++) {
      if (ess_dofs[i] < 0)
      {
         K.ScaleRow(i, 0.0);
      }
   }

   int smooth_steps;
   cout << "Enter number of smoothing steps --> "
        << flush;
   cin >> smooth_steps;

   // Smoothed mesh
   GridFunction xs(fespace);

   // Reasonable choices for this smoother
   double lambda = 0.6307;
   double mu = -0.68;
   int N = 10;
   TaubinSmoother S(N, lambda, mu);
   S.SetOperator(K);

   for (int n = 0; n < smooth_steps; n++) {

      {
         ostringstream os;
         os << "taubin-it" << n << ".mesh";
         ofstream mesh_ofs(os.str().c_str());
         mesh->Print(mesh_ofs);
      }

      S.Mult(*x, xs);
      *x = xs;
   }

   {
      ostringstream os;
      os << "taubin-final" << n << ".mesh";
      ofstream mesh_ofs(os.str().c_str());
      mesh->Print(mesh_ofs);
   }

}


void compute_window(
   const string& window,
   int N,
   Vector& wn)
{
   double a,b,c;

   if ("rectangular" == window) {
      a = 1.0;
      b = 0.0;
      c = 0.0;
   }
   else if ("hanning" == window) {
      a = 0.5;
      b = 0.5;
      c = 0.0;
   }
   else if ("hamming" == window) {
      a = 0.54;
      b = 0.46;
      c = 0.0;
   }
   else if ("blackman" == window) {
      a = 0.42;
      b = 0.50;
      c = 0.08;
   }
   else {
      printf("window unrecognized: %s\n",window.c_str());
      exit(1);
   }

   for (int i = 0; i <= N; i++) {
      double t = (i*M_PI)/(N+1);
      wn[i] = a + b*cos(t) +c*cos(2*t);
   }

}

void compute_chebyshev_coeffs(
   int N,
   Vector& fn,
   double k_pb)
{
   double theta_pb = acos(1.0 -0.5*k_pb);

   // This sigma offset value can (should) be optimized as a function
   // of N, kpb, and the window function.  This is a good value for
   // N=10, kpb = 0.1, and a Hamming window.

   double sigma = 0.482414167;

   fn[0] = (theta_pb +sigma)/M_PI;
   for (int i = 1; i <= N; i++) {
      double t = i*(theta_pb+sigma);
      fn[i] = 2.0*sin(t)/(i*M_PI);
   }
}

void smooth_fir(
   int dim,
   FiniteElementSpace *fespace,
   Mesh* mesh,
   FiniteElementCollection *fec,
   int mesh_poly_deg)
{
   GridFunction *x;
   x = mesh->GetNodes();

   FiniteElementCollection *grad_fec;
   FiniteElementSpace *scal_fes;
   FiniteElementSpace *grad_fes;
   DiscreteLinearOperator *grad;
   scal_fes = new FiniteElementSpace(mesh, fec);

   if (dim == 2)
   {
      grad_fec = new ND_FECollection(mesh_poly_deg, dim);
   }
   else
      grad_fec = new ND_FECollection(mesh_poly_deg, dim);
   grad_fes = new FiniteElementSpace(mesh, grad_fec);
   grad = new DiscreteLinearOperator(scal_fes, grad_fes);
   grad->AddDomainInterpolator(new GradientInterpolator);
   grad->Assemble();
   grad->Finalize();
   SparseMatrix &G = grad->SpMat();
   SparseMatrix *Gt = Transpose(G);
   SparseMatrix *GtG = Mult(*Gt, G);
   delete Gt;
   delete grad;
   delete grad_fes;
   delete grad_fec;

   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   Array<int> ess_dofs;
   scal_fes->GetEssentialVDofs(ess_bdr, ess_dofs);
   delete scal_fes;

   int n = GtG->Size();

   // Create K matrix from GtG.
   SparseMatrix K(*GtG);

   // Rescale
   int nrows = GtG->Size();
   for (int i = 0; i < nrows; i++) {
      double diag = K.Elem(i,i);
      K.ScaleRow(i, -1.0/diag);
   }

   // Zero rows of stationary points
   for (int i = 0; i < ess_dofs.Size(); i++) {
      if (ess_dofs[i] < 0)
      {
         K.ScaleRow(i, 0.0);
      }
   }

   int smooth_steps;
   cout << "Enter number of smoothing steps --> "
        << flush;
   cin >> smooth_steps;

   // Smoothed mesh
   GridFunction xs(fespace);

   // Reasonable choices for this smoother
   int N = 10;

   // Passband frequency
   double k_pb = 0.1;

   // Get window coefficients
   Vector wn(N+1);
   compute_window("hamming",N,wn);

   // Get chebyshev coefficients (without window factor)
   Vector fn(N+1);
   compute_chebyshev_coeffs(N, fn, k_pb);

   FIRSmoother S(N, wn, fn);
   S.SetOperator(K);

   {
      ostringstream os;
      os << "fir-it0.mesh";
      ofstream mesh_ofs(os.str().c_str());
      mesh->Print(mesh_ofs);
   }

   for (int n = 0; n < smooth_steps; n++) {

      S.Mult(*x, xs);
      *x = xs;

      {
         ostringstream os;
         os << "fir-it" << n+1 << ".mesh";
         ofstream mesh_ofs(os.str().c_str());
         mesh->Print(mesh_ofs);
      }

   }


}


int main (int argc, char *argv[])
{
   Mesh *mesh;
   char vishost[] = "localhost";
   int  visport   = 19916;
   int  ans;

   bool dump_iterations = false;

   if (argc == 1)
   {
      cout << "Usage: ex6 <mesh_file>" << endl;
      return 1;
   }

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral or hexahedral elements with the same code.
   ifstream imesh(argv[1]);
   if (!imesh)
   {
      cerr << "can not open mesh file: " << argv[1] << endl;
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();

   int dim = mesh->Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 1000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
      cout << "enter ref. levels [" << ref_levels << "] --> " << flush;
      cin >> ref_levels;
      for (int l = 0; l < ref_levels; l++)
         mesh->UniformRefinement();
   }

   // 5. Define a finite element space on the mesh. Here we use vector finite
   //    elements which are tensor products of quadratic finite elements. The
   //    dimensionality of the vector finite element space is specified by the
   //    last parameter of the FiniteElementSpace constructor.
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
   FiniteElementCollection *fec;
   if (mesh_poly_deg <= 0)
   {
      fec = new QuadraticPosFECollection;
      mesh_poly_deg = 2;
   }
   else
      fec = new H1_FECollection(mesh_poly_deg, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, dim);

   // 6. Make the mesh curved based on the above finite element space. This
   //    means that we define the mesh elements through a fespace-based
   //    transformation of the reference element.
   mesh->SetNodalFESpace(fespace);

   // 7. Set up the right-hand side vector b. In this case we do not need to use
   //    a LinearForm object because b=0.
   Vector b(fespace->GetVSize());
   b = 0.0;

   // 8. Get the mesh nodes (vertices and other quadratic degrees of freedom in
   //    the finite element space) as a finite element grid function in fespace.
   GridFunction *x;
   x = mesh->GetNodes();

   // 9. Define a vector representing the minimal local mesh size in the mesh
   //    nodes. We index the nodes using the scalar version of the degrees of
   //    freedom in fespace.
   Vector h0(fespace->GetNDofs());
   h0 = numeric_limits<double>::infinity();
   {
      Array<int> dofs;
      // loop over the mesh elements
      for (int i = 0; i < fespace->GetNE(); i++)
      {
         // get the local scalar element degrees of freedom in dofs
         fespace->GetElementDofs(i, dofs);
         // adjust the value of h0 in dofs based on the local mesh size
         for (int j = 0; j < dofs.Size(); j++)
            h0(dofs[j]) = min(h0(dofs[j]), mesh->GetElementSize(i));
      }
   }

   // 10. Add a random perturbation of the nodes in the interior of the domain.
   //     We define a random grid function of fespace and make sure that it is
   //     zero on the boundary and its values are locally of the order of h0.
   //     The latter is based on the DofToVDof() method which maps the scalar to
   //     the vector degrees of freedom in fespace.
   GridFunction rdm(fespace);
   double jitter = 0.25; // perturbation scaling factor
   cout << "Enter jitter --> " << flush;
   cin >> jitter;
   rdm.Randomize();
   rdm -= 0.5; // shift to random values in [-0.5,0.5]
   rdm *= jitter;
   {
      // scale the random values to be of order of the local mesh size
      for (int i = 0; i < fespace->GetNDofs(); i++)
         for (int d = 0; d < dim; d++)
            rdm(fespace->DofToVDof(i,d)) *= h0(i);

      Array<int> vdofs;
      // loop over the boundary elements
      for (int i = 0; i < fespace->GetNBE(); i++)
      {
         // get the vector degrees of freedom in the boundary element
         fespace->GetBdrElementVDofs(i, vdofs);
         // set the boundary values to zero
         for (int j = 0; j < vdofs.Size(); j++)
            rdm(vdofs[j]) = 0.0;
      }
   }
   *x -= rdm;

   // 11. Save the perturbed mesh to a file. This output can be viewed later
   //     using GLVis: "glvis -m perturbed.mesh".
   {
      ofstream mesh_ofs("perturbed.mesh");
      mesh->Print(mesh_ofs);
   }

   // 12. (Optional) Send the initially perturbed mesh with the vector field
   //     representing the displacements to the original mesh to GLVis.
   cout << "Visualize the initial random perturbation? [0/1] --> ";
   cin >> ans;
   if (ans)
   {
      osockstream sol_sock(visport, vishost);
      sol_sock << "solution\n";
      mesh->Print(sol_sock);
      rdm.Save(sol_sock);
      sol_sock.send();
   }

   int smoother;
   cout <<
      "Select smoother:\n"
      "0) Mesquite\n"
      "1) Harmonic, type 0\n"
      "2) Harmonic, type 1\n"
      "3) Harmonic, using G^t G\n"
      "4) Hyperelastic model\n"
      "5) Taubin Smoothing, using G^t G\n"
      "6) FIR Filter Smoothing, using G^t G\n"
      " --> " << flush;
   cin >> smoother;

   // 14. Simple mesh smoothing can be performed by relaxing the node coordinate
   //     grid function x with the matrix A and right-hand side b. This process
   //     converges to the solution of Ax=b, which we solve below with PCG. Note
   //     that the computed x is the A-harmonic extension of its boundary values
   //     (the coordinates of the boundary vertices). Furthermore, note that
   //     changing x automatically changes the shapes of the elements in the
   //     mesh. The vector field that gives the displacements to the perturbed
   //     positions is saved in the grid function x0.
   GridFunction x0(fespace);
   x0 = *x;

   if (smoother == 0)
   {
      int mesquite_option = 0;
      cout <<
         "Select Mesquite option:\n"
         "0) Laplace (local)\n"
         "1) Untangler (local)\n"
         "2) Shape Improver (global)\n"
         "3) Minimum Edge-Length Improver (global)\n"
         " --> " << flush;
      cin >> mesquite_option;
#ifdef MFEM_USE_MESQUITE
      mesh->MesquiteSmooth(mesquite_option);
#else
      cout << "Mesquite not compiled in." << endl;
      exit(1);
#endif
   }
   else if (smoother == 1 || smoother == 2)
   {
      // 13. Set up the bilinear form a(.,.) corresponding to the mesh Laplacian
      //     operator. The imposed boundary conditions mean that the nodes on
      //     the boundary of the domain will not be relaxed. After assembly and
      //     finalizing we extract the corresponding sparse matrix A.

      BilinearForm *a = new BilinearForm(fespace);
      a->AddDomainIntegrator(new VectorMeshLaplacianIntegrator(smoother-1));
      a->Assemble();
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      a->EliminateEssentialBC(ess_bdr, *x, b);
      a->Finalize();
      SparseMatrix &A = a->SpMat();

      int smooth_steps;
      cout << "Enter number of smoothing steps or 0 for global solve --> "
           << flush;
      cin >> smooth_steps;

      if (smooth_steps <= 0)
      {
         GSSmoother M(A);
         PCG(A, M, b, *x, 1, 2000, 1e-12, 0.0);
         // MINRES(A, M, b, *x, 1, 2000, 1e-12, 0.0);
      }
      else
      {
         // l1-Jacobi smoothing
         const double a = 1.;
         DSmoother S(A, 1, a, smooth_steps);
         S.iterative_mode = true;
         S.Mult(b, *x);
      }

      delete a;
   }
   else if (smoother == 3)
   {
      FiniteElementCollection *grad_fec;
      FiniteElementSpace *scal_fes;
      FiniteElementSpace *grad_fes;
      DiscreteLinearOperator *grad;
      scal_fes = new FiniteElementSpace(mesh, fec);
      if (dim == 2)
      {
         grad_fec = new ND_FECollection(mesh_poly_deg, dim);
         // grad_fec = new RT_FECollection(mesh_poly_deg, dim);
      }
      else
         grad_fec = new ND_FECollection(mesh_poly_deg, dim);
      grad_fes = new FiniteElementSpace(mesh, grad_fec);
      grad = new DiscreteLinearOperator(scal_fes, grad_fes);
      grad->AddDomainInterpolator(new GradientInterpolator);
      grad->Assemble();
      grad->Finalize();
      SparseMatrix &G = grad->SpMat();
      SparseMatrix *Gt = Transpose(G);
      SparseMatrix *GtG = Mult(*Gt, G);
      delete Gt;
      delete grad;
      delete grad_fes;
      delete grad_fec;

      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      Array<int> ess_dofs;
      scal_fes->GetEssentialVDofs(ess_bdr, ess_dofs);
      delete scal_fes;

      int n = GtG->Size(), dof_counter = 0;
      SparseMatrix GtGe(n);
      for (int i = 0; i < ess_dofs.Size(); i++)
         if (ess_dofs[i] < 0)
         {
            GtG->EliminateRowCol(i, GtGe);
            dof_counter++;
         }
      Array<int> ess_dof_list(dof_counter);
      dof_counter = 0;
      for (int i = 0; i < ess_dofs.Size(); i++)
         if (ess_dofs[i] < 0)
            ess_dof_list[dof_counter++] = i;

      int smooth_steps;
      cout << "Enter number of smoothing steps or 0 for global solve --> "
           << flush;
      cin >> smooth_steps;

      GSSmoother M(*GtG);
      // fespace->ordering is byNODES
      Vector xd(NULL, n), bd(NULL, n);
      for (int d = 0; d < dim; d++)
      {
         xd.SetData(&(*x)(d*n));
         bd.SetData(&b(d*n));
         GtGe.AddMult(xd, bd, -1.);
         GtG->PartMult(ess_dof_list, xd, bd);

         if (smooth_steps <= 0)
         {
            PCG(*GtG, M, bd, xd, 1, 200, 1e-12, 0.0);
         }
         else
         {

            if (dump_iterations) {
               ostringstream os;
               os << "GtG-it0.mesh";
               ofstream mesh_ofs(os.str().c_str());
               mesh->Print(mesh_ofs);
            }

            // Jacobi2 smoothing
            Vector yd(xd.Size());
            const double a = 1.;
            // cout << "Jacobi2 scaling factor = " << a << endl;
            for (int i = 0; i < smooth_steps; i++)
            {
               GtG->Jacobi2(bd, xd, yd, a);

               if (dump_iterations) {
                  ostringstream os;
                  os << "GtG-it" << i << ".mesh";
                  ofstream mesh_ofs(os.str().c_str());
                  mesh->Print(mesh_ofs);
               }

               if (++i >= smooth_steps)
               {
                  xd = yd;
                  break;
               }
               GtG->Jacobi2(bd, yd, xd, a);

               if (dump_iterations) {
                  ostringstream os;
                  os << "GtG-it" << i << ".mesh";
                  ofstream mesh_ofs(os.str().c_str());
                  mesh->Print(mesh_ofs);
               }


            }
         }
      }
   }
   else if (smoother == 4)
   {
      HyperelasticModel *model;

      cout <<
         "Choose hyperelastic model:\n"
         "0) Harmonic\n"
         "1) Inverse Harmonic (Winslow/Crowley)\n"
         " --> " << flush;
      cin >> ans;
      switch (ans)
      {
      case 0:  model = new HarmonicModel; break;
      default: model = new InverseHarmonicModel; break;
      }

      NonlinearForm a(fespace);
      a.AddDomainIntegrator(new HyperelasticNLFIntegrator(model));

      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      a.SetEssentialBC(ess_bdr);

      const double rtol = 1e-8;
      Solver *S;
      cout <<
         "Choose linear smoother:\n"
         "0) l1-Jacobi\n"
         "1) CG\n"
         "2) MINRES\n"
         " --> " << flush;
      cin >> ans;
      if (ans == 0)
      {
         cout << "Enter number of linear smoothing iterations --> " << flush;
         cin >> ans;
         S = new DSmoother(1, 1., ans);
      }
      else if (ans == 1)
      {
         cout << "Enter number of CG smoothing iterations --> " << flush;
         cin >> ans;
         CGSolver *cg = new CGSolver;
         cg->SetMaxIter(ans);
         cg->SetRelTol(rtol);
         cg->SetAbsTol(0.0);
         // cg->SetPrintLevel(3);
         S = cg;
      }
      else
      {
         cout << "Enter number of MINRES smoothing iterations --> " << flush;
         cin >> ans;
         MINRESSolver *minres = new MINRESSolver;
         minres->SetMaxIter(ans);
         minres->SetRelTol(rtol);
         minres->SetAbsTol(0.0);
         // minres->SetPrintLevel(3);
         S = minres;
      }

      cout << "Enter number of Newton iterations --> " << flush;
      cin >> ans;

      cout << "Initial strain energy : " << a.GetEnergy(*x) << endl;

      // note: (*x) are the mesh nodes
      NewtonSolver newt;
      newt.SetMaxIter(ans);
      newt.SetRelTol(rtol);
      newt.SetAbsTol(0.0);
      newt.SetPrintLevel(1);
      newt.SetOperator(a);
      newt.SetPreconditioner(*S);
      Vector b;
      newt.Mult(b, *x);

      if (!newt.GetConverged())
         cout << "NewtonIteration : rtol = " << rtol << " not achieved."
              << endl;

      cout << "Final strain energy   : " << a.GetEnergy(*x) << endl;

      delete S;
      delete model;
   }
   else if (smoother == 5)
   {
      smooth_taubin(dim, fespace, mesh, fec, mesh_poly_deg);
   }
   else if (smoother == 6)
   {
      smooth_fir(dim, fespace, mesh, fec, mesh_poly_deg);
   }
   else {
      printf("unknown smoothing option, smoother = %d\n",smoother);
      exit(1);
   }

   // Define mesh displacement
   x0 -= *x;

   // 15. Save the smoothed mesh to a file. This output can be viewed later
   //     using GLVis: "glvis -m smoothed.mesh".
   {
      ofstream mesh_ofs("smoothed.mesh");
      mesh_ofs.precision(14);
      mesh->Print(mesh_ofs);
   }
   // save subdivided VTK mesh?
   if (1)
   {
      cout << "Enter VTK mesh subdivision factor or 0 to skip --> " << flush;
      cin >> ans;
      if (ans > 0)
      {
         ofstream vtk_mesh("smoothed.vtk");
         vtk_mesh.precision(8);
         mesh->PrintVTK(vtk_mesh, ans);
      }
   }

   // 16. (Optional) Send the relaxed mesh with the vector field representing
   //     the displacements to the perturbed mesh by socket to a GLVis server.
   cout << "Visualize the smoothed mesh? [0/1] --> ";
   cin >> ans;
   if (ans)
   {
      osockstream sol_sock(visport, vishost);
      sol_sock << "solution\n";
      mesh->Print(sol_sock);
      x0.Save(sol_sock);
      sol_sock.send();
   }

   // 17. Free the used memory.
   delete fespace;
   delete fec;
   delete mesh;
}
