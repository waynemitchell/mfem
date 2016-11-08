#include "mfem.hpp"
#include <iostream>

using namespace std;
using namespace mfem;

double f1(const Vector & x);
double f2(const Vector & x);
double f3(const Vector & x);

void V2(const Vector & x, Vector & v);
void V3(const Vector & x, Vector & v);

void   F2(const Vector & x, Vector & v);
void   F3(const Vector & x, Vector & v);

double Grad_f1(const Vector & x);
void   Grad_f2(const Vector & x, Vector & df);
void   Grad_f3(const Vector & x, Vector & df);

double Curl_F2(const Vector & x);
void   Curl_F3(const Vector & x, Vector & df);

double Div_F2(const Vector & x);
double Div_F3(const Vector & x);

void   Vf2(const Vector & x, Vector & vf);
void   Vf3(const Vector & x, Vector & vf);

double VdotF2(const Vector & x);
double VdotF3(const Vector & x);

double VcrossF2(const Vector & x);
void   VcrossF3(const Vector & x, Vector & vf);

int main(int argc, char ** argv)
{
  int order = 2, w = 34, n = 1;
  
  OptionsParser args(argc, argv);
  args.AddOption(&order, "-o", "--order",
		 "Finite element order (polynomial degree) or -1 for"
		 " isoparametric space.");
  args.AddOption(&w, "-w", "--width",
		 "Width of output field.");
  args.AddOption(&n, "-n", "--num-elems",
		 "Number of elements in each direction.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh mesh1d(n, 2.0);
  // Mesh mesh1d(1, 1.0);
  Mesh mesh2d(n, n, Element::QUADRILATERAL, 1, 2.0, 3.0);
  // Mesh mesh3d(1, 1, 1, Element::HEXAHEDRON, 1, 2.0, 3.0, 5.0);
  Mesh mesh3d(n, n, n, Element::HEXAHEDRON, 1, 1.0, 1.0, 1.0);

  {
    FiniteElementCollection * fec_h1 = new H1_FECollection(order, 1);
    FiniteElementCollection * fec_nd = new ND_FECollection(order, 1);
    FiniteElementCollection * fec_l2 = new L2_FECollection(order - 1, 1);

    FiniteElementSpace fespace_h1(&mesh1d, fec_h1);
    FiniteElementSpace fespace_nd(&mesh1d, fec_nd);
    FiniteElementSpace fespace_l2(&mesh1d, fec_l2);

    cout << endl << "1D" << endl;
    cout << "Number H1 DoFs:  " << fespace_h1.GetNDofs() << endl;
    cout << "Number ND DoFs:  " << fespace_nd.GetNDofs() << endl;
    cout << "Number L2 DoFs:  " << fespace_l2.GetNDofs() << endl << endl;

    BilinearForm m_h1(&fespace_h1);
    m_h1.AddDomainIntegrator(new MassIntegrator());
    m_h1.Assemble();
    m_h1.Finalize();
    /*    
    BilinearForm m_nd(&fespace_nd);
    m_nd.AddDomainIntegrator(new MassIntegrator());
    m_nd.Assemble();
    m_nd.Finalize();
    */
    BilinearForm m_l2(&fespace_l2);
    m_l2.AddDomainIntegrator(new MassIntegrator());
    m_l2.Assemble();
    m_l2.Finalize();
 
    Vector tmp_h1(fespace_h1.GetNDofs());
    Vector tmp_l2(fespace_l2.GetNDofs());
    
    FunctionCoefficient f1_coef(f1);
    FunctionCoefficient df1_coef(Grad_f1);
    
    GridFunction f1_h1(&fespace_h1); f1_h1.ProjectCoefficient(f1_coef);
    GridFunction f1_l2(&fespace_l2); f1_l2.ProjectCoefficient(f1_coef);
    GridFunction g1_h1(&fespace_h1);
    GridFunction g1_l2(&fespace_l2);

     cout << setw(w) << setiosflags(std::ios::left)
	  << "Error in H1 projection: " << f1_h1.ComputeL2Error(f1_coef)
	  << endl;
     cout<< setw(w) << setiosflags(std::ios::left)
	 << "Error in L2 projection: " << f1_l2.ComputeL2Error(f1_coef)
	 << endl;
    {
      // cout << "Testing Mass Integrator in 1D" << endl;
      MixedBilinearForm blf(&fespace_h1, &fespace_l2);
      blf.AddDomainIntegrator(new MixedScalarMassIntegrator());
      blf.Assemble();
      blf.Finalize();

      blf.Mult(f1_h1,tmp_l2); g1_l2 = 0.0;
      CG(m_l2, tmp_l2, g1_l2, 0, 200, 1e-12, 0.0);

      cout << setw(w) << setiosflags(std::ios::left)
	   << "Error in H1 -> L2 mapping: " << g1_l2.ComputeL2Error(f1_coef)
	   << endl;

      MixedBilinearForm blfw(&fespace_l2, &fespace_h1);
      blfw.AddDomainIntegrator(new MixedScalarMassIntegrator());
      blfw.Assemble();
      blfw.Finalize();

      SparseMatrix * blfT = Transpose(blfw.SpMat());
      SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);
      cout << setw(w) << setiosflags(std::ios::left)
	   << "Max norm Op - (Weak Op)^T: " << diff->MaxNorm() << endl;
    }
    {
      // cout << "Testing Derivative Integrator in 1D" << endl;
      MixedBilinearForm blf(&fespace_h1, &fespace_l2);
      blf.AddDomainIntegrator(new MixedScalarDerivativeIntegrator());
      blf.Assemble();
      blf.Finalize();
      // blf.Print(cout);

      blf.Mult(f1_h1,tmp_l2); g1_l2 = 0.0;
      CG(m_l2, tmp_l2, g1_l2, 0, 200, 1e-12, 0.0);
      // g1_l2.Print(cout);
      cout << setw(w) << setiosflags(std::ios::left)
	   << "Error in Derivative: " << g1_l2.ComputeL2Error(df1_coef)
	   << endl;

      MixedBilinearForm blfw(&fespace_l2, &fespace_h1);
      blfw.AddDomainIntegrator(new MixedScalarWeakDerivativeIntegrator());
      blfw.Assemble();
      blfw.Finalize();

      SparseMatrix * blfT = Transpose(blfw.SpMat());
      SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);
      cout << setw(w) << setiosflags(std::ios::left)
	   << "Max norm Op + (Weak Op)^T: " << diff->MaxNorm() << endl;
    }
    /*
    {
      cout << "Testing Weak-Derivative Integrator in 1D" << endl;
      MixedBilinearForm blf(&fespace_l2, &fespace_h1);
      blf.AddDomainIntegrator(new MixedScalarWeakDerivativeIntegrator());
      blf.Assemble();
      blf.Finalize();
      blf.SpMat().Print(cout);

      blf.Mult(f1_l2,tmp_h1); g1_h1 = 0.0;
      CG(m_h1, tmp_h1, g1_h1, 0, 200, 1e-12, 0.0);
      f1_h1.Print(cout);
      f1_l2.Print(cout);
      tmp_h1.Print(cout);
      g1_h1.Print(cout);
      // g1_h1 *= -1.0;
      cout << "Error in Weak-Derivative:  " << g1_h1.ComputeL2Error(df1_coef)
	   << endl;
    }
    */
  }

  {
    FiniteElementCollection * fec_h1  = new H1_FECollection(order, 2);
    // FiniteElementCollection * fec_h1p = new H1_FECollection(order + 1, 2);
    FiniteElementCollection * fec_nd  = new ND_FECollection(order, 2);
    FiniteElementCollection * fec_ndp = new ND_FECollection(order + 1, 2);
    FiniteElementCollection * fec_rt  = new RT_FECollection(order - 1, 2);
    FiniteElementCollection * fec_l2  = new L2_FECollection(order - 1, 2);

    
    FiniteElementSpace fespace_h1(&mesh2d, fec_h1);
    // FiniteElementSpace fespace_h1p(&mesh2d, fec_h1p);
    FiniteElementSpace fespace_nd(&mesh2d, fec_nd);
    FiniteElementSpace fespace_ndp(&mesh2d, fec_ndp);
    FiniteElementSpace fespace_rt(&mesh2d, fec_rt);
    FiniteElementSpace fespace_l2(&mesh2d, fec_l2);

    cout << endl << "2D" << endl;
    cout << "Number H1 DoFs:   " << fespace_h1.GetNDofs() << endl;
    // cout << "Number H1p DoFs:  " << fespace_h1p.GetNDofs() << endl;
    cout << "Number ND DoFs:   " << fespace_nd.GetNDofs() << endl;
    cout << "Number NDp DoFs:  " << fespace_ndp.GetNDofs() << endl;
    cout << "Number RT DoFs:   " << fespace_rt.GetNDofs() << endl;
    cout << "Number L2 DoFs:   " << fespace_l2.GetNDofs() << endl << endl;

    BilinearForm m_h1(&fespace_h1);
    m_h1.AddDomainIntegrator(new MassIntegrator());
    m_h1.Assemble();
    m_h1.Finalize();
    
    // BilinearForm m_h1p(&fespace_h1p);
    // m_h1p.AddDomainIntegrator(new MassIntegrator());
    // m_h1p.Assemble();
    // m_h1p.Finalize();
    
    BilinearForm m_nd(&fespace_nd);
    m_nd.AddDomainIntegrator(new VectorFEMassIntegrator());
    m_nd.Assemble();
    m_nd.Finalize();
    
    BilinearForm m_ndp(&fespace_ndp);
    m_ndp.AddDomainIntegrator(new VectorFEMassIntegrator());
    m_ndp.Assemble();
    m_ndp.Finalize();
    
    BilinearForm m_rt(&fespace_rt);
    m_rt.AddDomainIntegrator(new VectorFEMassIntegrator());
    m_rt.Assemble();
    m_rt.Finalize();
    
    BilinearForm m_l2(&fespace_l2);
    m_l2.AddDomainIntegrator(new MassIntegrator());
    m_l2.Assemble();
    m_l2.Finalize();

    Vector tmp_h1(fespace_h1.GetNDofs());
    // Vector tmp_h1p(fespace_h1p.GetNDofs());
    Vector tmp_nd(fespace_nd.GetNDofs());
    Vector tmp_ndp(fespace_ndp.GetNDofs());
    Vector tmp_rt(fespace_rt.GetNDofs());
    Vector tmp_l2(fespace_l2.GetNDofs());

    FunctionCoefficient       f2_coef(f2);
    VectorFunctionCoefficient F2_coef(2, F2);
    
    VectorFunctionCoefficient grad_f2_coef(2, Grad_f2);
    FunctionCoefficient       curl_f2_coef(Curl_F2);
    FunctionCoefficient       div_f2_coef(Div_F2);
    
    VectorFunctionCoefficient V2_coef(2, V2);
    VectorFunctionCoefficient Vf2_coef(2, Vf2);
    FunctionCoefficient       VdotF2_coef(VdotF2);
	
    GridFunction f2_h1(&fespace_h1); f2_h1.ProjectCoefficient(f2_coef);
    // GridFunction f2_h1p(&fespace_h1p); f2_h1p.ProjectCoefficient(VdotF2_coef);
    GridFunction f2_nd(&fespace_nd); f2_nd.ProjectCoefficient(F2_coef);
    GridFunction f2_rt(&fespace_rt); f2_rt.ProjectCoefficient(F2_coef);
    GridFunction f2_l2(&fespace_l2); f2_l2.ProjectCoefficient(f2_coef);

    GridFunction g2_h1(&fespace_h1);
    // GridFunction g2_h1p(&fespace_h1p);
    GridFunction g2_nd(&fespace_nd);
    GridFunction g2_ndp(&fespace_ndp);
    GridFunction g2_rt(&fespace_rt);
    GridFunction g2_l2(&fespace_l2);

     cout << setw(w) << setiosflags(std::ios::left)
	   << "Error in H1 projection:  " << f2_h1.ComputeL2Error(f2_coef)
	   << endl;
     // cout << setw(w) << setiosflags(std::ios::left)
     //	   << "Error in H1p projection:  " << f2_h1p.ComputeL2Error(VdotF2_coef)
     //	   << endl;
     cout << setw(w) << setiosflags(std::ios::left)
	   << "Error in ND projection:  " << f2_nd.ComputeL2Error(F2_coef)
	   << endl;
     cout << setw(w) << setiosflags(std::ios::left)
	   << "Error in RT projection:  " << f2_rt.ComputeL2Error(F2_coef)
	   << endl;
     cout << setw(w) << setiosflags(std::ios::left)
	   << "Error in L2 projection:  " << f2_l2.ComputeL2Error(f2_coef)
	   << endl;

     {
       // cout << "Testing Scalar Mass Integrator in 2D" << endl;
      MixedBilinearForm blf(&fespace_h1, &fespace_l2);
      blf.AddDomainIntegrator(new MixedScalarMassIntegrator());
      blf.Assemble();
      blf.Finalize();
      // blf.Print(cout);

      blf.Mult(f2_h1,tmp_l2); g2_l2 = 0.0;
      CG(m_l2, tmp_l2, g2_l2, 0, 200, 1e-12, 0.0);

      cout << setw(w) << setiosflags(std::ios::left)
	   << "Error in H1 -> L2 mapping: " << g2_l2.ComputeL2Error(f2_coef)
	   << endl;

      MixedBilinearForm blfw(&fespace_l2, &fespace_h1);
      blfw.AddDomainIntegrator(new MixedScalarMassIntegrator());
      blfw.Assemble();
      blfw.Finalize();

      SparseMatrix * blfT = Transpose(blfw.SpMat());
      SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);
      cout << setw(w) << setiosflags(std::ios::left)
	   << "Max norm Op - (Weak Op)^T: " << diff->MaxNorm() << endl;
    }
    {
      // cout << "Testing Vector Mass Integrator in 2D" << endl;
      MixedBilinearForm blf(&fespace_nd, &fespace_rt);
      blf.AddDomainIntegrator(new MixedVectorMassIntegrator());
      blf.Assemble();
      blf.Finalize();
      // blf.SpMat().Print(cout);
      // cout << endl;

      blf.Mult(f2_nd,tmp_rt); g2_rt = 0.0;
      CG(m_rt, tmp_rt, g2_rt, 0, 200, 1e-12, 0.0);

      cout << setw(w) << setiosflags(std::ios::left)
	   << "Error in ND -> RT mapping: " << g2_rt.ComputeL2Error(F2_coef)
	   << endl;

      MixedBilinearForm blfw(&fespace_rt, &fespace_nd);
      blfw.AddDomainIntegrator(new MixedVectorMassIntegrator());
      blfw.Assemble();
      blfw.Finalize();

      SparseMatrix * blfT = Transpose(blfw.SpMat());
      SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);
      cout << setw(w) << setiosflags(std::ios::left)
	   << "Max norm Op - (Weak Op)^T: " << diff->MaxNorm() << endl;
    }
    {
      // cout << "Testing Gradient Integrator in 2D" << endl;
      MixedBilinearForm blf(&fespace_h1, &fespace_nd);
      blf.AddDomainIntegrator(new MixedVectorGradientIntegrator());
      blf.Assemble();
      blf.Finalize();
      // blf.SpMat().Print(cout);
      // cout << endl;

      blf.Mult(f2_h1,tmp_nd); g2_nd = 0.0;
      CG(m_nd, tmp_nd, g2_nd, 0, 200, 1e-12, 0.0);

      cout << setw(w) << setiosflags(std::ios::left)
	   << "Error in Gradient:  " << g2_nd.ComputeL2Error(grad_f2_coef)
	   << endl;

      MixedBilinearForm blfw(&fespace_nd, &fespace_h1);
      blfw.AddDomainIntegrator(new MixedVectorWeakDivergenceIntegrator());
      blfw.Assemble();
      blfw.Finalize();

      SparseMatrix * blfT = Transpose(blfw.SpMat());
      SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);
      cout << setw(w) << setiosflags(std::ios::left)
	   << "Max norm Op + (Weak Op)^T: " << diff->MaxNorm() << endl;
    }
    /*
    {
      cout << "Testing Weak-Divergence Integrator in 2D" << endl;
      MixedBilinearForm blf(&fespace_nd, &fespace_h1);
      blf.AddDomainIntegrator(new MixedVectorWeakDivergenceIntegrator());
      blf.Assemble();
      blf.Finalize();
      // blf.SpMat().Print(cout);
      // cout << endl;
    }
    */
    {
      // cout << "Testing Curl Integrator in 2D" << endl;
      MixedBilinearForm blf(&fespace_nd, &fespace_l2);
      blf.AddDomainIntegrator(new MixedScalarCurlIntegrator());
      blf.Assemble();
      blf.Finalize();
      // blf.Print(cout);

      blf.Mult(f2_nd,tmp_l2); g2_l2 = 0.0;
      CG(m_l2, tmp_l2, g2_l2, 0, 200, 1e-12, 0.0);

      cout << setw(w) << setiosflags(std::ios::left)
	   << "Error in Curl:  " << g2_l2.ComputeL2Error(curl_f2_coef)
	   << endl;

      MixedBilinearForm blfw(&fespace_l2, &fespace_nd);
      blfw.AddDomainIntegrator(new MixedScalarWeakCurlIntegrator());
      blfw.Assemble();
      blfw.Finalize();

      SparseMatrix * blfT = Transpose(blfw.SpMat());
      SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);
      cout << setw(w) << setiosflags(std::ios::left)
	   << "Max norm Op - (Weak Op)^T: " << diff->MaxNorm() << endl;
    }
    /*
    {
      cout << "Testing Weak-Curl Integrator in 2D" << endl;
      MixedBilinearForm blf(&fespace_l2, &fespace_nd);
      blf.AddDomainIntegrator(new MixedScalarWeakCurlIntegrator());
      blf.Assemble();
      blf.Finalize();
      // blf.Print(cout);
    }
    */
    {
      // cout << "Testing Divergence Integrator in 2D" << endl;
      MixedBilinearForm blf(&fespace_rt, &fespace_l2);
      blf.AddDomainIntegrator(new MixedScalarDivergenceIntegrator());
      blf.Assemble();
      blf.Finalize();
      // blf.Print(cout);

      blf.Mult(f2_rt,tmp_l2); g2_l2 = 0.0;
      CG(m_l2, tmp_l2, g2_l2, 0, 200, 1e-12, 0.0);

      cout << setw(w) << setiosflags(std::ios::left)
	   << "Error in Divergence:  " << g2_l2.ComputeL2Error(div_f2_coef)
	   << endl;

      MixedBilinearForm blfw(&fespace_l2, &fespace_rt);
      blfw.AddDomainIntegrator(new MixedScalarWeakGradientIntegrator());
      blfw.Assemble();
      blfw.Finalize();

      SparseMatrix * blfT = Transpose(blfw.SpMat());
      SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);
      cout << setw(w) << setiosflags(std::ios::left)
	   << "Max norm Op + (Weak Op)^T: " << diff->MaxNorm() << endl;
    }
    /*
    {
      cout << "Testing Weak-Gradient Integrator in 2D" << endl;
      MixedBilinearForm blf(&fespace_l2, &fespace_rt);
      blf.AddDomainIntegrator(new MixedScalarWeakGradientIntegrator());
      blf.Assemble();
      blf.Finalize();
      // blf.Print(cout);
    }
    */
    {
      // cout << "Testing Divergence Integrator in 2D" << endl;
      MixedBilinearForm blf(&fespace_h1, &fespace_ndp);
      blf.AddDomainIntegrator(new MixedVectorProductIntegrator(V2_coef));
      blf.Assemble();
      blf.Finalize();
      // blf.Print(cout);

      blf.Mult(f2_h1,tmp_ndp); g2_ndp = 0.0;
      CG(m_ndp, tmp_ndp, g2_ndp, 0, 200, 1e-12, 0.0);

      cout << setw(w) << setiosflags(std::ios::left)
	   << "Error in Vector Product:  " << g2_ndp.ComputeL2Error(Vf2_coef)
	   << endl;
    }
    {
      // cout << "Testing Divergence Integrator in 2D" << endl;
      MixedBilinearForm blf(&fespace_nd, &fespace_h1);
      blf.AddDomainIntegrator(new MixedDotProductIntegrator(V2_coef));
      blf.Assemble();
      blf.Finalize();
      // blf.Print(cout);

      blf.Mult(f2_nd,tmp_h1); g2_h1 = 0.0;
      CG(m_h1, tmp_h1, g2_h1, 0, 200, 1e-12, 0.0);

      cout << setw(w) << setiosflags(std::ios::left)
	   << "Error in Dot Product:  " << g2_h1.ComputeL2Error(VdotF2_coef)
	   << endl;
    }
  }
  
  {
    FiniteElementCollection * fec_h1  = new H1_FECollection(order, 3);
    FiniteElementCollection * fec_nd  = new ND_FECollection(order, 3);
    FiniteElementCollection * fec_rt  = new RT_FECollection(order - 1, 3);
    FiniteElementCollection * fec_l2  = new L2_FECollection(order - 1, 3);

    FiniteElementCollection * fec_h1p = new H1_FECollection(order + 1, 3);
    FiniteElementCollection * fec_ndp = new ND_FECollection(order + 1, 3);
    FiniteElementCollection * fec_rtp = new RT_FECollection(order, 3);

    FiniteElementSpace fespace_h1(&mesh3d, fec_h1);
    FiniteElementSpace fespace_nd(&mesh3d, fec_nd);
    FiniteElementSpace fespace_rt(&mesh3d, fec_rt);
    FiniteElementSpace fespace_l2(&mesh3d, fec_l2);

    FiniteElementSpace fespace_h1p(&mesh3d, fec_h1p);
    FiniteElementSpace fespace_ndp(&mesh3d, fec_ndp);
    FiniteElementSpace fespace_rtp(&mesh3d, fec_rtp);

    cout << endl << "3D" << endl;
    cout << "Number H1 DoFs:   " << fespace_h1.GetNDofs()  << endl;
    cout << "Number ND DoFs:   " << fespace_nd.GetNDofs()  << endl;
    cout << "Number RT DoFs:   " << fespace_rt.GetNDofs()  << endl;
    cout << "Number L2 DoFs:   " << fespace_l2.GetNDofs()  << endl << endl;

    cout << "Number H1p DoFs:  " << fespace_h1p.GetNDofs() << endl;
    cout << "Number NDp DoFs:  " << fespace_ndp.GetNDofs() << endl;
    cout << "Number RTp DoFs:  " << fespace_rtp.GetNDofs() << endl << endl;

    BilinearForm m_h1(&fespace_h1);
    m_h1.AddDomainIntegrator(new MassIntegrator());
    m_h1.Assemble();
    m_h1.Finalize();
    
    BilinearForm m_nd(&fespace_nd);
    m_nd.AddDomainIntegrator(new VectorFEMassIntegrator());
    m_nd.Assemble();
    m_nd.Finalize();
    
    BilinearForm m_rt(&fespace_rt);
    m_rt.AddDomainIntegrator(new VectorFEMassIntegrator());
    m_rt.Assemble();
    m_rt.Finalize();
    
    BilinearForm m_l2(&fespace_l2);
    m_l2.AddDomainIntegrator(new MassIntegrator());
    m_l2.Assemble();
    m_l2.Finalize();

    BilinearForm m_h1p(&fespace_h1p);
    m_h1p.AddDomainIntegrator(new MassIntegrator());
    m_h1p.Assemble();
    m_h1p.Finalize();
    
    BilinearForm m_ndp(&fespace_ndp);
    m_ndp.AddDomainIntegrator(new VectorFEMassIntegrator());
    m_ndp.Assemble();
    m_ndp.Finalize();
    
    BilinearForm m_rtp(&fespace_rtp);
    m_rtp.AddDomainIntegrator(new VectorFEMassIntegrator());
    m_rtp.Assemble();
    m_rtp.Finalize();
    
    Vector tmp_h1(fespace_h1.GetNDofs());
    Vector tmp_nd(fespace_nd.GetNDofs());
    Vector tmp_rt(fespace_rt.GetNDofs());
    Vector tmp_l2(fespace_l2.GetNDofs());

    Vector tmp_h1p(fespace_h1p.GetNDofs());
    Vector tmp_ndp(fespace_ndp.GetNDofs());
    Vector tmp_rtp(fespace_rtp.GetNDofs());

    FunctionCoefficient       f3_coef(f3);
    VectorFunctionCoefficient F3_coef(3, F3);

    VectorFunctionCoefficient grad_f3_coef(3, Grad_f3);
    VectorFunctionCoefficient curl_f3_coef(3, Curl_F3);
    FunctionCoefficient       div_f3_coef(Div_F3);

    VectorFunctionCoefficient V3_coef(3, V3);
    VectorFunctionCoefficient Vf3_coef(3, Vf3);
    FunctionCoefficient       VdotF3_coef(VdotF3);
    VectorFunctionCoefficient VcrossF3_coef(3, VcrossF3);

    GridFunction f3_h1(&fespace_h1); f3_h1.ProjectCoefficient(f3_coef);
    GridFunction f3_nd(&fespace_nd); f3_nd.ProjectCoefficient(F3_coef);
    GridFunction f3_rt(&fespace_rt); f3_rt.ProjectCoefficient(F3_coef);
    GridFunction f3_l2(&fespace_l2); f3_l2.ProjectCoefficient(f3_coef);

    GridFunction g3_h1(&fespace_h1);
    GridFunction g3_nd(&fespace_nd);
    GridFunction g3_rt(&fespace_rt);
    GridFunction g3_l2(&fespace_l2);

    GridFunction g3_h1p(&fespace_h1p); g3_h1p.ProjectCoefficient(VdotF3_coef);
    GridFunction g3_ndp(&fespace_ndp); g3_ndp.ProjectCoefficient(Vf3_coef);
    GridFunction g3_rtp(&fespace_rtp);
    g3_rtp.ProjectCoefficient(VcrossF3_coef);

     cout << setw(w) << setiosflags(std::ios::left)
	  << "Error in H1 projection of f3:  "
	  << f3_h1.ComputeL2Error(f3_coef)
	  << endl;
     cout << setw(w) << setiosflags(std::ios::left)
	  << "Error in ND projection of F3:  "
	  << f3_nd.ComputeL2Error(F3_coef)
	  << endl;
     cout << setw(w) << setiosflags(std::ios::left)
	  << "Error in RT projection of F3:  "
	  << f3_rt.ComputeL2Error(F3_coef)
	  << endl;
     cout << setw(w) << setiosflags(std::ios::left)
	  << "Error in L2 projection of f3:  "
	  << f3_l2.ComputeL2Error(f3_coef)
	  << endl;

     cout << setw(w) << setiosflags(std::ios::left)
	  << "Error in H1p projection of V.F3:  "
	  << g3_h1p.ComputeL2Error(VdotF3_coef)
	  << endl;
     cout << setw(w) << setiosflags(std::ios::left)
	  << "Error in NDp projection of Vf3:  "
	  << g3_ndp.ComputeL2Error(Vf3_coef)
	  << endl;
     cout << setw(w) << setiosflags(std::ios::left)
	  << "Error in RTp projection of VxF3:  "
	  << g3_rtp.ComputeL2Error(VcrossF3_coef)
	  << endl;
     {
       // cout << "Testing Scalar Mass Integrator in 3D" << endl;
      MixedBilinearForm blf(&fespace_h1, &fespace_l2);
      blf.AddDomainIntegrator(new MixedScalarMassIntegrator());
      blf.Assemble();
      blf.Finalize();
      // blf.Print(cout);

      blf.Mult(f3_h1,tmp_l2); g3_l2 = 0.0;
      CG(m_l2, tmp_l2, g3_l2, 0, 200, 1e-12, 0.0);

      cout << setw(w) << setiosflags(std::ios::left)
	   << "Error in H1 -> L2 mapping: " << g3_l2.ComputeL2Error(f3_coef)
	   << endl;

      MixedBilinearForm blfw(&fespace_l2, &fespace_h1);
      blfw.AddDomainIntegrator(new MixedScalarMassIntegrator());
      blfw.Assemble();
      blfw.Finalize();

      SparseMatrix * blfT = Transpose(blfw.SpMat());
      SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);
      cout << setw(w) << setiosflags(std::ios::left)
	   << "Max norm Op - (Weak Op)^T: " << diff->MaxNorm() << endl;
    }
     {
       // cout << "Testing Vector Mass Integrator in 3D" << endl;
      MixedBilinearForm blf(&fespace_nd, &fespace_rt);
      blf.AddDomainIntegrator(new MixedVectorMassIntegrator());
      blf.Assemble();
      blf.Finalize();
      // blf.SpMat().Print(cout);
      // cout << endl;

      blf.Mult(f3_nd,tmp_rt); g3_rt = 0.0;
      CG(m_rt, tmp_rt, g3_rt, 0, 200, 1e-12, 0.0);

      cout << setw(w) << setiosflags(std::ios::left)
	   << "Error in ND -> RT mapping: " << g3_rt.ComputeL2Error(F3_coef)
	   << endl;

      MixedBilinearForm blfw(&fespace_rt, &fespace_nd);
      blfw.AddDomainIntegrator(new MixedVectorMassIntegrator());
      blfw.Assemble();
      blfw.Finalize();

      SparseMatrix * blfT = Transpose(blfw.SpMat());
      SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);
      cout << setw(w) << setiosflags(std::ios::left)
	   << "Max norm Op - (Weak Op)^T: " << diff->MaxNorm() << endl;
    }
    {
      // cout << "Testing Gradient Integrator in 3D" << endl;
      MixedBilinearForm blf(&fespace_h1, &fespace_nd);
      blf.AddDomainIntegrator(new MixedVectorGradientIntegrator());
      blf.Assemble();
      blf.Finalize();
      // blf.SpMat().Print(cout);
      // cout << endl;

      blf.Mult(f3_h1,tmp_nd); g3_nd = 0.0;
      CG(m_nd, tmp_nd, g3_nd, 0, 200, 1e-12, 0.0);

      cout << setw(w) << setiosflags(std::ios::left)
	   << "Error in Gradient:  " << g3_nd.ComputeL2Error(grad_f3_coef)
	   << endl;

      MixedBilinearForm blfw(&fespace_nd, &fespace_h1);
      blfw.AddDomainIntegrator(new MixedVectorWeakDivergenceIntegrator());
      blfw.Assemble();
      blfw.Finalize();

      SparseMatrix * blfT = Transpose(blfw.SpMat());
      SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);
      cout << setw(w) << setiosflags(std::ios::left)
	   << "Max norm Op + (Weak Op)^T: " << diff->MaxNorm() << endl;
    }
    {
      // cout << "Testing Curl Integrator in 3D" << endl;
      MixedBilinearForm blf(&fespace_nd, &fespace_rt);
      blf.AddDomainIntegrator(new MixedVectorCurlIntegrator());
      blf.Assemble();
      blf.Finalize();
      // blf.SpMat().Print(cout);
      // cout << endl;

      blf.Mult(f3_nd,tmp_rt); g3_rt = 0.0;
      CG(m_rt, tmp_rt, g3_rt, 0, 200, 1e-12, 0.0);

      cout << setw(w) << setiosflags(std::ios::left)
	   << "Error in Curl:  " << g3_rt.ComputeL2Error(curl_f3_coef)
	   << endl;

      MixedBilinearForm blfw(&fespace_rt, &fespace_nd);
      blfw.AddDomainIntegrator(new MixedVectorWeakCurlIntegrator());
      blfw.Assemble();
      blfw.Finalize();

      SparseMatrix * blfT = Transpose(blfw.SpMat());
      SparseMatrix * diff = Add(1.0,blf.SpMat(),-1.0,*blfT);
      cout << setw(w) << setiosflags(std::ios::left)
	   << "Max norm Op - (Weak Op)^T: " << diff->MaxNorm() << endl;
    }
    {
      // cout << "Testing Divergence Integrator in 3D" << endl;
      MixedBilinearForm blf(&fespace_rt, &fespace_l2);
      blf.AddDomainIntegrator(new MixedScalarDivergenceIntegrator());
      blf.Assemble();
      blf.Finalize();
      //blf.Print(cout);

      blf.Mult(f3_rt,tmp_l2); g3_l2 = 0.0;
      CG(m_l2, tmp_l2, g3_l2, 0, 200, 1e-12, 0.0);

      cout << setw(w) << setiosflags(std::ios::left)
	   << "Error in Divergence:  " << g3_l2.ComputeL2Error(div_f3_coef)
	   << endl;

      MixedBilinearForm blfw(&fespace_l2, &fespace_rt);
      blfw.AddDomainIntegrator(new MixedScalarWeakGradientIntegrator());
      blfw.Assemble();
      blfw.Finalize();

      SparseMatrix * blfT = Transpose(blfw.SpMat());
      SparseMatrix * diff = Add(1.0,blf.SpMat(),1.0,*blfT);
      cout << setw(w) << setiosflags(std::ios::left)
	   << "Max norm Op + (Weak Op)^T: " << diff->MaxNorm() << endl;
    }
    /*
    {
      cout << "Testing Weak-Divergence Integrator in 3D" << endl;
      MixedBilinearForm blf(&fespace_nd, &fespace_h1);
      blf.AddDomainIntegrator(new MixedVectorWeakDivergenceIntegrator());
      blf.Assemble();
      blf.Finalize();
      // blf.SpMat().Print(cout);
      // cout << endl;
    }
    {
      cout << "Testing Weak-Curl Integrator in 3D" << endl;
      MixedBilinearForm blf(&fespace_rt, &fespace_nd);
      blf.AddDomainIntegrator(new MixedVectorWeakCurlIntegrator());
      blf.Assemble();
      blf.Finalize();
      // blf.SpMat().Print(cout);
      // cout << endl;
    }
    {
      cout << "Testing Weak-Gradient Integrator in 3D" << endl;
      MixedBilinearForm blf(&fespace_l2, &fespace_rt);
      blf.AddDomainIntegrator(new MixedScalarWeakGradientIntegrator());
      blf.Assemble();
      blf.Finalize();
      // blf.Print(cout);
    }
    */
    {
      // cout << "Testing Divergence Integrator in 2D" << endl;
      MixedBilinearForm blf(&fespace_h1, &fespace_ndp);
      blf.AddDomainIntegrator(new MixedVectorProductIntegrator(V3_coef));
      blf.Assemble();
      blf.Finalize();
      // blf.Print(cout);

      blf.Mult(f3_h1,tmp_ndp); g3_ndp = 0.0;
      CG(m_ndp, tmp_ndp, g3_ndp, 0, 200, 1e-12, 0.0);

      cout << setw(w) << setiosflags(std::ios::left)
	   << "Error in Vector Product:  " << g3_ndp.ComputeL2Error(Vf3_coef)
	   << endl;
    }
    {
      // cout << "Testing Divergence Integrator in 2D" << endl;
      MixedBilinearForm blf(&fespace_nd, &fespace_h1p);
      blf.AddDomainIntegrator(new MixedDotProductIntegrator(V3_coef));
      blf.Assemble();
      blf.Finalize();
      // blf.Print(cout);

      blf.Mult(f3_nd,tmp_h1p); g3_h1p = 0.0;
      CG(m_h1p, tmp_h1p, g3_h1p, 0, 200, 1e-12, 0.0);

      cout << setw(w) << setiosflags(std::ios::left)
	   << "Error in Dot Product:  " << g3_h1p.ComputeL2Error(VdotF3_coef)
	   << endl;
    }
    {
      // cout << "Testing Divergence Integrator in 2D" << endl;
      MixedBilinearForm blf(&fespace_nd, &fespace_rtp);
      blf.AddDomainIntegrator(new MixedCrossProductIntegrator(V3_coef));
      blf.Assemble();
      blf.Finalize();
      // blf.Print(cout);

      blf.Mult(f3_nd,tmp_rtp); g3_rtp = 0.0;
      CG(m_rtp, tmp_rtp, g3_rtp, 0, 200, 1e-12, 0.0);

      cout << setw(w) << setiosflags(std::ios::left)
	   << "Error in Cross Product:  "
	   << g3_rtp.ComputeL2Error(VcrossF3_coef)
	   << endl;
    }
  }
}

void V2(const Vector & x, Vector & v)
{
  v.SetSize(2);
  v[0] = 2.234 * x[0] + 1.357 * x[1];
  v[1] = 4.572 * x[0] + 3.321 * x[1];
}

void V3(const Vector & x, Vector & v)
{
  v.SetSize(3);
  v[0] = 4.234 * x[0] + 3.357 * x[1] + 1.572 * x[2];
  v[1] = 4.537 * x[0] + 1.321 * x[1] + 2.234 * x[2];
  v[2] = 1.572 * x[0] + 2.321 * x[1] + 3.234 * x[2];
}

double f1(const Vector & x) { return 2.345 * x[0]; }
double Grad_f1(const Vector & x) { return 2.345; }

double f2(const Vector & x) { return 2.345 * x[0] + 3.579 * x[1]; }
void Grad_f2(const Vector & x, Vector & df)
{
  df.SetSize(2);
  df[0] = 2.345;
  df[1] = 3.579;
}

double f3(const Vector & x)
{ return 2.345 * x[0] + 3.579 * x[1] + 4.680 * x[2]; }
void Grad_f3(const Vector & x, Vector & df)
{
  df.SetSize(3);
  df[0] = 2.345;
  df[1] = 3.579;
  df[2] = 4.680;
}

void F2(const Vector & x, Vector & v)
{
  v.SetSize(2);
  v[0] = 1.234 * x[0] - 2.357 * x[1];
  v[1] = 3.572 * x[0] + 4.321 * x[1];
}
double Div_F2(const Vector & x)
{ return 1.234 + 4.321; }
double Curl_F2(const Vector & x)
{ return 3.572 + 2.357; }

void Vf2(const Vector & x, Vector & vf)
{
  V2(x, vf);
  vf *= f2(x);
}

double VdotF2(const Vector & x)
{
  Vector v;
  Vector f;
  V2(x, v);
  F2(x, f);
  return v * f;
}

double VcrossF2(const Vector & x)
{
  Vector v; V2(x, v);
  Vector f; F2(x, f);
  return v(0) * f(1) - v(1) * f(0);
}

void F3(const Vector & x, Vector & v)
{
  v.SetSize(3);
  v[0] =  1.234 * x[0] - 2.357 * x[1] + 3.572 * x[2];
  v[1] =  2.537 * x[0] + 4.321 * x[1] - 1.234 * x[2];
  v[2] = -2.572 * x[0] + 1.321 * x[1] + 3.234 * x[2];
}
double Div_F3(const Vector & x)
{ return 1.234 + 4.321 + 3.234; }

void Curl_F3(const Vector & x, Vector & df)
{
  df.SetSize(3);
  df[0] = 1.321 + 1.234;
  df[1] = 3.572 + 2.572;
  df[2] = 2.537 + 2.357;
}

void Vf3(const Vector & x, Vector & vf)
{
  V3(x, vf);
  vf *= f3(x);
}

double VdotF3(const Vector & x)
{
  Vector v; V3(x, v);
  Vector f; F3(x, f);
  return v * f;
}

void VcrossF3(const Vector & x, Vector & vf)
{
  Vector v; V3(x, v);
  Vector f; F3(x, f);
  vf.SetSize(3);
  vf(0) = v(1) * f(2) - v(2) * f(1);
  vf(1) = v(2) * f(0) - v(0) * f(2);
  vf(2) = v(0) * f(1) - v(1) * f(0);
}
