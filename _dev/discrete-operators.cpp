
#include <fstream>
#include "mfem.hpp"

int main(int argc, char *argv[])
{
   Mesh *mesh;

   if (argc == 1)
   {
      cout << "\nUsage: " << argv[0] << " <mesh_file>\n" << endl;
      return 1;
   }

   // 1. Read the mesh from the given mesh file.
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
      cerr << "\nNot a 3D mesh!\n" << endl;
      delete mesh;
      return 3;
   }

   // 2. Refine the mesh to increase the resolution.
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

      // for ND(p) spaces on tets , p > 1
      if (mesh->MeshGenerator() & 1)
      {
         cout << "Reorienting tetrahedral mesh ... " << flush;
         mesh->ReorientTetMesh();
         cout << "done.\n" << endl;
      }

      mesh->PrintCharacteristics();
   }

   // 3. Define a finite element space on the mesh.
   int p;
   cout << "enter space order >= 1 --> " << flush;
   cin >> p;

   int prec = 18;

   cout << endl;
   FiniteElementCollection *h1_fec = new H1_FECollection(p, dim);
   FiniteElementSpace *h1_fes = new FiniteElementSpace(mesh, h1_fec);

   cout << "Number of H1 dofs: " << h1_fes->GetVSize() << endl;

   cout << endl;
   FiniteElementCollection *nd_fec = new ND_FECollection(p, dim);
   FiniteElementSpace *nd_fes = new FiniteElementSpace(mesh, nd_fec);

   cout << "Number of ND dofs: " << nd_fes->GetVSize() << endl;

   // discrete gradient matrix
   cout << "\nDiscrete gradient matrix, G ... " << flush;
   DiscreteLinearOperator *grad = new DiscreteLinearOperator(h1_fes, nd_fes);
   grad->AddDomainInterpolator(new GradientInterpolator);
   grad->Assemble();
   grad->Finalize();
   SparseMatrix &G = grad->SpMat();
   {
      char grad_mat_name[] = "G_matrix.txt";
      cout << '(' << grad_mat_name << ") ... " << flush;
      ofstream grad_mat_file(grad_mat_name);
      grad_mat_file.precision(prec);
      G.Print(grad_mat_file, 1);
   }
   cout << "done." << endl;

   {
      GridFunction one(h1_fes);
      GridFunction Gx1(nd_fes);
      one = 1.0;
      G.Mult(one, Gx1);
      cout << "\n(p=" << p << ") max_{i} |(G.1)_{i}| = "
           << Gx1.Normlinf() << endl;
   }

   // G^t G
   {
      cout << "\nG^t G ... " << flush;
      SparseMatrix *Gt = Transpose(G);
      SparseMatrix *GtG = Mult(*Gt, G);
      delete Gt;

      char gtg_mat_name[] = "GtG_matrix.txt";
      cout << '(' << gtg_mat_name << ") ... " << flush;
      ofstream gtg_mat_file(gtg_mat_name);
      gtg_mat_file.precision(prec);
      GtG->Print(gtg_mat_file, 1);
      cout << "done." << endl;
      delete GtG;
   }
   delete h1_fes;

   // Pi_h
   cout << "\nPi_h ... " << flush;
   FiniteElementSpace *h1v_fes = new FiniteElementSpace(mesh, h1_fec, dim);
   DiscreteLinearOperator *ident = new DiscreteLinearOperator(h1v_fes, nd_fes);
   ident->AddDomainInterpolator(new IdentityInterpolator);
   ident->Assemble();
   Array2D<SparseMatrix *> pi_h_blocks;
   ident->GetBlocks(pi_h_blocks);
   ident->Finalize();
   SparseMatrix &pi_h = ident->SpMat();
   {
      char pi_h_mat_name[] = "Pi_h_matrix.txt";
      cout << '(' << pi_h_mat_name << ") ... " << flush;
      ofstream pi_h_mat_file(pi_h_mat_name);
      pi_h_mat_file.precision(prec);
      pi_h.Print(pi_h_mat_file, 1);
      cout << "done." << endl;
   }
   for (int i = 0; i < dim; i++)
      delete pi_h_blocks(0, i);
   delete ident;
   delete h1v_fes;

   delete h1_fec;

   cout << endl;
   FiniteElementCollection *rt_fec = new RT_FECollection(p - 1, dim);
   FiniteElementSpace *rt_fes = new FiniteElementSpace(mesh, rt_fec);

   cout << "Number of RT dofs: " << rt_fes->GetVSize() << endl;

   // discrete curl matrix
   cout << "\nDiscrete curl matrix, C ... " << flush;
   DiscreteLinearOperator *curl = new DiscreteLinearOperator(nd_fes, rt_fes);
   curl->AddDomainInterpolator(new CurlInterpolator);
   curl->Assemble();
   curl->Finalize();
   SparseMatrix &C = curl->SpMat();
   {
      char curl_mat_name[] = "C_matrix.txt";
      cout << '(' << curl_mat_name << ") ... " << flush;
      ofstream curl_mat_file(curl_mat_name);
      curl_mat_file.precision(prec);
      C.Print(curl_mat_file, 1);
   }
   cout << "done." << endl;

   // check that C.G = 0
   {
      SparseMatrix *CxG = Mult(C, G);

      cout << "\n(p=" << p << ") max_{ij} |(C.G)_{ij}| = "
           << CxG->MaxNorm() << endl;

      delete CxG;
   }
   delete grad;

   delete nd_fes;
   delete nd_fec;

   cout << endl;
   FiniteElementCollection *l2_fec = new L2_FECollection(p - 1, dim);
   FiniteElementSpace *l2_fes = new FiniteElementSpace(mesh, l2_fec);

   cout << "Number of L2 dofs: " << l2_fes->GetVSize() << endl;

   cout << "\nDiscrete div matrix, D ... " << flush;
   DiscreteLinearOperator *div = new DiscreteLinearOperator(rt_fes, l2_fes);
   div->AddDomainInterpolator(new DivergenceInterpolator);
   div->Assemble();
   div->Finalize();
   SparseMatrix &D = div->SpMat();
   {
      char div_mat_name[] = "D_matrix.txt";
      cout << '(' << div_mat_name << ") ... " << flush;
      ofstream div_mat_file(div_mat_name);
      div_mat_file.precision(prec);
      D.Print(div_mat_file, 1);
   }
   cout << "done." << endl;

   // check that D.C = 0
   {
      SparseMatrix *DxC = Mult(D, C);

      cout << "\n(p=" << p << ") max_{ij} |(D.C)_{ij}| = "
           << DxC->MaxNorm() << endl;

      delete DxC;
   }

   delete div;
   delete l2_fes;
   delete l2_fec;

   delete curl;
   delete rt_fes;
   delete rt_fec;

   delete mesh;

   return 0;
}
