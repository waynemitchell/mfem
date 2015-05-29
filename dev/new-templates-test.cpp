
#include "mfem.hpp"
#include "new-templates.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

template <Geometry::Type GEOM, int P, int IR_ORDER>
void Test()
{
   std::cout << "-----------------------------------------------\n"
             << "GEOMETRY = " << Geometry::Name[GEOM] << "\n"
             << "ORDER    = " << P << "\n"
             << "IR_ORDER = " << IR_ORDER << "\n"
             << "-----------------------------------------------"
             << std::endl;
   typedef H1_FiniteElement<GEOM, P> h1_fe;

   typedef TIntegrationRule<GEOM, IR_ORDER> int_rule_1;
   // typedef GenericIntegrationRule<GEOM, int_rule_1::qpts, IR_ORDER> int_rule_2;

   typedef int_rule_1 int_rule;
   // typedef int_rule_2 int_rule;

   typedef ShapeEvaluator<h1_fe, int_rule> evaluator;

   H1_FiniteElement_Basis::Type basis_type;
   basis_type = H1_FiniteElement_Basis::GaussLobatto;
   // basis_type = H1_FiniteElement_Basis::Positive;
   h1_fe fe(basis_type);

   evaluator ev(fe);
   evaluator ev_copy(ev);
   typename evaluator::dof_data_type dof_data;
   typename evaluator::qpt_data_type qpt_data;
   dof_data.Set(1.0);
   ev.Calc(dof_data, qpt_data);
   ev.template CalcT<false>(qpt_data, dof_data);
   typename evaluator::grad_qpt_data_type grad_qpt_data;
   dof_data.Random();
   ev.CalcGrad(dof_data, grad_qpt_data);
   ev.template CalcGradT<false>(grad_qpt_data, dof_data);
   typename evaluator::asm_data_type M;
   qpt_data.Random();
   ev.Assemble(qpt_data, M);

   int nx = 1, ny = 1, nz = 1;
   switch (h1_fe::dim)
   {
      case 1:
         nx = (1200/P)*(800/P);
         break;
      case 2:
         nx = 1200/P;
         ny = 800/P;
         break;
      case 3:
         nx = 200/P;
         ny = 120/P;
         nz = 80/P;
   }
   int generate_edges = 1;
   Mesh *mesh;
   switch (GEOM)
   {
      case Geometry::SEGMENT:
         mesh = new Mesh(nx);
         break;
      case Geometry::TRIANGLE:
         mesh = new Mesh(nx, ny, Element::TRIANGLE, generate_edges);
         break;
      case Geometry::SQUARE:
         mesh = new Mesh(nx, ny, Element::QUADRILATERAL, generate_edges);
         break;
      case Geometry::TETRAHEDRON:
         mesh =  new Mesh(nx, ny, nz, Element::TETRAHEDRON, generate_edges);
         break;
      case Geometry::CUBE:
         mesh =  new Mesh(nx, ny, nz, Element::HEXAHEDRON, generate_edges);
         break;
   }
   std::cout << "Number of mesh elements = " << mesh->GetNE() << std::endl;
   H1_FECollection fec(P, h1_fe::dim);
   FiniteElementSpace fes(mesh, &fec);
   std::cout << "Number of dofs          = " << fes.GetNDofs() << std::endl;
   GridFunction x(&fes);
   x.Randomize();

   const int mesh_order = 1;
   H1_FECollection mesh_fec(mesh_order, h1_fe::dim);
   FiniteElementSpace mesh_fes(mesh, &mesh_fec, h1_fe::dim);
   mesh->SetNodalFESpace(&mesh_fes);

   const int scal_size = fes.GetNDofs();
   IndexVectorizer iv1(Ordering::byNODES, 1, scal_size);
   TIndexVectorizer_Ord<Ordering::byVDIM> iv2(1, scal_size);
   TIndexVectorizer<Ordering::byVDIM, 1> iv3(scal_size);

   typename h1_fe::dof_data_type vdof_data[1];

   if (0)
   {
      // Just test if it compliles
      // (execution will fail since fes is not a DG space)
      DG_ElementDofOperator<h1_fe> el_dof_1(fe, fes);
      el_dof_1.SetElement(0);
      el_dof_1.Extract(x, dof_data);
      el_dof_1.Assemble(dof_data, x);

      el_dof_1.VectorExtract(iv1, x, vdof_data);
      el_dof_1.VectorAssemble(iv1, vdof_data, x);

      el_dof_1.VectorExtract(iv2, x, vdof_data);
      el_dof_1.VectorAssemble(iv2, vdof_data, x);

      el_dof_1.VectorExtract(iv3, x, vdof_data);
      el_dof_1.VectorAssemble(iv3, vdof_data, x);
   }

   Table_ElementDofOperator<h1_fe> el_dof_2(fe, fes);
   el_dof_2.SetElement(0);
   el_dof_2.Extract(x, dof_data);
   el_dof_2.Assemble(dof_data, x);

   el_dof_2.VectorExtract(iv1, x, vdof_data);
   el_dof_2.VectorAssemble(iv1, vdof_data, x);

   el_dof_2.VectorExtract(iv2, x, vdof_data);
   el_dof_2.VectorAssemble(iv2, vdof_data, x);

   el_dof_2.VectorExtract(iv3, x, vdof_data);
   el_dof_2.VectorAssemble(iv3, vdof_data, x);

   typedef Table_ElementDofOperator<h1_fe> space_ElemDof;

   typedef H1_FiniteElement<GEOM, mesh_order> mesh_fe;
   typedef Table_ElementDofOperator<mesh_fe> mesh_ElemDof;
   typedef TIndexVectorizer<Ordering::byNODES, mesh_fe::dim> mesh_Vectorizer;

   typedef TMassAssembler<mesh_fe, mesh_ElemDof, mesh_Vectorizer,
           h1_fe, space_ElemDof, int_rule> mass_assembler_type;

   mass_assembler_type mass_oper(fes);
   Vector r(mass_oper.Height());
   r = 0.0; // touch 'r' to make sure the memory is actually allocated

   x.Randomize();

   tic_toc.Clear();
   tic_toc.Start();
   mass_oper.Mult(x, r);
   tic_toc.Stop();
   std::cout << "Unassembled Mult() time          = "
             << tic_toc.RealTime() << " sec" << std::endl;

   {
      BilinearForm mass_form(&fes);
      MassIntegrator *mass_int = new MassIntegrator;
      mass_int->SetIntRule(&int_rule::GetIntRule());
      mass_form.AddDomainIntegrator(mass_int);

      tic_toc.Clear();
      tic_toc.Start();
      mass_form.Assemble();
      mass_form.Finalize();
      tic_toc.Stop();
      std::cout << "CSR Assemble() + Finalize() time = "
                << tic_toc.RealTime() << " sec" << std::endl;

      Vector r2(mass_form.Height());
      r2 = 0.0; // touch 'r2' to make sure the memory is actually allocated

      tic_toc.Clear();
      tic_toc.Start();
      mass_form.Mult(x, r2);
      tic_toc.Stop();
      std::cout << "CSR Mult() time                  = "
                << tic_toc.RealTime() << " sec" << std::endl;

      r -= r2;
   }
   std::cout << "\n max residual norm = " << r.Normlinf() << '\n' << std::endl;

   delete mesh;
}

int main()
{
   static const int p = 2;
   static const int ir_order = 7;

   std::srand(std::time(0));

   Test<Geometry::SEGMENT,     p, ir_order>();
   Test<Geometry::TRIANGLE,    p, ir_order>();
   Test<Geometry::SQUARE,      p, ir_order>();
   Test<Geometry::TETRAHEDRON, p, ir_order>();
   Test<Geometry::CUBE,        p, ir_order>();


   TMatrix<5, 4> matrix_5x4;
   TCellData<1, 6> data_6;
   TCellData<2, 5> data_5x5;
   TCellData<3, 4> data_4x4x4;

   matrix_5x4(4,3) = 0.0;
   data_6[3] = 0.0;
   data_6.data[4] = 0.0;
   data_5x5(4,4) = 0.0;
   data_5x5.data[15] = 0.0;
   data_4x4x4(1,1,1) = 0.0;
   data_4x4x4.data[63] = 0.0;

   TMatrix<2,3> mA;
   TMatrix<3,4> mB;
   TMatrix<2,4> mC;
   TMatrix<4,2> mD;

   mA.Random();
   mB.Random();
   Mult_AB<false>(mA, mB, mC);

   Mult_AtB<false>(mA, mC, mB);

   Mult_ABt<false>(mC, mB, mA);

   Mult_AtBt<false>(mB, mA, mD);

   TVector<6> v;
   v.Set(mA);
   mA.Set(v);

   return 0;
}
