
#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "new-templates.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   typedef H1_FiniteElement<Geometry::SEGMENT, 1> h1_segment_fe;
   typedef H1_FiniteElement<Geometry::SQUARE, 1>  h1_quad_fe;
   typedef H1_FiniteElement<Geometry::CUBE, 1>    h1_hex_fe;

   h1_segment_fe fe1d;
   h1_quad_fe    fe2d;
   h1_hex_fe     fe3d;

   typedef TIntegrationRule<Geometry::SEGMENT, 7> tp_int_rule_1d;
   typedef TIntegrationRule<Geometry::SQUARE, 7>  tp_int_rule_2d;
   typedef TIntegrationRule<Geometry::CUBE, 7>    tp_int_rule_3d;
   typedef GenericIntegrationRule<Geometry::SEGMENT, 2, 3> g_int_rule_1d;
   typedef GenericIntegrationRule<Geometry::SQUARE, 4, 3>  g_int_rule_2d;
   typedef GenericIntegrationRule<Geometry::CUBE, 8, 3>    g_int_rule_3d;

#if 0
   typedef h1_segment_fe  h1_fe;
   typedef tp_int_rule_1d tp_int_rule;
   typedef g_int_rule_1d  g_int_rule;
#elif 0
   typedef h1_quad_fe     h1_fe;
   typedef tp_int_rule_2d tp_int_rule;
   typedef g_int_rule_2d  g_int_rule;
#else
   typedef h1_hex_fe      h1_fe;
   typedef tp_int_rule_3d tp_int_rule;
   typedef g_int_rule_3d  g_int_rule;
#endif

   typedef ShapeEvaluator<h1_fe, tp_int_rule> evaluator_1;
   typedef ShapeEvaluator<h1_fe, g_int_rule>  evaluator_2;

   std::srand(std::time(0));

   evaluator_1 ev1;
   evaluator_1 ev1_copy(ev1);
   evaluator_1::dof_data_type dof_data_1;
   evaluator_1::qpt_data_type qpt_data_1;
   dof_data_1.Set(1.0);
   ev1.Calc(dof_data_1, qpt_data_1);
   ev1.CalcT<false>(qpt_data_1, dof_data_1);
   evaluator_1::grad_qpt_data_type grad_qpt_data_1;
   dof_data_1.Random();
   ev1.CalcGrad(dof_data_1, grad_qpt_data_1);
   ev1.CalcGradT<false>(grad_qpt_data_1, dof_data_1);
   evaluator_1::asm_data_type M_1;
   qpt_data_1.Random();
   ev1.Assemble(qpt_data_1, M_1);

   evaluator_2 ev2;
   evaluator_2 ev2_copy(ev2);
   evaluator_2::dof_data_type dof_data_2;
   evaluator_2::qpt_data_type qpt_data_2;
   dof_data_2.Set(1.0);
   ev2.Calc(dof_data_2, qpt_data_2);
   ev2.CalcT<false>(qpt_data_2, dof_data_2);
   evaluator_2::grad_qpt_data_type grad_qpt_data_2;
   dof_data_2.Random();
   ev2.CalcGrad(dof_data_2, grad_qpt_data_2);
   ev2.CalcGradT<false>(grad_qpt_data_2, dof_data_2);
   evaluator_2::asm_data_type M_2;
   qpt_data_2.Random();
   ev2.Assemble(qpt_data_2, M_2);

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
