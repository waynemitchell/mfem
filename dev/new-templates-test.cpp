
#include "mfem.hpp"
#include "new-templates.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

template <Geometry::Type GEOM, int P, int IR_ORDER>
void Test()
{
   typedef H1_FiniteElement<GEOM, P> h1_fe;

   typedef TIntegrationRule<GEOM, IR_ORDER> int_rule_1;
   typedef GenericIntegrationRule<GEOM, int_rule_1::qpts, IR_ORDER> int_rule_2;

   typedef int_rule_1 int_rule;
   // typedef int_rule_2 int_rule;

   typedef ShapeEvaluator<h1_fe, int_rule> evaluator;

   evaluator ev;
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
}

int main()
{
   static const int p = 1;
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
