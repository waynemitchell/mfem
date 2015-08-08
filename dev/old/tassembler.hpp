
#include "tmatrix.hpp"
#include "tbasis.hpp"
#include "tinteg.hpp"
#include "mfem.hpp"

namespace mfem {

template <class Integ, class Basis, bool isDG>
class TAssembler : public Operator
{
protected:
   Mesh *mesh;
   FiniteElementSpace *fes;

   Basis basis;
   Integ integrator;
   double qpt_weights[Basis::qpts_1d]; // 1D quadrature weights
   Table elem_dof; // dofs in tensor product local ordering

   typename Integ::assembled_type *assembled_data;

   inline void ReorderBasisToIntegrator_Mesh(
      const int qpt_bi, const double *x_loc_qpt, const double *J_loc_qpt,
      double *integ) const;
   inline void ReorderBasisToIntegrator_Field(
      const int qpt_bi, const double *u_loc_qpt, const double *d_loc_qpt,
      double *integ) const;
   inline void ReorderBasisToIntegrator(
      const int qpt_bi, const double *x_loc_qpt, const double *J_loc_qpt,
      const double *u_loc_qpt, const double *d_loc_qpt, double *integ) const;
   inline void ReorderIntegratorToBasis(
      const int qpt_bi, const double *integ,
      double *u_loc_qpt, double *d_loc_qpt) const;

   inline void ApplyWeights(double *loc_qpt) const;

   void MultAssembled(const Vector &x, Vector &y) const;
   void MultNotAssembled(const Vector &x, Vector &y) const;

public:
   bool use_assembled_data;

   TAssembler(FiniteElementSpace *_fes);

   void Assemble();

   virtual void Mult(const Vector &x, Vector &y) const;

   const Table &GetElementToDofTable() const { return elem_dof; }

   void PrintIntegrator();
   void PrintBasis();

   virtual ~TAssembler() { delete [] assembled_data; }
};


template <class Integ, class Basis, bool isDG>
TAssembler<Integ,Basis,isDG>::TAssembler(FiniteElementSpace *_fes)
{
   // Assuming that 'Integ' uses 2 input fileds and 1 output:
   // The first input is the mesh;
   // The second input and the output are scalar.

   // Assuming 'mesh->Nodes' use the vector version of 'fes' with 'byNODES'
   // vector ordering (the default).

   mesh = _fes->GetMesh();
   fes = _fes;
   height = width = fes->GetVSize();

   if (Integ::num_inputs != 2)
      mfem_error("TAssembler<>::TAssembler: Integ::num_inputs != 2");
   if (Integ::num_outputs != 1)
      mfem_error("TAssembler<>::TAssembler: Integ::num_outputs != 1");
   if (Integ::dim != Basis::dim)
      mfem_error("TAssembler<>::TAssembler: Integ::dim != Basis::dim");
   if (mesh->Dimension() != Basis::dim)
      mfem_error("TAssembler<>::TAssembler: mesh->Dimension() != Basis::dim");
   // if (Basis::num_entries % Basis::dim != 0)
   //    mfem_error("TAssembler<>::TAssembler: "
   //               "Basis::num_entries % Basis::dim != 0");
   if ((Basis::total_qpts * Basis::num_entries) % Integ::num_qdr_points != 0)
      mfem_error("TAssembler<>::TAssembler:\n  "
                 "(Basis::total_qpts * Basis::num_entries) % "
                 "Integ::num_qdr_points != 0");
   if (mesh->GetNE() % Basis::num_entries != 0)
      mfem_error("TAssembler<>::TAssembler: "
                 "mesh->GetNE() % Basis::num_entries != 0");
   if (mesh->GetNodes() == NULL)
      mfem_error("TAssembler<>::TAssembler: the mesh has no Nodes");
   if (mesh->MeshGenerator() & 1)
      mfem_error("TAssembler<>::TAssembler: the mesh has triangles/tets");
   if (fes->GetVDim() != 1)
      mfem_error("TAssembler<>::TAssembler: fes->GetVDim() != 1");

   // Initialize 'qpt_weights'
   const int q_order = 2 * Basis::qpts_1d - 1;
   const IntegrationRule &ir = IntRules.Get(Geometry::SEGMENT, q_order);
   for (int i = 0; i < Basis::qpts_1d; i++)
      qpt_weights[i] = ir.IntPoint(i).weight;

   // Initialize 'basis'
   basis.SetInterp(*fes->GetFE(0), ir);

   if (!isDG)
   {
      // Set up 'elem_dof' with tensor product local dof ordering
      const FiniteElement *fe = fes->GetFE(0);
      const Array<int> *dof_map = NULL;

#if 0
      // implementation using a class ProductFiniteElement (TODO) that serves as
      // a base class for all tensor product FE and adds a method GetDofMap:
      //    const Array<int> &GetDofMap() const;
      const ProductFiniteElement *prod_fe =
         dynamic_cast<const ProductFiniteElement *>(fe);
      if (h1_quad_fe)
         dof_map = &prod_fe->GetDofMap();
#else
      // implementation using a method GetDofMap implemented for classes
      // H1_QuadrilateralElement (2D) and H1_HexahedronElement (3D)
      if (Basis::dim == 2)
      {
         const H1_QuadrilateralElement *h1_quad_fe =
            dynamic_cast<const H1_QuadrilateralElement *>(fe);
         if (h1_quad_fe)
            dof_map = &h1_quad_fe->GetDofMap();
      }
      else if (Basis::dim == 3)
      {
         const H1_HexahedronElement *h1_hex_fe =
            dynamic_cast<const H1_HexahedronElement *>(fe);
         if (h1_hex_fe)
            dof_map = &h1_hex_fe->GetDofMap();
      }
#endif

      if (dof_map == NULL)
         mfem_error("TAssembler<>::TAssembler: not a product FE");

      fes->BuildElementToDofTable();
      const Table &orig_elem_dof = fes->GetElementToDofTable();
      int dof = orig_elem_dof.RowSize(0);
      elem_dof.SetSize(orig_elem_dof.Size(), dof);
      for (int i = 0; i < orig_elem_dof.Size(); i++)
      {
         const int *dofs_in = orig_elem_dof.GetRow(i);
         int *dofs_tp = elem_dof.GetRow(i);
         for (int j = 0; j < dof; j++)
            dofs_tp[j] = dofs_in[(*dof_map)[j]]; // dof_map[tp_idx]==def_idx
      }
   }

   assembled_data = NULL;
   use_assembled_data = true;
}

template <class Integ, class Basis, bool isDG>
inline void TAssembler<Integ,Basis,isDG>::ReorderBasisToIntegrator_Mesh(
   const int qpt_bi, const double *x_loc_qpt, const double *J_loc_qpt,
   double *integ) const
{
   const int nc0 = Integ::num_input_comp[0];

   for (int qi = 0, off = 0; qi < Integ::num_qdr_points; qi++)
   {
      // values for input 0 (mesh)
      if (Integ::input_data[0] == Integ::VALUE ||
          Integ::input_data[0] == Integ::VALUE_AND_GRADIENT)
      {
         // x_loc_qpt is
         // Basis::total_qpts * Basis::num_entries * nc0
         for (int c = 0; c < nc0; c++)
         {
            integ[off + c] =
               x_loc_qpt[qi + qpt_bi * Integ::num_qdr_points +
                         c * (Basis::total_qpts * Basis::num_entries)];
         }
         off += nc0;
      }
      // gradient for input 0 (mesh)
      if (Integ::input_data[0] == Integ::GRADIENT ||
          Integ::input_data[0] == Integ::VALUE_AND_GRADIENT)
      {
         // J_loc_qpt is
         // Basis::total_qpts * Basis::num_entries * Basis::dim * nc0
         for (int c = 0; c < nc0; c++)
            for (int d = 0; d < Basis::dim; d++)
            {
               integ[off + c + d * nc0] =
                  J_loc_qpt[qi + qpt_bi * Integ::num_qdr_points +
                            (d + c * Basis::dim) *
                            (Basis::total_qpts * Basis::num_entries)];
            }
         off += nc0 * Basis::dim;
      }
   }
}

template <class Integ, class Basis, bool isDG>
inline void TAssembler<Integ,Basis,isDG>::ReorderBasisToIntegrator_Field(
   const int qpt_bi, const double *u_loc_qpt, const double *d_loc_qpt,
   double *integ) const
{
   for (int qi = 0, off = 0; qi < Integ::num_qdr_points; qi++)
   {
      // values for input 1 ('u')
      if (Integ::input_data[1] == Integ::VALUE ||
          Integ::input_data[1] == Integ::VALUE_AND_GRADIENT)
      {
         // u_loc_qpt is
         // Basis::total_qpts * Basis::num_entries
         integ[off] = u_loc_qpt[qi + qpt_bi * Integ::num_qdr_points];
         off++;
      }
      // gradient for input 1 ('u')
      if (Integ::input_data[1] == Integ::GRADIENT ||
          Integ::input_data[1] == Integ::VALUE_AND_GRADIENT)
      {
         // d_loc_qpt is
         // Basis::total_qpts * Basis::num_entries * Basis::dim
         for (int d = 0; d < Basis::dim; d++)
         {
            integ[off + d] =
               d_loc_qpt[qi + qpt_bi * Integ::num_qdr_points +
                         d * (Basis::total_qpts * Basis::num_entries)];
         }
         off += Basis::dim;
      }
   }
}

template <class Integ, class Basis, bool isDG>
inline void TAssembler<Integ,Basis,isDG>::ReorderBasisToIntegrator(
   const int qpt_bi, const double *x_loc_qpt, const double *J_loc_qpt,
   const double *u_loc_qpt, const double *d_loc_qpt, double *integ) const
{
   const int nc0 = Integ::num_input_comp[0];

   for (int qi = 0, off = 0; qi < Integ::num_qdr_points; qi++)
   {
      // values for input 0 (mesh)
      if (Integ::input_data[0] == Integ::VALUE ||
          Integ::input_data[0] == Integ::VALUE_AND_GRADIENT)
      {
         // x_loc_qpt is
         // Basis::total_qpts * Basis::num_entries * nc0
         for (int c = 0; c < nc0; c++)
         {
            integ[off + c] =
               x_loc_qpt[qi + qpt_bi * Integ::num_qdr_points +
                         c * (Basis::total_qpts * Basis::num_entries)];
         }
         off += nc0;
      }
      // gradient for input 0 (mesh)
      if (Integ::input_data[0] == Integ::GRADIENT ||
          Integ::input_data[0] == Integ::VALUE_AND_GRADIENT)
      {
         // J_loc_qpt is
         // Basis::total_qpts * Basis::num_entries * Basis::dim * nc0
         for (int c = 0; c < nc0; c++)
            for (int d = 0; d < Basis::dim; d++)
            {
               integ[off + c + d * nc0] =
                  J_loc_qpt[qi + qpt_bi * Integ::num_qdr_points +
                            (d + c * Basis::dim) *
                            (Basis::total_qpts * Basis::num_entries)];
            }
         off += nc0 * Basis::dim;
      }
      // values for input 1 ('u')
      if (Integ::input_data[1] == Integ::VALUE ||
          Integ::input_data[1] == Integ::VALUE_AND_GRADIENT)
      {
         // u_loc_qpt is
         // Basis::total_qpts * Basis::num_entries
         integ[off] =
            u_loc_qpt[qi + qpt_bi * Integ::num_qdr_points];
         off++;
      }
      // gradient for input 1 ('u')
      if (Integ::input_data[1] == Integ::GRADIENT ||
          Integ::input_data[1] == Integ::VALUE_AND_GRADIENT)
      {
         // d_loc_qpt is
         // Basis::total_qpts * Basis::num_entries * Basis::dim
         for (int d = 0; d < Basis::dim; d++)
         {
            integ[off + d] =
               d_loc_qpt[qi + qpt_bi * Integ::num_qdr_points +
                         d * (Basis::total_qpts * Basis::num_entries)];
         }
         off += Basis::dim;
      }
   }
}

template <class Integ, class Basis, bool isDG>
inline void TAssembler<Integ,Basis,isDG>::ReorderIntegratorToBasis(
   const int qpt_bi, const double *integ,
   double *u_loc_qpt, double *d_loc_qpt) const
{
   for (int qi = 0, off = 0; qi < Integ::num_qdr_points; qi++)
   {
      // values for output 0 ('v')
      if (Integ::output_data[0] == Integ::VALUE ||
          Integ::output_data[0] == Integ::VALUE_AND_GRADIENT)
      {
         // u_loc_qpt is
         // Basis::total_qpts * Basis::num_entries
         u_loc_qpt[qi + qpt_bi * Integ::num_qdr_points] = integ[off];
         off++;
      }
      // gradient for output 1 ('v')
      if (Integ::output_data[0] == Integ::GRADIENT ||
          Integ::output_data[0] == Integ::VALUE_AND_GRADIENT)
      {
         // d_loc_qpt is
         // Basis::total_qpts * Basis::num_entries * Basis::dim
         for (int d = 0; d < Basis::dim; d++)
         {
            d_loc_qpt[qi + qpt_bi * Integ::num_qdr_points +
                      d * (Basis::total_qpts * Basis::num_entries)] =
               integ[off + d];
         }
         off += Basis::dim;
      }
   }
}

template <class Integ, class Basis, bool isDG>
inline void TAssembler<Integ,Basis,isDG>::ApplyWeights(double *loc_qpt) const
{
   if (Basis::dim == 2)
   {
      for (int j = 0; j < Basis::qpts_1d; j++)
         for (int i = 0; i < Basis::qpts_1d; i++)
            loc_qpt[i + j * Basis::qpts_1d] *= qpt_weights[i] * qpt_weights[j];
   }
   else if (Basis::dim == 3)
   {
      // TODO
   }
}

template <class Integ, class Basis, bool isDG>
void TAssembler<Integ,Basis,isDG>::Assemble()
{
   int num_elem_blocks = mesh->GetNE() / Basis::num_entries;
   const double *mesh_node_data = mesh->GetNodes()->GetData();
   int comp_size = Height();
   const int *el_dof;

   const int num_integ_calls =
      (Basis::total_qpts * Basis::num_entries) / Integ::num_qdr_points;
   const int nc0 = Integ::num_input_comp[0];

   // quadrature point values and gradients
   double x_loc_qpt[Basis::total_qpts * Basis::num_entries * nc0];
   double J_loc_qpt[Basis::total_qpts * Basis::num_entries * Basis::dim * nc0];

   typename Integ::a_input_type integ_a_data_in[Integ::num_qdr_points];

   if (assembled_data == NULL)
      assembled_data =
         new typename Integ::assembled_type[Basis::total_qpts * mesh->GetNE()];
   typename Integ::assembled_type *asm_data;

   for (int el_bi = 0; el_bi < num_elem_blocks; el_bi++)
   {
      el_dof = elem_dof.GetRow(el_bi * Basis::num_entries);

      // Convert dof->qpt values for input 0 (mesh)
      if (Integ::input_data[0] == Integ::VALUE ||
          Integ::input_data[0] == Integ::VALUE_AND_GRADIENT)
      {
         for (int c = 0; c < nc0; c++)
            basis.Calc(el_dof, &mesh_node_data[c * comp_size],
                       &x_loc_qpt[c * (Basis::total_qpts *
                                       Basis::num_entries)]);
      }
      // Convert dof->qpt grad for input 0 (mesh)
      if (Integ::input_data[0] == Integ::GRADIENT ||
          Integ::input_data[0] == Integ::VALUE_AND_GRADIENT)
      {
         for (int c = 0; c < nc0; c++)
            basis.GradCalc(el_dof, &mesh_node_data[c * comp_size],
                           &J_loc_qpt[c * (Basis::total_qpts *
                                           Basis::num_entries *
                                           Basis::dim)]);
      }

      asm_data = &assembled_data[el_bi * (Basis::total_qpts *
                                          Basis::num_entries)];
      for (int qpt_bi = 0; qpt_bi < num_integ_calls; qpt_bi++)
      {
         // Reorder the 'basis' data for the 'integrator'
         // Can we eliminate this reordering?
         ReorderBasisToIntegrator_Mesh(qpt_bi, x_loc_qpt, J_loc_qpt,
                                       (double *)(&integ_a_data_in[0]));

         // Assemble
         integrator.Assemble(integ_a_data_in,
                             &asm_data[qpt_bi * Integ::num_qdr_points]);
      }
   }
}

template <class Integ, class Basis, bool isDG>
void TAssembler<Integ,Basis,isDG>::MultAssembled(
   const Vector &u, Vector &v) const
{
   int num_elem_blocks = mesh->GetNE() / Basis::num_entries;
   const double *u_data = u.GetData();
   double *v_data = v.GetData();
   const int *el_dof;

   const int num_integ_calls =
      (Basis::total_qpts * Basis::num_entries) / Integ::num_qdr_points;

   // quadrature point values and gradients
   double u_loc_qpt[Basis::total_qpts * Basis::num_entries];
   double d_loc_qpt[Basis::total_qpts * Basis::num_entries * Basis::dim];

   typename Integ::b_input_type integ_b_data_in[Integ::num_qdr_points];
   typename Integ::output_type integ_data_out[Integ::num_qdr_points];

   const typename Integ::assembled_type *asm_data;

   v = 0.0;

   for (int el_bi = 0; el_bi < num_elem_blocks; el_bi++)
   {
      if (isDG)
      {
         // TODO
      }
      else // isDG == false
      {
         el_dof = elem_dof.GetRow(el_bi * Basis::num_entries);

         // Convert dof->qpt values for input 1 ('u')
         if (Integ::input_data[1] == Integ::VALUE ||
             Integ::input_data[1] == Integ::VALUE_AND_GRADIENT)
         {
            basis.Calc(el_dof, u_data, u_loc_qpt);
         }
         // Convert dof->qpt grad for input 1 ('u')
         if (Integ::input_data[1] == Integ::GRADIENT ||
             Integ::input_data[1] == Integ::VALUE_AND_GRADIENT)
         {
            basis.GradCalc(el_dof, u_data, d_loc_qpt);
         }
      }

      asm_data = &assembled_data[el_bi * (Basis::total_qpts *
                                          Basis::num_entries)];
      for (int qpt_bi = 0; qpt_bi < num_integ_calls; qpt_bi++)
      {
         // Reorder the 'basis' data for the 'integrator'
         // Can we eliminate this reordering?
         ReorderBasisToIntegrator_Field(qpt_bi, u_loc_qpt, d_loc_qpt,
                                        (double *)(&integ_b_data_in[0]));

         // Apply the 'integrator'
         integrator.Calc(integ_b_data_in,
                         &asm_data[qpt_bi * Integ::num_qdr_points],
                         integ_data_out);

         // Reorder the 'integrator' data for the 'basis'
         // Can we eliminate this reordering?
         ReorderIntegratorToBasis(qpt_bi, (const double *)(&integ_data_out[0]),
                                  u_loc_qpt, d_loc_qpt);
      }

      // Multiply by the quadrature point weights
      if (Integ::output_data[0] == Integ::VALUE ||
          Integ::output_data[0] == Integ::VALUE_AND_GRADIENT)
      {
         for (int ei = 0; ei < Basis::num_entries; ei++)
            ApplyWeights(&u_loc_qpt[ei * Basis::total_qpts]);
      }
      if (Integ::output_data[0] == Integ::GRADIENT ||
          Integ::output_data[0] == Integ::VALUE_AND_GRADIENT)
      {
         for (int ei = 0; ei < (Basis::num_entries * Basis::dim); ei++)
            ApplyWeights(&d_loc_qpt[ei * Basis::total_qpts]);
      }

      if (isDG)
      {
         // TODO
      }
      else
      {
         // Add qpt values to dofs from output 0 ('v')
         if (Integ::output_data[0] == Integ::VALUE ||
             Integ::output_data[0] == Integ::VALUE_AND_GRADIENT)
         {
            basis.template CalcT<true>(el_dof, u_loc_qpt, v_data);
         }
         // Add qpt gradients to dofs from output 0 ('v')
         if (Integ::output_data[0] == Integ::GRADIENT ||
             Integ::output_data[0] == Integ::VALUE_AND_GRADIENT)
         {
            basis.template GradCalcT<true>(el_dof, d_loc_qpt, v_data);
         }
      }
   }
}

template <class Integ, class Basis, bool isDG>
void TAssembler<Integ,Basis,isDG>::MultNotAssembled(
   const Vector &u, Vector &v) const
{
   int num_elem_blocks = mesh->GetNE() / Basis::num_entries;
   const double *mesh_node_data = mesh->GetNodes()->GetData();
   const double *u_data = u.GetData();
   double *v_data = v.GetData();
   int comp_size = Height();
   const int *el_dof;

   const int num_integ_calls =
      (Basis::total_qpts * Basis::num_entries) / Integ::num_qdr_points;
   const int nc0 = Integ::num_input_comp[0];

   // quadrature point values and gradients
   double x_loc_qpt[Basis::total_qpts * Basis::num_entries * nc0];
   double J_loc_qpt[Basis::total_qpts * Basis::num_entries * Basis::dim * nc0];
   double u_loc_qpt[Basis::total_qpts * Basis::num_entries];
   double d_loc_qpt[Basis::total_qpts * Basis::num_entries * Basis::dim];

   typename Integ::input_type integ_data_in[Integ::num_qdr_points];
   typename Integ::output_type integ_data_out[Integ::num_qdr_points];

   v = 0.0;

   for (int el_bi = 0; el_bi < num_elem_blocks; el_bi++)
   {
      if (isDG)
      {
         // TODO
      }
      else // isDG == false
      {
         el_dof = elem_dof.GetRow(el_bi * Basis::num_entries);

         // Convert dof->qpt values for input 0 (mesh)
         if (Integ::input_data[0] == Integ::VALUE ||
             Integ::input_data[0] == Integ::VALUE_AND_GRADIENT)
         {
            for (int c = 0; c < nc0; c++)
               basis.Calc(el_dof, &mesh_node_data[c * comp_size],
                          &x_loc_qpt[c * (Basis::total_qpts *
                                          Basis::num_entries)]);
         }
         // Convert dof->qpt grad for input 0 (mesh)
         if (Integ::input_data[0] == Integ::GRADIENT ||
             Integ::input_data[0] == Integ::VALUE_AND_GRADIENT)
         {
            for (int c = 0; c < nc0; c++)
               basis.GradCalc(el_dof, &mesh_node_data[c * comp_size],
                              &J_loc_qpt[c * (Basis::total_qpts *
                                              Basis::num_entries *
                                              Basis::dim)]);
         }

         // Convert dof->qpt values for input 1 ('u')
         if (Integ::input_data[1] == Integ::VALUE ||
             Integ::input_data[1] == Integ::VALUE_AND_GRADIENT)
         {
            basis.Calc(el_dof, u_data, u_loc_qpt);
         }
         // Convert dof->qpt grad for input 1 ('u')
         if (Integ::input_data[1] == Integ::GRADIENT ||
             Integ::input_data[1] == Integ::VALUE_AND_GRADIENT)
         {
            basis.GradCalc(el_dof, u_data, d_loc_qpt);
         }
      }

      for (int qpt_bi = 0; qpt_bi < num_integ_calls; qpt_bi++)
      {
         // Reorder the 'basis' data for the 'integrator'
         // Can we eliminate this reordering?
         ReorderBasisToIntegrator(qpt_bi, x_loc_qpt, J_loc_qpt,
                                  u_loc_qpt, d_loc_qpt,
                                  (double *)(&integ_data_in[0]));

         // Apply the 'integrator'
         integrator.Calc(integ_data_in, integ_data_out);

         // Reorder the 'integrator' data for the 'basis'
         // Can we eliminate this reordering?
         ReorderIntegratorToBasis(qpt_bi, (const double *)(&integ_data_out[0]),
                                  u_loc_qpt, d_loc_qpt);
      }

      // Multiply by the quadrature point weights
      if (Integ::output_data[0] == Integ::VALUE ||
          Integ::output_data[0] == Integ::VALUE_AND_GRADIENT)
      {
         for (int ei = 0; ei < Basis::num_entries; ei++)
            ApplyWeights(&u_loc_qpt[ei * Basis::total_qpts]);
      }
      if (Integ::output_data[0] == Integ::GRADIENT ||
          Integ::output_data[0] == Integ::VALUE_AND_GRADIENT)
      {
         for (int ei = 0; ei < (Basis::num_entries * Basis::dim); ei++)
            ApplyWeights(&d_loc_qpt[ei * Basis::total_qpts]);
      }

      if (isDG)
      {
         // TODO
      }
      else
      {
         // Add qpt values to dofs from output 0 ('v')
         if (Integ::output_data[0] == Integ::VALUE ||
             Integ::output_data[0] == Integ::VALUE_AND_GRADIENT)
         {
            basis.template CalcT<true>(el_dof, u_loc_qpt, v_data);
         }
         // Add qpt gradients to dofs from output 0 ('v')
         if (Integ::output_data[0] == Integ::GRADIENT ||
             Integ::output_data[0] == Integ::VALUE_AND_GRADIENT)
         {
            basis.template GradCalcT<true>(el_dof, d_loc_qpt, v_data);
         }
      }
   }
}

template <class Integ, class Basis, bool isDG>
void TAssembler<Integ,Basis,isDG>::Mult(const Vector &u, Vector &v) const
{
   if (assembled_data == NULL || use_assembled_data == false)
      MultNotAssembled(u, v);
   else
      MultAssembled(u, v);
}

template <class Integ, class Basis, bool isDG>
void TAssembler<Integ,Basis,isDG>::PrintIntegrator()
{
   typedef typename Integ::input_type input_type;
   typedef typename Integ::output_type output_type;

   Integ integ;

   const int dim   = Integ::dim;
   const int n_qdr = Integ::num_qdr_points;
   const int n_in  = Integ::num_inputs;
   const int n_out = Integ::num_outputs;

   input_type data_in[n_qdr];
   output_type data_out[n_qdr];

   std::cout << "Integrator::" << std::endl;
   std::cout << "dim                 = " << Integ::dim << std::endl;
   std::cout << "num_qdr_points      = " << Integ::num_qdr_points << std::endl;
   std::cout << "num_inputs          = " << Integ::num_inputs << std::endl;
   std::cout << "num_outputs         = " << Integ::num_outputs << std::endl;
   std::cout << "num_input_comp[]    = {";
   for (int i = 0; i < n_in; i++)
      std::cout << ' ' << Integ::num_input_comp[i] << (i+1<n_in?',':' ');
   std::cout << '}' << std::endl;
   std::cout << "num_output_comp[]   = {";
   for (int i = 0; i < n_out; i++)
      std::cout << ' ' << Integ::num_output_comp[i] << (i+1<n_out?',':' ');
   std::cout << '}' << std::endl;

   std::cout << "input_data[]        = {";
   for (int i = 0; i < n_in; i++)
      std::cout << ' ' << Integ::field_name[Integ::input_data[i]]
                << (i+1<n_in?',':' ');
   std::cout << '}' << std::endl;
   std::cout << "output_data[]       = {";
   for (int i = 0; i < n_out; i++)
      std::cout << ' ' << Integ::field_name[Integ::output_data[i]]
                << (i+1<n_out?',':' ');
   std::cout << '}' << std::endl;
   std::cout << "sizeof(input_type)  = " << sizeof(input_type) << std::endl;
   std::cout << "sizeof(output_type) = " << sizeof(output_type) << std::endl;

   std::cout << "Calling Calc() ..." << std::flush;
   integ.Calc(data_in, data_out);
   std::cout << " done." << std::endl;
}

template <class Integ, class Basis, bool isDG>
void TAssembler<Integ,Basis,isDG>::PrintBasis()
{
   std::cout << "Basis::" << std::endl;
   std::cout << "dim         = " << Basis::dim << std::endl;
   std::cout << "degree      = " << Basis::degree << std::endl;
   std::cout << "dofs_1d     = " << Basis::dofs_1d << std::endl;
   std::cout << "qpts_1d     = " << Basis::qpts_1d << std::endl;
   std::cout << "num_entries = " << Basis::num_entries << std::endl;
   std::cout << "total_dofs  = " << Basis::total_dofs << std::endl;
   std::cout << "total_qpts  = " << Basis::total_qpts << std::endl;
}

}
