
#include "tmatrix.hpp"


// Base class for template integrator classes
class TIntegrator_base
{
public:
   enum field_data { VALUE, GRADIENT, VALUE_AND_GRADIENT };
   static const char *field_name[3];

   //*** Derived classes must provide compile time information about the
   //*** integrator class to generic assembly procedures templated on the class.

   // static const int dim;
   // static const int num_qdr_points;

   //*** Description of the input and output data:
   //    Input fields are split into two disjoint categories:
   //       a) a-input - used for quadrature point "assembly"
   //       b) b-input - used for performing the "assembled" action
   //    The input fields should be ordered by category: first all a-inputs
   //    followed by all b-inputs.
   // static const int num_inputs;     // number of input fields
   // static const int num_outputs;    // number of output fields
   // static const int num_a_inputs;   // number of a-input fields

   //*** Number of components for each input and output field
   // static const int num_input_comp[num_inputs];
   // static const int num_output_comp[num_outputs];

   //*** Field data required as input by the integrator
   // static const field_data input_data[num_inputs];
   //*** Field data returned by the integrator
   // static const field_data output_data[num_outputs];

   //*** Data types associated with a quadrature point: input, output,
   //    assembled, a-input, and b-input :
   // class input_type;
   // class output_type;
   // class assembled_type;
   // class a_input_type;
   // class b_input_type;

   //*** The input and output data types must use the following data layout:

   //*** [input type] = [field data]
   //*** [field data] = [field data 1] + ... + [field data num_inputs]
   //*** [field data i] = [field value data i]_opt + [field grad data i]_opt
   //*** [field value data i] = [num_input_comp[i-1]]
   //*** [field grad data i] = [num_input_comp[i-1]] x [dim]

   //*** Action of the integrator at quadrature points
   // void Calc(const input_type data_in[], output_type data_out[]) const;
   // *** Compute the assembled data at quadrature points
   // void Assemble(const a_input_type a_data_in[],
   //               assembled_type asm_data[]) const;
   //*** Action of the integrator at quadrature points (assembled form)
   // void Calc(const b_input_type b_data_in[], const assembled_type asm_data[],
   //           output_type data_out[]) const;
};


// Dim = spatial dimension
//   N = number of quadrature points
template <int Dim, int N>
class TMassIntegrator : public TIntegrator_base
{
public:

   //*** Traits: compile time information about the integrator class to enable
   //    the use inside template assembly procedures.
   static const int dim = Dim;
   static const int num_qdr_points = N;

   //*** Description of the input and output data:
   static const int num_inputs   = 2;    // mesh + one scalar field
   static const int num_outputs  = 1;    // one scalar field
   static const int num_a_inputs = 1;    // mesh only (a-input field)

   //*** Number of components for each input and output field
   static const int num_input_comp[num_inputs];   // = { Dim, 1 }
   static const int num_output_comp[num_outputs]; // = { 1 }

   //*** Field data required as input by the integrator
   static const field_data input_data[num_inputs]; // = { GRADIENT, VALUE }
   //*** Field data returned by the integrator
   static const field_data output_data[num_outputs]; // = { VALUE }

   //*** Data types associated with a quadrature point: input, output,
   //    assembled, a-input, and b-input :
   struct a_input_type { TMatrix<dim,dim> J; };

   struct b_input_type { double phi; };

   struct input_type { TMatrix<dim,dim> J; double phi; };

   struct output_type { double psi; };

   struct assembled_type { double detJ; };
   //*** End traits.

   //*** Definition of the mass integrator:

   //*** Action of the integrator at quadrature points
   void Calc(const input_type data_in[], output_type data_out[]) const;

   // *** Compute the assembled data at quadrature points
   void Assemble(const a_input_type a_data_in[],
                 assembled_type asm_data[]) const;

   //*** Action of the integrator at quadrature points (assembled form)
   void Calc(const b_input_type b_data_in[], const assembled_type asm_data[],
             output_type data_out[]) const;
};


// Dim = spatial dimension
//   N = number of quadrature points
template <int Dim, int N>
class TDiffusionIntegrator : public TIntegrator_base
{
public:

   //*** Traits: compile time information about the integrator class to enable
   //    the use inside template assembly procedures.
   static const int dim = Dim;
   static const int num_qdr_points = N;

   //*** Description of the input and output data:
   static const int num_inputs   = 2;    // mesh + one scalar field
   static const int num_outputs  = 1;    // one scalar field
   static const int num_a_inputs = 1;    // mesh only (a-input field)

   //*** Number of components for each input and output field
   static const int num_input_comp[num_inputs];   // = { Dim, 1 }
   static const int num_output_comp[num_outputs]; // = { 1 }

   //*** Field data required as input by the integrator
   static const field_data input_data[num_inputs]; // = { GRADIENT, GRADIENT }
   //*** Field data returned by the integrator
   static const field_data output_data[num_outputs]; // = { GRADIENT }

   //*** Data types associated with a quadrature point: input, output,
   //    assembled, a-input, and b-input :
   struct a_input_type { TMatrix<dim,dim> J; };

   struct b_input_type { TMatrix<dim,1> d_phi; };

   struct input_type { TMatrix<dim,dim> J; TMatrix<dim,1> d_phi; };

   struct output_type { TMatrix<dim,1> d_psi; };

   // TODO: 'diff' is symmetric --> we can save some data motion here
   struct assembled_type { TMatrix<dim,dim> diff; };
   //*** End traits.

   //*** Definition of the diffusion integrator:

   //*** Action of the integrator at quadrature points
   void Calc(const input_type data_in[], output_type data_out[]) const;

   // *** Compute the assembled data at quadrature points
   void Assemble(const a_input_type a_data_in[],
                 assembled_type asm_data[]) const;

   //*** Action of the integrator at quadrature points (assembled form)
   void Calc(const b_input_type b_data_in[], const assembled_type asm_data[],
             output_type data_out[]) const;
};


const char *TIntegrator_base::field_name[3] =
{ "value", "gradient", "value + gradient" };


template <int Dim, int N>
const int TMassIntegrator<Dim,N>::num_input_comp[num_inputs] = { Dim, 1 };

template <int Dim, int N>
const int TMassIntegrator<Dim,N>::num_output_comp[num_outputs] = { 1 };

template <int Dim, int N>
const TIntegrator_base::field_data TMassIntegrator<Dim,N>::
   input_data[num_inputs] = { GRADIENT, VALUE };

template <int Dim, int N>
const TIntegrator_base::field_data TMassIntegrator<Dim,N>::
   output_data[num_outputs] = { VALUE };

template <int Dim, int N>
void TMassIntegrator<Dim,N>::Calc(
   const input_type data_in[], output_type data_out[]) const
{
   for (int i = 0; i < N; i++)
      data_out[i].psi = data_in[i].phi * data_in[i].J.Det();
}

template <int Dim, int N>
void TMassIntegrator<Dim,N>::Assemble(
   const a_input_type a_data_in[], assembled_type asm_data[]) const
{
   for (int i = 0; i < N; i++)
      asm_data[i].detJ = a_data_in[i].J.Det();
}

template <int Dim, int N>
void TMassIntegrator<Dim,N>::Calc(
   const b_input_type b_data_in[], const assembled_type asm_data[],
   output_type data_out[]) const
{
   for (int i = 0; i < N; i++)
      data_out[i].psi = b_data_in[i].phi * asm_data[i].detJ;
}


template <int Dim, int N>
const int TDiffusionIntegrator<Dim,N>::num_input_comp[num_inputs] = { Dim, 1 };

template <int Dim, int N>
const int TDiffusionIntegrator<Dim,N>::num_output_comp[num_outputs] = { 1 };

template <int Dim, int N>
const TIntegrator_base::field_data TDiffusionIntegrator<Dim,N>::
   input_data[num_inputs] = { GRADIENT, GRADIENT };

template <int Dim, int N>
const TIntegrator_base::field_data TDiffusionIntegrator<Dim,N>::
   output_data[num_outputs] = { GRADIENT };

template <int Dim, int N>
void TDiffusionIntegrator<Dim,N>::Calc(
   const input_type data_in[], output_type data_out[]) const
{
   TMatrix<Dim,1> grad;
   TMatrix<Dim,Dim> adjJ;

   for (int i = 0; i < N; i++)
   {
      data_in[i].J.CalcAdjugate(adjJ);
      MultAtB(adjJ, data_in[i].d_phi, grad);
      grad.Scale(1.0 / data_in[i].J.Det(adjJ));
      Mult(adjJ, grad, data_out[i].d_psi);
   }
}


template <int Dim, int N>
void TDiffusionIntegrator<Dim,N>::Assemble(
   const a_input_type a_data_in[], assembled_type asm_data[]) const
{
   TMatrix<Dim,Dim> adjJ;

   for (int i = 0; i < N; i++)
   {
      a_data_in[i].J.CalcAdjugate(adjJ);
      MultABt(adjJ, adjJ, asm_data[i].diff);
      asm_data[i].diff.Scale(1.0 / a_data_in[i].J.Det(adjJ));
   }
}

template <int Dim, int N>
void TDiffusionIntegrator<Dim,N>::Calc(
   const b_input_type b_data_in[], const assembled_type asm_data[],
   output_type data_out[]) const
{
   for (int i = 0; i < N; i++)
   {
      Mult(asm_data[i].diff, b_data_in[i].d_phi, data_out[i].d_psi);
   }
}
