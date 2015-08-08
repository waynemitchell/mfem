
#include "tassembler.hpp"
#include "mfem.hpp"
// #include <iostream> // included by mfem.hpp

using namespace std;
using namespace mfem;

#ifdef BGQPROF
#include <mpi.h>
extern "C" HPM_Start(char*);
extern "C" HPM_Stop(char*);
#endif


int main(int argc, char *argv[])
{
#ifdef BGQPROF
   MPI_Init(&argc, &argv);
#endif

   const int dim = 2;   // space dimension (3D not implemented)
   const int p   = 2;   // polynomial degree
   const int q   = 4;   // quadrature points in each spatial direction
   const int ne  = 16;  // number of elements processed at a time
   const int nq  = 32;  // number of quadrature points processed at a time

   const int num_mults = 20;

   const int MiB = 1024*1024;
   const int GiB = 1024*MiB;

#ifndef BGQPROF
   const int max_mem = 512*MiB;

   const int mesh_2D_nx = 1200/p;
   const int mesh_2D_ny = 800/p;
#else
   const int max_mem = 32*MiB;

   const int mesh_2D_nx = 600/p;
   const int mesh_2D_ny = 400/p;
#endif

   const int mesh_2D_sx = 3.;
   const int mesh_2D_sy = 2.;

   typedef TMassIntegrator<dim, nq>      mass_integ_type;
   typedef TDiffusionIntegrator<dim, nq> diff_integ_type;
   // number of flops per quadrature point
   double flops_mass_integ = 0., flops_mass_integ_asm = 0.;
   double flops_diff_integ = 0., flops_diff_integ_asm = 0.;
   if (dim == 2)
   {
      flops_mass_integ = 3; // multiplies only

      flops_mass_integ_asm = 1; // flops in Calc (assembled form)

      flops_diff_integ += 0; // CalcAdjugate
      flops_diff_integ += 4; // adjJ^t.d_phi
      flops_diff_integ += 2; // det(J)
      flops_diff_integ += 1; // one division
      flops_diff_integ += 2; // grad*=val
      flops_diff_integ += 4; // adjJ.grad

      flops_diff_integ_asm = 4; // flops in Calc (assembled form)
   }
   else if (dim == 3)
   {
      flops_mass_integ = 10; // multiplies only

      flops_mass_integ_asm = 1; // flops in Calc (assembled form)
   }

   typedef TensorProductBasis<dim, p, q, ne> basis_type;
   // number of flops for a Calc or CalcT op on one element
   double flops_basis = 0.;
   if (dim == 2)
      flops_basis = (p+1)*q*(p+1+q); // multiplies only

   cout << '\n'
        << "dim = " << dim << '\n'
        << "p   = " << p << '\n'
        << "q   = " << q << '\n'
        << "ne  = " << ne << '\n'
        << "nq  = " << nq << '\n'
        << endl;

   Mesh *mesh = NULL;
   if (dim == 2)
      mesh = new Mesh(mesh_2D_nx, mesh_2D_ny, Element::QUADRILATERAL, 1,
                      mesh_2D_sx, mesh_2D_sy);
   else if (dim == 3)
      mesh = new Mesh(200/p, 120/p, 80/p, Element::HEXAHEDRON, 1, 5., 3., 2.);

   mesh->PrintCharacteristics();

   FiniteElementCollection *fec = new H1_FECollection(p, dim);
   FiniteElementSpace *fes = new FiniteElementSpace(mesh, fec);
   FiniteElementSpace *vfes = new FiniteElementSpace(mesh, fec, dim);
   mesh->SetNodalFESpace(vfes);

   int size = fes->GetVSize();
   double svec_dd = double(size)*sizeof(double);
   double vvec_dd = dim*svec_dd;
   cout << "size of the scalar space vectors = " << svec_dd/MiB << " MiB\n"
        << "size of the mesh nodes vector    = " << vvec_dd/MiB << " MiB\n"
        << endl;

   TAssembler<mass_integ_type, basis_type, false> mass_assembler(fes);
   TAssembler<diff_integ_type, basis_type, false> diff_assembler(fes);

   // mass_assembler.PrintIntegrator();
   // diff_assembler.PrintIntegrator();

   // mass_assembler.PrintBasis();
   // diff_assembler.PrintBasis();
   // cout << endl;

   double eldof_id = sizeof(int) *
      double(mass_assembler.GetElementToDofTable().Size_of_connections());
   cout << "size of element-to-dof J array = " << eldof_id/MiB << " MiB\n"
        << endl;

   double utime, rtime, bflops, qflops, flops, rmops, wmops;

   // mfem objects for comparison
   cout << "Assembling bilinear forms:\n"
        << "mass ...      " << flush;
   const IntegrationRule &ir =
      IntRules.Get(mesh->GetElementBaseGeometry(0), 2*q-1);
   // cout << endl << "ir.GetNPoints() = " << ir.GetNPoints() << endl;
   BilinearForm mass_form(fes);
   BilinearFormIntegrator *mass_integ = new MassIntegrator;
   mass_integ->SetIntRule(&ir);
   mass_form.AddDomainIntegrator(mass_integ);
   const int skip_zeros = 0; // keep zero entries during assemble/finalize
   mass_form.UsePrecomputedSparsity();
   mass_form.AllocateMatrix();
   tic();
   mass_form.Assemble(skip_zeros);
   // mass_form.Finalize(skip_zeros);
   tic_toc.Stop();
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   cout << " utime = " << utime << " s, rtime = " << rtime << " s" << endl;
   cout << "diffusion ... " << flush;
   BilinearForm diff_form(fes);
   BilinearFormIntegrator *diff_integ = new DiffusionIntegrator;
   diff_integ->SetIntRule(&ir);
   diff_form.AddDomainIntegrator(diff_integ);
   diff_form.UsePrecomputedSparsity();
   diff_form.AllocateMatrix();
   tic();
   diff_form.Assemble(skip_zeros);
   // diff_form.Finalize(skip_zeros);
   tic_toc.Stop();
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   cout << " utime = " << utime << " s, rtime = " << rtime << " s" << endl;
   NonlinearForm diff_actn(fes);
   diff_integ = new DiffusionIntegrator;
   diff_integ->SetIntRule(&ir);
   diff_actn.AddDomainIntegrator(diff_integ);

   const SparseMatrix &mass_mat = mass_form.SpMat();
   const SparseMatrix &diff_mat = diff_form.SpMat();

   int mass_nnz = mass_mat.NumNonZeroElems();
   double mass_dd = mass_nnz*double(sizeof(double));
   double mass_id = ((mass_nnz+mass_mat.Size()+1)*double(sizeof(int)));
   int diff_nnz = diff_mat.NumNonZeroElems();
   double diff_dd = diff_nnz*double(sizeof(double));
   double diff_id = ((diff_nnz+diff_mat.Size()+1)*double(sizeof(int)));
   cout << endl
        << "Mass matrix:         nonzeros = " << mass_nnz << "\n"
        << "Mass matrix:      double data = " << mass_dd/MiB << " MiB\n"
        << "Mass matrix:         int data = " << mass_id/MiB << " MiB\n"
        << "Mass matrix:       total data = " << (mass_dd+mass_id)/MiB
        << " MiB\n";
   cout << "Diffusion matrix:    nonzeros = " << diff_nnz << "\n"
        << "Diffusion matrix: double data = " << diff_dd/MiB << " MiB\n"
        << "Diffusion matrix:    int data = " << diff_id/MiB << " MiB\n"
        << "Diffusion matrix:  total data = " << (diff_dd+diff_id)/MiB
        << " MiB" << endl;

   // partial assembly in TAssembler
   cout << endl << "Partial assembly: TAssembler::Assemble:\n"
        << "mass ...      " << flush;
   tic();
   mass_assembler.Assemble();
   tic_toc.Stop();
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   cout << " utime = " << utime << " s, rtime = " << rtime << " s" << endl;
   cout << "diffusion ... " << flush;
   tic();
   diff_assembler.Assemble();
   tic_toc.Stop();
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   cout << " utime = " << utime << " s, rtime = " << rtime << " s" << endl;

   // size of the assembled data in TAssembler
   double mass_asm_dd = (double(sizeof(mass_integ_type::assembled_type)) *
                         basis_type::total_qpts * mesh->GetNE());
   double diff_asm_dd = (double(sizeof(diff_integ_type::assembled_type)) *
                         basis_type::total_qpts * mesh->GetNE());
   cout << endl
        << "Size of the partially assembled data:\n"
        << "Mass      = " << mass_asm_dd/MiB << " MiB\n"
        << "Diffusion = " << diff_asm_dd/MiB << " MiB" << endl;

#if 0
   // test the experimental implementation for correctness
   double max_mass_error = 0.0, max_diff_error = 0.0, max_diff_error_2 = 0.0;
   double max_mass_error_asm = 0.0, max_diff_error_asm = 0.0;
   GridFunction u(fes), v_tmpl(fes), v_mfem(fes);

   cout << endl << "Comparing Mult() on " << num_mults
        << " random vectors ..." << flush;
   for (int i = 0; i < num_mults; i++)
   {
      u.Randomize();
      mass_assembler.use_assembled_data = false;
      mass_assembler.Mult(u, v_tmpl);
      mass_form.Mult(u, v_mfem);
      v_tmpl -= v_mfem;
      max_mass_error = fmax(max_mass_error, v_tmpl.Normlinf());

      mass_assembler.use_assembled_data = true;
      mass_assembler.Mult(u, v_tmpl);
      v_tmpl -= v_mfem;
      max_mass_error_asm = fmax(max_mass_error_asm, v_tmpl.Normlinf());

      diff_form.Mult(u, v_mfem);
      diff_assembler.use_assembled_data = false;
      diff_assembler.Mult(u, v_tmpl);
      v_tmpl -= v_mfem;
      max_diff_error = fmax(max_diff_error, v_tmpl.Normlinf());

      diff_assembler.use_assembled_data = true;
      diff_assembler.Mult(u, v_tmpl);
      v_tmpl -= v_mfem;
      max_diff_error_asm = fmax(max_diff_error_asm, v_tmpl.Normlinf());

      diff_actn.Mult(u, v_tmpl);
      v_tmpl -= v_mfem;
      max_diff_error_2 = fmax(max_diff_error_2, v_tmpl.Normlinf());
   }

   cout << " done." << endl
        << "Errors relative to assembled MFEM:\n"
        << "max_mass_error = " << max_mass_error
        << " (experimental)" << endl
        << "max_mass_error = " << max_mass_error_asm
        << " (assembled experimental)" << endl
        << "max_diff_error = " << max_diff_error
        << " (experimental)" << endl
        << "max_diff_error = " << max_diff_error_asm
        << " (assembled experimental)" << endl
        << "max_diff_error = " << max_diff_error_2
        << " (action MFEM)" << endl;
#endif

   int num_vecs = max_mem/(sizeof(double)*size);

   Vector us(num_vecs*size), vs(num_vecs*size);
   Vector ui(NULL, size), vi(NULL, size);

   cout << endl;
   cout << "Comparing speed of matrix-vector products:" << endl
        << "number of matrix-vector products = " << num_vecs << endl
        << "size of all input/output vectors = "
        << svec_dd*num_vecs/MiB << " MiB" << endl;

   cout << endl;
   cout << "initializing the vectors ..." << flush;
   us.Randomize();
   vs = 0.0;
   cout << " done." << endl;

   cout << endl;
   cout << "Experimental assembler (NOT using assembled data):" << endl;

   mass_assembler.use_assembled_data = false;
#ifdef BGQPROF
   HPM_Start("Unassembled_mass");
#endif
   tic();
   for (int i = 0; i < num_vecs; i++)
   {
      ui.SetData(&us(i*size));
      vi.SetData(&vs(i*size));

      mass_assembler.Mult(ui, vi);
   }
   tic_toc.Stop();
#ifdef BGQPROF
   HPM_Stop("Unassembled_mass");
#endif
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   bflops = flops_basis*(dim*dim+2);
   bflops *= mesh->GetNE()*num_vecs;
   qflops = (flops_mass_integ+1)*basis_type::total_qpts;
   qflops *= mesh->GetNE()*num_vecs;
   flops = bflops + qflops;
   rmops = (svec_dd + vvec_dd + eldof_id)*num_vecs;
   wmops = svec_dd*num_vecs;
   cout << "Mass assembler:         utime = " << utime << " s" << endl;
   cout << "Mass assembler:         rtime = " << rtime << " s" << endl;
   cout << "Mass assembler:        Gflops = " << flops/1e9
        << " = " << bflops/1e9 << " basis + " << qflops/1e9 << " qpts" << endl;
   cout << "Mass assembler:      Gflops/s = " << flops/1e9/rtime << endl;
   cout << "Mass assembler:           GiB = " << (rmops+wmops)/GiB
        << " = " << rmops/GiB << " read + " << wmops/GiB << " write" << endl;
   cout << "Mass assembler:         GiB/s = " << (rmops+wmops)/GiB/rtime
        << " = " << rmops/GiB/rtime << " read + " << wmops/GiB/rtime
        << " write" << endl;

   cout << endl;

   diff_assembler.use_assembled_data = false;
#ifdef BGQPROF
   HPM_Start("Unassembled_diff");
#endif
   tic();
   for (int i = 0; i < num_vecs; i++)
   {
      ui.SetData(&us(i*size));
      vi.SetData(&vs(i*size));

      diff_assembler.Mult(ui, vi);
   }
   tic_toc.Stop();
#ifdef BGQPROF
   HPM_Stop("Unassembled_diff");
#endif
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   bflops = flops_basis*(dim*(dim+2));
   bflops *= mesh->GetNE()*num_vecs;
   qflops = (flops_diff_integ+dim)*basis_type::total_qpts;
   qflops *= mesh->GetNE()*num_vecs;
   flops = bflops + qflops;
   rmops = (svec_dd + vvec_dd + eldof_id)*num_vecs;
   wmops = svec_dd*num_vecs;
   cout << "Diffusion assembler:    utime = " << utime << " s" << endl;
   cout << "Diffusion assembler:    rtime = " << rtime << " s" << endl;
   cout << "Diffusion assembler:   Gflops = " << flops/1e9
        << " = " << bflops/1e9 << " basis + " << qflops/1e9 << " qpts" << endl;
   cout << "Diffusion assembler: Gflops/s = " << flops/1e9/rtime << endl;

   cout << "Diffusion assembler:      GiB = " << (rmops+wmops)/GiB
        << " = " << rmops/GiB << " read + " << wmops/GiB << " write" << endl;
   cout << "Diffusion assembler:    GiB/s = " << (rmops+wmops)/GiB/rtime
        << " = " << rmops/GiB/rtime << " read + " << wmops/GiB/rtime
        << " write" << endl;

   cout << endl;
   cout << "Experimental assembler (using assembled data):" << endl;

   mass_assembler.use_assembled_data = true;
#ifdef BGQPROF
   HPM_Start("Partial_assembled_mass");
#endif
   tic();
   for (int i = 0; i < num_vecs; i++)
   {
      ui.SetData(&us(i*size));
      vi.SetData(&vs(i*size));

      mass_assembler.Mult(ui, vi);
   }
   tic_toc.Stop();
#ifdef BGQPROF
   HPM_Stop("Partial_assembled_mass");
#endif
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   bflops = flops_basis*(2);
   bflops *= mesh->GetNE()*num_vecs;
   qflops = (flops_mass_integ_asm+1)*basis_type::total_qpts;
   qflops *= mesh->GetNE()*num_vecs;
   flops = bflops + qflops;
   rmops = (svec_dd + mass_asm_dd + eldof_id)*num_vecs;
   wmops = svec_dd*num_vecs;
   cout << "Mass assembler:         utime = " << utime << " s" << endl;
   cout << "Mass assembler:         rtime = " << rtime << " s" << endl;
   cout << "Mass assembler:        Gflops = " << flops/1e9
        << " = " << bflops/1e9 << " basis + " << qflops/1e9 << " qpts" << endl;
   cout << "Mass assembler:      Gflops/s = " << flops/1e9/rtime << endl;
   cout << "Mass assembler:           GiB = " << (rmops+wmops)/GiB
        << " = " << rmops/GiB << " read + " << wmops/GiB << " write" << endl;
   cout << "Mass assembler:         GiB/s = " << (rmops+wmops)/GiB/rtime
        << " = " << rmops/GiB/rtime << " read + " << wmops/GiB/rtime
        << " write" << endl;

   cout << endl;

   diff_assembler.use_assembled_data = true;
#ifdef BGQPROF
   HPM_Start("Partial_assembled_diff");
#endif
   tic();
   for (int i = 0; i < num_vecs; i++)
   {
      ui.SetData(&us(i*size));
      vi.SetData(&vs(i*size));

      diff_assembler.Mult(ui, vi);
   }
   tic_toc.Stop();
#ifdef BGQPROF
   HPM_Stop("Partial_assembled_diff");
#endif
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   bflops = flops_basis*(2*dim);
   bflops *= mesh->GetNE()*num_vecs;
   qflops = (flops_diff_integ_asm+dim)*basis_type::total_qpts;
   qflops *= mesh->GetNE()*num_vecs;
   flops = bflops + qflops;
   rmops = (svec_dd + diff_asm_dd + eldof_id)*num_vecs;
   wmops = svec_dd*num_vecs;
   cout << "Diffusion assembler:    utime = " << utime << " s" << endl;
   cout << "Diffusion assembler:    rtime = " << rtime << " s" << endl;
   cout << "Diffusion assembler:   Gflops = " << flops/1e9
        << " = " << bflops/1e9 << " basis + " << qflops/1e9 << " qpts" << endl;
   cout << "Diffusion assembler: Gflops/s = " << flops/1e9/rtime << endl;

   cout << "Diffusion assembler:      GiB = " << (rmops+wmops)/GiB
        << " = " << rmops/GiB << " read + " << wmops/GiB << " write" << endl;
   cout << "Diffusion assembler:    GiB/s = " << (rmops+wmops)/GiB/rtime
        << " = " << rmops/GiB/rtime << " read + " << wmops/GiB/rtime
        << " write" << endl;

   cout << endl;
   cout << "Assembled bilinear forms (csr format):" << endl;

#ifdef BGQPROF
   HPM_Start("MFEM_assembled_mass");
#endif
   tic();
   for (int i = 0; i < num_vecs; i++)
   {
      ui.SetData(&us(i*size));
      vi.SetData(&vs(i*size));

      mass_form.Mult(ui, vi);
   }
   tic_toc.Stop();
#ifdef BGQPROF
   HPM_Stop("MFEM_assembled_mass");
#endif
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   flops = double(mass_mat.NumNonZeroElems())*num_vecs; // multiplies only
   rmops = (svec_dd + mass_dd + mass_id)*num_vecs;
   wmops = svec_dd*num_vecs;
   cout << "Mass form:              utime = " << utime << " s" << endl;
   cout << "Mass form:              rtime = " << rtime << " s" << endl;
   cout << "Mass form:             Gflops = " << flops/1e9 << endl;
   cout << "Mass form:           Gflops/s = " << flops/1e9/rtime << endl;
   cout << "Mass form:                GiB = " << (rmops+wmops)/GiB
        << " = " << rmops/GiB << " read + " << wmops/GiB << " write" << endl;
   cout << "Mass form:              GiB/s = " << (rmops+wmops)/GiB/rtime
        << " = " << rmops/GiB/rtime << " read + " << wmops/GiB/rtime
        << " write" << endl;

   cout << endl;

#ifdef BGQPROF
   HPM_Start("MFEM_assembled_diff");
#endif
   tic();
   for (int i = 0; i < num_vecs; i++)
   {
      ui.SetData(&us(i*size));
      vi.SetData(&vs(i*size));

      diff_form.Mult(ui, vi);
   }
   tic_toc.Stop();
#ifdef BGQPROF
   HPM_Stop("MFEM_assembled_diff");
#endif
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   flops = double(diff_mat.NumNonZeroElems())*num_vecs; // multiplies only
   rmops = (svec_dd + diff_dd + diff_id)*num_vecs;
   wmops = svec_dd*num_vecs;
   cout << "Diffusion form:         utime = " << utime << " s" << endl;
   cout << "Diffusion form:         rtime = " << rtime << " s" << endl;
   cout << "Diffusion form:        Gflops = " << flops/1e9 << endl;
   cout << "Diffusion form:      Gflops/s = " << flops/1e9/rtime << endl;
   cout << "Diffusion form:           GiB = " << (rmops+wmops)/GiB
        << " = " << rmops/GiB << " read + " << wmops/GiB << " write" << endl;
   cout << "Diffusion form:         GiB/s = " << (rmops+wmops)/GiB/rtime
        << " = " << rmops/GiB/rtime << " read + " << wmops/GiB/rtime
        << " write" << endl;

#if 1
   cout << endl;
   cout << "MFEM matrix-vector product (no matrix assembly):" << endl;
   int actn_num_vecs = num_vecs/10;
   cout << "number of matrix-vector products = " << actn_num_vecs << endl;

#ifdef BGQPROF
   HPM_Start("MFEM_unassembled_diff");
#endif
   tic();
   for (int i = 0; i < actn_num_vecs; i++)
   {
      ui.SetData(&us(i*size));
      vi.SetData(&vs(i*size));

      diff_actn.Mult(ui, vi);
   }
   tic_toc.Stop();
#ifdef BGQPROF
   HPM_Stop("MFEM_unassembled_diff");
#endif
   utime = tic_toc.UserTime();
   rtime = tic_toc.RealTime();
   flops = 0.;
   rmops = (svec_dd + vvec_dd + eldof_id)*actn_num_vecs;
   wmops = svec_dd*actn_num_vecs;
   cout << "Diffusion action:       utime = " << utime << " s" << endl;
   cout << "Diffusion action:       rtime = " << rtime << " s" << endl;
   cout << "Diffusion action:      Gflops = " << flops/1e9 << endl;
   cout << "Diffusion action:    Gflops/s = " << flops/1e9/rtime << endl;
   cout << "Diffusion action:         GiB = " << (rmops+wmops)/GiB
        << " = " << rmops/GiB << " read + " << wmops/GiB << " write" << endl;
   cout << "Diffusion action:       GiB/s = " << (rmops+wmops)/GiB/rtime
        << " = " << rmops/GiB/rtime << " read + " << wmops/GiB/rtime
        << " write" << endl;
#endif

   cout << endl;

   delete vfes;
   delete fes;
   delete fec;
   delete mesh;

#ifdef BGQPROF
   MPI_Finalize();
#endif

   return 0;
}
