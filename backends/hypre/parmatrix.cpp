#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_MPI)

#include "parmatrix.hpp"
#include "_hypre_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"

namespace mfem
{

namespace hypre
{

   // Make a block-diagonal matrix
ParMatrix::ParMatrix(Layout &layout, mfem::SparseMatrix &spmat)
   : mfem::Operator(layout), x_vec(NULL), y_vec(NULL) {

   // Create the HYPRE ParCSRMatrix
   mat = hypre_ParCSRMatrixCreate(
      layout.GetEngine().GetComm(),   // comm
      layout.GlobalSize(),            // global_num_rows
      layout.GlobalSize(),            // global_num_cols
      layout.Offsets(),               // row_starts
      layout.Offsets(),               // col_starts
      0,                              // num_cols_offd
      spmat.NumNonZeroElems(),        // num_nonzeros_diag
      0);                             // num_nonzeros_offd

   // The HYPRE ParCSRMatrix owns everything (for now)
   hypre_ParCSRMatrixOwnsData(mat) = (HYPRE_Int) true;
   hypre_ParCSRMatrixOwnsRowStarts(mat) = (HYPRE_Int) false;
   hypre_ParCSRMatrixOwnsColStarts(mat) = (HYPRE_Int) false;

   // Recall that this asssumed that the engine device memory space
   // matches that of HYPRE, so we can EITHER use hypre_CTAlloc OR
   // the engine's allocation (MakeArray or MakeVector). When using
   // hypre_CTAlloc, the ownership should be set so that HYPRE
   // destroys the data (owned = true).

   // ----- Initialize the on-processor part -----
   // Get the on-processor CSRMatrix
   hypre_CSRMatrix *mat_diag = hypre_ParCSRMatrixDiag(mat);

   // We provide and hence deallocate i, j, and data
   hypre_CSRMatrixOwnsData(mat_diag) = (HYPRE_Int) true;

   // spmat.I, spmat.J, and spmat.data are all in host memory and need to be copied over
   {
      const std::size_t length = spmat.Height() + 1;
      hypre_CSRMatrixI(mat_diag) = hypre_TAlloc(HYPRE_Int, length, HYPRE_MEMORY_SHARED);
      MFEM_ASSERT(sizeof(HYPRE_Int) == sizeof(int), "");
      std::memcpy(hypre_CSRMatrixI(mat_diag), spmat.GetI(), length * sizeof(HYPRE_Int));
   }

   {
      const std::size_t length = spmat.NumNonZeroElems();
      hypre_CSRMatrixJ(mat_diag) = hypre_TAlloc(HYPRE_Int, length, HYPRE_MEMORY_SHARED);
      MFEM_ASSERT(sizeof(HYPRE_Int) == sizeof(int), "");
      std::memcpy(hypre_CSRMatrixJ(mat_diag), spmat.GetJ(), length * sizeof(HYPRE_Int));
   }

   {
      const std::size_t length = spmat.NumNonZeroElems();
      hypre_CSRMatrixData(mat_diag) = hypre_TAlloc(HYPRE_Real, length, HYPRE_MEMORY_SHARED);
      MFEM_ASSERT(sizeof(HYPRE_Real) == sizeof(double), "");
      std::memcpy(hypre_CSRMatrixData(mat_diag), spmat.GetData(), length * sizeof(HYPRE_Real));
   }

   // Populate the CSRMatrix's rownnz
   // NOTE: CSRMatrix's rownnz is always owned by HYPRE
   hypre_CSRMatrixSetRownnz(mat_diag);
   // ----------------------

   // ----- Initialize the off-processor part (all zeros) -----
   // Get the off-processor CSRMatrix
   hypre_CSRMatrix *mat_offd = hypre_ParCSRMatrixOffd(mat);

   hypre_CSRMatrixOwnsData(mat_offd) = (HYPRE_Int) true;
   hypre_CSRMatrixI(mat_offd) = hypre_CTAlloc(HYPRE_Int, spmat.Height() + 1, HYPRE_MEMORY_SHARED);

   // Leave this matrix zeroed
   // ----------------------

   hypre_ParCSRMatrixSetNumNonzeros(mat);

   /* Make sure that the first entry in each row is the diagonal one. */
   hypre_CSRMatrixReorder(mat_diag);

   hypre_MatvecCommPkgCreate(mat);

   x_vec = InitializeVector(layout);
   y_vec = InitializeVector(layout);
}

ParMatrix::ParMatrix(Layout &in_layout, Layout &out_layout,
                     HYPRE_Int *i_diag, HYPRE_Int *j_diag,
                     HYPRE_Int *i_offd, HYPRE_Int *j_offd,
                     HYPRE_Int *cmap, HYPRE_Int cmap_size)
   : mfem::Operator(in_layout, out_layout)
{
   HYPRE_Int diag_nnz, offd_nnz;

   MPI_Comm comm = in_layout.GetEngine().GetComm();

   int id, np;
   MPI_Comm_rank(comm, &id);
   MPI_Comm_size(comm, &np);

   HYPRE_Int *row = out_layout.Offsets();
   HYPRE_Int *col = in_layout.Offsets();

   if (HYPRE_AssumedPartitionCheck())
   {
      diag_nnz = i_diag[row[1]-row[0]];
      offd_nnz = i_offd[row[1]-row[0]];

      mat = hypre_ParCSRMatrixCreate(comm, row[2], col[2], row, col,
                                   cmap_size, diag_nnz, offd_nnz);
   }
   else
   {
      diag_nnz = i_diag[row[id+1]-row[id]];
      offd_nnz = i_offd[row[id+1]-row[id]];

      mat = hypre_ParCSRMatrixCreate(comm, row[np], col[np], row, col,
                                     cmap_size, diag_nnz, offd_nnz);
   }

   hypre_ParCSRMatrixOwnsData(mat) = (HYPRE_Int) true;
   hypre_ParCSRMatrixOwnsRowStarts(mat) = (HYPRE_Int) false;
   hypre_ParCSRMatrixOwnsColStarts(mat) = (HYPRE_Int) false;

   // ----- Initialize the on-processor part -----
   hypre_CSRMatrix *mat_diag = hypre_ParCSRMatrixDiag(mat);
   hypre_CSRMatrixOwnsData(mat_diag) = (HYPRE_Int) true;

   {
      const std::size_t length = out_layout.Size()+1;
      hypre_CSRMatrixI(mat_diag) = hypre_TAlloc(HYPRE_Int, length, HYPRE_MEMORY_SHARED);
      std::memcpy(hypre_CSRMatrixI(mat_diag), i_diag, length * sizeof(HYPRE_Int));
   }

   {
      const std::size_t length = diag_nnz;
      hypre_CSRMatrixJ(mat_diag) = hypre_TAlloc(HYPRE_Int, length, HYPRE_MEMORY_SHARED);
      std::memcpy(hypre_CSRMatrixJ(mat_diag), j_diag, length * sizeof(HYPRE_Int));
   }

   {
      const int length = diag_nnz;
      HYPRE_Real *data_diag = hypre_TAlloc(HYPRE_Real, length, HYPRE_MEMORY_SHARED);
      for (HYPRE_Int i = 0; i < length; i++) data_diag[i] = 1.0;
      hypre_CSRMatrixData(mat_diag) = data_diag;
   }

   hypre_CSRMatrixSetRownnz(mat_diag);
   // ----------------------

   // ----- Initialize the off-processor part -----
   hypre_CSRMatrix *mat_offd = hypre_ParCSRMatrixOffd(mat);
   hypre_CSRMatrixOwnsData(mat_offd) = (HYPRE_Int) true;

   {
      const std::size_t length = out_layout.Size()+1;
      hypre_CSRMatrixI(mat_offd) = hypre_TAlloc(HYPRE_Int, length, HYPRE_MEMORY_SHARED);
      std::memcpy(hypre_CSRMatrixI(mat_offd), i_offd, length * sizeof(HYPRE_Int));
   }

   {
      const std::size_t length = offd_nnz;
      hypre_CSRMatrixJ(mat_offd) = hypre_TAlloc(HYPRE_Int, length, HYPRE_MEMORY_SHARED);
      std::memcpy(hypre_CSRMatrixJ(mat_offd), j_offd, length * sizeof(HYPRE_Int));
   }

   {
      const int length = offd_nnz;
      HYPRE_Real *data_offd = hypre_TAlloc(HYPRE_Real, length, HYPRE_MEMORY_SHARED);
      for (HYPRE_Int i = 0; i < length; i++) data_offd[i] = 1.0;
      hypre_CSRMatrixData(mat_offd) = data_offd;
   }

   hypre_CSRMatrixSetRownnz(mat_offd);

   {
      const std::size_t length = cmap_size;
      hypre_ParCSRMatrixColMapOffd(mat) = hypre_TAlloc(HYPRE_Int, length, HYPRE_MEMORY_HOST);
      std::memcpy(hypre_ParCSRMatrixColMapOffd(mat), cmap, length * sizeof(HYPRE_Int));
   }
   // ----------------------

   hypre_ParCSRMatrixSetNumNonzeros(mat);

   // Make sure that the first entry in each row is the diagonal one.
   if (row == col) hypre_CSRMatrixReorder(mat_diag);

   hypre_MatvecCommPkgCreate(mat);

   x_vec = InitializeVector(in_layout);
   y_vec = InitializeVector(out_layout);
}

ParMatrix::ParMatrix(mfem::hypre::Layout &in_layout, mfem::hypre::Layout &out_layout, hypre_ParCSRMatrix *mat_)
   : mfem::Operator(in_layout, out_layout), mat(mat_)
{
   x_vec = InitializeVector(in_layout);
   y_vec = InitializeVector(out_layout);
}

ParMatrix::ParMatrix(ParMatrix& other)
   : mfem::Operator(other.in_layout->As<Layout>(), other.out_layout->As<Layout>())
{
   mat = hypre_ParCSRMatrixCompleteClone(other.mat);
   x_vec = InitializeVector(in_layout->As<Layout>());
   y_vec = InitializeVector(out_layout->As<Layout>());
}

void ParMatrix::HypreAxpy(const double alpha, const ParMatrix& A, const double beta, const ParMatrix& B)
{
   hypre_CSRMatrix *transpose = hypre_ParCSRMatrixDiagT(A.HypreMatrix());
   if (transpose != NULL) mfem_error("Does not work with transposed matrices");

   // diag
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A.HypreMatrix());
   hypre_CSRMatrix *B_diag = hypre_ParCSRMatrixDiag(B.HypreMatrix());
   hypre_CSRMatrix *C_diag = hypre_ParCSRMatrixDiag(mat);

   {
      double *A_data = hypre_CSRMatrixData(A_diag);
      double *B_data = hypre_CSRMatrixData(B_diag);
      double *C_data = hypre_CSRMatrixData(C_diag);

      int nnz = hypre_CSRMatrixNumNonzeros(C_diag);
      // TODO: Check where this is and prefetch if needed
      for (int i = 0; i < nnz; i++) C_data[i] = A_data[i] * alpha + B_data[i] * beta;
   }

   // offd
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A.HypreMatrix());
   hypre_CSRMatrix *B_offd = hypre_ParCSRMatrixOffd(B.HypreMatrix());
   hypre_CSRMatrix *C_offd = hypre_ParCSRMatrixOffd(mat);

   {
      double *A_data = hypre_CSRMatrixData(A_offd);
      double *B_data = hypre_CSRMatrixData(B_offd);
      double *C_data = hypre_CSRMatrixData(C_offd);

      int nnz = hypre_CSRMatrixNumNonzeros(C_offd);
      // TODO: Check where this is and prefetch if needed
      for (int i = 0; i < nnz; i++) C_data[i] = A_data[i] * alpha + B_data[i] * beta;
   }
}

hypre_ParVector *InitializeVector(Layout &layout)
{

   hypre_ParVector *vec = hypre_ParVectorCreate(layout.GetEngine().GetComm(), layout.GlobalSize(), layout.Offsets());
   hypre_ParVectorInitialize(vec);

   // Set when used
   hypre_VectorData(hypre_ParVectorLocalVector(vec)) = NULL;

   hypre_VectorSize(hypre_ParVectorLocalVector(vec)) = layout.Size();
   hypre_ParVectorOwnsPartitioning(vec) = (HYPRE_Int) false;
   hypre_ParVectorOwnsData(vec) = (HYPRE_Int) false;

   return vec;
}

ParMatrix *MakePtAP(const ParMatrix &P, const ParMatrix &A)
{
   const bool P_owns_its_col_starts = hypre_ParCSRMatrixOwnsColStarts(P.HypreMatrix());

   hypre_ParCSRMatrix *rap;
   hypre_BoomerAMGBuildCoarseOperator(P.HypreMatrix(), A.HypreMatrix(), P.HypreMatrix(), &rap);
   hypre_ParCSRMatrixSetNumNonzeros(rap);
   // hypre_MatvecCommPkgCreate(rap);

   /* Warning: hypre_BoomerAMGBuildCoarseOperator steals the col_starts
      from P (even if it does not own them)! */
   hypre_ParCSRMatrixOwnsRowStarts(rap) = (HYPRE_Int) false;
   hypre_ParCSRMatrixOwnsColStarts(rap) = (HYPRE_Int) false;

   if (P_owns_its_col_starts)
   {
      hypre_ParCSRMatrixOwnsColStarts(P.HypreMatrix()) = (HYPRE_Int) true;
   }

   return new ParMatrix(P.InLayout()->As<Layout>(), P.InLayout()->As<Layout>(), rap);
}

ParMatrix *Add(const double alpha, const ParMatrix &A, const double beta, const ParMatrix &B)
{
   hypre_ParCSRMatrix *mat;
   hypre_ParcsrAdd(alpha, A.HypreMatrix(), beta, B.HypreMatrix(), &mat);
   return new ParMatrix(A.InLayout()->As<Layout>(), A.OutLayout()->As<Layout>(), mat);
}


} // namespace mfem::hypre

} // namespace mfem

#endif
