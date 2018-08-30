#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_MPI)

#include "fespace.hpp"

namespace mfem {

namespace hypre {

ParMatrix *FiniteElementSpace::MakeProlongation() const
{
   // ParMatrix *prolong = new ParMatrix(t_layout, l_layout);
   MFEM_ASSERT(pfespace->Conforming(), "Only implemented for conforming case!");

   const std::size_t ldof = l_layout->Size();
   const std::size_t tdof = t_layout->Size();

   mfem::Array<int> i_diag(ldof+1);
   mfem::Array<int> j_diag(tdof);
   int diag_counter;

   mfem::Array<int> i_offd(ldof+1);
   mfem::Array<int> j_offd(ldof-tdof);
   int offd_counter;

   mfem::Array<int> cmap(ldof-tdof);

   mfem::Array<mfem::Pair<HYPRE_Int, int> > cmap_j_offd(ldof-tdof);

   i_diag[0] = i_offd[0] = 0;
   diag_counter = offd_counter = 0;
   for (std::size_t i = 0; i < ldof; i++)
   {
      int ltdof = pfespace->GetLocalTDofNumber(i);
      if (ltdof >= 0)
      {
         j_diag[diag_counter++] = ltdof;
      }
      else
      {
         cmap_j_offd[offd_counter].one = pfespace->GetGlobalTDofNumber(i);
         cmap_j_offd[offd_counter].two = offd_counter;
         offd_counter++;
      }
      i_diag[i+1] = diag_counter;
      i_offd[i+1] = offd_counter;
   }

   mfem::SortPairs<HYPRE_Int, int>(cmap_j_offd, offd_counter);

   for (int i = 0; i < offd_counter; i++)
   {
      cmap[i] = cmap_j_offd[i].one;
      j_offd[cmap_j_offd[i].two] = i;
   }

   return new mfem::hypre::ParMatrix(*t_layout, *l_layout,
                                     i_diag, j_diag,
                                     i_offd, j_offd,
                                     cmap, offd_counter);
}

} // namespace mfem::hypre

} // namespace mfem

#endif
