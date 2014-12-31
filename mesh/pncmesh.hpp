#ifndef MFEM_PNCMESH
#define MFEM_PNCMESH

#include "../config.hpp"

#ifdef MFEM_USE_MPI

#include <cstring>
#include <stdint.h> // <cstdint>

#include "ncmesh.hpp"

namespace mfem
{

/** Represents a unique leaf element ID, expressed as a root (coarse)
 *  element index and a refinement path to the leaf. Used to identify
 *  elements/refinements across processors. The path is stored as a sequence
 *  of child numbers, each taking half a uint8_t (i.e., elements can have
 *  up to 16 children).
 */
class ElementId
{
   int root_index;
   enum { PathBytes = 24 };
   uint8_t path[PathBytes];
   int length;

public:
   /// Initialize path root, set path length to zero.
   ElementId(int root_index)
      : root_index(root_index), length(0)
   {
      std::memset(path, 0, sizeof(path));
   }

   /// Return path length.
   int Length() const { return length; }

   /// Append child index to the refinement path.
   void AppendPath(int child)
   {
      if (length >= PathBytes*2)
         MFEM_ABORT("Refinement path too long, increase PathBytes.");

      path[length / 2] |= (uint8_t) (child << (4*(length & 1)));
      length++;
   }

   /// Return child index at path position 'pos', 0 <= pos < Length().
   int GetChild(int pos) const
   {
      return (path[pos / 2] >> (4*(pos & 1))) & 0xf;
   }

   /// Dump content to a buffer, update buffer pointer.
   void ToBuffer(uint8_t*& buffer) const;

   /// Load content from a buffer, update buffer pointer.
   void FromBuffer(uint8_t*& buffer);

   /// Return the maximum number of bytes 'ToBuffer' will ever write.
   int MaxBufferBytes() const { return 3 + 1 + PathBytes; }
};


/** \brief
 *
 */
class ParNCMesh : public NCMesh
{
public:
   ParNCMesh(MPI_Comm comm, const Mesh* coarse_mesh);

protected:

   MPI_Comm MyComm;
   int NRanks, MyRank;

   void InitialPartition();

   virtual void AssignLeafIndices(); // override


};


} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_PNCMESH
