//                       Create parallel mesh file set.
//
// Compile with: make refine_mesh
//
// Sample runs:  mpirun -np 4 refine_mesh -s ../data/beam-tri.mesh -d ../data/beam-tri-4
//
// Description:  This example code will refine a serial mesh into a resulting serial or parallel mesh.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   int num_procs = 1;
   int myid = 0;

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *src_mesh_file = "../data/beam-tri.mesh";
   const char *par_mesh_prefix = "./beam-tri";
   int ser_ref_levels = 0;
   int par_ref_levels = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&src_mesh_file, "-s", "--src",
                  "Source mesh file path.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_mesh_prefix, "-p", "--parallel-mesh-prefix",
                  "File name prefix for parallel mesh files that will be generated.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // Read the (serial) mesh from the given mesh file on all processors.
   Mesh *mesh = new Mesh(src_mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // Refine the serial mesh on all processors to increase the resolution.
   for (int l = 0; l < ser_ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh->UniformRefinement();
   }

   // Write out resulting mesh.
   {
      std::ostringstream pmesh_name;
      pmesh_name << par_mesh_prefix << '.' << setfill('0') << setw(6) << myid;
      ofstream pmesh_ofs(pmesh_name.str().c_str());
      pmesh_ofs.precision(8);
      pmesh->ParPrint(pmesh_ofs);
   }

   delete pmesh;
   MPI_Finalize();

   return 0;
}
