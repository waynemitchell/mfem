//                                MFEM Example 6
//
// Compile with: make ex6
//
// Sample runs:  ex6 square-disc.mesh2d
//               ex6 star.mesh2d
//
// Description: This example code performs a simple mesh smoothing based on a
//              topologically defined "mesh Laplacian" matrix.
//
//              The example highlights meshes with curved elements, the
//              assembling of a custom finite element matrix, the use of vector
//              finite element spaces, the definition of different spaces and
//              grid functions on the same mesh, and the setting of values by
//              iterating over the interior and the boundary elements.

#include <fstream>
#include "mfem.hpp"

// 1. Define the bilinear form corresponding to a mesh Laplacian operator. This
//    will be used to assemble the global mesh Laplacian matrix based on the
//    local matrix provided in the AssembleElementMatrix method. More examples
//    of bilinear integrators can be found in ../fem/bilininteg.hpp.
class VectorMeshLaplacianIntegrator : public BilinearFormIntegrator
{
public:
   VectorMeshLaplacianIntegrator() {};
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   ~VectorMeshLaplacianIntegrator() {};
};

// 2. Implement the local stiffness matrix of the mesh Laplacian. This is a
//    block-diagonal matrix with each block having a unit diagonal and constant
//    negative off-diagonal entries, such that the row sums are zero.
void VectorMeshLaplacianIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   int dim = el.GetDim(); // space dimension
   int dof = el.GetDof(); // number of element degrees of freedom

   elmat.SetSize(dim*dof); // block-diagonal element matrix

   for (int d = 0; d < dim; d++)
      for (int k = 0; k < dof; k++)
         for (int l = 0; l < dof; l++)
            if (k==l)
               elmat (dof*d+k, dof*d+l) = 1.0;
            else
               elmat (dof*d+k, dof*d+l) = -1.0/(dof-1);
}


int main (int argc, char *argv[])
{
   Mesh *mesh;
   char vishost[] = "localhost";
   int  visport   = 19916;
   int  ans;

   if (argc == 1)
   {
      cout << "Usage: ex6 <mesh_file>" << endl;
      return 1;
   }

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral or hexahedral elements with the same code.
   ifstream imesh(argv[1]);
   if (!imesh)
   {
      cerr << "can not open mesh file: " << argv[1] << endl;
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();

   int dim = mesh->Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 1000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
         mesh->UniformRefinement();
   }

   // 5. Define a finite element space on the mesh. Here we use vector finite
   //    elements which are tensor products of quadratic finite elements. The
   //    dimensionality of the vector finite element space is specified by the
   //    last parameter of the FiniteElementSpace constructor.
   FiniteElementCollection *fec = new QuadraticFECollection;
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, dim);

   // 6. Make the mesh curved based on the above finite element space. This
   //    means that we define the mesh elements through a fespace-based
   //    transformation of the reference element.
   mesh->SetNodalFESpace(fespace);

   // 7. Set up the right-hand side vector b. In this case we do not need to use
   //    a LinearForm object because b=0.
   Vector b(fespace->GetVSize());
   b = 0.0;

   // 8. Get the mesh nodes (vertices and other quadratic degrees of freedom in
   //    the finite element space) as a finite element grid function in fespace.
   GridFunction *x;
   x = mesh->GetNodes();

   // 9. Define a vector representing the minimal local mesh size in the mesh
   //    nodes. We index the nodes using the scalar version of the degrees of
   //    freedom in fespace.
   Vector h0(fespace->GetNDofs());
   h0 = numeric_limits<double>::infinity();
   {
      Array<int> dofs;
      // loop over the mesh elements
      for (int i = 0; i < fespace->GetNE(); i++)
      {
         // get the local scalar element degrees of freedom in dofs
         fespace->GetElementDofs(i, dofs);
         // adjust the value of h0 in dofs based on the local mesh size
         for (int j = 0; j < dofs.Size(); j++)
            h0(dofs[j]) = min(h0(dofs[j]), mesh->GetElementSize(i));
      }
   }

   // 10. Add a random perturbation of the nodes in the interior of the domain.
   //     We define a random grid function of fespace and make sure that it is
   //     zero on the boundary and its values are locally of the order of h0.
   //     The latter is based on the DofToVDof() method which maps the scalar to
   //     the vector degrees of freedom in fespace.
   GridFunction rdm(fespace);
   double jitter = 0.25; // perturbation scaling factor
   rdm.Randomize();
   rdm -= 0.5; // shift to random values in [-0.5,0.5]
   rdm *= jitter;
   {
      // scale the random values to be of order of the local mesh size
      for (int i = 0; i < fespace->GetNDofs(); i++)
         for (int d = 0; d < dim; d++)
            rdm(fespace->DofToVDof(i,d)) *= h0(i);

      Array<int> vdofs;
      // loop over the boundary elements
      for (int i = 0; i < fespace->GetNBE(); i++)
      {
         // get the vector degrees of freedom in the boundary element
         fespace->GetBdrElementVDofs(i, vdofs);
         // set the boundary values to zero
         for (int j = 0; j < vdofs.Size(); j++)
            rdm(vdofs[j]) = 0.0;
      }
   }
   *x -= rdm;

   // 11. Save the perturbed mesh to a file. This output can be viewed later
   //     using GLVis: "glvis -m perturbed.mesh".
   {
      ofstream mesh_ofs("perturbed.mesh");
      mesh->Print(mesh_ofs);
   }

   // 12. (Optional) Send the initially perturbed mesh with the vector field
   //     representing the displacements to the original mesh to GLVis.
   cout << "Visualize the initial random perturbation? [0/1] --> ";
   cin >> ans;
   if (ans)
   {
      osockstream sol_sock(visport, vishost);
      if (dim == 2)
         sol_sock << "vfem2d_gf_data\n";
      else
         sol_sock << "vfem3d_gf_data\n";
      mesh->Print(sol_sock);
      rdm.Save(sol_sock);
      sol_sock.send();
   }

   // 13. Set up the bilinear form a(.,.) corresponding to the mesh Laplacian
   //     operator. The imposed boundary conditions mean that the nodes on the
   //     boundary of the domain will not be relaxed. After assembly and
   //     finalizing we extract the corresponding sparse matrix A.
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new VectorMeshLaplacianIntegrator);
   a->Assemble();
   Array<int> ess_bdr(mesh->bdr_attributes.Size());
   ess_bdr = 1;
   a->EliminateEssentialBC(ess_bdr, *x, b);
   a->Finalize();
   SparseMatrix &A = a->SpMat();

   // 14. Simple mesh smoothing can be perform by relaxing the node coordinate
   //     grid function x with the matrix A and right-hand side b. This process
   //     converges to the solution of Ax=b, which we solve below with PCG. Note
   //     that the computed x is the A-harmonic extension of its boundary values
   //     (the coordinates of the boundary vertices). Furthermore, note that
   //     changing x automatically changes the shapes of the elements in the
   //     mesh. The vector field that gives the displacements to the perturbed
   //     positions is saved in the grid function x0.
   GridFunction x0(fespace);
   GSSmoother M(A);
   x0 = *x;
   PCG(A, M, b, *x, 1, 200, 1e-12, 0.0);
   x0 -= *x;

   // 15. Save the smoothed mesh to a file. This output can be viewed later using
   //     GLVis: "glvis -m smoothed.mesh".
   {
      ofstream mesh_ofs("smoothed.mesh");
      mesh->Print(mesh_ofs);
   }

   // 16. (Optional) Send the relaxed mesh with the vector field representing
   //     the displacements to the perturbed mesh by socket to a GLVis server.
   cout << "Visualize the smoothed mesh? [0/1] --> ";
   cin >> ans;
   if (ans)
   {
      osockstream sol_sock(visport, vishost);
      if (dim == 2)
         sol_sock << "vfem2d_gf_data\n";
      else
         sol_sock << "vfem3d_gf_data\n";
      mesh->Print(sol_sock);
      x0.Save(sol_sock);
      sol_sock.send();
   }

   // 17. Free the used memory.
   delete a;
   delete fespace;
   delete fec;
   delete mesh;
}
