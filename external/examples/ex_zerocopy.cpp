#include "mfem.hpp"
#include <fstream>

using namespace std;
using namespace mfem;

/*
 * Geometry::POINT       Point
 * Geometry::SEGMENT     Segment
 * Geometry::TRIANGLE    Triangle
 * Geometry::SQUARE      Quadrilateral
 * Geometry::CUBE        Hexahedron
 * Geometry::TETRAHEDRON Tetrahedron
*/

int main(int argc, char *argv[])
{

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   printf("Geometry::Point is %d\n", Geometry::POINT);
   printf("Geometry::Segment is %d\n", Geometry::SEGMENT);
   printf("Geometry::Triangle is %d\n", Geometry::TRIANGLE);
   printf("Geometry::Square is %d\n", Geometry::SQUARE);
   printf("Geometry::Cube is %d\n", Geometry::CUBE);
   printf("Geometry::Tetrahedron is %d\n", Geometry::TETRAHEDRON);
   printf("\n");

   Mesh *mesh;
   double vertices[12] = {0,0,1,0,1,1,0,1,2,0,2,1};
   int num_vertices = 6;

   int elem_indices[8] = {1,3,4,2,3,5,6,4};
   int elem_attributes[2] = {0,1};
   int num_elem = 2;

   int bound_indices[12] = {1,2,1,3,3,4,5,6,6,4,4,2};
   int bound_attributes[6] = {1,1,1,1,1,1};
   int num_bound = 6;

   /// The reinit constructor
   /*
   Mesh(double *vertices, int num_vertices,
        int *element_indices, Geometry::Type element_type, 
        int *element_attributes, int num_elements,
        int *boundary_indices, Geometry::Type boundary_type,
        int *boundary_attributes, int num_boundary_elements,
        int dimension, int space_dimension= -1);
   */
   mesh = new Mesh(vertices, num_vertices,
         elem_indices, Geometry::SQUARE, elem_attributes, num_elem,
         bound_indices, Geometry::SEGMENT, bound_attributes, num_bound,
         2);

   // a bunch of debug printing stuff
   int dim = mesh->Dimension();
   printf("Mesh Dimension: %d\n", dim);
   // print the elements (type Element)
   const Element*const* e = mesh->GetElementsArray();
   int ne = mesh->GetNE();
   for (int i = 0; i < ne; i++) { 
      printf("Element %d is type %d attibute %d and has vertices [", 
            i, e[i]->GetType(), e[i]->GetAttribute());
      const int *is = e[i]->GetIndices();
      int ni = e[i]->GetNVertices();
      for (int j = 0; j < ni; j++) {
         printf(j == ni - 1 ? "%d" : "%d,", is[j]);
      }
      printf("]\n");
   }

   // print the boundary (type Element)
   int nb = mesh->GetNBE();
   for (int i = 0; i < nb; i++) { 
      const Element* b = mesh->GetBdrElement(i);
      printf("Boundary Element %d is type %d attibute %d and has vertices [", 
            i, b->GetType(), b->GetAttribute());
      const int *is = b->GetIndices();
      int ni = b->GetNVertices();
      for (int j = 0; j < ni; j++) {
         printf(j == ni - 1 ? "%d" : "%d,", is[j]);
      }
      printf("]\n");
   }

   // print the vertices (type Vertex (double array))
   for (int i = 0; i < mesh->GetNV(); i++) {
      double *v = mesh->GetVertex(i);
      printf("Vertex %d is [", i);
      for (int j = 0; j < dim; j++) {
         printf(j == dim - 1 ? "%f" : "%f,", v[j]);
      }
      printf("]\n");
   }

   // 14. Free the used memory.
   delete mesh;

   return 0;
}
