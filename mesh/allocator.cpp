#include "allocator.hpp"

int ElementAllocator::update_elements(int *old_indices, int *old_attributes) {
   if (!elements) {
      throw std::runtime_error("This ElementAllocator doesn't have"
                               "an associated array of Elements");
   }
   mfem::Array<mfem::Element*> &elms = *elements;
   // the number of allocated ints we have gone
   // through. a good stopping condition since we know
   // the capacity
   for (size_t i = 0; i < count; i++) {
      if (elms[i] == NULL || elms[i]->IsSelfAlloc()) {
         continue;
      }
      size_t indices_offset = elms[i]->GetIndices() - old_indices;
      size_t attribute_offset = elms[i]->GetAttributePtr() - old_attributes;

      const int *this_indices = elms[i]->GetIndices();
      const int *this_attribute = elms[i]->GetAttributePtr();

      elms[i]->SetIndices(indices + indices_offset);
      elms[i]->SetAttributePtr(attributes + attribute_offset);
      {
         printf("Moving indices for Element %zu at %p"
                " with offset of %zu from %p\n", i, this_indices, 
                indices_offset, old_indices);
         this_indices = elms[i]->GetIndices();
         indices_offset = this_indices - indices;
         printf("New indices location for Element %zu at %p"
                " with offset of %zu from %p\n", i, this_indices,
                indices_offset, indices);

         printf("Moving attribute for Element %zu at %p "
                "with offset of %zu from %p\n", i, this_attribute, 
                attribute_offset, old_attributes);
         this_attribute = elms[i]->GetAttributePtr();
         attribute_offset = this_attribute - attributes;
         printf("New attribute location for Element %zu %p at "
                "with offset of %zu from %p\n", i, this_attribute,
                attribute_offset, attributes);
      }
   }
   return 0;
}

InternalElementAllocator::InternalElementAllocator(size_t _capacity, size_t _indices_count)
   : ElementAllocator() 
{
   capacity = _capacity;
   indices_capacity = capacity * _indices_count;
   indices = (int*)malloc(indices_capacity * sizeof(int));
   if (!indices) {
      indices_capacity = 0;
      throw std::bad_alloc();
   }
   attributes = (int*)malloc(capacity * sizeof(int));
   if (!attributes) {
      capacity = 0;
      throw std::bad_alloc();
   }
   //printf("allocated %zu ints\n", capacity);
}


InternalElementAllocator::~InternalElementAllocator() {
   free(indices);
   free(attributes);
}


int_ptr_pair InternalElementAllocator::alloc(size_t _indices_count) {
   // default return (nil)
   if (!indices || !attributes) {
      return int_ptr_pair(NULL, NULL);
   }
   // realloc if either array is full
   if (indices_count + _indices_count > indices_capacity ||
         count == capacity) {
      indices_capacity *= 2;
      capacity *= 2;
      printf("reallocating indices from size %zu to %zu\n", 
            capacity / 2, capacity);
      int *old_indices = indices;
      int *old_attributes = attributes;
      indices = (int*)realloc(indices, indices_capacity * sizeof(int));
      if (!indices) {
         capacity = 0;
         throw std::bad_alloc();
      }
      attributes = (int*)realloc(attributes, capacity * sizeof(int));
      if (!attributes) {
         capacity = 0;
         throw std::bad_alloc();
      }
      update_elements(old_indices, old_attributes);
   }
   indices_count += _indices_count;
   count++;
   // we have already incremented incides_count to include the
   // new set of indices so add the total index count to the 
   // base pointer and move back 'count' entires in indices
   printf("allocated element %zu with indices %p len %zu and boundary %p\n",
         count - 1, indices + indices_count - _indices_count, _indices_count,
         attributes + count - 1);
   return int_ptr_pair(indices + indices_count - _indices_count, 
         attributes + count - 1);
}


AliasElementAllocator::AliasElementAllocator(int *indices, int *attributes) {
   if (!indices || !attributes) {
      throw std::runtime_error("You cannot reinit a mesh with NULL element or"
            " boundary element pointers");
   }
   this->indices = indices;
   this->attributes = attributes;
}

int_ptr_pair AliasElementAllocator::alloc(size_t _indices_count) {
   indices_count += _indices_count; 
   count++;
   return int_ptr_pair(indices + indices_count - _indices_count, 
         attributes + count - 1); 
};

