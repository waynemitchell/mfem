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
      int scale = 100;
      indices_capacity *= scale;
      capacity *= scale;
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

#ifdef MFEM_USE_SIDRE
SidreElementAllocator::SidreElementAllocator(size_t _shape,
      asctoolkit::sidre::DataView *_indices_view,
      asctoolkit::sidre::DataView *_attribute_view)
   : ElementAllocator(), indices_view(_indices_view), 
   attributes_view(_attribute_view), shape(_shape) {
   capacity = 0; 
   indices_capacity = 0;
   scale = 2;
}


int SidreElementAllocator::setsize(size_t _capacity, size_t _shape) {
   if (_shape == 0) _shape = shape;
   // set initial size
   if (indices == NULL && attributes == NULL) {
      if (count == 0 && indices_count == 0) {
         capacity = _capacity;
         indices_capacity = _shape * _capacity;
         attributes_view->allocate(
               asctoolkit::sidre::detail::SidreTT<int>::id,
               capacity);
         attributes = attributes_view->getArray();
         indices_view->allocate(
               asctoolkit::sidre::detail::SidreTT<int>::id,
               indices_capacity);
         indices = indices_view->getArray();
         printf("capacity is now %zu\n", capacity);
         return 1;
      }
      else {
         throw std::runtime_error("Can't have > 0 count with no storage!");
      }
   }
   // this is a resize, we must call update_elements
   else if (indices_capacity < _shape * _capacity ||
         capacity < _capacity) {
      int *old_indices = indices;
      int *old_attributes = attributes;
      capacity = _capacity;
      indices_capacity = _capacity * _shape;
      attributes_view->reallocate(capacity);
      attributes = attributes_view->getArray();
      indices_view->reallocate(indices_capacity);
      indices = indices_view->getArray();
      update_elements(old_indices, old_attributes);
      printf("capacity is now %zu\n", capacity);
      return 1;
   }
   return 0;
}


int_ptr_pair SidreElementAllocator::alloc(size_t _indices_count) {
   // this is super inefficient
   if (!indices && !attributes) {
      setsize(1, _indices_count);
   }
   else if (indices_count + _indices_count > indices_capacity ||
         count == capacity) {
      setsize(capacity * scale);
   }
   count += 1;
   indices_count += _indices_count;
   return int_ptr_pair(indices + indices_count - _indices_count,
         attributes + count - 1);
}



#endif
