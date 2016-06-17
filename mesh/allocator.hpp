#ifndef MFEM_ALLOCATOR
#define MFEM_ALLOCATOR

#include <stdexcept>
#include "element.hpp"

// packed type for returning a index and attribute pointer
typedef std::pair<int*, int*> int_ptr_pair; 

// Element allocator class.  Holds an array of ints and a pointer
// to an mfem::Array of mfem::Element* so that we can update their
// `indices` pointer on a reallocation
class ElementAllocator {
   protected:
      mfem::Array<mfem::Element*> *elements;
      int *indices;
      int *attributes;
      size_t indices_count;
      size_t count;
      // needs to be called on a realloc (if the
      // address of data moves) otherwise the Elements
      // in elements are invalid (the indices pointer
      // is not valid)
      int update_elements(int *old_indices, int *old_attributes);
   public:
      ElementAllocator() { elements = NULL; indices = NULL; 
         attributes = NULL; indices_count = 0; count = 0; };
      virtual ~ElementAllocator() {};

      // a nicety so we can call the object (functor fun)
      inline int_ptr_pair operator()(size_t _indices_count) { return alloc(_indices_count); };
      inline virtual int_ptr_pair alloc(size_t) { return int_ptr_pair(NULL, NULL); };

      inline int *get_indices() { return indices; };
      inline int *get_attributes() { return attributes; };
      inline int get_indices_count() { return indices_count; };
      inline int get_count() { return count; };

      inline const mfem::Array<mfem::Element*>* get_elements() const
         { return elements; };
      inline void set_elements(mfem::Array<mfem::Element*> *_elements) 
         { elements = _elements; };
};


// extension of Element_allocator that holds it's own data.
//  It will realloc on the fly
class InternalElementAllocator : public ElementAllocator {
   protected:
      // maybe this should be an Element_allocator entry?
      size_t indices_capacity;
      size_t capacity;
   public:
      InternalElementAllocator(size_t _capacity, size_t element_size);
      ~InternalElementAllocator();
      virtual int_ptr_pair alloc(size_t _indices_count);
};


class AliasElementAllocator : public ElementAllocator {
   public:
      AliasElementAllocator(int *_indices, int *_attributes);
      ~AliasElementAllocator() {};
      int_ptr_pair alloc(size_t _indices_count);
};

#endif
