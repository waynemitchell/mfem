#ifndef MFEM_ALLOCATOR
#define MFEM_ALLOCATOR

#include <stdexcept>
#include "element.hpp"

#ifdef MFEM_USE_SIDRE
#include "sidre/sidre.hpp"
#endif

class Allocator {
   protected:
      size_t count;
      size_t capacity;
      void *data;

   public:
      Allocator() { count = 0; capacity = 0; };
      virtual ~Allocator() {};
      virtual void *operator()(size_t _count) { return alloc(_count); }
      virtual void *getdata() { return data; }
      size_t getcapacity() { return capacity; }
      virtual void *alloc(size_t _count) { return NULL; }
      virtual int setsize(size_t _capacity) { return 0; }
};

#ifdef MFEM_USE_SIDRE
template <class T>
class SidreAllocator : public Allocator {
   private:
      asctoolkit::sidre::DataView *view;
      T *data;
      size_t scale;
   public:
      SidreAllocator(asctoolkit::sidre::DataView *_view,
            size_t _capacity = 0, size_t _scale = 2) 
         : view(_view), scale(_scale) {
            capacity = 0;
            setsize(_capacity);
         }
      ~SidreAllocator() {};
      void *alloc(size_t _count) {
         if (count + _count > capacity ) {
            setsize(count + _count);
         }
         count += _count;
         return data + count - _count;
      }
      int setsize(size_t _capacity) {
         if (_capacity == 0 || _capacity < capacity) {
            return 1;
         }
         if (capacity == 0) {
            view->allocate(
                  asctoolkit::sidre::detail::SidreTT<T>::id,
                  _capacity);
         }
         else {
            view->reallocate(_capacity);
         }
         data = view->getArray();
         capacity = _capacity;
         return 0;
      }

};
#endif


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
      inline int_ptr_pair operator()(size_t _indices_count) 
      { return alloc(_indices_count); };
      inline virtual int_ptr_pair alloc(size_t) 
      { return int_ptr_pair(NULL, NULL); };

      inline int *get_indices() { return indices; };
      inline int *get_attributes() { return attributes; };
      inline int get_indices_count() { return indices_count; };
      inline int get_count() { return count; };
      virtual int setsize(size_t _capacity, size_t _shape = 0) { return -1; };

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

#ifdef MFEM_USE_SIDRE
class SidreElementAllocator : public ElementAllocator {
   private:
      asctoolkit::sidre::DataView *indices_view;
      asctoolkit::sidre::DataView *attributes_view;
      size_t shape;
      size_t capacity;
      size_t indices_capacity;
      size_t scale;

   public:
      SidreElementAllocator(size_t _shape,
            asctoolkit::sidre::DataView *indices_view,
            asctoolkit::sidre::DataView *attribute_view);
      ~SidreElementAllocator() {};
      int_ptr_pair alloc(size_t _indices_count);
      int setsize(size_t _capacity, size_t _shape = 0);
};
#endif

#endif
