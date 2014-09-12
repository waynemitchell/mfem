// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_HASH
#define MFEM_HASH

#include "idgenerator.hpp"


/** A concept for items that should be used in HashTable and be accessible by
 *  hashing two IDs.
 */
struct Hashed2
{
   int id;
   int p1, p2;
   Hashed2* next;

   Hashed2(int id) : id(id) {}
};

/** A concept for items that should be used in HashTable and be accessible by
 *  hashing four IDs.
 */
struct Hashed4
{
   int id;
   int p1, p2, p3; // p4 redundant
   Hashed4* next;

   Hashed4(int id) : id(id) {}
};


/** TODO
 *
 */
template<typename ItemT>
class HashTable
{
public:
   HashTable(int size = 128*1024);
   ~HashTable();

   /// Get an item whose parents are p1, p2... Create it if it doesn't exist.
   ItemT* Get(int p1, int p2);
   ItemT* Get(int p1, int p2, int p3, int p4);

   /// Get an item whose parents are p1, p2... Return NULL if it doesn't exist.
   ItemT* Peek(int p1, int p2);
   ItemT* Peek(int p1, int p2, int p3, int p4);

   // item pointer variants of the above for convenience
   ItemT* Get(ItemT* i1, ItemT* i2) { return Get(i1->id, i2->id); }
   ItemT* Get(ItemT* i1, ItemT* i2, ItemT* i3, ItemT* i4)
      { return Get(i1->id, i2->id, i3->id, i4->id); }

   ItemT* Peek(ItemT* i1, ItemT* i2) { return Peek(i1->id, i2->id); }
   ItemT* Peek(ItemT* i1, ItemT* i2, ItemT* i3, ItemT* i4)
      { return Peek(i1->id, i2->id, i3->id, i4->id); }

   /// Remove the item from the hash table and delete the item itself.
   void Remove(ItemT* item);

   // TODO: iterator

protected:

   ItemT** table;
   int mask;
   int nqueries, ncollisions; // stats

   inline int hash(int p1, int p2)
      { return (984120265*p1 + 125965121*p2) & mask; }

   inline int hash(int p1, int p2, int p3)
      { return (984120265*p1 + 125965121*p2 + 495698413*p3) & mask; }

   ItemT* SearchList(ItemT* item, int p1, int p2);
   ItemT* SearchList(ItemT* item, int p1, int p2, int p3);

   IdGenerator id_gen; ///< id generator for new items
   Array<int> used_bins; ///< bins in 'table' that contain something
};


template<typename ItemT>
HashTable<ItemT>::HashTable(int size)
{
   mask = size-1;
   if (size & mask)
      mfem_error("Hash table size must be a power of two.");

   table = new ItemT*[size];
   memset(table, 0, size * sizeof(ItemT*));

   nqueries = ncollisions = 0;
}

template<typename ItemT>
HashTable<ItemT>::~HashTable()
{
   // TODO!!
}

namespace detail {

inline void sort3(int &a, int &b, int &c)
{
   if (a > b) std::swap(a, b);
   if (a > c) std::swap(a, c);
   if (b > c) std::swap(b, c);
}

inline void sort4(int &a, int &b, int &c, int &d)
{
   if (a > b) std::swap(a, b);
   if (a > c) std::swap(a, c);
   if (a > d) std::swap(a, d);
   sort3(b, c, d);
}

} // detail

template<typename ItemT>
ItemT* HashTable<ItemT>::Peek(int p1, int p2)
{
  if (p1 > p2) std::swap(p1, p2);
  return SearchList(table[hash(p1, p2)], p1, p2);
}

template<typename ItemT>
ItemT* HashTable<ItemT>::Peek(int p1, int p2, int p3, int p4)
{
  detail::sort4(p1, p2, p3, p4);
  return SearchList(table[hash(p1, p2, p3)], p1, p2, p3);
}

template<typename ItemT>
ItemT* HashTable<ItemT>::Get(int p1, int p2)
{
  // search for the item in the hashtable
  if (p1 > p2) std::swap(p1, p2);
  int idx = hash(p1, p2);
  ItemT* node = SearchList(table[idx], p1, p2);
  if (node) return node;

  // not found - create a new one
  ItemT* newitem = new ItemT(id_gen.Get());
  newitem->p1 = p1;
  newitem->p2 = p2;

  // insert into hashtable
  newitem->next = table[idx];
  table[idx] = newitem;

  return newitem;
}

template<typename ItemT>
ItemT* HashTable<ItemT>::Get(int p1, int p2, int p3, int p4)
{
  // search for the item in the hashtable
   detail::sort4(p1, p2, p3, p4);
  int idx = hash(p1, p2, p3);
  ItemT* node = SearchList(table[idx], p1, p2, p3);
  if (node) return node;

  // not found - create a new one
  ItemT* newitem = new ItemT(id_gen.Get());
  newitem->p1 = p1;
  newitem->p2 = p2;

  // insert into hashtable
  newitem->next = table[idx];
  table[idx] = newitem;

  return newitem;
}

template<typename ItemT>
ItemT* HashTable<ItemT>::SearchList(ItemT* item, int p1, int p2)
{
   nqueries++;
   while (item != NULL)
   {
      if (item->p1 == p1 && item->p2 == p2) return item;
      item = (ItemT*) item->next;
      ncollisions++;
   }
   return NULL;
}

template<typename ItemT>
ItemT* HashTable<ItemT>::SearchList(ItemT* item, int p1, int p2, int p3)
{
   nqueries++;
   while (item != NULL)
   {
      if (item->p1 == p1 && item->p2 == p2 && item->p3 == p3) return item;
      item = (ItemT*) item->next;
      ncollisions++;
   }
   return NULL;
}


#endif
