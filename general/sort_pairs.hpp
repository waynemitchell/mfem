// Copyright (c) 2010,  Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// This file is part of the MFEM library.  See file COPYRIGHT for details.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_SORT_PAIRS
#define MFEM_SORT_PAIRS

/// A pair of objects
template <class A, class B>
class Pair
{
public:
   A one;
   B two;
};

/// Compare the first element of the pairs
template <class A, class B>
int ComparePairs (const void *_p, const void *_q);

/// Sort with respect to the first element
template <class A, class B>
void SortPairs (Pair<A, B> *pairs, int size);

#endif
