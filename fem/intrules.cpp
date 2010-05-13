// Copyright (c) 2010,  Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// This file is part of the MFEM library.  See file COPYRIGHT for details.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

// Implementation of IntegrationRule(s) classes

#include <math.h>
#include "fem.hpp"

IntegrationRule::IntegrationRule (int NP)
{
   NPoints   = NP;
   IntPoints = new IntegrationPoint[NP];
}

IntegrationRule::IntegrationRule (IntegrationRule &irx, IntegrationRule &iry)
{
   int i, j, nx, ny;

   nx = irx.GetNPoints();
   ny = iry.GetNPoints();
   NPoints = nx * ny;
   IntPoints = new IntegrationPoint[NPoints];

   for (j = 0; j < ny; j++)
   {
      IntegrationPoint &ipy = iry.IntPoint(j);
      for (i = 0; i < nx; i++)
      {
         IntegrationPoint &ipx = irx.IntPoint(i);
         IntegrationPoint &ip  = IntPoints[j*nx+i];

         ip.x = ipx.x;
         ip.y = ipy.x;
         ip.weight = ipx.weight * ipy.weight;
      }
   }
}

void IntegrationRule::GaussianRule()
{
   int n = NPoints;
   int m = (n+1)/2;
   int i, j;
   double p1, p2, p3;
   double pp, z, z1;
   for (i = 1; i <= m; i++)
   {
      z = cos ( M_PI * (i - 0.25) / (n + 0.5));

      while(1)
      {
         p1 = 1;
         p2 = 0;
         for (j = 1; j <= n; j++)
         {
            p3 = p2;
            p2 = p1;
            p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
         }
         // p1 is Legendre polynomial

         pp = n * (z*p1-p2) / (z*z - 1);
         z1 = z;
         z = z1-p1/pp;

         if (fabs (z - z1) < 1e-14) break;
      }

      IntPoints[i-1].x  = 0.5 * (1 - z);
      IntPoints[n-i].x  = 0.5 * (1 + z);
      IntPoints[i-1].weight = IntPoints[n-i].weight =
         1.0 / ( (1  - z * z) * pp * pp);
   }
}

void IntegrationRule::UniformRule()
{
   int i;
   double h;

   h = 1.0 / (NPoints - 1);
   for (i = 0; i < NPoints; i++)
   {
      IntPoints[i].x = double(i) / (NPoints - 1);
      IntPoints[i].weight = h;
   }
   IntPoints[0].weight = 0.5 * h;
   IntPoints[NPoints-1].weight = 0.5 * h;
}

IntegrationRule::~IntegrationRule ()
{
   delete [] IntPoints;
}



IntegrationRules IntRules(0);

IntegrationRules RefinedIntRules(1);

IntegrationRules::IntegrationRules (int refined)
{
   if (refined < 0) { own_rules = 0; return; }

   own_rules = 1;
   PointIntegrationRules();
   SegmentIntegrationRules(refined);
   TriangleIntegrationRules(0);
   SquareIntegrationRules();
   TetrahedronIntegrationRules(0);
   CubeIntegrationRules();
}

const IntegrationRule & IntegrationRules::Get (int GeomType, int Order)
{
   switch (GeomType)
   {
   case Geometry::POINT:    return *PointIntRules[Order];
   case Geometry::SEGMENT:  return *SegmentIntRules[Order];
   case Geometry::TRIANGLE: return *TriangleIntRules[Order];
   case Geometry::SQUARE:   return *SquareIntRules[Order];
   case Geometry::TETRAHEDRON: return *TetrahedronIntRules[Order];
   case Geometry::CUBE:        return *CubeIntRules[Order];
   default:
#ifdef MFEM_DEBUG
      mfem_error ("IntegrationRules::Get (...)");
#endif
      ;
   }
   return *TriangleIntRules[Order];
}

void IntegrationRules::Set (int GeomType, int Order, IntegrationRule &IntRule)
{
   Array<IntegrationRule *> *ir_array;

   switch (GeomType)
   {
   case Geometry::POINT:       ir_array = &PointIntRules; break;
   case Geometry::SEGMENT:     ir_array = &SegmentIntRules; break;
   case Geometry::TRIANGLE:    ir_array = &TriangleIntRules; break;
   case Geometry::SQUARE:      ir_array = &SquareIntRules; break;
   case Geometry::TETRAHEDRON: ir_array = &TetrahedronIntRules; break;
   case Geometry::CUBE:        ir_array = &CubeIntRules; break;
   default:
#ifdef MFEM_DEBUG
      mfem_error ("IntegrationRules::Set (...)");
#endif
      ;
   }

   if (ir_array -> Size() <= Order)
   {
      int i = ir_array -> Size();

      ir_array -> SetSize (Order + 1);

      for ( ; i < Order; i++)
         (*ir_array)[i] = NULL;
   }

   (*ir_array)[Order] = &IntRule;
}

IntegrationRules::~IntegrationRules ()
{
   int i;

   if (!own_rules) return;

   for (i = 0; i < PointIntRules.Size(); i++)
      if (PointIntRules[i] != NULL)
         delete PointIntRules[i];

   for (i = 0; i < SegmentIntRules.Size(); i++)
      if (SegmentIntRules[i] != NULL)
         delete SegmentIntRules[i];

   for (i = 0; i < TriangleIntRules.Size(); i++)
      if (TriangleIntRules[i] != NULL)
         delete TriangleIntRules[i];

   for (i = 0; i < SquareIntRules.Size(); i++)
      if (SquareIntRules[i] != NULL)
         delete SquareIntRules[i];

   for (i = 0; i < TetrahedronIntRules.Size(); i++)
      if (TetrahedronIntRules[i] != NULL)
         delete TetrahedronIntRules[i];

   for (i = 0; i < CubeIntRules.Size(); i++)
      if (CubeIntRules[i] != NULL)
         delete CubeIntRules[i];
}

// Integration rules for a point
void IntegrationRules::PointIntegrationRules()
{
   PointIntRules.SetSize(2);

   PointIntRules[0] = new IntegrationRule (1);
   PointIntRules[0] -> IntPoint(0).x = .0;
   PointIntRules[0] -> IntPoint(0).weight = 1.;

   PointIntRules[1] = new IntegrationRule (1);
   PointIntRules[1] -> IntPoint(0).x = .0;
   PointIntRules[1] -> IntPoint(0).weight = 1.;
}

// Integration rules for line segment [0,1]
void IntegrationRules::SegmentIntegrationRules(int refined)
{
   int i, j;

   SegmentIntRules.SetSize(32);

   if (refined) {
      int n;
      IntegrationRule * tmp;
      for (i = 0; i < SegmentIntRules.Size(); i++) {
         n = i/2+1; tmp = new IntegrationRule(n); tmp -> GaussianRule();
         SegmentIntRules[i] = new IntegrationRule (2*n);
         for (j = 0; j < n; j++) {
            SegmentIntRules[i]->IntPoint(j).x = tmp->IntPoint(j).x/2.0;
            SegmentIntRules[i]->IntPoint(j).weight = tmp->IntPoint(j).weight/2.0;
            SegmentIntRules[i]->IntPoint(j+n).x = 0.5 + tmp->IntPoint(j).x/2.0;
            SegmentIntRules[i]->IntPoint(j+n).weight = tmp->IntPoint(j).weight/2.0;
         }
         delete tmp;
      }
      return;
   }

   for (i = 0; i < SegmentIntRules.Size(); i++)
      SegmentIntRules[i] = NULL;

   // 1 point - 1 degree
   SegmentIntRules[0] = new IntegrationRule (1);

   SegmentIntRules[0] -> IntPoint(0).x = .5;
   SegmentIntRules[0] -> IntPoint(0).weight = 1.;

   // 1 point - 1 degree
   SegmentIntRules[1] = new IntegrationRule (1);

   SegmentIntRules[1] -> IntPoint(0).x = .5;
   SegmentIntRules[1] -> IntPoint(0).weight = 1.;

   // 2 point - 3 degree
   SegmentIntRules[2] = new IntegrationRule (2);

   SegmentIntRules[2] -> IntPoint(0).x = 0.211324865405187;
   SegmentIntRules[2] -> IntPoint(0).weight = .5;
   SegmentIntRules[2] -> IntPoint(1).x = 0.788675134594812;
   SegmentIntRules[2] -> IntPoint(1).weight = .5;

   // 2 point - 3 degree
   SegmentIntRules[3] = new IntegrationRule (2);

   SegmentIntRules[3] -> IntPoint(0).x = 0.211324865405187;
   SegmentIntRules[3] -> IntPoint(0).weight = .5;
   SegmentIntRules[3] -> IntPoint(1).x = 0.788675134594812;
   SegmentIntRules[3] -> IntPoint(1).weight = .5;

   // 3 point - 5 degree
   SegmentIntRules[4] = new IntegrationRule (3);
   SegmentIntRules[4] -> GaussianRule();

   SegmentIntRules[4] -> IntPoint(0).x = 0.11270166537925831148;
   SegmentIntRules[4] -> IntPoint(0).weight = 0.2777777777777777777777778;
   SegmentIntRules[4] -> IntPoint(1).x = 0.5;
   SegmentIntRules[4] -> IntPoint(1).weight = 0.4444444444444444444444444;
   SegmentIntRules[4] -> IntPoint(2).x = 0.88729833462074168852;
   SegmentIntRules[4] -> IntPoint(2).weight = 0.2777777777777777777777778;

   // 3 point - 5 degree
   SegmentIntRules[5] = new IntegrationRule (3);

   SegmentIntRules[5] -> IntPoint(0).x = 0.11270166537925831148;
   SegmentIntRules[5] -> IntPoint(0).weight = 0.2777777777777777777777778;
   SegmentIntRules[5] -> IntPoint(1).x = 0.5;
   SegmentIntRules[5] -> IntPoint(1).weight = 0.4444444444444444444444444;
   SegmentIntRules[5] -> IntPoint(2).x = 0.88729833462074168852;
   SegmentIntRules[5] -> IntPoint(2).weight = 0.2777777777777777777777778;

   /*
   // 4 point - 7 degree
   SegmentIntRules[6] = new IntegrationRule (4);

   SegmentIntRules[6] -> IntPoint(0).x = 0.069431844202973;
   SegmentIntRules[6] -> IntPoint(0).weight = .1739274226587269286;
   SegmentIntRules[6] -> IntPoint(1).x = 0.330009478207572;
   SegmentIntRules[6] -> IntPoint(1).weight = .3260725774312730713;
   SegmentIntRules[6] -> IntPoint(2).x = 0.669990521792428;
   SegmentIntRules[6] -> IntPoint(2).weight = .3260725774312730713;
   SegmentIntRules[6] -> IntPoint(3).x = 0.930568155797026;
   SegmentIntRules[6] -> IntPoint(3).weight = .1739274226587269286;

   // 4 point - 7 degree
   SegmentIntRules[7] = new IntegrationRule (4);

   SegmentIntRules[7] -> IntPoint(0).x = 0.069431844202973;
   SegmentIntRules[7] -> IntPoint(0).weight = .1739274226587269286;
   SegmentIntRules[7] -> IntPoint(1).x = 0.330009478207572;
   SegmentIntRules[7] -> IntPoint(1).weight = .3260725774312730713;
   SegmentIntRules[7] -> IntPoint(2).x = 0.669990521792428;
   SegmentIntRules[7] -> IntPoint(2).weight = .3260725774312730713;
   SegmentIntRules[7] -> IntPoint(3).x = 0.930568155797026;
   SegmentIntRules[7] -> IntPoint(3).weight = .1739274226587269286;

   // 5 point - 9 degree
   SegmentIntRules[8] = new IntegrationRule (5);

   SegmentIntRules[8] -> IntPoint(0).x = 0.046910077030668;
   SegmentIntRules[8] -> IntPoint(0).weight = .1655506920379504390;
   SegmentIntRules[8] -> IntPoint(1).x = 0.230765344947158;
   SegmentIntRules[8] -> IntPoint(1).weight = .3344378060412287839;
   SegmentIntRules[8] -> IntPoint(2).x = 0.5;
   SegmentIntRules[8] -> IntPoint(2).weight = .0000230038416415541;
   SegmentIntRules[8] -> IntPoint(3).x = 0.769234655052842;
   SegmentIntRules[8] -> IntPoint(3).weight = .3344378060412287839;
   SegmentIntRules[8] -> IntPoint(4).x = 0.953089922969332;
   SegmentIntRules[8] -> IntPoint(4).weight = .1655506920379504390;

   // 5 point - 9 degree
   SegmentIntRules[9] = new IntegrationRule (5);

   SegmentIntRules[9] -> IntPoint(0).x = 0.046910077030668;
   SegmentIntRules[9] -> IntPoint(0).weight = .1655506920379504390;
   SegmentIntRules[9] -> IntPoint(1).x = 0.230765344947158;
   SegmentIntRules[9] -> IntPoint(1).weight = .3344378060412287839;
   SegmentIntRules[9] -> IntPoint(2).x = 0.5;
   SegmentIntRules[9] -> IntPoint(2).weight = .0000230038416415541;
   SegmentIntRules[9] -> IntPoint(3).x = 0.769234655052842;
   SegmentIntRules[9] -> IntPoint(3).weight = .3344378060412287839;
   SegmentIntRules[9] -> IntPoint(4).x = 0.953089922969332;
   SegmentIntRules[9] -> IntPoint(4).weight = .1655506920379504390;

   // 6 point - 11 degree
   SegmentIntRules[10] = new IntegrationRule (6);

   SegmentIntRules[10] -> IntPoint(0).x = 0.033765242898424;
   SegmentIntRules[10] -> IntPoint(0).weight = .0856622461895851724;
   SegmentIntRules[10] -> IntPoint(1).x = 0.169395306766868;
   SegmentIntRules[10] -> IntPoint(1).weight = .1803807865240693038;
   SegmentIntRules[10] -> IntPoint(2).x = 0.380690406958402;
   SegmentIntRules[10] -> IntPoint(2).weight = .2339569672863455237;
   SegmentIntRules[10] -> IntPoint(3).x = 0.619309593041598;
   SegmentIntRules[10] -> IntPoint(3).weight = .2339569672863455237;
   SegmentIntRules[10] -> IntPoint(4).x = 0.830604693233132;
   SegmentIntRules[10] -> IntPoint(4).weight = .1803807865240693038;
   SegmentIntRules[10] -> IntPoint(5).x = 0.966234757101576;
   SegmentIntRules[10] -> IntPoint(5).weight = .0856622461895851724;

   // 6 point - 11 degree
   SegmentIntRules[11] = new IntegrationRule (6);

   SegmentIntRules[11] -> IntPoint(0).x = 0.033765242898424;
   SegmentIntRules[11] -> IntPoint(0).weight = .0856622461895851724;
   SegmentIntRules[11] -> IntPoint(1).x = 0.169395306766868;
   SegmentIntRules[11] -> IntPoint(1).weight = .1803807865240693038;
   SegmentIntRules[11] -> IntPoint(2).x = 0.380690406958402;
   SegmentIntRules[11] -> IntPoint(2).weight = .2339569672863455237;
   SegmentIntRules[11] -> IntPoint(3).x = 0.619309593041598;
   SegmentIntRules[11] -> IntPoint(3).weight = .2339569672863455237;
   SegmentIntRules[11] -> IntPoint(4).x = 0.830604693233132;
   SegmentIntRules[11] -> IntPoint(4).weight = .1803807865240693038;
   SegmentIntRules[11] -> IntPoint(5).x = 0.966234757101576;
   SegmentIntRules[11] -> IntPoint(5).weight = .0856622461895851724;

   // 7 point - 13 degree
   SegmentIntRules[12] = new IntegrationRule (7);

   SegmentIntRules[12] -> IntPoint(0).x = 0.025446043828621;
   SegmentIntRules[12] -> IntPoint(0).weight = .0818467915961606551;
   SegmentIntRules[12] -> IntPoint(1).x = 0.129234407200303;
   SegmentIntRules[12] -> IntPoint(1).weight = .1768003619484993849;
   SegmentIntRules[12] -> IntPoint(2).x = 0.297077424311301;
   SegmentIntRules[12] -> IntPoint(2).weight = .2413528419051118830;
   SegmentIntRules[12] -> IntPoint(3).x = 0.5;
   SegmentIntRules[12] -> IntPoint(3).weight = 9.100456214e-9;
   SegmentIntRules[12] -> IntPoint(4).x = 0.702922575688699;
   SegmentIntRules[12] -> IntPoint(4).weight = .2413528419051118830;
   SegmentIntRules[12] -> IntPoint(5).x = 0.870765592799697;
   SegmentIntRules[12] -> IntPoint(5).weight = .1768003619484993849;
   SegmentIntRules[12] -> IntPoint(6).x = 0.974553956171379;
   SegmentIntRules[12] -> IntPoint(6).weight = .0818467915961606551;

   // 7 point - 13 degree
   SegmentIntRules[13] = new IntegrationRule (7);

   SegmentIntRules[13] -> IntPoint(0).x = 0.025446043828621;
   SegmentIntRules[13] -> IntPoint(0).weight = .0818467915961606551;
   SegmentIntRules[13] -> IntPoint(1).x = 0.129234407200303;
   SegmentIntRules[13] -> IntPoint(1).weight = .1768003619484993849;
   SegmentIntRules[13] -> IntPoint(2).x = 0.297077424311301;
   SegmentIntRules[13] -> IntPoint(2).weight = .2413528419051118830;
   SegmentIntRules[13] -> IntPoint(3).x = 0.5;
   SegmentIntRules[13] -> IntPoint(3).weight = 9.100456214e-9;
   SegmentIntRules[13] -> IntPoint(4).x = 0.702922575688699;
   SegmentIntRules[13] -> IntPoint(4).weight = .2413528419051118830;
   SegmentIntRules[13] -> IntPoint(5).x = 0.870765592799697;
   SegmentIntRules[13] -> IntPoint(5).weight = .1768003619484993849;
   SegmentIntRules[13] -> IntPoint(6).x = 0.974553956171379;
   SegmentIntRules[13] -> IntPoint(6).weight = .0818467915961606551;
   */

   for (i = 6; i < SegmentIntRules.Size(); i++)
   {
      SegmentIntRules[i] = new IntegrationRule (i/2+1);
      SegmentIntRules[i] -> GaussianRule();
   }
}

// Integration rules for reference triangle {[0,0],[1,0],[0,1]}
void IntegrationRules::TriangleIntegrationRules(int refined)
{
   TriangleIntRules.SetSize(8);

   if (refined)
      mfem_error ("Refined TriangleIntegrationRules are not implemented!");

   for (int i = 0; i < TriangleIntRules.Size(); i++)
      TriangleIntRules[i] = NULL;

   // 1 point - 0 degree
   TriangleIntRules[0] = new IntegrationRule (1);

   TriangleIntRules[0] -> IntPoint(0).x = 0.33333333333333333333;
   TriangleIntRules[0] -> IntPoint(0).y = 0.33333333333333333333;
   TriangleIntRules[0] -> IntPoint(0).weight = 0.5;

   /*
   // 3 point - 1 degree (vertices)
   TriangleIntRules[1] = new IntegrationRule (3);

   TriangleIntRules[1] -> IntPoint(0).x      = 0.;
   TriangleIntRules[1] -> IntPoint(0).y      = 0.;
   TriangleIntRules[1] -> IntPoint(0).weight = 0.16666666666667;

   TriangleIntRules[1] -> IntPoint(1).x      = 1.;
   TriangleIntRules[1] -> IntPoint(1).y      = 0.;
   TriangleIntRules[1] -> IntPoint(1).weight = 0.16666666666667;

   TriangleIntRules[1] -> IntPoint(2).x      = 0.;
   TriangleIntRules[1] -> IntPoint(2).y      = 1.;
   TriangleIntRules[1] -> IntPoint(2).weight = 0.16666666666667;
   */

   // 1 point - 1 degree
   TriangleIntRules[1] = new IntegrationRule (1);

   TriangleIntRules[1] -> IntPoint(0).x = 0.33333333333333333333;
   TriangleIntRules[1] -> IntPoint(0).y = 0.33333333333333333333;
   TriangleIntRules[1] -> IntPoint(0).weight = 0.5;

   // 3 point - 2 degree (midpoints)
   TriangleIntRules[2] = new IntegrationRule (3);

   TriangleIntRules[2] -> IntPoint(0).x      = 0.5;
   TriangleIntRules[2] -> IntPoint(0).y      = 0;
   TriangleIntRules[2] -> IntPoint(0).weight = 0.16666666666667;

   TriangleIntRules[2] -> IntPoint(1).x      = 0.5;
   TriangleIntRules[2] -> IntPoint(1).y      = 0.5;
   TriangleIntRules[2] -> IntPoint(1).weight = 0.16666666666667;

   TriangleIntRules[2] -> IntPoint(2).x      = 0;
   TriangleIntRules[2] -> IntPoint(2).y      = 0.5;
   TriangleIntRules[2] -> IntPoint(2).weight = 0.16666666666667;

   // 4 point - 3 degree (has one negative weight)
   TriangleIntRules[3] = new IntegrationRule (4);

   TriangleIntRules[3] -> IntPoint(0).x      = 0.33333333333333;
   TriangleIntRules[3] -> IntPoint(0).y      = 0.33333333333333;
   TriangleIntRules[3] -> IntPoint(0).weight = -0.28125;

   TriangleIntRules[3] -> IntPoint(1).x      = 0.2;
   TriangleIntRules[3] -> IntPoint(1).y      = 0.2;
   TriangleIntRules[3] -> IntPoint(1).weight = 0.26041666666665;

   TriangleIntRules[3] -> IntPoint(2).x      = 0.6;
   TriangleIntRules[3] -> IntPoint(2).y      = 0.2;
   TriangleIntRules[3] -> IntPoint(2).weight = 0.26041666666665;

   TriangleIntRules[3] -> IntPoint(3).x      = 0.2;
   TriangleIntRules[3] -> IntPoint(3).y      = 0.6;
   TriangleIntRules[3] -> IntPoint(3).weight = 0.26041666666665;

   // 6 point - 4 degree
   TriangleIntRules[4] = new IntegrationRule (6);

   TriangleIntRules[4] -> IntPoint(0).x      = 0.091576213509771;
   TriangleIntRules[4] -> IntPoint(0).y      = 0.091576213509771;
   TriangleIntRules[4] -> IntPoint(0).weight = 0.054975871827661;

   TriangleIntRules[4] -> IntPoint(1).x      = 0.091576213509771;
   TriangleIntRules[4] -> IntPoint(1).y      = 0.816847572980459;
   TriangleIntRules[4] -> IntPoint(1).weight = 0.054975871827661;

   TriangleIntRules[4] -> IntPoint(2).x      = 0.816847572980459;
   TriangleIntRules[4] -> IntPoint(2).y      = 0.091576213509771;
   TriangleIntRules[4] -> IntPoint(2).weight = 0.054975871827661;

   TriangleIntRules[4] -> IntPoint(3).x      = 0.445948490915965;
   TriangleIntRules[4] -> IntPoint(3).y      = 0.445948490915965;
   TriangleIntRules[4] -> IntPoint(3).weight = 0.1116907948390055;

   TriangleIntRules[4] -> IntPoint(4).x      = 0.445948490915965;
   TriangleIntRules[4] -> IntPoint(4).y      = 0.108103018168070;
   TriangleIntRules[4] -> IntPoint(4).weight = 0.1116907948390055;

   TriangleIntRules[4] -> IntPoint(5).x      = 0.108103018168070;
   TriangleIntRules[4] -> IntPoint(5).y      = 0.445948490915965;
   TriangleIntRules[4] -> IntPoint(5).weight = 0.1116907948390055;

   // 7 point - 5 degree
   TriangleIntRules[5] = new IntegrationRule (7);

   TriangleIntRules[5] -> IntPoint(0).x      = 0.3333333333333333333333333333333;
   TriangleIntRules[5] -> IntPoint(0).y      = 0.3333333333333333333333333333333;
   TriangleIntRules[5] -> IntPoint(0).weight = 0.1125;

   TriangleIntRules[5] -> IntPoint(1).x      = 0.1012865073234563388009873619151;
   TriangleIntRules[5] -> IntPoint(1).y      = 0.1012865073234563388009873619151;
   TriangleIntRules[5] -> IntPoint(1).weight = 0.06296959027241357629784197275009;

   TriangleIntRules[5] -> IntPoint(2).x      = 0.1012865073234563388009873619151;
   TriangleIntRules[5] -> IntPoint(2).y      = 0.7974269853530873223980252761698;
   TriangleIntRules[5] -> IntPoint(2).weight = 0.06296959027241357629784197275009;

   TriangleIntRules[5] -> IntPoint(3).x      = 0.7974269853530873223980252761698;
   TriangleIntRules[5] -> IntPoint(3).y      = 0.1012865073234563388009873619151;
   TriangleIntRules[5] -> IntPoint(3).weight = 0.06296959027241357629784197275009;

   TriangleIntRules[5] -> IntPoint(4).x      = 0.4701420641051150897704412095134;
   TriangleIntRules[5] -> IntPoint(4).y      = 0.4701420641051150897704412095134;
   TriangleIntRules[5] -> IntPoint(4).weight = 0.06619707639425309036882469391658;

   TriangleIntRules[5] -> IntPoint(5).x      = 0.4701420641051150897704412095134;
   TriangleIntRules[5] -> IntPoint(5).y      = 0.0597158717897698204591175809731;
   TriangleIntRules[5] -> IntPoint(5).weight = 0.06619707639425309036882469391658;

   TriangleIntRules[5] -> IntPoint(6).x      = 0.0597158717897698204591175809731;
   TriangleIntRules[5] -> IntPoint(6).y      = 0.4701420641051150897704412095134;
   TriangleIntRules[5] -> IntPoint(6).weight = 0.06619707639425309036882469391658;

   // 12 point - 6 degree
   TriangleIntRules[6] = new IntegrationRule (12);

   TriangleIntRules[6] -> IntPoint(0).x      = 0.063089014491502;
   TriangleIntRules[6] -> IntPoint(0).y      = 0.063089014491502;
   TriangleIntRules[6] -> IntPoint(0).weight = 0.0254224531851035;

   TriangleIntRules[6] -> IntPoint(1).x      = 0.063089014491502;
   TriangleIntRules[6] -> IntPoint(1).y      = 0.873821971016996;
   TriangleIntRules[6] -> IntPoint(1).weight = 0.0254224531851035;

   TriangleIntRules[6] -> IntPoint(2).x      = 0.873821971016996;
   TriangleIntRules[6] -> IntPoint(2).y      = 0.063089014491502;
   TriangleIntRules[6] -> IntPoint(2).weight = 0.0254224531851035;

   TriangleIntRules[6] -> IntPoint(3).x      = 0.249286745170911;
   TriangleIntRules[6] -> IntPoint(3).y      = 0.249286745170911;
   TriangleIntRules[6] -> IntPoint(3).weight = 0.0583931378631895;

   TriangleIntRules[6] -> IntPoint(4).x      = 0.249286745170911;
   TriangleIntRules[6] -> IntPoint(4).y      = 0.501426509658179;
   TriangleIntRules[6] -> IntPoint(4).weight = 0.0583931378631895;

   TriangleIntRules[6] -> IntPoint(5).x      = 0.501426509658179;
   TriangleIntRules[6] -> IntPoint(5).y      = 0.249286745170911;
   TriangleIntRules[6] -> IntPoint(5).weight = 0.0583931378631895;

   TriangleIntRules[6] -> IntPoint(6).x      = 0.310352451033785;
   TriangleIntRules[6] -> IntPoint(6).y      = 0.053145049844816;
   TriangleIntRules[6] -> IntPoint(6).weight = 0.041425537809187;

   TriangleIntRules[6] -> IntPoint(7).x      = 0.310352451033785;
   TriangleIntRules[6] -> IntPoint(7).y      = 0.636502499121399;
   TriangleIntRules[6] -> IntPoint(7).weight = 0.041425537809187;

   TriangleIntRules[6] -> IntPoint(8).x      = 0.053145049844816;
   TriangleIntRules[6] -> IntPoint(8).y      = 0.310352451033785;
   TriangleIntRules[6] -> IntPoint(8).weight = 0.041425537809187;

   TriangleIntRules[6] -> IntPoint(9).x      = 0.053145049844816;
   TriangleIntRules[6] -> IntPoint(9).y      = 0.636502499121399;
   TriangleIntRules[6] -> IntPoint(9).weight = 0.041425537809187;

   TriangleIntRules[6] -> IntPoint(10).x      = 0.636502499121399;
   TriangleIntRules[6] -> IntPoint(10).y      = 0.310352451033785;
   TriangleIntRules[6] -> IntPoint(10).weight = 0.041425537809187;

   TriangleIntRules[6] -> IntPoint(11).x      = 0.636502499121399;
   TriangleIntRules[6] -> IntPoint(11).y      = 0.053145049844816;
   TriangleIntRules[6] -> IntPoint(11).weight = 0.041425537809187;

   // 13 point - 7 degree
   TriangleIntRules[7] = new IntegrationRule (13);

   TriangleIntRules[7] -> IntPoint(0).x      = 0.33333333333333;
   TriangleIntRules[7] -> IntPoint(0).y      = 0.33333333333333;
   TriangleIntRules[7] -> IntPoint(0).weight = -0.074785022233835;

   TriangleIntRules[7] -> IntPoint(1).x      = 0.2603459661;
   TriangleIntRules[7] -> IntPoint(1).y      = 0.2603459661;
   TriangleIntRules[7] -> IntPoint(1).weight = 0.087807628716602;

   TriangleIntRules[7] -> IntPoint(2).x      = 0.4793080678;
   TriangleIntRules[7] -> IntPoint(2).y      = 0.2603459661;
   TriangleIntRules[7] -> IntPoint(2).weight = 0.087807628716602;

   TriangleIntRules[7] -> IntPoint(3).x      = 0.2603459661;
   TriangleIntRules[7] -> IntPoint(3).y      = 0.4793080678;
   TriangleIntRules[7] -> IntPoint(3).weight = 0.087807628716602;

   TriangleIntRules[7] -> IntPoint(4).x      = 0.0651301029;
   TriangleIntRules[7] -> IntPoint(4).y      = 0.0651301029;
   TriangleIntRules[7] -> IntPoint(4).weight = 0.026673617804419;

   TriangleIntRules[7] -> IntPoint(5).x      = 0.8697397942;
   TriangleIntRules[7] -> IntPoint(5).y      = 0.0651301029;
   TriangleIntRules[7] -> IntPoint(5).weight = 0.026673617804419;

   TriangleIntRules[7] -> IntPoint(6).x      = 0.0651301029;
   TriangleIntRules[7] -> IntPoint(6).y      = 0.8697397942;
   TriangleIntRules[7] -> IntPoint(6).weight = 0.026673617804419;

   TriangleIntRules[7] -> IntPoint(7).x      = 0.31286549600487;
   TriangleIntRules[7] -> IntPoint(7).y      = 0.04869031542532;
   TriangleIntRules[7] -> IntPoint(7).weight = 0.038556880445128;

   TriangleIntRules[7] -> IntPoint(8).x      = 0.04869031542532;
   TriangleIntRules[7] -> IntPoint(8).y      = 0.31286549600487;
   TriangleIntRules[7] -> IntPoint(8).weight = 0.038556880445128;

   TriangleIntRules[7] -> IntPoint(9).x      = 0.63844418856981;
   TriangleIntRules[7] -> IntPoint(9).y      = 0.04869031542532;
   TriangleIntRules[7] -> IntPoint(9).weight = 0.038556880445128;

   TriangleIntRules[7] -> IntPoint(10).x      = 0.63844418856981;
   TriangleIntRules[7] -> IntPoint(10).y      = 0.31286549600487;
   TriangleIntRules[7] -> IntPoint(10).weight = 0.038556880445128;

   TriangleIntRules[7] -> IntPoint(11).x      = 0.04869031542532;
   TriangleIntRules[7] -> IntPoint(11).y      = 0.63844418856981;
   TriangleIntRules[7] -> IntPoint(11).weight = 0.038556880445128;

   TriangleIntRules[7] -> IntPoint(12).x      = 0.31286549600487;
   TriangleIntRules[7] -> IntPoint(12).y      = 0.63844418856981;
   TriangleIntRules[7] -> IntPoint(12).weight = 0.038556880445128;
}

// Integration rules for unit square
void IntegrationRules::SquareIntegrationRules()
{
   SquareIntRules.SetSize(20);

   int i,k,s,np;

   for (i = 0; i < SquareIntRules.Size(); i++)
   {
      np = SegmentIntRules[i] -> GetNPoints();
      SquareIntRules[i] = new IntegrationRule(np*np);
      for (k = 0; k < np; k++)
         for (s = 0; s < np; s++)
         {
            SquareIntRules[i] -> IntPoint(k*np+s).x
               = SegmentIntRules[i] -> IntPoint(k).x ;

            SquareIntRules[i] -> IntPoint(k*np+s).y
               = SegmentIntRules[i] -> IntPoint(s).x ;

            SquareIntRules[i] -> IntPoint(k*np+s).weight
               = SegmentIntRules[i] -> IntPoint(k).weight
               * SegmentIntRules[i] -> IntPoint(s).weight;
         }
   }
}

/** Integration rules for reference tetrahedron
    {[0,0,0],[1,0,0],[0,1,0],[0,0,1]}          */
void IntegrationRules::TetrahedronIntegrationRules(int refined)
{
   TetrahedronIntRules.SetSize(9);

   if (refined)
      mfem_error ("Refined TetrahedronIntegrationRules are not implemented!");

   for (int i = 0; i < TetrahedronIntRules.Size(); i++)
      TetrahedronIntRules[i] = NULL;


   // 1 point - degree 1
   TetrahedronIntRules[0] = new IntegrationRule (1);

   TetrahedronIntRules[0] -> IntPoint(0).x = 0.25;
   TetrahedronIntRules[0] -> IntPoint(0).y = 0.25;
   TetrahedronIntRules[0] -> IntPoint(0).z = 0.25;
   TetrahedronIntRules[0] -> IntPoint(0).weight = 0.1666666666666666667;

   // 1 point - degree 1
   TetrahedronIntRules[1] = new IntegrationRule (1);

   TetrahedronIntRules[1] -> IntPoint(0).x = 0.25;
   TetrahedronIntRules[1] -> IntPoint(0).y = 0.25;
   TetrahedronIntRules[1] -> IntPoint(0).z = 0.25;
   TetrahedronIntRules[1] -> IntPoint(0).weight = 0.1666666666666666667;

   // 4 points - degree 2
   TetrahedronIntRules[2] = new IntegrationRule (4);

   TetrahedronIntRules[2] -> IntPoint(0).x = 0.58541019662496845446;
   TetrahedronIntRules[2] -> IntPoint(0).y = 0.13819660112501051518;
   TetrahedronIntRules[2] -> IntPoint(0).z = 0.13819660112501051518;
   TetrahedronIntRules[2] -> IntPoint(0).weight = 0.041666666666666666667;
   TetrahedronIntRules[2] -> IntPoint(1).x = 0.13819660112501051518;
   TetrahedronIntRules[2] -> IntPoint(1).y = 0.58541019662496845446;
   TetrahedronIntRules[2] -> IntPoint(1).z = 0.13819660112501051518;
   TetrahedronIntRules[2] -> IntPoint(1).weight = 0.041666666666666666667;
   TetrahedronIntRules[2] -> IntPoint(2).x = 0.13819660112501051518;
   TetrahedronIntRules[2] -> IntPoint(2).y = 0.13819660112501051518;
   TetrahedronIntRules[2] -> IntPoint(2).z = 0.58541019662496845446;
   TetrahedronIntRules[2] -> IntPoint(2).weight = 0.041666666666666666667;
   TetrahedronIntRules[2] -> IntPoint(3).x = 0.13819660112501051518;
   TetrahedronIntRules[2] -> IntPoint(3).y = 0.13819660112501051518;
   TetrahedronIntRules[2] -> IntPoint(3).z = 0.13819660112501051518;
   TetrahedronIntRules[2] -> IntPoint(3).weight = 0.041666666666666666667;

   // 5 points - degree 3
   TetrahedronIntRules[3] = new IntegrationRule (5);

   TetrahedronIntRules[3] -> IntPoint(0).x = 0.25;
   TetrahedronIntRules[3] -> IntPoint(0).y = 0.25;
   TetrahedronIntRules[3] -> IntPoint(0).z = 0.25;
   TetrahedronIntRules[3] -> IntPoint(0).weight = -0.13333333333333333333;
   TetrahedronIntRules[3] -> IntPoint(1).x = 0.5;
   TetrahedronIntRules[3] -> IntPoint(1).y = 0.1666666666666666667;
   TetrahedronIntRules[3] -> IntPoint(1).z = 0.1666666666666666667;
   TetrahedronIntRules[3] -> IntPoint(1).weight = 0.075;
   TetrahedronIntRules[3] -> IntPoint(2).x = 0.1666666666666666667;
   TetrahedronIntRules[3] -> IntPoint(2).y = 0.5;
   TetrahedronIntRules[3] -> IntPoint(2).z = 0.1666666666666666667;
   TetrahedronIntRules[3] -> IntPoint(2).weight = 0.075;
   TetrahedronIntRules[3] -> IntPoint(3).x = 0.1666666666666666667;
   TetrahedronIntRules[3] -> IntPoint(3).y = 0.1666666666666666667;
   TetrahedronIntRules[3] -> IntPoint(3).z = 0.5;
   TetrahedronIntRules[3] -> IntPoint(3).weight = 0.075;
   TetrahedronIntRules[3] -> IntPoint(4).x = 0.1666666666666666667;
   TetrahedronIntRules[3] -> IntPoint(4).y = 0.1666666666666666667;
   TetrahedronIntRules[3] -> IntPoint(4).z = 0.1666666666666666667;
   TetrahedronIntRules[3] -> IntPoint(4).weight = 0.075;

   // 11 points - degree 4
   TetrahedronIntRules[4] = new IntegrationRule (11);

   TetrahedronIntRules[4] -> IntPoint(0).x = 0.25;
   TetrahedronIntRules[4] -> IntPoint(0).y = 0.25;
   TetrahedronIntRules[4] -> IntPoint(0).z = 0.25;
   TetrahedronIntRules[4] -> IntPoint(0).weight = -0.013155555555555555556;
   TetrahedronIntRules[4] -> IntPoint(1).x = 0.78571428571428571429;
   TetrahedronIntRules[4] -> IntPoint(1).y = 0.071428571428571428571;
   TetrahedronIntRules[4] -> IntPoint(1).z = 0.071428571428571428571;
   TetrahedronIntRules[4] -> IntPoint(1).weight = 0.0076222222222222222222;
   TetrahedronIntRules[4] -> IntPoint(2).x = 0.071428571428571428571;
   TetrahedronIntRules[4] -> IntPoint(2).y = 0.78571428571428571429;
   TetrahedronIntRules[4] -> IntPoint(2).z = 0.071428571428571428571;
   TetrahedronIntRules[4] -> IntPoint(2).weight = 0.0076222222222222222222;
   TetrahedronIntRules[4] -> IntPoint(3).x = 0.071428571428571428571;
   TetrahedronIntRules[4] -> IntPoint(3).y = 0.071428571428571428571;
   TetrahedronIntRules[4] -> IntPoint(3).z = 0.78571428571428571429;
   TetrahedronIntRules[4] -> IntPoint(3).weight = 0.0076222222222222222222;
   TetrahedronIntRules[4] -> IntPoint(4).x = 0.071428571428571428571;
   TetrahedronIntRules[4] -> IntPoint(4).y = 0.071428571428571428571;
   TetrahedronIntRules[4] -> IntPoint(4).z = 0.071428571428571428571;
   TetrahedronIntRules[4] -> IntPoint(4).weight = 0.0076222222222222222222;
   TetrahedronIntRules[4] -> IntPoint(5).x = 0.39940357616679920500;
   TetrahedronIntRules[4] -> IntPoint(5).y = 0.39940357616679920500;
   TetrahedronIntRules[4] -> IntPoint(5).z = 0.10059642383320079500;
   TetrahedronIntRules[4] -> IntPoint(5).weight = 0.024888888888888888889;
   TetrahedronIntRules[4] -> IntPoint(6).x = 0.39940357616679920500;
   TetrahedronIntRules[4] -> IntPoint(6).y = 0.10059642383320079500;
   TetrahedronIntRules[4] -> IntPoint(6).z = 0.39940357616679920500;
   TetrahedronIntRules[4] -> IntPoint(6).weight = 0.024888888888888888889;
   TetrahedronIntRules[4] -> IntPoint(7).x = 0.39940357616679920500;
   TetrahedronIntRules[4] -> IntPoint(7).y = 0.10059642383320079500;
   TetrahedronIntRules[4] -> IntPoint(7).z = 0.10059642383320079500;
   TetrahedronIntRules[4] -> IntPoint(7).weight = 0.024888888888888888889;
   TetrahedronIntRules[4] -> IntPoint(8).x = 0.10059642383320079500;
   TetrahedronIntRules[4] -> IntPoint(8).y = 0.39940357616679920500;
   TetrahedronIntRules[4] -> IntPoint(8).z = 0.39940357616679920500;
   TetrahedronIntRules[4] -> IntPoint(8).weight = 0.024888888888888888889;
   TetrahedronIntRules[4] -> IntPoint(9).x = 0.10059642383320079500;
   TetrahedronIntRules[4] -> IntPoint(9).y = 0.39940357616679920500;
   TetrahedronIntRules[4] -> IntPoint(9).z = 0.10059642383320079500;
   TetrahedronIntRules[4] -> IntPoint(9).weight = 0.024888888888888888889;
   TetrahedronIntRules[4] -> IntPoint(10).x = 0.10059642383320079500;
   TetrahedronIntRules[4] -> IntPoint(10).y = 0.10059642383320079500;
   TetrahedronIntRules[4] -> IntPoint(10).z = 0.39940357616679920500;
   TetrahedronIntRules[4] -> IntPoint(10).weight = 0.024888888888888888889;

   // 15 points - degree 5
   TetrahedronIntRules[5] = new IntegrationRule (15);

   TetrahedronIntRules[5] -> IntPoint( 0).weight = +0.0060267857142857;
   TetrahedronIntRules[5] -> IntPoint( 1).weight = +0.0060267857142857;
   TetrahedronIntRules[5] -> IntPoint( 2).weight = +0.0060267857142857;
   TetrahedronIntRules[5] -> IntPoint( 3).weight = +0.0060267857142857;
   TetrahedronIntRules[5] -> IntPoint( 4).weight = +0.0302836780970892;
   TetrahedronIntRules[5] -> IntPoint( 5).weight = +0.0116452490860290;
   TetrahedronIntRules[5] -> IntPoint( 6).weight = +0.0116452490860290;
   TetrahedronIntRules[5] -> IntPoint( 7).weight = +0.0116452490860290;
   TetrahedronIntRules[5] -> IntPoint( 8).weight = +0.0116452490860290;
   TetrahedronIntRules[5] -> IntPoint( 9).weight = +0.0109491415613865;
   TetrahedronIntRules[5] -> IntPoint(10).weight = +0.0109491415613865;
   TetrahedronIntRules[5] -> IntPoint(11).weight = +0.0109491415613865;
   TetrahedronIntRules[5] -> IntPoint(12).weight = +0.0109491415613865;
   TetrahedronIntRules[5] -> IntPoint(13).weight = +0.0109491415613865;
   TetrahedronIntRules[5] -> IntPoint(14).weight = +0.0109491415613865;
   TetrahedronIntRules[5] -> IntPoint( 0).x = +0.3333333333333333;
   TetrahedronIntRules[5] -> IntPoint( 0).y = +0.3333333333333333;
   TetrahedronIntRules[5] -> IntPoint( 0).z = +0.3333333333333333;
   TetrahedronIntRules[5] -> IntPoint( 1).x = +0.3333333333333333;
   TetrahedronIntRules[5] -> IntPoint( 1).y = +0.3333333333333333;
   TetrahedronIntRules[5] -> IntPoint( 1).z = +0.0000000000000000;
   TetrahedronIntRules[5] -> IntPoint( 2).x = +0.3333333333333333;
   TetrahedronIntRules[5] -> IntPoint( 2).y = +0.0000000000000000;
   TetrahedronIntRules[5] -> IntPoint( 2).z = +0.3333333333333333;
   TetrahedronIntRules[5] -> IntPoint( 3).x = +0.0000000000000000;
   TetrahedronIntRules[5] -> IntPoint( 3).y = +0.3333333333333333;
   TetrahedronIntRules[5] -> IntPoint( 3).z = +0.3333333333333333;
   TetrahedronIntRules[5] -> IntPoint( 4).x = +0.2500000000000000;
   TetrahedronIntRules[5] -> IntPoint( 4).y = +0.2500000000000000;
   TetrahedronIntRules[5] -> IntPoint( 4).z = +0.2500000000000000;
   TetrahedronIntRules[5] -> IntPoint( 5).x = +0.0909090909090909;
   TetrahedronIntRules[5] -> IntPoint( 5).y = +0.0909090909090909;
   TetrahedronIntRules[5] -> IntPoint( 5).z = +0.0909090909090909;
   TetrahedronIntRules[5] -> IntPoint( 6).x = +0.0909090909090909;
   TetrahedronIntRules[5] -> IntPoint( 6).y = +0.0909090909090909;
   TetrahedronIntRules[5] -> IntPoint( 6).z = +0.7272727272727273;
   TetrahedronIntRules[5] -> IntPoint( 7).x = +0.0909090909090909;
   TetrahedronIntRules[5] -> IntPoint( 7).y = +0.7272727272727273;
   TetrahedronIntRules[5] -> IntPoint( 7).z = +0.0909090909090909;
   TetrahedronIntRules[5] -> IntPoint( 8).x = +0.7272727272727273;
   TetrahedronIntRules[5] -> IntPoint( 8).y = +0.0909090909090909;
   TetrahedronIntRules[5] -> IntPoint( 8).z = +0.0909090909090909;
   TetrahedronIntRules[5] -> IntPoint( 9).x = +0.0665501535736643;
   TetrahedronIntRules[5] -> IntPoint( 9).y = +0.0665501535736643;
   TetrahedronIntRules[5] -> IntPoint( 9).z = +0.4334498464263357;
   TetrahedronIntRules[5] -> IntPoint(10).x = +0.0665501535736643;
   TetrahedronIntRules[5] -> IntPoint(10).y = +0.4334498464263357;
   TetrahedronIntRules[5] -> IntPoint(10).z = +0.4334498464263357;
   TetrahedronIntRules[5] -> IntPoint(11).x = +0.4334498464263357;
   TetrahedronIntRules[5] -> IntPoint(11).y = +0.4334498464263357;
   TetrahedronIntRules[5] -> IntPoint(11).z = +0.0665501535736643;
   TetrahedronIntRules[5] -> IntPoint(12).x = +0.4334498464263357;
   TetrahedronIntRules[5] -> IntPoint(12).y = +0.0665501535736643;
   TetrahedronIntRules[5] -> IntPoint(12).z = +0.0665501535736643;
   TetrahedronIntRules[5] -> IntPoint(13).x = +0.0665501535736643;
   TetrahedronIntRules[5] -> IntPoint(13).y = +0.4334498464263357;
   TetrahedronIntRules[5] -> IntPoint(13).z = +0.0665501535736643;
   TetrahedronIntRules[5] -> IntPoint(14).x = +0.4334498464263357;
   TetrahedronIntRules[5] -> IntPoint(14).y = +0.0665501535736643;
   TetrahedronIntRules[5] -> IntPoint(14).z = +0.4334498464263357;

   // 24 points - degree 6
   TetrahedronIntRules[6] = new IntegrationRule (24);

   TetrahedronIntRules[6] -> IntPoint( 0).weight = +0.0066537917096946;
   TetrahedronIntRules[6] -> IntPoint( 1).weight = +0.0066537917096946;
   TetrahedronIntRules[6] -> IntPoint( 2).weight = +0.0066537917096946;
   TetrahedronIntRules[6] -> IntPoint( 3).weight = +0.0066537917096946;
   TetrahedronIntRules[6] -> IntPoint( 4).weight = +0.0016795351758868;
   TetrahedronIntRules[6] -> IntPoint( 5).weight = +0.0016795351758868;
   TetrahedronIntRules[6] -> IntPoint( 6).weight = +0.0016795351758868;
   TetrahedronIntRules[6] -> IntPoint( 7).weight = +0.0016795351758868;
   TetrahedronIntRules[6] -> IntPoint( 8).weight = +0.0092261969239424;
   TetrahedronIntRules[6] -> IntPoint( 9).weight = +0.0092261969239424;
   TetrahedronIntRules[6] -> IntPoint(10).weight = +0.0092261969239424;
   TetrahedronIntRules[6] -> IntPoint(11).weight = +0.0092261969239424;
   TetrahedronIntRules[6] -> IntPoint(12).weight = +0.0080357142857143;
   TetrahedronIntRules[6] -> IntPoint(13).weight = +0.0080357142857143;
   TetrahedronIntRules[6] -> IntPoint(14).weight = +0.0080357142857143;
   TetrahedronIntRules[6] -> IntPoint(15).weight = +0.0080357142857143;
   TetrahedronIntRules[6] -> IntPoint(16).weight = +0.0080357142857143;
   TetrahedronIntRules[6] -> IntPoint(17).weight = +0.0080357142857143;
   TetrahedronIntRules[6] -> IntPoint(18).weight = +0.0080357142857143;
   TetrahedronIntRules[6] -> IntPoint(19).weight = +0.0080357142857143;
   TetrahedronIntRules[6] -> IntPoint(20).weight = +0.0080357142857143;
   TetrahedronIntRules[6] -> IntPoint(21).weight = +0.0080357142857143;
   TetrahedronIntRules[6] -> IntPoint(22).weight = +0.0080357142857143;
   TetrahedronIntRules[6] -> IntPoint(23).weight = +0.0080357142857143;
   TetrahedronIntRules[6] -> IntPoint( 0).x = +0.2146028712591517;
   TetrahedronIntRules[6] -> IntPoint( 0).y = +0.2146028712591517;
   TetrahedronIntRules[6] -> IntPoint( 0).z = +0.2146028712591517;
   TetrahedronIntRules[6] -> IntPoint( 1).x = +0.2146028712591517;
   TetrahedronIntRules[6] -> IntPoint( 1).y = +0.2146028712591517;
   TetrahedronIntRules[6] -> IntPoint( 1).z = +0.3561913862225449;
   TetrahedronIntRules[6] -> IntPoint( 2).x = +0.2146028712591517;
   TetrahedronIntRules[6] -> IntPoint( 2).y = +0.3561913862225449;
   TetrahedronIntRules[6] -> IntPoint( 2).z = +0.2146028712591517;
   TetrahedronIntRules[6] -> IntPoint( 3).x = +0.3561913862225449;
   TetrahedronIntRules[6] -> IntPoint( 3).y = +0.2146028712591517;
   TetrahedronIntRules[6] -> IntPoint( 3).z = +0.2146028712591517;
   TetrahedronIntRules[6] -> IntPoint( 4).x = +0.0406739585346113;
   TetrahedronIntRules[6] -> IntPoint( 4).y = +0.0406739585346113;
   TetrahedronIntRules[6] -> IntPoint( 4).z = +0.0406739585346113;
   TetrahedronIntRules[6] -> IntPoint( 5).x = +0.0406739585346113;
   TetrahedronIntRules[6] -> IntPoint( 5).y = +0.0406739585346113;
   TetrahedronIntRules[6] -> IntPoint( 5).z = +0.8779781243961660;
   TetrahedronIntRules[6] -> IntPoint( 6).x = +0.0406739585346113;
   TetrahedronIntRules[6] -> IntPoint( 6).y = +0.8779781243961660;
   TetrahedronIntRules[6] -> IntPoint( 6).z = +0.0406739585346113;
   TetrahedronIntRules[6] -> IntPoint( 7).x = +0.8779781243961660;
   TetrahedronIntRules[6] -> IntPoint( 7).y = +0.0406739585346113;
   TetrahedronIntRules[6] -> IntPoint( 7).z = +0.0406739585346113;
   TetrahedronIntRules[6] -> IntPoint( 8).x = +0.3223378901422757;
   TetrahedronIntRules[6] -> IntPoint( 8).y = +0.3223378901422757;
   TetrahedronIntRules[6] -> IntPoint( 8).z = +0.3223378901422757;
   TetrahedronIntRules[6] -> IntPoint( 9).x = +0.3223378901422757;
   TetrahedronIntRules[6] -> IntPoint( 9).y = +0.3223378901422757;
   TetrahedronIntRules[6] -> IntPoint( 9).z = +0.0329863295731731;
   TetrahedronIntRules[6] -> IntPoint(10).x = +0.3223378901422757;
   TetrahedronIntRules[6] -> IntPoint(10).y = +0.0329863295731731;
   TetrahedronIntRules[6] -> IntPoint(10).z = +0.3223378901422757;
   TetrahedronIntRules[6] -> IntPoint(11).x = +0.0329863295731731;
   TetrahedronIntRules[6] -> IntPoint(11).y = +0.3223378901422757;
   TetrahedronIntRules[6] -> IntPoint(11).z = +0.3223378901422757;
   TetrahedronIntRules[6] -> IntPoint(12).x = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(12).y = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(12).z = +0.2696723314583159;
   TetrahedronIntRules[6] -> IntPoint(13).x = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(13).y = +0.2696723314583159;
   TetrahedronIntRules[6] -> IntPoint(13).z = +0.6030056647916491;
   TetrahedronIntRules[6] -> IntPoint(14).x = +0.2696723314583159;
   TetrahedronIntRules[6] -> IntPoint(14).y = +0.6030056647916491;
   TetrahedronIntRules[6] -> IntPoint(14).z = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(15).x = +0.6030056647916491;
   TetrahedronIntRules[6] -> IntPoint(15).y = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(15).z = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(16).x = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(16).y = +0.6030056647916491;
   TetrahedronIntRules[6] -> IntPoint(16).z = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(17).x = +0.6030056647916491;
   TetrahedronIntRules[6] -> IntPoint(17).y = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(17).z = +0.2696723314583159;
   TetrahedronIntRules[6] -> IntPoint(18).x = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(18).y = +0.2696723314583159;
   TetrahedronIntRules[6] -> IntPoint(18).z = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(19).x = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(19).y = +0.6030056647916491;
   TetrahedronIntRules[6] -> IntPoint(19).z = +0.2696723314583159;
   TetrahedronIntRules[6] -> IntPoint(20).x = +0.2696723314583159;
   TetrahedronIntRules[6] -> IntPoint(20).y = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(20).z = +0.6030056647916491;
   TetrahedronIntRules[6] -> IntPoint(21).x = +0.6030056647916491;
   TetrahedronIntRules[6] -> IntPoint(21).y = +0.2696723314583159;
   TetrahedronIntRules[6] -> IntPoint(21).z = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(22).x = +0.2696723314583159;
   TetrahedronIntRules[6] -> IntPoint(22).y = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(22).z = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(23).x = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(23).y = +0.0636610018750175;
   TetrahedronIntRules[6] -> IntPoint(23).z = +0.6030056647916491;

   // 31 points - degree 7
   TetrahedronIntRules[7] = new IntegrationRule (31);

   TetrahedronIntRules[7] -> IntPoint( 0).weight = +0.0009700176366843;
   TetrahedronIntRules[7] -> IntPoint( 1).weight = +0.0009700176366843;
   TetrahedronIntRules[7] -> IntPoint( 2).weight = +0.0009700176366843;
   TetrahedronIntRules[7] -> IntPoint( 3).weight = +0.0009700176366843;
   TetrahedronIntRules[7] -> IntPoint( 4).weight = +0.0009700176366843;
   TetrahedronIntRules[7] -> IntPoint( 5).weight = +0.0009700176366843;
   TetrahedronIntRules[7] -> IntPoint( 6).weight = +0.0182642234661088;
   TetrahedronIntRules[7] -> IntPoint( 7).weight = +0.0105999415244142;
   TetrahedronIntRules[7] -> IntPoint( 8).weight = +0.0105999415244142;
   TetrahedronIntRules[7] -> IntPoint( 9).weight = +0.0105999415244142;
   TetrahedronIntRules[7] -> IntPoint(10).weight = +0.0105999415244142;
   TetrahedronIntRules[7] -> IntPoint(11).weight = -0.0625177401143300;
   TetrahedronIntRules[7] -> IntPoint(12).weight = -0.0625177401143300;
   TetrahedronIntRules[7] -> IntPoint(13).weight = -0.0625177401143300;
   TetrahedronIntRules[7] -> IntPoint(14).weight = -0.0625177401143300;
   TetrahedronIntRules[7] -> IntPoint(15).weight = +0.0048914252630735;
   TetrahedronIntRules[7] -> IntPoint(16).weight = +0.0048914252630735;
   TetrahedronIntRules[7] -> IntPoint(17).weight = +0.0048914252630735;
   TetrahedronIntRules[7] -> IntPoint(18).weight = +0.0048914252630735;
   TetrahedronIntRules[7] -> IntPoint(19).weight = +0.0275573192239851;
   TetrahedronIntRules[7] -> IntPoint(20).weight = +0.0275573192239851;
   TetrahedronIntRules[7] -> IntPoint(21).weight = +0.0275573192239851;
   TetrahedronIntRules[7] -> IntPoint(22).weight = +0.0275573192239851;
   TetrahedronIntRules[7] -> IntPoint(23).weight = +0.0275573192239851;
   TetrahedronIntRules[7] -> IntPoint(24).weight = +0.0275573192239851;
   TetrahedronIntRules[7] -> IntPoint(25).weight = +0.0275573192239851;
   TetrahedronIntRules[7] -> IntPoint(26).weight = +0.0275573192239851;
   TetrahedronIntRules[7] -> IntPoint(27).weight = +0.0275573192239851;
   TetrahedronIntRules[7] -> IntPoint(28).weight = +0.0275573192239851;
   TetrahedronIntRules[7] -> IntPoint(29).weight = +0.0275573192239851;
   TetrahedronIntRules[7] -> IntPoint(30).weight = +0.0275573192239851;
   TetrahedronIntRules[7] -> IntPoint( 0).x = +0.5000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 0).y = +0.5000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 0).z = +0.0000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 1).x = +0.5000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 1).y = +0.0000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 1).z = +0.0000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 2).x = +0.0000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 2).y = +0.0000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 2).z = +0.5000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 3).x = +0.0000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 3).y = +0.5000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 3).z = +0.5000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 4).x = +0.5000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 4).y = +0.0000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 4).z = +0.5000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 5).x = +0.0000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 5).y = +0.5000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 5).z = +0.0000000000000000;
   TetrahedronIntRules[7] -> IntPoint( 6).x = +0.2500000000000000;
   TetrahedronIntRules[7] -> IntPoint( 6).y = +0.2500000000000000;
   TetrahedronIntRules[7] -> IntPoint( 6).z = +0.2500000000000000;
   TetrahedronIntRules[7] -> IntPoint( 7).x = +0.0782131923303186;
   TetrahedronIntRules[7] -> IntPoint( 7).y = +0.0782131923303186;
   TetrahedronIntRules[7] -> IntPoint( 7).z = +0.0782131923303186;
   TetrahedronIntRules[7] -> IntPoint( 8).x = +0.0782131923303186;
   TetrahedronIntRules[7] -> IntPoint( 8).y = +0.0782131923303186;
   TetrahedronIntRules[7] -> IntPoint( 8).z = +0.7653604230090441;
   TetrahedronIntRules[7] -> IntPoint( 9).x = +0.0782131923303186;
   TetrahedronIntRules[7] -> IntPoint( 9).y = +0.7653604230090441;
   TetrahedronIntRules[7] -> IntPoint( 9).z = +0.0782131923303186;
   TetrahedronIntRules[7] -> IntPoint(10).x = +0.7653604230090441;
   TetrahedronIntRules[7] -> IntPoint(10).y = +0.0782131923303186;
   TetrahedronIntRules[7] -> IntPoint(10).z = +0.0782131923303186;
   TetrahedronIntRules[7] -> IntPoint(11).x = +0.1218432166639044;
   TetrahedronIntRules[7] -> IntPoint(11).y = +0.1218432166639044;
   TetrahedronIntRules[7] -> IntPoint(11).z = +0.1218432166639044;
   TetrahedronIntRules[7] -> IntPoint(12).x = +0.1218432166639044;
   TetrahedronIntRules[7] -> IntPoint(12).y = +0.1218432166639044;
   TetrahedronIntRules[7] -> IntPoint(12).z = +0.6344703500082868;
   TetrahedronIntRules[7] -> IntPoint(13).x = +0.1218432166639044;
   TetrahedronIntRules[7] -> IntPoint(13).y = +0.6344703500082868;
   TetrahedronIntRules[7] -> IntPoint(13).z = +0.1218432166639044;
   TetrahedronIntRules[7] -> IntPoint(14).x = +0.6344703500082868;
   TetrahedronIntRules[7] -> IntPoint(14).y = +0.1218432166639044;
   TetrahedronIntRules[7] -> IntPoint(14).z = +0.1218432166639044;
   TetrahedronIntRules[7] -> IntPoint(15).x = +0.3325391644464206;
   TetrahedronIntRules[7] -> IntPoint(15).y = +0.3325391644464206;
   TetrahedronIntRules[7] -> IntPoint(15).z = +0.3325391644464206;
   TetrahedronIntRules[7] -> IntPoint(16).x = +0.3325391644464206;
   TetrahedronIntRules[7] -> IntPoint(16).y = +0.3325391644464206;
   TetrahedronIntRules[7] -> IntPoint(16).z = +0.0023825066607383;
   TetrahedronIntRules[7] -> IntPoint(17).x = +0.3325391644464206;
   TetrahedronIntRules[7] -> IntPoint(17).y = +0.0023825066607383;
   TetrahedronIntRules[7] -> IntPoint(17).z = +0.3325391644464206;
   TetrahedronIntRules[7] -> IntPoint(18).x = +0.0023825066607383;
   TetrahedronIntRules[7] -> IntPoint(18).y = +0.3325391644464206;
   TetrahedronIntRules[7] -> IntPoint(18).z = +0.3325391644464206;
   TetrahedronIntRules[7] -> IntPoint(19).x = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(19).y = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(19).z = +0.2000000000000000;
   TetrahedronIntRules[7] -> IntPoint(20).x = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(20).y = +0.2000000000000000;
   TetrahedronIntRules[7] -> IntPoint(20).z = +0.6000000000000000;
   TetrahedronIntRules[7] -> IntPoint(21).x = +0.2000000000000000;
   TetrahedronIntRules[7] -> IntPoint(21).y = +0.6000000000000000;
   TetrahedronIntRules[7] -> IntPoint(21).z = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(22).x = +0.6000000000000000;
   TetrahedronIntRules[7] -> IntPoint(22).y = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(22).z = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(23).x = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(23).y = +0.6000000000000000;
   TetrahedronIntRules[7] -> IntPoint(23).z = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(24).x = +0.6000000000000000;
   TetrahedronIntRules[7] -> IntPoint(24).y = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(24).z = +0.2000000000000000;
   TetrahedronIntRules[7] -> IntPoint(25).x = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(25).y = +0.2000000000000000;
   TetrahedronIntRules[7] -> IntPoint(25).z = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(26).x = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(26).y = +0.6000000000000000;
   TetrahedronIntRules[7] -> IntPoint(26).z = +0.2000000000000000;
   TetrahedronIntRules[7] -> IntPoint(27).x = +0.2000000000000000;
   TetrahedronIntRules[7] -> IntPoint(27).y = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(27).z = +0.6000000000000000;
   TetrahedronIntRules[7] -> IntPoint(28).x = +0.6000000000000000;
   TetrahedronIntRules[7] -> IntPoint(28).y = +0.2000000000000000;
   TetrahedronIntRules[7] -> IntPoint(28).z = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(29).x = +0.2000000000000000;
   TetrahedronIntRules[7] -> IntPoint(29).y = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(29).z = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(30).x = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(30).y = +0.1000000000000000;
   TetrahedronIntRules[7] -> IntPoint(30).z = +0.6000000000000000;

   // 45 points - degree 8
   TetrahedronIntRules[8] = new IntegrationRule (45);

   TetrahedronIntRules[8] -> IntPoint( 0).weight = -0.0393270066412926;
   TetrahedronIntRules[8] -> IntPoint( 1).weight = +0.0040813160593427;
   TetrahedronIntRules[8] -> IntPoint( 2).weight = +0.0040813160593427;
   TetrahedronIntRules[8] -> IntPoint( 3).weight = +0.0040813160593427;
   TetrahedronIntRules[8] -> IntPoint( 4).weight = +0.0040813160593427;
   TetrahedronIntRules[8] -> IntPoint( 5).weight = +0.0006580867733043;
   TetrahedronIntRules[8] -> IntPoint( 6).weight = +0.0006580867733043;
   TetrahedronIntRules[8] -> IntPoint( 7).weight = +0.0006580867733043;
   TetrahedronIntRules[8] -> IntPoint( 8).weight = +0.0006580867733043;
   TetrahedronIntRules[8] -> IntPoint( 9).weight = +0.0043842588251228;
   TetrahedronIntRules[8] -> IntPoint(10).weight = +0.0043842588251228;
   TetrahedronIntRules[8] -> IntPoint(11).weight = +0.0043842588251228;
   TetrahedronIntRules[8] -> IntPoint(12).weight = +0.0043842588251228;
   TetrahedronIntRules[8] -> IntPoint(13).weight = +0.0043842588251228;
   TetrahedronIntRules[8] -> IntPoint(14).weight = +0.0043842588251228;
   TetrahedronIntRules[8] -> IntPoint(15).weight = +0.0138300638425098;
   TetrahedronIntRules[8] -> IntPoint(16).weight = +0.0138300638425098;
   TetrahedronIntRules[8] -> IntPoint(17).weight = +0.0138300638425098;
   TetrahedronIntRules[8] -> IntPoint(18).weight = +0.0138300638425098;
   TetrahedronIntRules[8] -> IntPoint(19).weight = +0.0138300638425098;
   TetrahedronIntRules[8] -> IntPoint(20).weight = +0.0138300638425098;
   TetrahedronIntRules[8] -> IntPoint(21).weight = +0.0042404374246837;
   TetrahedronIntRules[8] -> IntPoint(22).weight = +0.0042404374246837;
   TetrahedronIntRules[8] -> IntPoint(23).weight = +0.0042404374246837;
   TetrahedronIntRules[8] -> IntPoint(24).weight = +0.0042404374246837;
   TetrahedronIntRules[8] -> IntPoint(25).weight = +0.0042404374246837;
   TetrahedronIntRules[8] -> IntPoint(26).weight = +0.0042404374246837;
   TetrahedronIntRules[8] -> IntPoint(27).weight = +0.0042404374246837;
   TetrahedronIntRules[8] -> IntPoint(28).weight = +0.0042404374246837;
   TetrahedronIntRules[8] -> IntPoint(29).weight = +0.0042404374246837;
   TetrahedronIntRules[8] -> IntPoint(30).weight = +0.0042404374246837;
   TetrahedronIntRules[8] -> IntPoint(31).weight = +0.0042404374246837;
   TetrahedronIntRules[8] -> IntPoint(32).weight = +0.0042404374246837;
   TetrahedronIntRules[8] -> IntPoint(33).weight = +0.0022387397396142;
   TetrahedronIntRules[8] -> IntPoint(34).weight = +0.0022387397396142;
   TetrahedronIntRules[8] -> IntPoint(35).weight = +0.0022387397396142;
   TetrahedronIntRules[8] -> IntPoint(36).weight = +0.0022387397396142;
   TetrahedronIntRules[8] -> IntPoint(37).weight = +0.0022387397396142;
   TetrahedronIntRules[8] -> IntPoint(38).weight = +0.0022387397396142;
   TetrahedronIntRules[8] -> IntPoint(39).weight = +0.0022387397396142;
   TetrahedronIntRules[8] -> IntPoint(40).weight = +0.0022387397396142;
   TetrahedronIntRules[8] -> IntPoint(41).weight = +0.0022387397396142;
   TetrahedronIntRules[8] -> IntPoint(42).weight = +0.0022387397396142;
   TetrahedronIntRules[8] -> IntPoint(43).weight = +0.0022387397396142;
   TetrahedronIntRules[8] -> IntPoint(44).weight = +0.0022387397396142;
   TetrahedronIntRules[8] -> IntPoint( 0).x = +0.2500000000000000;
   TetrahedronIntRules[8] -> IntPoint( 0).y = +0.2500000000000000;
   TetrahedronIntRules[8] -> IntPoint( 0).z = +0.2500000000000000;
   TetrahedronIntRules[8] -> IntPoint( 1).x = +0.1274709365666390;
   TetrahedronIntRules[8] -> IntPoint( 1).y = +0.1274709365666390;
   TetrahedronIntRules[8] -> IntPoint( 1).z = +0.1274709365666390;
   TetrahedronIntRules[8] -> IntPoint( 2).x = +0.1274709365666390;
   TetrahedronIntRules[8] -> IntPoint( 2).y = +0.1274709365666390;
   TetrahedronIntRules[8] -> IntPoint( 2).z = +0.6175871903000830;
   TetrahedronIntRules[8] -> IntPoint( 3).x = +0.1274709365666390;
   TetrahedronIntRules[8] -> IntPoint( 3).y = +0.6175871903000830;
   TetrahedronIntRules[8] -> IntPoint( 3).z = +0.1274709365666390;
   TetrahedronIntRules[8] -> IntPoint( 4).x = +0.6175871903000830;
   TetrahedronIntRules[8] -> IntPoint( 4).y = +0.1274709365666390;
   TetrahedronIntRules[8] -> IntPoint( 4).z = +0.1274709365666390;
   TetrahedronIntRules[8] -> IntPoint( 5).x = +0.0320788303926323;
   TetrahedronIntRules[8] -> IntPoint( 5).y = +0.0320788303926323;
   TetrahedronIntRules[8] -> IntPoint( 5).z = +0.0320788303926323;
   TetrahedronIntRules[8] -> IntPoint( 6).x = +0.0320788303926323;
   TetrahedronIntRules[8] -> IntPoint( 6).y = +0.0320788303926323;
   TetrahedronIntRules[8] -> IntPoint( 6).z = +0.9037635088221031;
   TetrahedronIntRules[8] -> IntPoint( 7).x = +0.0320788303926323;
   TetrahedronIntRules[8] -> IntPoint( 7).y = +0.9037635088221031;
   TetrahedronIntRules[8] -> IntPoint( 7).z = +0.0320788303926323;
   TetrahedronIntRules[8] -> IntPoint( 8).x = +0.9037635088221031;
   TetrahedronIntRules[8] -> IntPoint( 8).y = +0.0320788303926323;
   TetrahedronIntRules[8] -> IntPoint( 8).z = +0.0320788303926323;
   TetrahedronIntRules[8] -> IntPoint( 9).x = +0.0497770956432810;
   TetrahedronIntRules[8] -> IntPoint( 9).y = +0.0497770956432810;
   TetrahedronIntRules[8] -> IntPoint( 9).z = +0.4502229043567190;
   TetrahedronIntRules[8] -> IntPoint(10).x = +0.0497770956432810;
   TetrahedronIntRules[8] -> IntPoint(10).y = +0.4502229043567190;
   TetrahedronIntRules[8] -> IntPoint(10).z = +0.4502229043567190;
   TetrahedronIntRules[8] -> IntPoint(11).x = +0.4502229043567190;
   TetrahedronIntRules[8] -> IntPoint(11).y = +0.4502229043567190;
   TetrahedronIntRules[8] -> IntPoint(11).z = +0.0497770956432810;
   TetrahedronIntRules[8] -> IntPoint(12).x = +0.4502229043567190;
   TetrahedronIntRules[8] -> IntPoint(12).y = +0.0497770956432810;
   TetrahedronIntRules[8] -> IntPoint(12).z = +0.0497770956432810;
   TetrahedronIntRules[8] -> IntPoint(13).x = +0.0497770956432810;
   TetrahedronIntRules[8] -> IntPoint(13).y = +0.4502229043567190;
   TetrahedronIntRules[8] -> IntPoint(13).z = +0.0497770956432810;
   TetrahedronIntRules[8] -> IntPoint(14).x = +0.4502229043567190;
   TetrahedronIntRules[8] -> IntPoint(14).y = +0.0497770956432810;
   TetrahedronIntRules[8] -> IntPoint(14).z = +0.4502229043567190;
   TetrahedronIntRules[8] -> IntPoint(15).x = +0.1837304473985499;
   TetrahedronIntRules[8] -> IntPoint(15).y = +0.1837304473985499;
   TetrahedronIntRules[8] -> IntPoint(15).z = +0.3162695526014501;
   TetrahedronIntRules[8] -> IntPoint(16).x = +0.1837304473985499;
   TetrahedronIntRules[8] -> IntPoint(16).y = +0.3162695526014501;
   TetrahedronIntRules[8] -> IntPoint(16).z = +0.3162695526014501;
   TetrahedronIntRules[8] -> IntPoint(17).x = +0.3162695526014501;
   TetrahedronIntRules[8] -> IntPoint(17).y = +0.3162695526014501;
   TetrahedronIntRules[8] -> IntPoint(17).z = +0.1837304473985499;
   TetrahedronIntRules[8] -> IntPoint(18).x = +0.3162695526014501;
   TetrahedronIntRules[8] -> IntPoint(18).y = +0.1837304473985499;
   TetrahedronIntRules[8] -> IntPoint(18).z = +0.1837304473985499;
   TetrahedronIntRules[8] -> IntPoint(19).x = +0.1837304473985499;
   TetrahedronIntRules[8] -> IntPoint(19).y = +0.3162695526014501;
   TetrahedronIntRules[8] -> IntPoint(19).z = +0.1837304473985499;
   TetrahedronIntRules[8] -> IntPoint(20).x = +0.3162695526014501;
   TetrahedronIntRules[8] -> IntPoint(20).y = +0.1837304473985499;
   TetrahedronIntRules[8] -> IntPoint(20).z = +0.3162695526014501;
   TetrahedronIntRules[8] -> IntPoint(21).x = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(21).y = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(21).z = +0.0229177878448171;
   TetrahedronIntRules[8] -> IntPoint(22).x = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(22).y = +0.0229177878448171;
   TetrahedronIntRules[8] -> IntPoint(22).z = +0.5132800333608811;
   TetrahedronIntRules[8] -> IntPoint(23).x = +0.0229177878448171;
   TetrahedronIntRules[8] -> IntPoint(23).y = +0.5132800333608811;
   TetrahedronIntRules[8] -> IntPoint(23).z = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(24).x = +0.5132800333608811;
   TetrahedronIntRules[8] -> IntPoint(24).y = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(24).z = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(25).x = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(25).y = +0.5132800333608811;
   TetrahedronIntRules[8] -> IntPoint(25).z = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(26).x = +0.5132800333608811;
   TetrahedronIntRules[8] -> IntPoint(26).y = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(26).z = +0.0229177878448171;
   TetrahedronIntRules[8] -> IntPoint(27).x = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(27).y = +0.0229177878448171;
   TetrahedronIntRules[8] -> IntPoint(27).z = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(28).x = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(28).y = +0.5132800333608811;
   TetrahedronIntRules[8] -> IntPoint(28).z = +0.0229177878448171;
   TetrahedronIntRules[8] -> IntPoint(29).x = +0.0229177878448171;
   TetrahedronIntRules[8] -> IntPoint(29).y = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(29).z = +0.5132800333608811;
   TetrahedronIntRules[8] -> IntPoint(30).x = +0.5132800333608811;
   TetrahedronIntRules[8] -> IntPoint(30).y = +0.0229177878448171;
   TetrahedronIntRules[8] -> IntPoint(30).z = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(31).x = +0.0229177878448171;
   TetrahedronIntRules[8] -> IntPoint(31).y = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(31).z = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(32).x = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(32).y = +0.2319010893971509;
   TetrahedronIntRules[8] -> IntPoint(32).z = +0.5132800333608811;
   TetrahedronIntRules[8] -> IntPoint(33).x = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(33).y = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(33).z = +0.7303134278075384;
   TetrahedronIntRules[8] -> IntPoint(34).x = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(34).y = +0.7303134278075384;
   TetrahedronIntRules[8] -> IntPoint(34).z = +0.1937464752488044;
   TetrahedronIntRules[8] -> IntPoint(35).x = +0.7303134278075384;
   TetrahedronIntRules[8] -> IntPoint(35).y = +0.1937464752488044;
   TetrahedronIntRules[8] -> IntPoint(35).z = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(36).x = +0.1937464752488044;
   TetrahedronIntRules[8] -> IntPoint(36).y = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(36).z = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(37).x = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(37).y = +0.1937464752488044;
   TetrahedronIntRules[8] -> IntPoint(37).z = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(38).x = +0.1937464752488044;
   TetrahedronIntRules[8] -> IntPoint(38).y = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(38).z = +0.7303134278075384;
   TetrahedronIntRules[8] -> IntPoint(39).x = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(39).y = +0.7303134278075384;
   TetrahedronIntRules[8] -> IntPoint(39).z = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(40).x = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(40).y = +0.1937464752488044;
   TetrahedronIntRules[8] -> IntPoint(40).z = +0.7303134278075384;
   TetrahedronIntRules[8] -> IntPoint(41).x = +0.7303134278075384;
   TetrahedronIntRules[8] -> IntPoint(41).y = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(41).z = +0.1937464752488044;
   TetrahedronIntRules[8] -> IntPoint(42).x = +0.1937464752488044;
   TetrahedronIntRules[8] -> IntPoint(42).y = +0.7303134278075384;
   TetrahedronIntRules[8] -> IntPoint(42).z = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(43).x = +0.7303134278075384;
   TetrahedronIntRules[8] -> IntPoint(43).y = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(43).z = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(44).x = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(44).y = +0.0379700484718286;
   TetrahedronIntRules[8] -> IntPoint(44).z = +0.1937464752488044;
}

/// Integration rules for reference cube
void IntegrationRules::CubeIntegrationRules()
{
   int i, k, l, m, np;

   CubeIntRules.SetSize(8);

   for (i = 0; i < CubeIntRules.Size(); i++)
      CubeIntRules[i] = NULL;


   for(i = 0; i < CubeIntRules.Size(); i++)
   {
      np = SegmentIntRules[i] -> GetNPoints();
      CubeIntRules[i] = new IntegrationRule(np*np*np);
      for (k = 0; k < np; k++)
         for (l = 0; l < np; l++)
            for (m = 0; m < np; m++)
            {
               CubeIntRules[i] -> IntPoint((k*np+l)*np+m).x =
                  SegmentIntRules[i] -> IntPoint(k).x;

               CubeIntRules[i] -> IntPoint((k*np+l)*np+m).y =
                  SegmentIntRules[i] -> IntPoint(l).x;

               CubeIntRules[i] -> IntPoint((k*np+l)*np+m).z =
                  SegmentIntRules[i] -> IntPoint(m).x;

               CubeIntRules[i] -> IntPoint((k*np+l)*np+m).weight =
                  SegmentIntRules[i] -> IntPoint(k).weight *
                  SegmentIntRules[i] -> IntPoint(l).weight *
                  SegmentIntRules[i] -> IntPoint(m).weight;
            }
   }

}
