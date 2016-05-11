// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

// Implementation of QuadratureFunctions1D class


#include <cmath>
#include "quadraturefunc.hpp"

namespace mfem
{
    QuadratureFunctions1D::QuadratureFunctions1D() {}
    QuadratureFunctions1D::~QuadratureFunctions1D() {}

    void QuadratureFunctions1D::GaussLegendre(const int np, IntegrationRule* ir)
    {
       ir->SetSize(np);

       int n = np;
       int m = (n+1)/2;
       for(int i = 1; i <= m; i++)
       {
          double z = cos(M_PI * (i - 0.25) / (n + 0.5));
          double pp, p1;
          while (1)
          {
             p1 = 1;
             double p2 = 0;
             for (int j = 1; j <= n; j++)
             {
                double p3 = p2;
                p2 = p1;
                p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
             }
             // p1 is Legendre polynomial

             pp = n * (z*p1-p2) / (z*z - 1);

             if (fabs(p1/pp) < 2e-16) { break; }

             z = z - p1/pp;
          }

          z = ((1 - z) + p1/pp)/2;

          ir->IntPoint(i-1).x  = z;
          ir->IntPoint(n-i).x  = 1 - z;
          ir->IntPoint(i-1).weight =
                  ir->IntPoint(n-i).weight = 1./(4*z*(1 - z)*pp*pp);
       }
    }

    void QuadratureFunctions1D::GaussLobatto(const int np, IntegrationRule* ir)
    {
        /**
        * An np point Gauss-Lobatto quadrature has (np - 2) free abscissa
        * the other (2) abscissa are the interval endpoints.
        *
        *  The interior x_i are the zeros of P'_{np-1}(x)
        *  The weights of the interior points on ther interval [-1,1] are:
        *  w_i = 2/(np*(np-1)*[P_{np-1}(x_i)]^2)
        *
        * The end point weights (on [-1,1]) are:
        *  w_{end} = 2/(np*(np-1))
        *
        *  The interior absicca are found via a nonlinear solve
        *  the initial guess for each point is the corresponding Chebyshev point
        *
        *  After we find all points on the interval [-1,1], we will map and
        *  scale the points and weights to the MFEM natural interval [0,1]
        *
        *  For reference purposes, see Appendix A of:
        *  1) E. E. Lewis and W. F. Millier, "Computational Methods of Neutron Transport"
        *
        *  and
        *
        *  2) the QUADRULE software by John Burkardt:
        *       https://people.sc.fsu.edu/~jburkardt/cpp_src/quadrule/quadrule.cpp
        *
        **/
        ir->SetSize(np);
        if( np ==1 )
        {
            ir->IntPoint(0).x = 0.;
            ir->IntPoint(0).weight = 2.;
        }
        else
        {
            /// endpoints and respective weights
            ir->IntPoint(0).x = -1.;
            ir->IntPoint(np-1).x = 1.;
            ir->IntPoint(0).weight = ir->IntPoint(np-1).weight = 2./ double(np*(np-1));

            /// interior points and weights
            for(int i = 1 ; i < (np-1) ; ++i)
            {
                /// initial guess is the corresponding Chebyshev point
                double x_i = cos(M_PI * double (np -1 -i) / double(np -1) );
                double p_l = 0.;
                for(int iter = 0 ; ; true)
                {
                    /// build Legendre polynomials, up to P_{np}(x_i)
                    double p_lm1 = 1.;
                    p_l = x_i;
                    double p_lp1 = 0.;

                    /// derivatives of Legendre polynomials, up to D_{np}(x_i)
                    double d_lm1 = 0.;
                    double d_l = 1.;
                    double d_lp1 = 0.;

                    for(int l = 1 ; l < (np-1) ; ++l)
                    {
                        double ell = (double) l;
                        /** The legendre polynomials can be built be recursion:
                         x * P_l(x) = 1/(2*l+1)*[ (l+1)*P_{l+1}(x) + l*P_{l-1} ]
                        */
                        p_lp1 = ( (2.*ell + 1.)*x_i*p_l - ell*p_lm1)/(ell + 1.);

                        /**
                         * Taking the derivative of the Legendre recursion
                         * (we will want lower order derivatives to evaluate the jacobian)
                         */
                        d_lp1 = ( (2.*ell + 1.)*(p_l + x_i*d_l) - ell*d_lm1)/(ell + 1.);

                        d_lm1 = d_l;
                        d_l = d_lp1;
                        p_lm1 = p_l;
                        p_l = p_lp1;
                    }
                    /// after this loop, p_l holds P_{np-1}(x_i)
                    /// resid = (x^2-1)*P'_l(x_i)
                    /// but use the recurrence relationship
                    /// (x^2 -1)P'_l(x) = l*[ x*P_l(x) - P_{l-1}(x) ]
                    double resid = (double) (np-1) * (x_i*p_l - p_lm1);

                    /** The derivative of the residual is:
                     *
                     * \frac{d}{d x} \left[ l*[ x*P_l(x) - P_{l-1}(x) ] \right] =
                     * l*( P_l(x) + x*P_l'(x) - P_{l-1}'(x) )
                     */
                    double deriv = (double) (np-1) * (p_l + x_i*d_l - d_lm1);

                    double x_new = x_i - resid/deriv;
                    if(std::fabs(x_new - x_i) < 2.0E-16)
                        break;
                    else
                        x_i = x_new;
                }
                ir->IntPoint(i).x = x_i;
                /// w_i = 2/[ n*(n-1)*[P_{n-1}(x_i)]^2 ]
                ir->IntPoint(i).weight = 2./( double(np*(np-1)) * p_l * p_l);
            }
        }

        /// Map to the interval [0,1] and scale the weights
        for(int i = 0 ; i < np ; ++i)
        {
          ir->IntPoint(i).x = (1. + ir->IntPoint(i).x)/2. ;
          ir->IntPoint(i).weight /= 2.;
        }
    }

    void QuadratureFunctions1D::OpenEquallySpaced(const int np, IntegrationRule* ir)
    {
        ir->SetSize(np);

        /**
         * The Newton-Cotes quadrature is based on exactly integrating the interpolatory
         * polynomial exactly that goes through the equally spaced quadrature points
         *
         */
        double dx = 1./(double (np+1));
        for(int i = 0; i < np ; ++i)
            ir->IntPoint(i).x = double(i+1)*dx;

        CalculateLagrangeWeights(ir);
    }

    void QuadratureFunctions1D::ClosedEquallySpaced(const int np, IntegrationRule* ir)
    {
        ir->SetSize(np);
        MFEM_ASSERT(np != 1, "");

        double dx = 1./(double (np-1) );
        for(int i = 0; i < np ; ++i)
            ir->IntPoint(i).x = double(i) * dx;

        CalculateLagrangeWeights(ir);
    }

    void QuadratureFunctions1D::CalculateLagrangeWeights(IntegrationRule *ir)
    {
        /**
        * The Lagrange polynomials are:
        * p_i = \prod_{j \neq i}{     \frac{x - x_j }{x_i - x_j}    }
        *
        * The weight associated with each abcissa is the integral of p_i over [0,1]
        *
        * To calculate the integral of p_i, we first expand p_i
        *
        */
        const int np = ir->Size();
        std::cout << "Np: " << np << std::endl;

        /// array to actually store the coefficients as we expand the numerator
        double *coeff_store1;
        double *coeff_store2;
        
        double *start_ptr;
        double *dest_ptr;
        double *temp_ptr;
        
        coeff_store1 = new double[np];
        coeff_store2 = new double[np];

        /// loop over all quadrature absicca
        for(int i = 0 ; i < np ; ++i)
        {
          /// clear out the working arrays
          for(int e = 0 ; e < np ; ++e)
          {
            coeff_store1[e] = 0.;
            coeff_store2[e] = 0.;
          }
          
          start_ptr = coeff_store1;
          dest_ptr = coeff_store2;
          start_ptr[0] = 1.;
          
          /// polynomial order we will expand to after adding the next (x - x_j) term
          int l = 1;
          
          double denom = 1.;
          double x_i = ir->IntPoint(i).x; 
          for(int j=0; j < np ; ++j)
          {
            if(j==i)
              continue;
            
            /// x * previously expanded terms
            for(int p = 0; p < l ; ++p)
              dest_ptr[p+1] = start_ptr[p];
            
            /// -x_j * previously expanded terms
            double x_j = ir->IntPoint(j).x;
            for(int p = 0; p < l ; ++p)
              dest_ptr[p] -= start_ptr[p]*x_j;
           
            /// track the denominator as well
            denom *= x_i - x_j;
              
            /// swap where we are writing to and reading from
            temp_ptr = start_ptr;
            start_ptr = dest_ptr;
            dest_ptr = temp_ptr;
            
            /// record that we now have a higher polynomial degree expansion
            ++l;  
            
            /// clean out the destiation array
            for(int p = 0 ; p < np ; ++p)
              dest_ptr[p] = 0.;
          }
          
          /// start_ptr now has the fully expanded numerator, integrate over[0,1] to get weight
          double w = 0.;
          for(int p = 0 ; p < np ; ++p)
            w += start_ptr[p]/( double(p+1) );
            
          ir->IntPoint(i).weight = w/denom;
        }
        delete[] coeff_store1;
        delete[] coeff_store2;
    }
    
}
