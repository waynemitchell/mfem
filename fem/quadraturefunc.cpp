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
        for(int i = 0; i < np ; ++i)
            ir->IntPoint(i).x = double(i+1) / double(np + 1);

        if(np == 1)
        {
            ir->IntPoint(0).weight = 1.;
            return;
        }

        if(np < 11)
            NewtonPolynomialNewtonCotesWeights(ir,true);
        else
            CalculateLagrangeWeights(ir);
    }

    void QuadratureFunctions1D::ClosedEquallySpaced(const int np, IntegrationRule* ir)
    {
        ir->SetSize(np);
        MFEM_ASSERT(np != 1, "");

        double dx = 1./(double (np-1) );
        for(int i = 0; i < np ; ++i)
            ir->IntPoint(i).x = double(i) * dx;

       if(np < 14)
           NewtonPolynomialNewtonCotesWeights(ir,false);
       else
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
        *        *
        *  This algorithm suffers from round-off error that increases with larger n
        *
        *  Compiler options for optimization exacerbate the roundoff error
        *
        *  This was tested by comparing the difference of the MFEM library result
        *  to the same code, but compiled in -O0 mode in a separate program
        */
        const int n = ir->Size();

        /// Coding by Maginot
        /// array to actually store the coefficients as we expand the numerator
        double *coeff_store1;
        double *coeff_store2;

        double *start_ptr;
        double *dest_ptr;
        double *temp_ptr;

        coeff_store1 = new double[n];
        coeff_store2 = new double[n];

        /// loop over all quadrature absicca
        for(int i = 0 ; i < n ; ++i)
        {
          /// clear out the working arrays
          for(int e = 0 ; e < n ; ++e)
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
          for(int j=0; j < n ; ++j)
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
            for(int p = 0 ; p < n ; ++p)
              dest_ptr[p] = 0.;
          }

          /// start_ptr now has the fully expanded numerator, integrate over[0,1] to get weight
          double w = 0.;
          for(int p = 0 ; p < n ; ++p)
            w += start_ptr[p]/( double(p+1) );

          ir->IntPoint(i).weight = w/denom;
        }
        delete[] coeff_store1;
        delete[] coeff_store2;
    }
    
    void QuadratureFunctions1D::NewtonPolynomialNewtonCotesWeights(IntegrationRule *ir ,
                    const bool is_open)
    {
       /* Calculate the Newton-Cotes rational number weights
        *
        *  Use Newton-polynomials, and special relations applicable only to
        *  equally spaced quadrature points.  Work with integer math as long
        *  as possible.
        *
        *  For info on divided differences and Newton polynomials, see:
        *
        *  https://en.wikipedia.org/wiki/Newton_polynomial
        *  https://en.wikipedia.org/wiki/Divided_differences
        *
        *  This algorithm suffers from overflow errors
        *      for intermediate (>13) points
        */

        /*
         *  Let the Lagrange polynomial of degree n, which interpolates at point i
         *  be P_i^n(x).  Newton polynomials, represent P_i^n(x) as:
         *
         *  P_i^n(x) = \sum_{j=0}^n{ d_j * L_j }
         *
         *  d_j is the corresponding element of the divided difference table
         *  L_j is the polynomial expansion of interpolation points:
         *
         *  L_j = \prod_{k=0}^{j-1} { (x - x_k) }
         *
         *  where x_k are the Lagrange interpolation (Newton-Cotes quadrature) points
         *
         *  With equally spaced interpolation points in the interval [0,1], x_k can be represented as:
         *
         *  x_k = 0 + m h
         *
         *  Where for Open Newton-Cotes, quadrature
         *
         *  h = 1/(np + 1) and m=k+1
         *
         *  and closed Newton-Cotes quadrature:
         *
         *  h = 1/(np - 1) and m=k
         *
         */

        /// Expand the L polynomial (does not change with interpolation point)
        /*
         * We can visualize the expansion of the L polynomials as follows [for a 3 point example]
         *
         *      For Open Netwon Cotes,
         *         L(x) = (x-x_1) * (x - x_2) * (x - x_3) ...
         *         L(x) = (x - h) * (x - 2h ) * (x - 3h ) ...
         *         L(x) = (x + c_1 h) * (x + c_2 h) * (x + c_3 h) ...
         *
         * The first term is:
         *
         *        x^0   | x^1    | x^2
         * h^0 | ----   |   1    |
         * h^1 |   -1   |
         * h^2 |
         *
         * The second term is:
         *
         *        x^0   | x^1        | x^2
         * h^0 | ----   |  ---       |  1
         * h^1 | ---    |-1 + 1*(-2)
         * h^2 | -1*(-2)
         *
         * Simplifying, the second term is:
         *
         *        x^0   | x^1    | x^2
         * h^0 | ----   |  ---   |  1
         * h^1 | ---    | -3
         * h^2 |   2
         *
         * The third term is:
         *
         *         x^0   | x^1        | x^2          |    x^3
         * h^0 |  ---   |  ---        |   ---        | 1
         * h^1 |  ---   |  ---        | 1*(-3) -3
         * h^2 |   ---  |(-3)*(-3)+ 2
         * h^3 |  2*(-3)
         *
         */
        const int np = ir->Size();

        /// lPoly will store the succession of L polynomials
        int** lPoly = new int*[np];
        for(int j = 0 ; j<np ; ++j)
            lPoly[j] = new int[np];

        /// zero out data
        for(int j = 0 ; j < np ; ++j)
            for(int k =0 ; k < np ; ++k)
                lPoly[j][k] = 0;


        /// Each column of lPoly will hold one L polynomial
        /// highest power of h will be at lPoly[0][col]
        if(is_open)
        {
            /// linear polynomial is (x - h)
            lPoly[0][0] = -1;
            lPoly[1][0] = 1;
        }
        else
        {
            /// linear Polynomial is (x - 0)
            lPoly[0][0] = 0;
            lPoly[1][0] = 1;
        }

        /** Build the L Polynomials greater than linear that we need
         *  lPoly is ordered [row][column]
         *
         *  Each column is the next L polynomial
         *
         *  Column = 0 is the linear polynomial
         *
         *  row = 0 holds the highest degree power of h for a given L polynomial
         *
         */
        for(int col = 1; col < (np-1) ; ++col)
        {
            /// multiply the last lPoly by x
            for(int row = 0; row <= col; ++row)
                lPoly[row+1][col] = lPoly[row][col-1];

            /// multiply the last lPoly by coeff*h
            int coeff = (is_open ? -(col+1)  : -col );
            for(int row = 0 ; row <= col; ++row)
                lPoly[row][col] += coeff*lPoly[row][col-1];
        }

        /// Allocate space that will track the numerator of the divided difference tables
        int **DivTable = new int*[np];
        for(int i = 0 ; i < np ; ++i)
            DivTable[i] = new int[np];

        /// array to hold the rational numerator coefficients of x^p
        /// element [0] holds the x^0 power coefficients
        long long int *num_coeff = new long long int[np];

        /// Loop over all quadrature points  (divided difference table changes with interpolation point)
        for(int i = 0 ; i < np; ++i)
        {
            /// Clear out the divided difference table
            for(int m = 0; m < np; ++m)
            {
                for(int n = 0 ; n<np ; ++n)
                    DivTable[m][n] = 0;
            }

            /// This Lagrange polynomial is non-zero only at the interpolation point
            DivTable[i][0] = 1;

            /// Calculate the numerator of the divide difference, column by column
            for(int m = 1; m < np ; ++m)
            {
                for(int n = 0; n < (np - m); ++n)
                {
                    DivTable[n][m] = DivTable[n+1][m-1] - DivTable[n][m-1];
                }
            }

            /**
             * Lagrange polynomial =
             *
             *    P_i^{np} = DivTable[0][0] + \sum_{j=1}^{np-1} LPoly_j * DivTable[0][j] / (j! h^j)
             *
             *    LPoly_j = \sum_{i = 0}^j{ x^i h^(j-i) lPoly[j-i][j-1] }
             *
             *    We now wish to find the integer coefficients in front of each polynomial of x in P_i
             *
             *    The highest order LPoly_j comes when j = np - 1
             *
             *    This also has the largest denominator
             *
             */

            // highest factorial used is also the denominator
            unsigned long long int denom = 1;
            for(int p = 1 ; p <np; ++p)
                denom *= p;

            /// initialize the numerator
            for(int p = 0 ; p<np ; ++p)
                num_coeff[p] = 0;

            int h_inv = (is_open ? np+1 : np-1 );

            /// sum all terms of the Newton Polynomial
            /// use denom as a common denomintator
            num_coeff[0] = DivTable[0][0]*denom;

            /// access all polynomials stored in lPoly, in order
            unsigned long long int curr_fac = 1;
            for(int p = 1 ; p < np ; ++p)
            {
                int dt = DivTable[0][p];
                /// P_i = \sum{ dt * lPoly_p / (p! h^p) }
                unsigned long long int lmc = denom/curr_fac;

                unsigned long long int h_track = 1;
                for(int j = 0 ; j <= p ; ++j)
                {
                    num_coeff[j] += lmc*dt*lPoly[j][p-1]*h_track;

                    h_track *= h_inv;
                }

                curr_fac *= (p+1);
            }

            /**
             *  we now have the rational number coefficients of the powers of P_i^{np-1}
             *  integrate to get the quadrature weight
             *
             *  Integrate the polynomial.  If possible, reduce the numerator coefficients
             *  Otherwise, increase the denominator and multiply all terms of the numerator
             *  by (j+1), except for the offending polynomial integrand)
             */
            for(int j = 1 ; j <np ;  ++j)
            {
                int div = j+1;
                if( (num_coeff[j] % div) == 0 )
                {
                    num_coeff[j] /= div;
                }
                else
                {
                    denom *= div;
                    for(int k = 0 ; k < np ; ++k)
                    {
                        if( k==j)
                            continue;
                        else
                            num_coeff[k] *= div;
                    }
                }
            }
            long long int sum = 0;
            for(int p = 0 ; p < np ; ++p)
                sum += num_coeff[p];

            ir->IntPoint(i).weight = double(sum) / double(denom);

        }
        /// Delete working arrays
        for(int j = 0 ; j < np ; ++j)
        {
            delete[] DivTable[j];
            delete[] lPoly[j];
        }

        delete[] DivTable;
        delete[] lPoly;
        delete[] num_coeff;

        return;
    }
}
