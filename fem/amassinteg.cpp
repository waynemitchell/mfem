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
#include "../config/config.hpp"
#ifdef MFEM_USE_ACROTENSOR

#include "amassinteg.hpp"

namespace mfem {

AcroMassIntegrator::AcroMassIntegrator(Coefficient &q, FiniteElementSpace &f, bool gpu) :
  PAIntegrator(q,f,gpu)
{
    if (onGPU) 
    {
        TE.SetExecutorType("SMChunkPerBlock");
        //acro::setCudaContext(occa::cuda::getContext(device));
    } 
    else 
    {
        TE.SetExecutorType("CPUInterpreted");
    }

    const IntegrationRule *ir1D = &IntRules.Get(Geometry::SEGMENT, ir->GetOrder());
    nDof1D = FEOrder + 1;
    nQuad1D = ir1D->GetNPoints();

    if (hasTensorBasis) {
        H1_FECollection fec(FEOrder,1);
        const FiniteElement *fe = fec.FiniteElementForGeometry(Geometry::SEGMENT);
        Vector eval(nDof1D);
        DenseMatrix deval(nDof1D,1);
        B.Init(nQuad1D, nDof1D);
        std::vector<int> wdims(nDim, nQuad1D);
        W.Init(wdims);

        Vector w(nQuad1D);
        for (int k = 0; k < nQuad1D; ++k)
        {
            const IntegrationPoint &ip = ir1D->IntPoint(k);
            fe->CalcShape(ip, eval);

            B(k,0) = eval(0);
            B(k,nDof1D-1) = eval(1);        
            for (int i = 1; i < nDof1D-1; ++i)
            {
                B(k,i) = eval(i+1);
            }
            w(k) = ip.weight;
        }

        if (nDim == 1)
        {
            for (int k1 = 0; k1 < nQuad1D; ++ k1)
            {
                W(k1) = w(k1);
            }
        }
        else if (nDim == 2)
        {
            for (int k1 = 0; k1 < nQuad1D; ++ k1)
            {
                for (int k2 = 0; k2 < nQuad1D; ++ k2)
                {
                    W(k1,k2) = w(k1)*w(k2);
                }
            }
        }
        else if (nDim == 3)
        {
            for (int k1 = 0; k1 < nQuad1D; ++ k1)
            {
                for (int k2 = 0; k2 < nQuad1D; ++ k2)
                {
                    for (int k3 = 0; k3 < nQuad1D; ++ k3)
                    {
                        W(k1,k2,k3) = w(k1)*w(k2)*w(k3);
                    }                       
                }
            }
        }        
    } else {
        H1_FECollection fec(FEOrder,1);
        const FiniteElement *fe = fec.FiniteElementForGeometry(Geometry::SEGMENT);
        Vector eval(nDof);
        DenseMatrix deval(nDof,nDim);
        B.Init(nQuad, nDof);
        W.Init(nQuad);
        for (int k = 0; k < nQuad; ++k)
        {
            const IntegrationPoint &ip = ir->IntPoint(k);
            fe->CalcShape(ip, eval);
            for (int i = 0; i < nDof; ++i)
            {
                for (int d = 0; d < nDim; ++d)
                {
                    B(k,i) = eval(i);
                }
            }
            W(k) = ip.weight;
        }
    }

    if (onGPU)
    {
        B.MapToGPU();
        W.MapToGPU();
    }
}

AcroMassIntegrator::~AcroMassIntegrator() 
{

}


void AcroMassIntegrator::BatchedPartialAssemble() 
{

    //Initilze the tensors
    acro::Tensor J,Jdet,q;
    if (hasTensorBasis)
    {
        if (nDim == 1)
        {
            D.Init(nElem, nDim, nDim, nQuad1D);
            J.Init(nElem, nQuad1D, nQuad1D, nDim, nDim);         
            Jdet.Init(nElem, nQuad1D);
            q.Init(nElem, nQuad1D);
        }
        else if (nDim == 2)
        {
            D.Init(nElem, nDim, nDim, nQuad1D, nQuad1D);
            J.Init(nElem, nQuad1D, nQuad1D, nDim, nDim);
            Jdet.Init(nElem, nQuad1D, nQuad1D);
            q.Init(nElem, nQuad1D, nQuad1D);
        }
        else if (nDim == 3)
        {
            D.Init(nElem, nDim, nDim, nQuad1D, nQuad1D, nQuad1D);
            J.Init(nElem, nQuad1D, nQuad1D, nQuad1D, nDim, nDim);
            Jdet.Init(nElem, nQuad1D, nQuad1D, nQuad1D);
            q.Init(nElem, nQuad1D, nQuad1D,nQuad1D);
        }
        else
        {
            mfem_error("AcroDiffusionIntegrator tensor computations don't support dim > 3.");
        }
    }
    else
    {
        D.Init(nElem, nDim, nDim, nQuad);
        J.Init(nElem, nQuad, nDim, nDim);
        Jdet.Init(nElem, nQuad);
        q.Init(nElem, nQuad);
    }

    //Fill the jacobians and coefficients
    int idx = 0;
    for (int e = 0; e < nElem; ++e)
    {
        ElementTransformation *Trans = fes->GetElementTransformation(e);
        for (int k = 0; k < nQuad; ++k)
        {
            const IntegrationPoint &ip = ir->IntPoint(k);
            Trans->SetIntPoint(&ip);
            q[e*nQuad+k] = Q->Eval(*Trans, ip);
            const DenseMatrix &JMat = Trans->Jacobian();
            for (int m = 0; m < nDim; ++m)
            {
                for (int n = 0; n < nDim; ++n)
                {
                    J[idx] = JMat(m,n);
                    idx ++;
                }
            }
        }
    }
    TE.BatchMatrixDet(Jdet, J);


    if (hasTensorBasis) 
    {
        if (nDim == 1) {
            TE["D_e_k = W_k Q_e_k Jdet_e_k"]
              (D, W, q, Jdet);
        } 
        else if (nDim == 2) 
        {
            TE["D_e_k1_k2 = W_k1_k2 Q_e_k1_k2 Jdet_e_k1_k2"]
              (D, W, q, Jdet);
        } 
        else if (nDim == 3)
        {
            TE["D_e_k1_k2_k3 = W_k1_k2_k3 Q_e_k1_k2_k3 Jdet_e_k1_k2_k3"]
              (D, W, q, Jdet);
        } 
        else 
        {
          mfem_error("AcroMassIntegrator tensor computations don't support dim > 3.");
        }
    } 
    else 
    {
        TE["D_e_k = W_k Q_e_k Jdet_e_k"]
          (D, W, q, Jdet);
    }
}


void AcroMassIntegrator::BatchedAssembleMatrix() {

    if (!M.IsInitialized()) 
    {
        if (hasTensorBasis) 
        {
            if (nDim == 1) 
            {
                M.Init(nElem, nDof1D, nDof1D);
                if (onGPU) {M.SwitchToGPU();}
            } 
            else if (nDim == 2) 
            {
                M.Init(nElem, nDof1D, nDof1D, nDof1D, nDof1D);
                if (onGPU) {M.SwitchToGPU();}
            } 
            else if (nDim == 3) 
            {
                M.Init(nElem, nDof1D, nDof1D, nDof1D, nDof1D, nDof1D, nDof1D);
                if (onGPU) {M.SwitchToGPU();}
            }
        } 
        else 
        {
            M.Init(nElem, nDof, nDof);
            if (onGPU) {M.SwitchToGPU();}
        }
    }

    if (hasTensorBasis) {
        if (nDim == 1) {
            TE["M_e_i1_j1 = B_k1_i1 B_k1_j1 D_e_k1"]
                (M, B, B, D);
        } 
        else if (nDim == 2) 
        {
            TE["M_e_i1_i2_j1_j2 = B_k1_i1 B_k1_j1 B_k2_i2 B_k2_j2 D_e_k1_k2"]
                (M, B, B, B, B, D);
        } 
        else if (nDim == 3) 
        {
            TE["M_e_i1_i2_i3_j1_j2_j3 = B_k1_i1 B_k1_j1 B_k2_i2 B_k2_j2 B_k3_i3 B_k3_j3 D_e_k1_k2_k3"]
                (M, B, B, B, B, B, B, D);
        }
    } 
    else 
    {
        TE["M_e_i_j = B_k_i B_k_j D_e_k"](M, B, B, D);
    }
}

void AcroMassIntegrator::PAMult(const Vector &x, Vector &y) {

  if (!T1.IsInitialized() && hasTensorBasis) {
    if (nDim == 1) {
      T1.Init(nElem, nQuad1D);
      if (onGPU) {
        T1.SwitchToGPU();
      }
    } else if (nDim == 2) {
      T1.Init(nElem, nQuad1D, nDof1D);
      T2.Init(nElem, nQuad1D, nQuad1D);
      if (onGPU) {
        T1.SwitchToGPU();
        T2.SwitchToGPU();
      }
    } else if (nDim == 3) {
      T1.Init(nElem, nQuad1D, nDof1D, nDof1D);
      T2.Init(nElem, nQuad1D, nQuad1D, nDof1D);
      T3.Init(nElem, nQuad1D, nQuad1D, nQuad1D);
      if (onGPU) {
        T1.SwitchToGPU();
        T2.SwitchToGPU();
        T3.SwitchToGPU();
      }
    }
  }

  double *x_ptr = const_cast<double*>(x.GetData());
  double *y_ptr = y.GetData();
  if (hasTensorBasis) {
    if (nDim == 1) {
      acro::Tensor X(nElem, nDof1D, x_ptr, x_ptr, onGPU);
      acro::Tensor Y(nElem, nDof1D, y_ptr, y_ptr, onGPU);
      TE["T1_e_k1 = D_e_k1 B_k1_j1 X_e_j1"](T1, D, B, X);
      TE["Y_e_i1 = B_k1_i1 T1_e_k1"](Y, B, T1);
    } else if (nDim == 2) {
      acro::Tensor X(nElem, nDof1D, nDof1D, x_ptr, x_ptr, onGPU);
      acro::Tensor Y(nElem, nDof1D, nDof1D, y_ptr, y_ptr, onGPU);
      TE["T1_e_k2_j1 = B_k2_j2 X_e_j1_j2"](T1, B, X);
      TE["T2_e_k1_k2 = D_e_k1_k2 B_k1_j1 T1_e_k2_j1"](T2, D, B, T1);
      TE["T1_e_k1_i2 = B_k2_i2 T2_e_k1_k2"](T1, B, T2);
      TE["Y_e_i1_i2 = B_k1_i1 T1_e_k1_i2"](Y, B, T1);
    } else if (nDim == 3) {
      acro::Tensor X(nElem, nDof1D, nDof1D, nDof1D, x_ptr, x_ptr, onGPU);
      acro::Tensor Y(nElem, nDof1D, nDof1D, nDof1D, y_ptr, y_ptr, onGPU);
      TE["T1_e_k3_j1_j2 = B_k3_j3 X_e_j1_j2_j3"](T1, B, X);
      TE["T2_e_k2_k3_j1 = B_k2_j2 T1_e_k3_j1_j2"](T2, B, T1);
      TE["T3_e_k1_k2_k3 = D_e_k1_k2_k3 B_k1_j1 T2_e_k2_k3_j1"](T3, D, B, T2);
      TE["T2_e_k1_k2_i3 = B_k3_i3 T3_e_k1_k2_k3"](T2, B, T3);
      TE["T1_e_k1_i2_i3 = B_k2_i2 T2_e_k1_k2_i3"](T1, B, T2);
      TE["Y_e_i1_i2_i3 = B_k1_i1 T1_e_k1_i2_i3"](Y, B, T1);
    }
  } else {
    mfem_error("AcroMassIntegrator PAMult on simplices not supported");
  }

}

}

#endif
