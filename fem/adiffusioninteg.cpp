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
#if defined(MFEM_USE_ACROTENSOR)
#include "../config/config.hpp"
#include "adiffusioninteg.hpp"

namespace mfem 
{

AcroDiffusionIntegrator::AcroDiffusionIntegrator(Coefficient &q, FiniteElementSpace &f, bool gpu) :
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
        G.Init(nQuad1D, nDof1D);
        std::vector<int> wdims(nDim, nQuad1D);
        W.Init(wdims);

        Vector w(nQuad1D);
        for (int k = 0; k < nQuad1D; ++k)
        {
            const IntegrationPoint &ip = ir1D->IntPoint(k);
            fe->CalcShape(ip, eval);
            fe->CalcDShape(ip, deval);

            B(k,0) = eval(0);
            B(k,nQuad1D-1) = eval(1);
            G(k,0) = deval(0,0);
            G(k,nQuad1D-1) = deval(1,0);            
            for (int i = 2; i < nDof1D-1; ++i)
            {
                B(k,i-1) = eval(i);
                G(k,i-1) = deval(i,0);
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
        G.Init(nQuad, nDof,nDim);
        W.Init(nQuad);
        for (int k = 0; k < nQuad; ++k)
        {
            const IntegrationPoint &ip = ir->IntPoint(k);
            fe->CalcDShape(ip, deval);
            for (int i = 0; i < nDof; ++i)
            {
                for (int d = 0; d < nDim; ++d)
                {
                    G(k,i,d) = deval(i,d);
                }
            }
            W(k) = ip.weight;
        }
    }

    if (onGPU)
    {
        B.MapToGPU();
        G.MapToGPU();
        W.MapToGPU();
    }
}


AcroDiffusionIntegrator::~AcroDiffusionIntegrator() {

}


void AcroDiffusionIntegrator::ComputeBTilde() {
    Btil.SetSize(nDim);
    for (int d = 0; d < nDim; ++d) 
    {
        Btil[d] = new acro::Tensor(nDim, nDim, nQuad1D, nDof1D, nDof1D);
        if (onGPU) 
        {
            Btil[d]->SwitchToGPU();
        }
        acro::Tensor Bsub(nQuad1D, nDof1D, nDof1D, Btil[d]->GetCurrentData(), Btil[d]->GetCurrentData(), onGPU);
        for (int mi = 0; mi < nDim; ++mi) 
        {
            for (int ni = 0; ni < nDim; ++ni) 
            {
                int offset = (nDim*mi + ni) * nQuad1D*nDof1D*nDof1D;
                Bsub.Retarget(Btil[d]->GetCurrentData() + offset, Btil[d]->GetCurrentData() + offset);
                acro::Tensor &BGM = (mi == d) ? G : B;
                acro::Tensor &BGN = (ni == d) ? G : B;
                TE["Bsub_k1_i1_j1 = M_k1_i1 N_k1_j1"](Bsub, BGM, BGN);
            }
        }
    }
}


void AcroDiffusionIntegrator::BatchedPartialAssemble() 
{
    //Initilze the tensors
    acro::Tensor J,Jinv,Jdet,q;
    if (hasTensorBasis)
    {
        if (nDim == 1)
        {
            D.Init(nElem, nDim, nDim, nQuad1D);
            J.Init(nElem, nQuad1D, nQuad1D, nDim, nDim);
            Jinv.Init(nElem, nQuad1D, nQuad1D, nDim, nDim);            
            Jdet.Init(nElem, nQuad1D);
            q.Init(nElem, nQuad1D);
        }
        else if (nDim == 2)
        {
            D.Init(nElem, nDim, nDim, nQuad1D, nQuad1D);
            J.Init(nElem, nQuad1D, nQuad1D, nDim, nDim);
            Jinv.Init(nElem, nQuad1D, nQuad1D, nDim, nDim);
            Jdet.Init(nElem, nQuad1D, nQuad1D);
            q.Init(nElem, nQuad1D, nQuad1D);
        }
        else if (nDim == 3)
        {
            D.Init(nElem, nDim, nDim, nQuad1D, nQuad1D, nQuad1D);
            J.Init(nElem, nQuad1D, nQuad1D, nQuad1D, nDim, nDim);
            Jinv.Init(nElem, nQuad1D, nQuad1D, nQuad1D, nDim, nDim);
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
        Jinv.Init(nElem, nQuad, nDim, nDim);
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
    TE.BatchMatrixInvDet(Jinv, Jdet, J);

    if (hasTensorBasis) 
    {
        if (nDim == 1) 
        {
            TE["D_e_m_n_k = W_k Q_e_k Jdet_e_k Jinv_e_k_m_n Jinv_e_k_n_m"]
              (D, W, q, Jdet, Jinv, Jinv);
        } 
        else if (nDim == 2) 
        {
            TE["D_e_m_n_k1_k2 = W_k1_k2 Q_e_k1_k2 Jdet_e_k1_k2 Jinv_e_k1_k2_m_n Jinv_e_k1_k2_n_m"]
                (D, W, q, Jdet, Jinv, Jinv);
        } 
        else if (nDim == 3)
        {
            TE["D_e_m_n_k1_k2_k3 = W_k1_k2_k3 Q_e_k1_k2_k3 Jdet_e_k1_k2_k3 Jinv_e_k1_k2_k3_m_n Jinv_e_k1_k2_k3_n_m"]
                (D, W, q, Jdet, Jinv, Jinv);
        } 
    } 
    else 
    {
        TE["D_e_m_n_k = W_k Q_e_k Jdet_e_k Jinv_e_k_m_n Jinv_e_k_n_m"]
          (D, W, q, Jdet, Jinv, Jinv);
    }
}


void AcroDiffusionIntegrator::BatchedAssembleMatrix() {

    if (hasTensorBasis && Btil.Size() == 0) 
    {
        ComputeBTilde();
    }

    if (!S.IsInitialized()) 
    {
        if (hasTensorBasis) 
        {
            if (nDim == 1) 
            {
                S.Init(nElem, nDof1D, nDof1D);
                if (onGPU) {S.SwitchToGPU();}
            } 
            else if (nDim == 2) 
            {
                S.Init(nElem, nDof1D, nDof1D, nDof1D, nDof1D);
                if (onGPU) {S.SwitchToGPU();}
            } 
            else if (nDim == 3) 
            {
                S.Init(nElem, nDof1D, nDof1D, nDof1D, nDof1D, nDof1D, nDof1D);
                if (onGPU) {S.SwitchToGPU();}
            }
        } 
        else 
        {
            S.Init(nElem, nDof, nDof);
            if (onGPU) {S.SwitchToGPU();}
        }
    }


    if (hasTensorBasis) {
        if (nDim == 1) {
            TE["S_e_i1_j1 = Btil_m_n_k1_i1_j1 D_e_m_n_k1"]
                (S, *Btil[0], D);
        } 
        else if (nDim == 2) 
        {
            TE["S_e_i1_i2_j1_j2 = Btil1_m_n_k1_i1_j1 Btil2_m_n_k2_i2_j2 D_e_m_n_k1_k2"]
                (S, *Btil[0], *Btil[1], D);
        } 
        else if (nDim == 3) 
        {
            TE["S_e_i1_i2_i3_j1_j2_j3 = Btil1_m_n_k1_i1_j1 Btil2_m_n_k2_i2_j2 Btil3_m_n_k3_i3_j3 D_e_m_n_k1_k2_k3"]
                (S, *Btil[0], *Btil[1], *Btil[2], D);
        }
    } 
    else 
    {
        TE["S_e_i_j = G_k_i_m G_k_i_n D_e_m_n_k"]
            (S, G, G, D);
    }
}

void AcroDiffusionIntegrator::PAMult(const Vector &x, Vector &y) 
{
    //TODO:  shuffle the vectors to and from the tensor product ordering


  if (!U.IsInitialized() && hasTensorBasis) {
    if (nDim == 1) {
      U.Init(nDim, nElem, nQuad1D);
      Z.Init(nDim, nElem, nQuad1D);
      if (onGPU) {
        U.SwitchToGPU();
        Z.SwitchToGPU();
      }
    } else if (nDim == 2) {
      U.Init(nDim, nElem, nQuad1D, nQuad1D);
      Z.Init(nDim, nElem, nQuad1D, nQuad1D);
      T1.Init(nElem,nDof1D,nQuad1D);
      if (onGPU) {
        U.SwitchToGPU();
        Z.SwitchToGPU();
        T1.SwitchToGPU();
      }
    } else if (nDim == 3) {
      U.Init(nDim, nElem, nQuad1D, nQuad1D, nQuad1D);
      Z.Init(nDim, nElem, nQuad1D, nQuad1D, nQuad1D);
      T1.Init(nElem,nDof1D,nQuad1D,nQuad1D);
      T2.Init(nElem,nDof1D,nDof1D,nQuad1D);
      if (onGPU) {
        U.SwitchToGPU();
        Z.SwitchToGPU();
        T1.SwitchToGPU();
        T2.SwitchToGPU();
      }
    }
  }

  double *x_ptr = const_cast<double*>(x.GetData());
  double *y_ptr = y.GetData();
  if (hasTensorBasis) {
    if (nDim == 1) {
      acro::Tensor X(nElem, nDof1D,
                     x_ptr, x_ptr, onGPU);
      acro::Tensor Y(nElem, nDof1D,
                     y_ptr, y_ptr, onGPU);
      TE["U_n_e_k1 = G_k1_i1 X_e_i1"](U, G, X);
      TE["Z_m_e_k1 = D_e_m_n_k1 U_n_e_k1"](Z, D, U);
      TE["Y_e_i1 += G_k1_i1 Z_m_e_k1"](Y, G, Z);
    } else if (nDim == 2) {
      acro::Tensor X(nElem, nDof1D, nDof1D,
                     x_ptr, x_ptr, onGPU);
      acro::Tensor Y(nElem, nDof1D, nDof1D,
                     y_ptr, y_ptr, onGPU);
      acro::SliceTensor U1(U, 0), U2(U, 1);
      acro::SliceTensor Z1(Z, 0), Z2(Z, 1);

      //U1_e_k1_k2 = G_k1_i1 B_k2_i2 X_e_i1_i2
      TE["BX_e_i1_k2 = B_k2_i2 X_e_i1_i2"](T1, B, X);
      TE["U1_e_k1_k2 = G_k1_i1 BX_e_i1_k2"](U1, G, T1);

      //U2_e_k1_k2 = B_k1_i1 G_k2_i2 X_e_i1_i2
      TE["GX_e_i1_k2 = G_k2_i2 X_e_i1_i2"](T1, G, X);
      TE["U2_e_k1_k2 = B_k1_i1 GX_e_i1_k2"](U2, B, T1);

      TE["Z_m_e_k1_k2 = D_e_m_n_k1_k2 U_n_e_k1_k2"](Z, D, U);

      //Y_e_i1_i2 += G_k1_i1 B_k2_i2 Z1_e_k1_k2
      TE["BZ1_e_i2_k1 = B_k2_i2 Z1_e_k1_k2"](T1, B, Z1);
      TE["Y_e_i1_i2 += G_k1_i1 BZ1_e_i2_k1"](Y, G, T1);

      //Y_e_i1_i2 += B_k1_i1 G_k2_i2 Z2_e_k1_k2
      TE["GZ2_e_i2_k1 = G_k2_i2 Z2_e_k1_k2"](T1, G, Z2);
      TE["Y_e_i1_i2 += B_k1_i1 GZ2_e_i2_k1"](Y, B, T1);
    } else if (nDim == 3) {
      acro::Tensor X(nElem, nDof1D, nDof1D, nDof1D,
                     x_ptr, x_ptr, onGPU);
      acro::Tensor Y(nElem, nDof1D, nDof1D, nDof1D,
                     y_ptr, y_ptr, onGPU);
      acro::SliceTensor U1(U, 0), U2(U, 1), U3(U, 2);
      acro::SliceTensor Z1(Z, 0), Z2(Z, 1), Z3(Z, 2);

      //U1_e_k1_k2_k3 = G_k1_i1 B_k2_i2 B_k3_i3 X_e_i1_i2_i3
      TE["BX_e_i1_i2_k3 = B_k3_i3 X_e_i1_i2_i3"](T2, B, X);
      TE["BBX_e_i1_k2_k3 = B_k2_i2 BX_e_i1_i2_k3"](T1, B, T2);
      TE["U1_e_k1_k2_k3 = G_k1_i1 BBX_e_i1_k2_k3"](U1, G, T1);

      //U2_e_k1_k2_k3 = B_k1_i1 G_k2_i2 B_k3_i3 X_e_i1_i2_i3
      TE["GBX_e_i1_k2_k3 = G_k2_i2 BX_e_i1_i2_k3"](T1, G, T2);
      TE["U2_e_k1_k2_k3 = B_k1_i1 GBX_e_i1_k2_k3"](U2, B, T1);

      //U3_e_k1_k2_k3 = B_k1_i1 B_k2_i2 G_k3_i3 X_e_i1_i2_i3
      TE["GX_e_i1_i2_k3 = G_k3_i3 X_e_i1_i2_i3"](T2, G, X);
      TE["BGX_e_i1_k2_k3 = B_k2_i2 GX_e_i1_i2_k3"](T1, B, T2);
      TE["U3_e_k1_k2_k3 = B_k1_i1 BGX_e_i1_k2_k3"](U3, B, T1);

      TE["Z_m_e_k1_k2_k3 = D_e_m_n_k1_k2_k3 U_n_e_k1_k2_k3"](Z, D, U);

      //Y_e_i1_i2_i3 =  G_k1_i1 B_k2_i2 B_k3_i3 Z1_e_k1_k2_k3
      TE["BZ1_e_i3_k1_k2 = B_k3_i3 Z1_e_k1_k2_k3"](T1, B, Z1);
      TE["BBZ1_e_i2_i3_k1 = B_k2_i2 BZ1_e_i3_k1_k2"](T2, B, T1);
      TE["Y_e_i1_i2_i3 += G_k1_i1 BBZ1_e_i2_i3_k1"](Y, G, T2);

      //Y_e_i1_i2_i3 =  B_k1_i1 G_k2_i2 B_k3_i3 Z2_e_k1_k2_k3
      TE["BZ2_e_i3_k1_k2 = B_k3_i3 Z2_e_k1_k2_k3"](T1, B, Z2);
      TE["GBZ2_e_i2_i3_k1 = G_k2_i2 BZ2_e_i3_k1_k2"](T2, G, T1);
      TE["Y_e_i1_i2_i3 += B_k1_i1 GBZ2_e_i2_i3_k1"](Y, B, T2);

      //Y_e_i1_i2_i3 =  B_k1_i1 B_k2_i2 G_k3_i3 Z3_e_k1_k2_k3
      TE["GZ3_e_i3_k1_k2 = G_k3_i3 Z3_e_k1_k2_k3"](T1, G, Z3);
      TE["BGZ3_e_i2_i3_k1 = B_k2_i2 GZ3_e_i3_k1_k2"](T2, B, T1);
      TE["Y_e_i1_i2_i3 += B_k1_i1 BGZ3_e_i2_i3_k1"](Y, B, T2);
    }
  } else {
    mfem_error("AcroDiffusionIntegrator MultAdd on simplices not supported");
  }

}

}

#endif
