#!/usr/bin/env python3 
"""
Derive Elastic forces for Yakoub 2-nodes elements
"""
import numpy as np

import scipy.integrate as sp_int

import yakoub_shape_function as shape_fun
maxsubs=10
rtol=1e-3
def tpl_int(fun, low_lim, up_lim):
    ## aux_fun allows fun to return vectors of vectors
    def aux_fun(X,fun):
        if X.ndim > 1:
            res0 = fun(X[0,:])
            res = np.empty((X.shape[0],) + res0.shape)
            res[0,:,:] = res0
            for i in range(1,X.shape[0]):
                res[i,:,:] = fun(X[i,:])
        else:
            res = fun(X[0,:])
        return res

    return sp_int.cubature(lambda X: aux_fun(X,fun), low_lim, up_lim,
                           max_subdivisions=maxsubs, rtol=rtol)

class Fe():
    def __init__(self, a, b, l , e0, lam, G):
        self.a = a
        self.b = b
        self.l = l
        self.e0 = e0
        self.lam = lam
        self.G = G
        self.ncoord = e0.size
        self.dS_dr0 = [lambda x, y, z: shape_fun.eval_dS_dr0_0(x,y,z,l,e0),
                       lambda x, y, z: shape_fun.eval_dS_dr0_1(x,y,z,l,e0),
                       lambda x, y, z: shape_fun.eval_dS_dr0_2(x,y,z,l,e0)]
                       
    def dS_dr0TxdS_dr0(self,x,y,z,i):
        aux = self.dS_dr0[i](x,y,z)
        return aux.T@aux

    def S_alfiTxS_alfj(self,x,y,z,alf,i,j):
        aux = self.dS_dr0TxdS_dr0(x,y,z,alf)
        A = aux[i,:].reshape(aux.shape[0],1)
        B = aux[j,:].reshape(1,aux.shape[0])
        res = A@B
        return res

    def eval_K1(self):
        print(f"Evaluating K1 with maxsubs = {maxsubs} and rtol = {rtol}.",flush=True)
        K1 = np.zeros((self.ncoord,self.ncoord))
        for i in range(3):
            aux_int = tpl_int(lambda X: self.dS_dr0TxdS_dr0(X[0],X[1],X[2],i),
                          (0, -self.a/2,-self.b/2), (self.l, self.a/2, self.b/2))
            if aux_int.status == "not_converged":
                print(f"K1 integration at alf = {i} has not converged.",flush=True)
            else:
                print(f"K1 integration at alf = {i} has {aux_int.status}.",flush=True)
            K1 += aux_int.estimate
        K1 *= -(3*self.lam+2*self.G)/2

        np.save(f"K1_matrix{maxsubs}_{rtol}.npy", K1) # all this metadata should be added to the file

        self.K1 = K1

    def load_K1(self):
        self.K1 = np.load("K1_matrix.npy")

    def eval_CK2(self):
        print(f"Evaluating CK2 with maxsubs = {maxsubs}.",flush=True)
        ncoord = self.ncoord
        CK2 = np.zeros((ncoord**2,ncoord**2))
        for i in range(ncoord):
            auxi = i*ncoord
            for j in range(ncoord):
                auxj=j*ncoord
                for alf in range(3):
                    aux_int = tpl_int(lambda X: self.S_alfiTxS_alfj(X[0],X[1],X[2],alf,i,j),
                                      (0, -self.a/2,-self.b/2), (self.l, self.a/2, self.b/2))
                    if aux_int.status == "not_converged":
                        print(f"CK2 first integration at i = {i}, j = {j} and alf = {alf} has not converged.",flush=True)
                    else:
                        print(f"CK2 first integration at i = {i}, j = {j} and alf = {alf} has {aux_int.status}.",flush=True)
                    CK2[auxi:auxi+ncoord,auxj:auxj+ncoord] += (self.lam + 2*self.G)/2*aux_int.estimate
                    for bet in range(3):
                        if alf != bet:
                            aux_int = tpl_int(lambda X:
                                              self.dS_dr0TxdS_dr0(X[0],X[1],X[2],alf)[i,:].reshape(ncoord,1)@
                                              self.dS_dr0TxdS_dr0(X[0],X[1],X[2],bet)[j,:].reshape(1,ncoord),
                                              (0, -self.a/2,-self.b/2), (self.l, self.a/2, self.b/2))
                            if aux_int.status == "not_converged":
                                print(f"CK2 second integration at i = {i}, j = {j}, alf = {alf} and bet = {bet} has not converged.",flush=True)
                            else:
                                print(f"CK2 second integration at i = {i}, j = {j}, alf = {alf} and bet = {bet} has {aux_int.status}.",flush=True)
                            CK2[auxi:auxi+ncoord,auxj:auxj+ncoord] += self.lam/2*aux_int.estimate
                            aux_int = tpl_int(lambda X: (self.dS_dr0[alf](X[0],X[1],X[2]).T@
                                                         self.dS_dr0[bet](X[0],X[1],X[2]))[i,:].reshape(ncoord,1)@
                                              (self.dS_dr0[bet](X[0],X[1],X[2]).T@
                                               self.dS_dr0[alf](X[0],X[1],X[2]))[j,:].reshape(1,ncoord),
                                              (0, -self.a/2,-self.b/2), (self.l, self.a/2, self.b/2))
                            if aux_int.status == "not_converged":
                                print(f"CK2 third integration at i = {i}, j = {j}, alf = {alf} and bet = {bet} has not converged.",flush=True)
                            else:
                                print(f"CK2 third integration at i = {i}, j = {j}, alf = {alf} and bet = {bet} has {aux_int.status}.",flush=True)
                            CK2[auxi:auxi+ncoord,auxj:auxj+ncoord] += self.G*aux_int.estimate

        np.save(f"CK2_matrix{maxsubs}.npy", CK2)

        self.CK2 = CK2

    def load_CK2(self):
        self.CK2 = np.load("CK2_matrix.npy")

                            
    def eval_K2(self,e):
        CK2 = self.CK2
        ncoord = self.ncoord
        K2 = np.empty((ncoord,ncoord))
        for i in range(ncoord):
            for j in range(ncoord):
                K2[i,j] = e.T@CK2[i*ncoord:(i+1)*ncoord,j*ncoord:(j+1)*ncoord]@e
        return K2

    def eval_Fe(self,e):
        Ke = self.eval_K2(e) + self.K1
        return Ke@e

# """
# Further expressions
# """

# #K2[i,j] = np.Matrix(e.T*CK2[i,j]*e) #each CK2[i,j] are 24x24 matrices, and there should be 24x24 of them.

# #Ke = K2 + K1 # Ke should be 24x24

# #Fe =  # Fe should be 24x1
