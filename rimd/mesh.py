#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 17:05:45 2022

@author: cyberguli
"""
from igl import cotmatrix
import igl
import numpy as np
import math
import scipy
from queue import Queue

class Mesh:
    def buildCotWeights(self):
        C=cotmatrix(self.verts,self.faces)
        num_verts=self.verts.shape[0]
        for i in range(num_verts):
            self.cot_weights.append([])
        
        for i in range(num_verts):
            num_neighnbours=len(self.neighbors[i])
            for j in range(num_neighnbours):
                self.cot_weights[i].append(0)
            for j in range(num_neighnbours):
                v_j=self.neighbors[i][j]
                wij=C[i,v_j]
                eps=1e-06
                cotan_max=math.cos(eps)/math.sin(eps)
                if wij>=cotan_max:
                    wij=cotan_max
                self.cot_weights[i][j]=wij
        
    def __init__(self,verts=np.array([[0,0,0]]),faces=np.array([[0]])):
        self.verts=verts
        self.faces=faces
        self.neighbors=igl.adjacency_list(faces)
        self.cot_weights=[]
        self.buildCotWeights()
        

class RIMD:
    
    def solveTransform(self):
        T_matrices=[];
        num_verts=self.baseMesh.verts.shape[0]
        # for each vertex vi, compute Ti
        for i in range(num_verts):
            num_edges = len(self.baseMesh.neighbors[i])
            A = np.zeros([3 * num_edges, 9]);
            weights = np.zeros([3 * num_edges, 3 * num_edges]);
            b = np.zeros(3 * num_edges);
    
            #  We want to solve t11, t12, t13, t21, ......, t32, t33
            # The shape of A matrix is 3n * 9 and the shape of b vector is 3n * 1
            # n denotes the degree number of vi
    
            for j in range(num_edges):
                v_j = self.baseMesh.neighbors[i][j];
    
                e_ij = self.baseMesh.verts[i,:] - self.baseMesh.verts[v_j,:];
                e_ij_prime = self.targetMesh.verts[i,:] - self.targetMesh.verts[v_j,:]
    
                # Filling the A matrix
                A[3 * j, 0] = e_ij[0];
                A[3 * j, 3] = e_ij[1];
                A[3 * j, 6 ] = e_ij[2];
                A[3 * j + 1, 1 ] = e_ij[0];
                A[3 * j + 1, 4 ] = e_ij[1];
                A[3 * j + 1, 7 ] = e_ij[2];
                A[3 * j + 2, 2 ] = e_ij[0];
                A[3 * j + 2, 5 ] = e_ij[1];
                A[3 * j + 2, 8 ] = e_ij[2];
    
                # Filling the b vector
                b[j * 3 ] = e_ij_prime[0 ];
                b[j * 3 + 1 ] = e_ij_prime[1];
                b[j * 3 + 2 ] = e_ij_prime[2];
    
                # Filling the matrix of cotangent weights
                w_ij = self.baseMesh.cot_weights[i][j];
                if (w_ij < 0):
                    w_ij *= -1
    
                weights[j * 3, j * 3] = math.sqrt(w_ij);
                weights[j * 3 + 1, j * 3 + 1] = math.sqrt(w_ij);
                weights[j * 3 + 2, j * 3 + 2] = math.sqrt(w_ij);
            
    
            # Solve the least-square problem using the OR decomposition
            solution = scipy.linalg.solve(weights@A,weights@b) 
            T=np.zeros([3,3])
            for m in range(3):
                for n in range(3):
                    T[m, n] = solution[m * 3 + n]
            T_matrices.append(T.transpose()) #T.transpose()
        self.transforms = T_matrices
        

    def PolarDecomposition(self,M):
        U, s, V = np.linalg.svd(M)
        SV=np.diag(s)
        R=U@V
        SVT=SV@V
        if(np.linalg.det(R)<0):
            W=V
            W[-1,:]*=-1
            R=U*W
            S=W*SVT
        else:
            S=V*SVT
        
        decomposition=[R,S]
        return decomposition
    
    def solveRIMD(self):
        rimd_features=[]
        num_verts=self.baseMesh.verts.shape[0]
        for i in range(num_verts):
            R_i,S_i=self.PolarDecomposition(self.transforms[i])
            feature=[]
            for j in range(len(self.baseMesh.neighbors[i])):
                v_j=self.baseMesh.neighbors[i][j]
                T_j=self.transforms[v_j]
                R_j=self.PolarDecomposition(T_j)[0]
                log_dR_ij=scipy.linalg.logm(R_i.transpose()@R_j)
                feature.append(log_dR_ij)
            feature.append(S_i)
            rimd_features.append(feature)
        return rimd_features                
                
    
    def BFS_initialize(self, rimd_vectors):
    
        num_verts = self.baseMesh.verts.shape[0];
        # Create a list of R matrices
        R=[0]*num_verts
        #Create the color list which denotes the state of visitation
        Color=[0]*num_verts
        # Create the parent list
        Parent=[0]*num_verts

    
        # Define some constant
        white = -2; # undiscover
        gray = -1; # if u is discovered, the adjacent vertex v is undiscovered
        black = 0; # discovered
        unknown = -3;
    
        # Initialization
        for i in range(num_verts):
            Color[i] = white;
            Parent[i] = -1; # Parent is unknown
        
        # Visit the start vertex
        #I take the first vertex as the start vertex
        s = 0;
        Color[s] = gray;
        Parent[s] = unknown;
    
        #Set to the identity matrix
        R[s]=np.eye(3);
    
        Q=Queue();
        # put the start vertex into the queue
        Q.put(s);
    
        while (not Q.empty()):
            u = Q.get();
            for i in range(len(self.baseMesh.neighbors[u])):
                # Compute the initial value
                Ru = R[u];
                dRuv = scipy.linalg.expm(rimd_vectors[u][i]);

    
                v = self.baseMesh.neighbors[u][i];
                if (Color[v] == white):
                    Color[v] = gray;
                    Parent[v] = u;
                    Q.put(v);
                    R[v] = Ru @ dRuv;
                
            
            Color[u] = black;
        
        return R
    
    
    
    def quickReconstruct(self, rimd_vectors):
        num_verts = self.baseMesh.verts.shape[0];
        P=np.zeros([num_verts,3])
        P_prime=np.zeros([num_verts,3])
        for i in range(num_verts):
            P[i,:] = self.baseMesh.verts[i,:]
    
        R=[0]*num_verts
        #Create the color list which denotes the state of visitation
        Color=[0]*num_verts
        # Create the parent list
        Parent=[0]*num_verts
        # Define some constant
        white = -2; # undiscover
        gray = -1; # if u is discovered, the adjacent vertex v is undiscovered
        black = 0; # discovered
        unknown = -3;
        # Initialization
        for i in range(num_verts):
            Color[i] = white;
            Parent[i] = -1; # Parent is unknown

        # Choose an arbitrary vertex
        # I take the first vertex as the start vertex
        s = 0;
        Color[s] = gray;
        Parent[s] = unknown;
        P_prime[s,:] = P[s,:];
    
        # Set to the identity matrix
        R[s]=np.eye(3)
    
        # Create an empty queue
        Q=Queue();
        # put the start vertex into the queue
        Q.put(s);

        while (not Q.empty()):
            u = Q.get();
            for i in range(len(self.baseMesh.neighbors[u])):
                # Compute the initial value
                Ru = R[u];
                dRuv = scipy.linalg.expm(rimd_vectors[u][i]);
    
                v=self.baseMesh.neighbors[u][i]
                if (Color[v] == white):
                    Color[v] = gray;
                    Parent[v] = u;
                    Q.put(v);
                    R[v] = Ru @ dRuv;
                    evu=np.zeros(3)
                    temp=np.zeros(3)
                    temp = self.baseMesh.verts[v,:] - self.baseMesh.verts[u,:];
                    evu[0] = temp[0];
                    evu[1] = temp[1];
                    evu[2] = temp[2];
                    Sv = rimd_vectors[v][-1];
                    evu_prime = R[v] @ Sv @ evu;
    
                    temp[0] = evu_prime[0];
                    temp[1] = evu_prime[1];
                    temp[2] = evu_prime[2]
                    # Update vertex
                    P_prime[v, 0] = temp[0] + P_prime[u, 0];
                    P_prime[v, 0] = temp[1] + P_prime[u, 1];
                    P_prime[v, 2] = temp[2] + P_prime[u, 2];
                
            
            Color[u] = black;
        
    
        return P_prime;
    
    
    def __init__(self, base, target):
        self.baseMesh=base
        self.targetMesh=target
        self.transforms=[]
        self.solveTransform()
        self.features=self.solveRIMD()
        
a=Mesh(np.array([[ 1.,  0., -0.70710678], [-1.,  0., -0.70710678],[ 0. ,1. ,  0.70710678],[ 0., -1.,  0.70710678]]),np.array([[0, 1, 2],[0, 1, 3], [1, 2, 3],[0, 2, 3]]))
b=Mesh(3*np.array([[ 1.,  0., -0.70710678], [-1.,  0., -0.70710678],[ 0. ,1. ,  0.70710678],[ 0., -1.,  0.70710678]]),np.array([[0, 1, 2],[0, 1, 3], [1, 2, 3],[0, 2, 3]]))
c=RIMD(a,b)