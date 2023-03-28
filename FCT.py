# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:50:49 2020

@author: kwik
"""

import numpy as np

def sigmo(z):
   
    return 2*1.716/(1 + np.exp(-0.667*z)) -1.716

def Der_sigmo(z):
   
    return 2*1.716*0.667*np.exp(-0.667*z)/(1+np.exp(-0.667*z))**2

def valeur_init_W(Ni,Nc,Ns):
#W1,W2=valeur_init_W()
    W1=np.random.rand(Ni,Nc)
    W2=np.random.rand(Nc,Ns)
    #W1=np.zeros((2,3))+0.01
    #W2=np.zeros((3,1))+0.01
    return W1,W2


def valeur_init_W_2NC(Ni,Nc,Ns):
#W1,W2,W3=valeur_init_W()
    W1=np.random.rand(Ni,Nc)
    W2=np.random.rand(Nc,Nc)
    W3=np.random.rand(Nc,Ns)
    #W1=np.zeros((2,3))+0.01
    #W2=np.zeros((3,1))+0.01
    return W1,W2,W3

def propagation(X,W1,W2,Y):

    #V2,Y2s,V1,Y1=propagation(X,W1,W2,Y)
    V1=np.dot(X,W1)
    Y1=sigmo(V1)
    V2=np.dot(Y1,W2)
    Y2s=sigmo(V2) 
    E=Y-Y2s
    #print(E.T)
    return V2,Y2s,V1,Y1

def propagation_2NC(X,W1,W2,W3,Y):

    #V1,Y1,V2,Y2,V3,Y3s=propagation_2NC(X,W1,W2,Y)
    V1=np.dot(X,W1)
    Y1=sigmo(V1)
    V2=np.dot(Y1,W2)
    
    Y2=sigmo(V2) 
    V3=np.dot(Y2,W3)
    Y3s=sigmo(V3)
    
    
    E=Y-Y3s
    #print(E.T)
    return V1,Y1,V2,Y2,V3,Y3s


def Back_prop(V2,Y2s,V1,Y1,X,Y,W1,W2,LR):
   
       #deltaS,DeDW2,W2,W1,E,delta1,deDW1=Back_prop(V2,Y2s,V1,Y1,X,Y,W1,W2,LR)
    E=Y-Y2s
    deltaS=E*Der_sigmo(V2)
    #DeDW2 = np.dot(-Y1.T,deltaS)
   
    DeDW2 = np.dot(Y1.T,-deltaS)
    W2=W2-(LR*DeDW2)
    
    
    delta2=np.dot(deltaS,W2.T)*Der_sigmo(V1)
    #delta2=np.dot(deltaS,W2.T)*sigmo(V1)
    deDW1=np.dot(-X.T,delta2)
    W1=W1-(LR*deDW1)
    
    
    
    
    return deltaS,DeDW2,W2,W1,E,delta2,deDW1

def Back_prop_2NC(V3,Y3s,V2,Y2,V1,Y1,X,Y,W1,W2,W3,LR):
   
       #deltaS,DeDW2,W3,W2,W1,E,delta1,deDW1=Back_prop(V2,Y2s,V1,Y1,X,Y,W1,W2,LR)
    E=Y-Y3s
    deltaS=E*Der_sigmo(V3)
    #DeDW2 = np.dot(-Y1.T,deltaS)
   
    DeDW3 = np.dot(Y2.T,-deltaS)
    W3=W3-(LR*DeDW3)
    
    
    delta2=np.dot(deltaS,W3.T)*Der_sigmo(V2)
    
    deDW2=np.dot(Y1.T,-delta2)
    W2=W2-(LR*deDW2)
    
    
    delta1=np.dot(delta2,W2.T)*sigmo(V1)
    deDW1=np.dot(-X.T,delta2)
    W1=W1-(LR*deDW1)
    
    
    
    
    return deltaS,deDW2,W3,W2,W1,E,delta2,deDW1