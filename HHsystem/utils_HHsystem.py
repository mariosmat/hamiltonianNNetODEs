#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 22:37:17 2020

@author: marios
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy.integrate import odeint
   
    
   
    
   
    
###################
# Symplectic Euler
####################
def symEuler(Ns, x0,y0, px0, py0, t0, t_max,lam):
    t_s = np.linspace(t0, t_max, Ns+1)
    ts = t_max/Ns
    dts = t_max/Ns; 
    
    x_s = np.zeros(Ns+1); px_s = np.zeros(Ns+1);
    y_s = np.zeros(Ns+1); py_s = np.zeros(Ns+1)
     
    x_s[0], px_s[0], y_s[0], py_s[0] = x0,  px0,y0, py0
    for n in range(Ns):
        x_s[n+1] = x_s[n] + dts*px_s[n]
        y_s[n+1] = y_s[n] + dts*py_s[n]
        
        px_s[n+1] = px_s[n] - dts*(x_s[n+1] + 2*lam*x_s[n+1]*y_s[n+1])
        py_s[n+1] = py_s[n] - dts*(y_s[n+1] + lam*(x_s[n+1]**2-y_s[n+1]**2))    
        # E_euler = energy( x_s, y_s, px_s, py_s, lam)

    E_euler = energy( x_s, y_s, px_s, py_s, lam)
    return E_euler, x_s,y_s, px_s, py_s, t_s
 
   
    
   
# Use below in the Scipy Solver   
def f(u, t ,lam=1):
    x, y, px, py = u      # unpack current values of u
    derivs = [px, py, -x -2*lam*x*y, -y -lam*(x**2-y**2) ]     # list of dy/dt=f functions
    return derivs

# Scipy Solver   
def HHsolution(N,t, x0, y0, px0, py0,lam=1):
    u0 = [x0, y0, px0, py0]
    # Call the ODE solver
    solPend = odeint(f, u0, t, args=(lam,))
    xP = solPend[:,0];    yP  = solPend[:,1];
    pxP = solPend[:,2];   pyP = solPend[:,3]
    return xP,yP, pxP, pyP

# Energy of Henon Heiles system
def energy(x, y, px, py, lam=1):    
    Nx=len(x); 
    x=x.reshape(Nx);      y=y.reshape(Nx)
    px=px.reshape(Nx);    py=py.reshape(Nx)
    E = 0.5*(px**2 + py**2) + 0.5*(x**2+y**2)+lam*(x**2 *y - y**3/3)
    E = E.reshape(Nx)
    return E

# initial energy
def HH_exact(N,x0, y0, vx0, vy0, lam):
    E0 = 0.5*(vx0**2+vy0**2) + 0.5*(x0**2+y0**2)+lam*(x0**2 *y0 - y0**3/3)
    E_ex = E0*np.ones(N);
    return E0, E_ex





def saveData(path, t, x, y, px,py, E):
    np.savetxt(path+"t.txt",t)
    np.savetxt(path+"x.txt",x)
    np.savetxt(path+"y.txt",y)
    np.savetxt(path+"px.txt",px)
    np.savetxt(path+"py.txt",py)
    np.savetxt(path+"E.txt",E)

    
    