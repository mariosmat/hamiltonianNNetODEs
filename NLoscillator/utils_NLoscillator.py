#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 18:56:15 2020

@author: marios

This file contains all the functions used for numerical integrators
"""

import numpy as np
from scipy.integrate import odeint



###################
# Symplectic Euler
####################
def symEuler(Ns, x0,px0,t0, t_max, lam=1):
    t_s = np.linspace(t0, t_max, Ns+1)
    x_s = np.zeros(Ns+1); p_s = np.zeros(Ns+1)
    x_s[0], p_s[0] = x0, px0
    dts = t_max/Ns; 

    for n in range(Ns):
        x_s[n+1] = x_s[n] + dts*p_s[n]
        p_s[n+1] = p_s[n] - dts*(x_s[n+1] + lam*x_s[n+1]**3)

    E_euler = energy(x_s, p_s, lam=1)
    return E_euler, x_s, p_s, t_s


# Energy of nonlinear oscillator
def energy(x, px, lam=1):    
    Nx=len(x); 
    x=x.reshape(Nx);        px=px.reshape(Nx);    
    E = 0.5*px**2 + 0.5*x**2 + lam*x**4/4
    E = E.reshape(Nx)
    return E



#####################################
# Scipy Solver   
######################################
def f(u, t ,lam=1):
    x,  px = u      # unpack current values of u
    derivs = [px, -x - lam*x**3]     # list of derivatives
    return derivs
# Scipy Solver   
def NLosc_solution(N,t, x0,  px0, lam=1):
    u0 = [x0, px0]
    # Call the ODE solver
    solPend = odeint(f, u0, t, args=(lam,))
    xP = solPend[:,0];        pxP = solPend[:,1];   
    return xP, pxP
# initial energy
def NLosc_exact(N,x0, px0, lam):
    E0 = 0.5*px0**2 + 0.5*x0**2 + lam*x0**4/4
    E_ex = E0*np.ones(N);
    return E0, E_ex

## END:  FUNCTIONS FOR THE GROUND TRUTH SOLUTIONS



## Save data function
def saveData(path, t, x, px, E):
    np.savetxt(path+"t.txt",t)
    np.savetxt(path+"x.txt",x)
    np.savetxt(path+"px.txt",px)
    np.savetxt(path+"E.txt",E)
    
