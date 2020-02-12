#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:13:26 2020
@author: marios mattheakis

In this code a Hamiltonian Neural Network is designed and employed
to solve a system of two differential equations obtained by Hamilton's
equations for the the Hamiltonian of nonlinear oscillator.
"""

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import grad

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import copy

from scipy.integrate import odeint

dtype=torch.float


## Define the Functions

# Define the sin() activation function
class mySin(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.sin(input)

# Use below in the Scipy Solver   
def f(u, t ,lam=1):
    x,  px = u      # unpack current values of u
    derivs = [px, -x - lam*x**3]     # list of dy/dt=f functions
    return derivs
# Scipy Solver   
def NLosc_solution(N,t, x0,  px0, lam=1):
    u0 = [x0, px0]
    # Call the ODE solver
    solPend = odeint(f, u0, t, args=(lam,))
    xP = solPend[:,0];        pxP = solPend[:,1];   
    return xP, pxP
# Energy of nonlinear oscillator
def energy(x, px, lam=1):    
    Nx=len(x); 
    x=x.reshape(Nx);        px=px.reshape(Nx);    
    E = 0.5*px**2 + 0.5*x**2 + lam*x**4/4
    E = E.reshape(Nx)
    return E
# initial energy
def NLosc_exact(N,x0, px0, lam):
    E0 = 0.5*px0**2 + 0.5*x0**2 + lam*x0**4/4
    E_ex = E0*np.ones(N);
    return E0, E_ex

# Set the initial state. lam controls the nonlinearity
x0, px0,  lam =  1.3, 1., 1; 
t0, t_max, N = 0.,4*np.pi, 200; dt = t_max/N; 
X0 = [t0, x0, px0, lam]
t_num = np.linspace(t0, t_max, N)
E0, E_ex = NLosc_exact(N,x0, px0, lam)

# Solution obtained by Scipy solver
x_num,  px_num  = NLosc_solution(N,t_num, x0,  px0,  lam)
E_num = energy( x_num,  px_num, lam)


#####################################
# Hamiltonian Neural Network
####################################

# Define some more general functions
def dfx(x,f):
    # Calculate the derivatice with auto-differention
    return grad([f], [x], grad_outputs=torch.ones(x.shape, dtype=dtype), create_graph=True)[0]

def perturbPoints(grid,t0,tf,sig=0.5):
#   stochastic perturbation of the evaluation points
#   force t[0]=t0  & force points to be in the t-interval
    delta_t = grid[1] - grid[0]  
    noise = delta_t * torch.randn_like(grid)*sig
    t = grid + noise
    t.data[2] = torch.ones(1,1)*(-1)
    t.data[t<t0]=t0 - t.data[t<t0]
    t.data[t>tf]=2*tf - t.data[t>tf]
    t.data[0] = torch.ones(1,1)*t0
    t.requires_grad = False
    return t

def saveData(path, t, x, px, E, loss):
    np.savetxt(path+"t.txt",t)
    np.savetxt(path+"x.txt",x)
    np.savetxt(path+"px.txt",px)
    np.savetxt(path+"E.txt",E)
    np.savetxt(path+"Loss.txt",loss)

# Define some functions used by the Hamiltonian network
    
def parametricSolutions(t, nn, X0):
    # parametric solutions
    t0, x0, px0,  lam = X0[0],X0[1],X0[2],X0[3]
    N1, N2  = nn(t)
    dt =t-t0
#### THERE ARE TWO PARAMETRIC SOLUTIONS. Uncomment f=dt 
    f = (1-torch.exp(-dt)) 
#     f = dt
    x_hat  = x0  + f*N1
    px_hat = px0 + f*N2
    return x_hat, px_hat

def hamEqs_Loss(t,x,px,lam):
    # Define the loss function by Hamilton Eqs., write explicitely the Ham. Equations
    xd,pxd= dfx(t,x),dfx(t,px)
    fx  = xd - px; 
    fpx = pxd + x + lam*x.pow(3)
    Lx  = (fx.pow(2)).mean();     Lpx = (fpx.pow(2)).mean();
    L = Lx  + Lpx
    return L


def hamEqs_Loss_byH(t,x,px,lam):
    # This is an alternative way to define the loss function:
    # Define the loss function by Hamilton Eqs. directly from Hamiltonian H
    #
    # Potential and Kinetic Energy
    V = 0.5*x.pow(2)  + lam*x.pow(4)/4
    K = 0.5*px.pow(2)
    ham = K + V
    xd,pxd= dfx(t,x),dfx(t,px)
    # calculate the partial spatial derivatives of H
    hx  = grad([ham], [x], grad_outputs=torch.ones(x.shape, dtype=dtype), create_graph=True)[0]
    hpx = grad([ham], [px], grad_outputs=torch.ones(px.shape, dtype=dtype), create_graph=True)[0]

    # Hamilton Eqs
    fx  = xd - hpx;     fpx = pxd + hx;     
    Lx  = (fx.pow(2)).mean();  Lpx = (fpx.pow(2)).mean(); 
    L = Lx +  Lpx
    return L


# NETWORK ARCHITECTURE
    
# A two hidden layer NN, 1 input & two output
class odeNet_NLosc_MM(torch.nn.Module):
    def __init__(self, D_hid=10):
        super(odeNet_NLosc_MM,self).__init__()

#####    CHOOCE THE ACTIVATION FUNCTION
        self.actF = mySin()
#         self.actF = torch.nn.Sigmoid()   
# define layers
        self.Lin_1   = torch.nn.Linear(1, D_hid)
        self.Lin_2   = torch.nn.Linear(D_hid, D_hid)
        self.Lin_out = torch.nn.Linear(D_hid, 2)

    def forward(self,t):
        # layer 1
        l = self.Lin_1(t);    h = self.actF(l)
        # layer 2
        l = self.Lin_2(h);    h = self.actF(l)

        # output layer
        r = self.Lin_out(h)
        xN  = (r[:,0]).reshape(-1,1); pxN = (r[:,1]).reshape(-1,1);
        return xN, pxN

# FUNCTION NETWORK TRAINING 
def run_odeNet_NLosc_MM(X0, tf, neurons, epochs, n_train,lr,
                    minibatch_number = 1):
    fc0 = odeNet_NLosc_MM(neurons)
    fc1=0; # fc1 will be a deepcopy of the network with the lowest training loss
    # optimizer
    betas = [0.999, 0.9999]    
    optimizer = optim.Adam(fc0.parameters(), lr=lr, betas=betas)
    Loss_history = [];     Llim =  1 
    
    t0=X0[0];
    grid = torch.linspace(t0, tf, n_train).reshape(-1,1)    
    
##  TRAINING ITERATION    
    TeP0 = time.time()
    
    for tt in range(epochs):                
# Perturbing the evaluation points & forcing t[0]=t0
        t=perturbPoints(grid,t0,tf,sig=0.03*tf)
            
# BATCHING
        batch_size = int(n_train/minibatch_number)
        batch_start, batch_end = 0, batch_size

        idx = np.random.permutation(n_train)
        t_b = t[idx]
        t_b.requires_grad = True

        loss=0.0
        for nbatch in range(minibatch_number): 
# batch time set
            t_mb = t_b[batch_start:batch_end]
#  Network solutions 
            x,px =parametricSolutions(t_mb,fc0,X0)
# LOSS
#  Loss function defined by Hamilton Eqs. (symplectic): Writing explicitely the Eqs (faster)
            Ltot = hamEqs_Loss(t_mb,x,px,lam)

#  Loss function defined by Hamilton Eqs. (symplectic): Calculating with auto-diff the Eqs (slower)
#             Ltot = hamEqs_Loss_byH(t_mb,x,y,px,py,lam)
    
# OPTIMIZER
            Ltot.backward(retain_graph=False); #True
            optimizer.step(); loss += Ltot.data.numpy()
            optimizer.zero_grad()

            batch_start +=batch_size
            batch_end +=batch_size

# keep the loss function history
        Loss_history.append(loss)       
    
#Keep the best model (lowest loss) by using a deep copy
        if  tt > 0.8*epochs  and Ltot < Llim:
            fc1 =  copy.deepcopy(fc0)
            Llim=Ltot 

    TePf = time.time()
    runTime = TePf - TeP0     
    return fc1, Loss_history, runTime


# END OF FUNCTIONS DEFINITION


# TRAIN THE NETWORK. 
# Here, we use one mini-batch. NO significant different in using more
n_train, neurons, epochs, lr,mb = 200, 50, int(5e4), 8e-3,  1
model,loss,runTime = run_odeNet_NLosc_MM(X0, t_max, 
                                      neurons, epochs, n_train,lr,mb)



print('Training time (minutes):', runTime/60)
plt.figure()
plt.loglog(loss,'-b',alpha=0.975);
plt.tight_layout()
plt.ylabel('Loss');plt.xlabel('t')

#plt.savefig('../results/nonlinearOscillator_loss.png')
plt.savefig('nonlinearOscillator_loss.png')




# TEST THE PREDICTED SOLUTIONS
nTest = N ; tTest = torch.linspace(t0,t_max,nTest)
tTest = tTest.reshape(-1,1);
tTest.requires_grad=True
t_net = tTest.detach().numpy()

x,px = parametricSolutions(tTest,model,X0)


# HERE WE CALCULATE THE ell_max (maximum loss in time)
xd,pxd= dfx(tTest,x),dfx(tTest,px) # derivatives obtained by back-propagation
fx  = xd - px; 
fpx = pxd + x + x.pow(3)
ell_sq = fx.pow(2) + fpx.pow(2)
ell_max = np.max(np.sqrt( ell_sq.data.numpy() ) )
print('The maximum in time loss is ', ell_max)



###################
# Symplectic Euler
####################
def symEuler(Ns, x0,px0,t_max,lam):
    t_s = np.linspace(t0, t_max, Ns+1)
    x_s = np.zeros(Ns+1); p_s = np.zeros(Ns+1)
    x_s[0], p_s[0] = x0, px0
    dts = t_max/Ns; 

    for n in range(Ns):
        x_s[n+1] = x_s[n] + dts*p_s[n]
        p_s[n+1] = p_s[n] - dts*(x_s[n+1] + lam*x_s[n+1]**3)

    E_euler = energy(x_s, p_s, lam=1)
    return E_euler, x_s, p_s, t_s


Ns = n_train -1; 
E_s, x_s, p_s, t_s = symEuler(Ns, x0,px0,t_max,lam)
Ns100 = 100*n_train ; 
E_s100, x_s100, p_s100, t_s100 = symEuler(Ns100, x0,px0,t_max,lam)



################
# Make the plots
#################
x=x.data.numpy(); px=px.data.numpy();
E  = energy(x, px, lam)

# Figure for trajectories: x(t), p(t), energy in time E(t), 
#          and phase space trajectory p(x)

lineW = 2 # Line thickness

plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
plt.plot(t_num,x_num,'-g',linewidth=lineW, label='Ground truth'); 
plt.plot(t_net, x,'--b', label='Neural Network')
plt.plot(t_s,x_s,':k',linewidth=lineW, label='Symplectic Euler')
plt.plot(t_s100,x_s100,'-.r',linewidth=lineW, label='Symplectic Euler X 100 points')
plt.ylabel('x');plt.xlabel('t')
#plt.legend()

plt.subplot(2,2,2)
plt.plot(t_num,E_ex,'-g',linewidth=lineW, label='Ground truth'); 
plt.plot(t_net, E_num,'--b', label='Neural Network')
plt.plot(t_s,E_s,':k',linewidth=lineW,label='Symplectic Euler'); 
plt.plot(t_s100,E_s100,'-.r',linewidth=lineW,label='Symplectic Euler x 100 points'); 
plt.ylabel('E');plt.xlabel('t')
plt.ylim([0.5*E0,1.5*E0])
plt.legend()

plt.subplot(2,2,3)
plt.plot(t_num,px_num,'-g',linewidth=lineW); 
plt.plot(t_net, px,'--b')
plt.plot(t_s,p_s,':k',linewidth=lineW); 
plt.plot(t_s100,p_s100,'-.r',linewidth=lineW); 
plt.ylabel('px');plt.xlabel('t')

plt.subplot(2,2,4)
plt.plot(x_num,px_num,'-g',linewidth=lineW); 
plt.plot(x, px,'--b')
plt.plot(x_s,p_s,':k',linewidth=lineW); 
plt.plot(x_s,p_s,'-.r',linewidth=lineW); 
plt.ylabel('p');plt.xlabel('x');

#plt.savefig('../results/nonlinearOscillator_trajectories.png')
plt.savefig('nonlinearOscillator_trajectories.png')



## Figure for the error in the predicted solutions: delta_x and delta_p, 
# and the energy again

# calculate the errors for the solutions obtained by network and euler
dx_num =x_num-x_num;       dp_num=px_num-px_num
dx = x_num - x[:,0];            dp = px_num - px[:,0]
dx_s = x_num - x_s;        dp_s = px_num - p_s
# find the exact solution for more points used in Euler x 100
x_num100,  px_num100  = NLosc_solution(N,t_s100, x0,  px0,  lam)
dx_s100 = x_num100 - x_s100;  dp_s100 = px_num100 - p_s100


plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.plot(dx_num,dp_num,'-g',linewidth=lineW); 
#plt.plot(dx, dp,'--b')
plt.plot(dx_s,dp_s,':k',linewidth=lineW); 
plt.plot(dx_s100,dp_s100,'-.r',linewidth=lineW); 
plt.ylabel('$\delta_p$');plt.xlabel('$\delta_x$');
plt.ylim([-1e-2,1e-2])
plt.xlim([-1e-2,1e-2])
#plt.legend()

plt.subplot(2,2,2)
plt.plot(t_num,E_ex,'-g',linewidth=lineW, label='Ground truth'); 
plt.plot(t_net, E,'--b', label='Neural Network')
plt.plot(t_s,E_s,':k',linewidth=lineW,label='symplectic Euler'); 
plt.plot(t_s100,E_s100,'-.r',linewidth=lineW,label='symplectic Euler x 100 points'); 
plt.ylabel('E');plt.xlabel('t')
plt.ylim([0.9*E0,1.1*E0])
plt.legend()

plt.subplot(2,2,3)
plt.plot(t_num,dx_num,'-g',linewidth=lineW, label='Ground truth'); 
plt.plot(t_net, dx,'--b', label='Neural Network')
plt.plot(t_s,dx_s,':k',linewidth=lineW, label='symplectic Euler')
plt.plot(t_s100,dx_s100,'-.r',linewidth=lineW, label='symplectic Euler X 100 points')
plt.ylabel('$\delta_x$');plt.xlabel('t')

plt.subplot(2,2,4)
plt.plot(t_num,dp_num,'-g',linewidth=lineW); 
plt.plot(t_net, dp,'--b')
plt.plot(t_s,dp_s,':k',linewidth=lineW); 
plt.plot(t_s100,dp_s100,'-.r',linewidth=lineW); 
plt.ylabel('$\delta_p$');plt.xlabel('t')


#plt.savefig('../results/nonlinearOscillator_error.png')
plt.savefig('nonlinearOscillator_error.png')

