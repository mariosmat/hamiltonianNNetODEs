#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:10:49 2020
@author: marios mattheakis

In this code a Hamiltonian Neural Network is designed and employed
to solve a system of four differential equations obtained by Hamilton's
equations for the the Hamiltonian of Henon-Heiles chaotic dynamical.
"""

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
import time
import copy
from scipy.integrate import odeint
dtype=torch.float



# Define the sin() activation function
class mySin(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.sin(input)
   
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

# Energy of nonlinear oscillator
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
    E_ex = E0*np.ones(N+1);
    return E0, E_ex




# Set the initial state. lam controls the nonlinearity
x0, y0, px0, py0, lam =  0.3,-0.3, 0.3, 0.15, 1; 
t0, t_max, N = 0.,6*np.pi, 200; dt = t_max/N; 
X0 = [t0, x0, y0, px0, py0, lam]
t_num = np.linspace(t0, t_max, N+1)
E0, E_ex = HH_exact(N,x0, y0, px0, py0, lam)

x_num, y_num, px_num, py_num = HHsolution(N,t_num, x0, y0, px0, py0, lam)
# E_num = energy( x_ex, y_ex, vx_ex, vy_ex, lam)





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

def saveData(path, t, x,y, px,py, E, loss):
    np.savetxt(path+"t.txt",t)
    np.savetxt(path+"x.txt",x)
    np.savetxt(path+"y.txt",y)
    np.savetxt(path+"px.txt",px)
    np.savetxt(path+"py.txt",py)
    np.savetxt(path+"E.txt",E)
    np.savetxt(path+"Loss.txt",loss)
    
    
    
# Define some functions used by the Hamiltonian network

def parametricSolutions(t, nn, X0):
    # parametric solutions
    t0, x0, y0, px0, py0, lam = X0[0],X0[1],X0[2],X0[3],X0[4],X0[5]
    N1, N2, N3, N4 = nn(t)
    dt =t-t0
#### THERE ARE TWO PARAMETRIC SOLUTIONS. Uncomment f=dt 
    f = (1-torch.exp(-dt))
#     f=dt
    x_hat  = x0  + f*N1
    y_hat  = y0  + f*N2
    px_hat = px0 + f*N3
    py_hat = py0 + f*N4
    return x_hat, y_hat, px_hat, py_hat

def hamEqs_Loss(t,x,y,px,py,lam):
    # Define the loss function by Hamilton Eqs., write explicitely the Ham. Equations
    xd,yd,pxd,pyd= dfx(t,x),dfx(t,y),dfx(t,px),dfx(t,py)
    fx  = xd - px; 
    fy  = yd - py; 
    fpx = pxd + x + 2.*lam*x*y
    fpy = pyd + y + lam*(x.pow(2) - y.pow(2))
    Lx  = (fx.pow(2)).mean();  Ly  = (fy.pow(2)).mean(); 
    Lpx = (fpx.pow(2)).mean(); Lpy = (fpy.pow(2)).mean();
    L = Lx + Ly + Lpx + Lpy
    return L


def hamEqs_Loss_byH(t,x,y,px,py,lam):
    # This is an alternative way to define the loss function:
    # Define the loss function by Hamilton Eqs. directly from Hamiltonian H
    #
    # Potential and Kinetic Energy
    V = 0.5*(x.pow(2) + y.pow(2)) + lam*(x.pow(2)*y - y.pow(3)/3)
    K = 0.5*(px.pow(2)+py.pow(2))
    ham = K + V
    xd,yd,pxd,pyd= dfx(t,x),dfx(t,y),dfx(t,px),dfx(t,py)
    # calculate the partial spatial derivatives of H
    hx  = grad([ham], [x], grad_outputs=torch.ones(x.shape, dtype=dtype), create_graph=True)[0]
    hy  = grad([ham], [y], grad_outputs=torch.ones(y.shape, dtype=dtype), create_graph=True)[0]
    hpx = grad([ham], [px], grad_outputs=torch.ones(px.shape, dtype=dtype), create_graph=True)[0]
    hpy = grad([ham], [py], grad_outputs=torch.ones(py.shape, dtype=dtype), create_graph=True)[0]
    # Hamilton Eqs
    fx  = xd - hpx;      fy  = yd - hpy
    fpx = pxd + hx;      fpy = pyd + hy
    Lx  = (fx.pow(2)).mean();  Ly  = (fy.pow(2)).mean(); 
    Lpx = (fpx.pow(2)).mean(); Lpy = (fpy.pow(2)).mean();
    L = Lx + Ly + Lpx + Lpy
    return L


def hamiltonian_Loss(t,x,y,px,py,lam):
# Define the loss function as the time derivative of the hamiltonian
    xd,yd,pxd,pyd= dfx(t,x),dfx(t,y),dfx(t,px),dfx(t,py)
    ham = 0.5*(px.pow(2)+py.pow(2)+x.pow(2)+y.pow(2))+lam*(x.pow(2)*y-y.pow(3)/3)
    hx  = grad([ham], [x], grad_outputs=torch.ones(x.shape, dtype=dtype), create_graph=True)[0]
    hy  = grad([ham], [y], grad_outputs=torch.ones(y.shape, dtype=dtype), create_graph=True)[0]
    hpx = grad([ham], [px], grad_outputs=torch.ones(px.shape, dtype=dtype), create_graph=True)[0]
    hpy = grad([ham], [py], grad_outputs=torch.ones(py.shape, dtype=dtype), create_graph=True)[0]
    ht = hx*xd + hy*yd + hpx*pxd + hpy*pyd
    L = (ht.pow(2)).mean()
    return L


# NETWORK ARCHITECTURE

# A two hidden layer NN, 1 input & two output
class odeNet_HH_MM(torch.nn.Module):
    def __init__(self, D_hid=10):
        super(odeNet_HH_MM,self).__init__()

        # Define the Activation
#         self.actF = torch.nn.Sigmoid()   
        self.actF = mySin()
        
        # define layers
        self.Lin_1   = torch.nn.Linear(1, D_hid)
        self.Lin_2   = torch.nn.Linear(D_hid, D_hid)
        self.Lin_out = torch.nn.Linear(D_hid, 4)

    def forward(self,t):
        # layer 1
        l = self.Lin_1(t);    h = self.actF(l)
        # layer 2
        l = self.Lin_2(h);    h = self.actF(l)
        # output layer
        r = self.Lin_out(h)
        xN  = (r[:,0]).reshape(-1,1); yN  = (r[:,1]).reshape(-1,1)
        pxN = (r[:,2]).reshape(-1,1); pyN = (r[:,3]).reshape(-1,1)
        return xN, yN, pxN, pyN

# Train the NN
def run_odeNet_HH_MM(X0, tf, neurons, epochs, n_train,lr,
                    minibatch_number = 1):
    fc0 = odeNet_HH_MM(neurons)
    fc1=0; # fc1 will be a deepcopy of the network with the lowest training loss
    # optimizer
    betas = [0.999, 0.9999]
    
    optimizer = optim.Adam(fc0.parameters(), lr=lr, betas=betas)
    Loss_history = [];     Llim =  1 
    
    t0=X0[0];
    grid = torch.linspace(t0, tf, n_train).reshape(-1,1)
    
    
    
## TRAINING ITERATION    
    TeP0 = time.time()
    for tt in range(epochs):                
# Perturbing the evaluation points & forcing t[0]=t0
        t=perturbPoints(grid,t0,tf,sig=.03*tf)
            
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
            x,y,px,py =parametricSolutions(t_mb,fc0,X0)
# LOSS
#  Loss function defined by Hamilton Eqs. (symplectic): Writing explicitely the Eqs (faster)
            Ltot = hamEqs_Loss(t_mb,x,y,px,py,lam)

#  Loss function defined by Hamilton Eqs. (symplectic): Calculating with auto-diff the Eqs (slower)
#             Ltot = hamEqs_Loss_byH(t_mb,x,y,px,py,lam)
    
#  Alternatively, Loss function defined by Hamiltonian (slower)
#             if tt>1e3:
#                 Ltot += hamiltonian_Loss(t_mb,x,y,px,py,lam)
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



###
    
## TRAIN THE NETWORK

n_train, neurons, epochs, lr,mb = 100, 50, int(3e4), 8e-3, 1 
model,loss,runTime = run_odeNet_HH_MM(X0, t_max, 
                                      neurons, epochs, n_train,lr,mb)

# Loss function
print('Training time (minutes):', runTime/60)
plt.loglog(loss,'-b',alpha=0.975);
plt.tight_layout()


# TEST THE PREDICTED SOLUTIONS
nTest = n_train; tTest = torch.linspace(t0,t_max,nTest)
tTest = tTest.reshape(-1,1);
tTest.requires_grad=True
t_net = tTest.detach().numpy()
x,y,px,py =parametricSolutions(tTest,model,X0)
x=x.data.numpy(); y=y.data.numpy()
px=px.data.numpy(); py=py.data.numpy()
E  = energy(x, y, px, py, lam)




###################
# Symplectic Euler
####################
Ns = n_train;
# Ns = 10*n_train;
t_s = np.linspace(t0, t_max, Ns+1)
dts = t_max/Ns

x_s = np.zeros(Ns+1); px_s = np.zeros(Ns+1);
y_s = np.zeros(Ns+1); py_s = np.zeros(Ns+1)

x_s[0], px_s[0], y_s[0], py_s[0] = x0,  px0,y0, py0
for n in range(Ns):
    x_s[n+1] = x_s[n] + dts*px_s[n]
    y_s[n+1] = y_s[n] + dts*py_s[n]
    
    px_s[n+1] = px_s[n] - dts*(x_s[n+1] + 2*lam*x_s[n+1]*y_s[n+1])
    py_s[n+1] = py_s[n] - dts*(y_s[n+1] + lam*(x_s[n+1]**2-y_s[n+1]**2))    
E_s = energy( x_s, y_s, px_s, py_s, lam)



################
# Make the plots
#################
lineW = 2 # Line thickness

plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
plt.plot(t_num,x_num,'-r',linewidth=lineW, label='Ground truth'); 
plt.plot(t_net, x,'--b', label='Neural Net'); 
plt.plot(t_s,x_s,':y',linewidth=lineW, label='Symplectic Euler'); 
plt.ylabel('x');plt.xlabel('t')

plt.subplot(2,2,2)
plt.plot(t_num,E_ex,'-r',linewidth=lineW); 
plt.plot(t_net, E,'--b')
plt.plot(t_s,E_s,':y',linewidth=lineW); 
plt.ylabel('E');plt.xlabel('t')
plt.ylim([0.5*E0,1.5*E0])

plt.subplot(2,2,3)
plt.plot(t_num,px_num,'-r',linewidth=lineW); 
plt.plot(t_net, px,'--b')
plt.plot(t_s,px_s,':y',linewidth=lineW); 
plt.ylabel('px');plt.xlabel('t')

plt.subplot(2,2,4)
plt.plot(x_num,px_num,'-r',linewidth=lineW); 
plt.plot(x, px,'--b')
plt.plot(x_s,px_s,'--y',linewidth=lineW); 
plt.ylabel('p');plt.xlabel('x');

















