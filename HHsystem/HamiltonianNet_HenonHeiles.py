#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:10:49 2020
@author: marios mattheakis

In this code a Hamiltonian Neural Network is designed and employed
to solve a system of four differential equations obtained by Hamilton's
equations for the Hamiltonian of Henon-Heiles chaotic dynamical.
"""

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
import time
import copy
from os import path
import sys

from utils_HHsystem import symEuler, HH_exact, HHsolution, energy ,saveData

dtype=torch.float


# %matplotlib inline
plt. close('all')


# Define the sin() activation function
class mySin(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.sin(input)
   
    
   #####################################
# Hamiltonian Neural Network (HNN) class
####################################


# Calculate the derivatice with auto-differention
def dfx(x,f):
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
    # t.data[0] = torch.ones(1,1)*t0
    t.requires_grad = False
    return t

    
def parametricSolutions(t, nn, X0):
    # parametric solutions
    t0, x0, y0, px0, py0, _ = X0[0],X0[1],X0[2],X0[3],X0[4],X0[5]
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




def hamiltonian(x,y,px,py,lam):
    #returns the hamiltonian ham for Kinetic (K)  and Potential (V) Energies
    V = 0.5*(x**2 + y**2) + lam*(x**2*y - y**3/3)
    K = 0.5*(px**2+py**2)
    ham = K + V
    return ham


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
def run_odeNet_HH_MM(X0, tf, neurons, epochs, n_train,lr, PATH= "models/model_HH", loadWeights=False,
                     minLoss=1e-3):
                    
    fc0 = odeNet_HH_MM(neurons)
    fc1 =  copy.deepcopy(fc0) # fc1 is a deepcopy of the network with the lowest training loss
    # optimizer
    betas = [0.999, 0.9999]
    
    optimizer = optim.Adam(fc0.parameters(), lr=lr, betas=betas)
    Loss_history = [];     Llim =  1 
    Loss_erg_history= []
    
    t0=X0[0];
    x0, y0, px0, py0, lam = X0[1], X0[2], X0[3], X0[4], X0[5]
    # Initial Energy that should be convserved 
    
    ham0 = hamiltonian(x0,y0,px0,py0,lam)    
    grid = torch.linspace(t0, tf, n_train).reshape(-1,1)
    
    
    
## LOADING WEIGHTS PART if PATH file exists and loadWeights=True
    if path.exists(PATH) and loadWeights==True:
        checkpoint = torch.load(PATH)
        fc0.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        tt = checkpoint['epoch']
        Ltot = checkpoint['loss']
        fc0.train(); # or model.eval
    
    
## TRAINING ITERATION    
    TeP0 = time.time()
    for tt in range(epochs):                
# Perturbing the evaluation points & forcing t[0]=t0
        # t=perturbPoints(grid,t0,tf,sig=.03*tf)
        t=perturbPoints(grid,t0,tf,sig= 0.3*tf)
        t.requires_grad = True

#  Network solutions 
        x,y,px,py =parametricSolutions(t,fc0,X0)

# LOSS FUNCTION
    #  Loss function defined by Hamilton Eqs.
        Ltot = hamEqs_Loss(t,x,y,px,py,lam)
            
    # ENERGY REGULARIZATION
        ham  = hamiltonian(x,y,px,py,lam)
        L_erg =  .5*( ( ham - ham0).pow(2) ).mean() 
        Ltot=Ltot+ L_erg
    

# OPTIMIZER
        Ltot.backward(retain_graph=False); #True
        optimizer.step() 
        loss = Ltot.data.numpy()
        loss_erg=L_erg.data.numpy()
        optimizer.zero_grad()

# keep the loss function history
        Loss_history.append(loss)       
        Loss_erg_history.append(loss_erg)

#Keep the best model (lowest loss) by using a deep copy
        if  tt > 0.8*epochs  and Ltot < Llim:
            fc1 =  copy.deepcopy(fc0)
            Llim=Ltot 

# break the training after a thresold of accuracy
        if Ltot < minLoss :
            fc1 =  copy.deepcopy(fc0)
            print('Reach minimum requested loss')
            break



    TePf = time.time()
    runTime = TePf - TeP0     
    
    
    torch.save({
    'epoch': tt,
    'model_state_dict': fc1.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': Ltot,
    }, PATH)

    return fc1, Loss_history, runTime, Loss_erg_history

###

def trainModel(X0, t_max, neurons, epochs, n_train, lr,  loadWeights=True, minLoss=1e-6, showLoss=True, PATH ='models/'):
    model,loss,runTime, loss_erg = run_odeNet_HH_MM(X0, t_max, neurons, epochs, n_train,lr,  loadWeights=loadWeights, minLoss=minLoss)

    np.savetxt('data/loss.txt',loss)
    
    if showLoss==True :
        print('Training time (minutes):', runTime/60)
        print('Training Loss: ',  loss[-1] )
        plt.figure()
        plt.loglog(loss,'-b',alpha=0.975, label='Total loss');
        plt.loglog(loss_erg,'-r',alpha=0.75, label='Energy penalty');
        plt.legend()
        plt.tight_layout()
        plt.ylabel('Loss');plt.xlabel('t')
    
        plt.savefig('HenonHeiles_loss.png')
    

def loadModel(PATH="models/model_HH"):
    if path.exists(PATH):
        fc0 = odeNet_HH_MM(neurons)
        checkpoint = torch.load(PATH)
        fc0.load_state_dict(checkpoint['model_state_dict'])
        fc0.train(); # or model.eval
    else:
        print('Warning: There is not any trained model. Terminate')
        sys.exit()

    return fc0    



############
# Set the initial state. lam controls the nonlinearity
t0, x0, y0, px0, py0, lam =  0, 0.3,-0.3, 0.3, 0.15, 1; 
X0 = [t0, x0, y0, px0, py0, lam]

# Run first a short time prediction. 
# Then load the model and train for longer time

# ## SHORT TIME
# t_max, N =  6*np.pi, 500; 
# print(t_max * 0.069, ' Lyapunov times prediction'); dt = t_max/N; 
# n_train, neurons, epochs, lr = N, 80, int(2e4 ), 8e-3
# trainModel(X0, t_max, neurons, epochs, n_train, lr,  loadWeights=False, minLoss=1e-8, showLoss=True)


## LONG TIME: use loadWeights=True
t_max, N =  12*np.pi, 500; 
print(t_max * 0.069, ' Lyapunov times prediction'); dt = t_max/N; 
n_train, neurons, epochs, lr = N, 80, int(5e4 ), 5e-3
# TRAIN THE NETWORK. 
trainModel(X0, t_max, neurons, epochs, n_train, lr,  loadWeights=True, minLoss=1e-8, showLoss=True)

model = loadModel()



#####################################
# TEST THE PREDICTED SOLUTIONS
#######################################3
# 

nTest = N ; t_max_test = 1.0*t_max
tTest = torch.linspace(t0,t_max_test,nTest)

tTest = tTest.reshape(-1,1);
tTest.requires_grad=True
t_net = tTest.detach().numpy()


x,y,px,py =parametricSolutions(tTest,model,X0)
x=x.data.numpy(); y=y.data.numpy()
px=px.data.numpy(); py=py.data.numpy()
E  = energy(x, y, px, py, lam)




# ####################
# Scipy solver
######################
t_num = np.linspace(t0, t_max_test, N)
E0, E_ex = HH_exact(N,x0, y0, px0, py0, lam)
x_num, y_num, px_num, py_num = HHsolution(N,t_num, x0, y0, px0, py0, lam)
E_num = energy(x_num, y_num, px_num, py_num, lam)



# ###################
# # Symplectic Euler
# # ####################
Ns = n_train -1; 
E_s, x_s, y_s, px_s, py_s, t_s = symEuler(Ns, x0,y0, px0,py0,t0,t_max_test,lam)
# # 10 times more time points

Ns10 = 10*n_train ; 

T0 = time.time()
E_s10, x_s10, y_s10, px_s10, py_s10, t_s10 = symEuler(Ns10, x0,y0, px0,py0,t0,t_max_test,lam)
runTimeEuler = time.time() - T0
print('Euler runtime is ', runTimeEuler/60)

################
# Make the plots
#################

# Figure for trajectories: x(t), p(t), energy in time E(t), 
#          and phase space trajectory p(x)

lineW = 2 # Line thickness

plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
plt.plot(t_num,x_num,'-g',linewidth=lineW, label='Ground truth'); 
plt.plot(t_net, x,'--b', label='Neural Net'); 
plt.plot(t_s,x_s,':k',linewidth=lineW, label='Symplectic Euler'); 
plt.plot(t_s10,x_s10,'-.r',linewidth=lineW, label='Symplectic Euler x 10 points'); 
plt.ylabel('x');plt.xlabel('t')
plt.legend()

plt.subplot(2,2,2)
plt.plot(t_num,E_ex,'-g',linewidth=lineW); 
plt.plot(t_net, E,'--b')
plt.plot(t_s,E_s,':k',linewidth=lineW); 
plt.plot(t_s10,E_s10,'-.r',linewidth=lineW); 
plt.ylabel('E');plt.xlabel('t')
plt.ylim([1.1*E0,0.9*E0])

plt.subplot(2,2,3)
plt.plot(t_num,px_num,'-g',linewidth=lineW); 
plt.plot(t_net, px,'--b')
plt.plot(t_s,px_s,':k',linewidth=lineW); 
plt.plot(t_s10,px_s10,'-.r',linewidth=lineW); 
plt.ylabel('$p_x$');plt.xlabel('t')

plt.subplot(2,2,4)
plt.plot(x_num,y_num,'-g',linewidth=lineW); 
plt.plot(x, y,'--b')
plt.plot(x_s,y_s,'--k',linewidth=lineW); 
plt.plot(x_s10, y_s10,'-.r',linewidth=lineW); 
plt.ylabel('y');plt.xlabel('x');

plt.tight_layout()
plt.savefig('HenonHeiles_trajectories.png')




# calculate the errors for the solutions obtained by network 
dx_num =x_num-x_num;       dp_num=px_num-px_num
dx = x_num - x[:,0];       dpx = px_num - px[:,0]
dy = y_num- y[:,0];        dpy = py_num - py[:,0]

# # calculate the errors for the solutions obtained by Euler
x_numN, y_numN, px_numN, py_numN   = HHsolution(Ns,t_s, x0, y0, px0, py0, lam) 
dx_s = x_numN - x_s;        dpx_s = px_numN - px_s
dy_s = y_numN - y_s;        dpy_s = py_numN - py_s

x_numN, y_numN, px_numN, py_numN   = HHsolution(Ns10,t_s10, x0, y0, px0, py0, lam) 
dx_s10 =  x_numN - x_s10;      dpx_s10 = px_numN - px_s10
dy_s10 = y_numN -  y_s10;       dpy_s10 = py_numN - py_s10





plt.figure(figsize=(10,8))


plt.subplot(2,2,1)
plt.plot(t_net,dx, 'b', label='Neural Net')
plt.plot(t_s, dx_s, ':k', label='Symplectic Euler')
plt.plot(t_s10, dx_s10, '-.r', label='Symplectic Euler x 10')
plt.ylabel('$\delta_x$');plt.xlabel('t')
plt.legend()



plt.subplot(2,2,2)
plt.plot(dx,dpx,'b')
plt.plot(dx_s, dpx_s, ':k')
plt.plot(dx_s10, dpx_s10, '-.r')
plt.ylabel('$\delta_{p_x}$'); plt.xlabel('$\delta_x$');


plt.subplot(2,2,3)
plt.plot(t_net,dy, 'b')
plt.plot(t_s, dy_s, ':k')
plt.plot(t_s10, dy_s10, '-.r')
plt.ylabel('$\delta_y$');plt.xlabel('t')

plt.subplot(2,2,4)
plt.plot(dy,dpy,'b')
plt.plot(dy_s, dpy_s, ':k')
plt.plot(dy_s10, dpy_s10, '-.r')
plt.ylabel('$\delta_{p_y}$'); plt.xlabel('$\delta_y$');

plt.tight_layout()
plt.savefig('HenonHeiles_trajectories_error.png')



# saveData('data/', t_net, x, y, px,py, E)
# saveData('data/Euler10/', t_s10, x_s10, y_s10, px_s10,py_s10, E_s10)
# saveData('data/solver/', t_num, x_num, y_num, px_num, py_num, E_num)

# np.savetxt('data/'+"dx.txt",dx)
# np.savetxt('data/'+"dy.txt",dy)
# np.savetxt('data/'+"dpx.txt",dpx)
# np.savetxt('data/'+"dpy.txt",dpy)

# np.savetxt('data/Euler10/'+"dx.txt",dx_s10)
# np.savetxt('data/Euler10/'+"dy.txt",dy_s10)
# np.savetxt('data/Euler10/'+"dpx.txt",dpx_s10)
# np.savetxt('data/Euler10/'+"dpy.txt",dpy_s10)


# saveData('data/Euler200/', t_s10, x_s10, y_s10, px_s10,py_s10, E_s10)
# np.savetxt('data/Euler200/'+"dx.txt",dx_s10)
# np.savetxt('data/Euler200/'+"dy.txt",dy_s10)
# np.savetxt('data/Euler200/'+"dpx.txt",dpx_s10)
# np.savetxt('data/Euler200/'+"dpy.txt",dpy_s10)
