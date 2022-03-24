
**Hamiltonian Neural Networks for Solving Equations of Motion.**

Data-free Hamiltonian Neural Network suggests an alternative way to solve the equations of motion (Hamilton's equations) for dynamical system that conserve energy.   This is an equation-driven machine learning method since no data are used during the network optimization training. For more information check the papers:


**arXiv:2001.11107** : Hamiltonian Neural Networks for solving differential equations  

Two systems are solved and presented in the above paper, the 1D nonlinear oscillator and the 2D Henon-Heiles chaotic system. There are two main directories one for each of the systems.


**Basic Usage**
Simply run the python codes, as:

> python HNN_NLoscillator

> python HamiltonianNet_HenonHeiles



**Description**
Each directory has the main code (HNN_NLoscillator.py and HamiltonianNet_HenonHeiles.py) and a supplementary code (utils_NLoscillator.py and utils_HHsystem.py), corresponding to the wor different problems that are discussed in the manuscript. The code employ a Neural Network for solving Hamilton's equations, a system of ordinary differential equations, that govern the  temporal motion of dynamical systems. For comparison, a Scipy solver is used and its solutions are considered as the ground truth. In addition, an Euler symplectic integrator is used. The solutions obtained by the Hamiltonian neural network conserve the energy better than the solutions obtained by the symplectic Euler when same number of evaluation points are considered. 

You can investigate different activation functions and parametric solutions. Check for the comments # Define the Activation and # parametric solutions, respectively.  By default an energy penalty is used to enhanced the learning. This can be disabled by comment out the Ltot = Ltot+Lreg in the training loop.

 After running the python codes (by reproducible run) figures will be saved. More specifically:

In nonlinear oscillator: (a) A figure with the training loss as a function of the training epochs; (b) The predicted trajectories x(t), y(t), the energy in time E(t), and the phase space trajectory p(x); (c) The errors of the predicted solutions delta_x(t), delta_p(t), the phase space error delta_p(delta_x), and again the energy in time.

In Henon-Heiles:  (a) A figure with the training loss as a function of the training epochs; (b) The  predicted trajectories x(t), y(t), the energy in time E(t),  the orbit in x-y plane y(x), the errors for the position delta_x and delta_y, and the phase space errors  delta_px(delta_x) and delta_py(delta_y).


