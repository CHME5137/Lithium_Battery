# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:30:14 2022

@author: Dominick and Nora

~~~~~~~~~~~~~~~~~~~~
https://tinyurl.com/2ny7n938
~~~~~~~~~~~~~~~~~~~~
"""
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import numpy as np
from SALib.sample import morris as ms
from SALib.analyze import morris as ma
from SALib.plotting import morris as mp
from tqdm import tqdm 

# Physical constants 
R = 8.314 # J/mol-K, gas constant

# Electrochemical constants
alpha_a = 0.5 # anodic symmetry parameter
alpha_c = 0.5 # cathodic symmetry parameter
n = 1 # Number of electrons
F = 96485 # C/mol, Faraday's Constant

def odeSys(x,y,p):
    """
    Function describing the system of ODEs, including Butler-Volmar kinetics,
    Ohm's law for the ionic and electronic phases, and charge conservation 
    
    Input:
        x - position within the electrode
        y - array containing the ionic and electronic potentials, the electronic
            current, and the rate of change of the electronic current
        p - array of parameters to describe the system, including electronic 
            and ionic conductivities, electrode specific area, exchange 
            current density, temperature, current, and electrode thickness 
    
    Output:
        array containing the rate of change for all varaibels in the y array 
    """
    sigma, kappa, a, i_0, T, i_tot, L = p # Unpack parameters
    phi_s, phi_l, i_s, didx_s = y # Unpack variables 
    # Based on Butler-Volmar electrode kinetics 
    didx_l = a*i_0*(np.exp(alpha_a*n*F*(phi_s-phi_l)/R/T)-\
                    np.exp(-alpha_c*n*F*(phi_s-phi_l)/R/T))
    didx_s = -didx_l # Conservation of charge 
    dphidx_s = -i_s/sigma # Ohm's Law for electronic phase 
    dphidx_l = -(i_tot-i_s)/kappa # Ohm's Law for ionic phase 
    # Based on the derivative of Butler-Volmar 
    d2idx2_s = -a*i_0*alpha_a*n*F*(-i_s/sigma+(i_tot-i_s)/kappa)*\
        (np.exp(alpha_a*n*F*(phi_s-phi_l)/R/T)+\
         np.exp(-alpha_c*n*F*(phi_s-phi_l)/R/T))/R/T
    return [dphidx_s,dphidx_l,didx_s, d2idx2_s]

def BC(y0, y1, p):
    """
    Function that sets up the boundary conditions for the system of ODEs set up
    in the function odeSys. 
    
    Input: 
        y0 - array of values for y at the left bound for odeSys, i.e. at x=0
        y1 - array of values for y at the right bound for odeSys, i.e. at x=L
        p - array of parameters to describe the system, including electronic 
            and ionic conductivities, electrode specific area, exchange 
            current density, temperature, current, and electrode thickness 
    
    Output:
        array of residules based on the applied boundary conditions 
    """
    sigma, kappa, a, i_0, T, i_tot, L = p # Unpack parameters 
    
    BCs = [
           # Arbitrary ref point, electronic potential is 0V at seperator (x=L)
           y1[0], 
           # Electron current is total current at current collector (x=0)
           i_tot-y0[2],
           # Electronic current is 0 at separator (x=L)
           y1[2], 
           # Electronic current is constant at current collector (x=0)
           y0[3]] 
    return BCs

def solveSys(params=[10**-2,10**-2,10**6,1,298,10**-3,10**-4]):
    """
    Function that acts as a wrapper for solving the system of ODEs with only 
    passing the values of the relevant parameters. The function can be passed
    a single array of parameters and will return two numbers; one describing 
    the overall heterogeneity and a second value describing the location 
    and relative value of heterogeneity. If multiple arrays of parameters are 
    passed in a matrix, the function will return an array of values describing 
    the overall heterogeneity of the electrode. 
    
    Input:
        p - array of parameters to describe the system, including electronic 
            and ionic conductivities, electrode specific area, exchange 
            current density, temperature, current, and electrode thickness 
            
    Output:
        array containing numbers that describe the heterogeneity of the system
    """
    x = np.linspace(0,params[-1],10) # Initialize 10 position nodes for solving
    if len(x.shape) > 1: # Checks if multiple sets of parameters were recieved 
        heterogeneity_factor = list() # Initialize list for returning 
        # Loop through each set of params, using tqdm for progress bar 
        for i in tqdm(range(np.size(x,1))): 
            xi = x[:,i] # Get a 1-D array of position nodes 
            # Get matrix of initial y values at each position node 
            yi = np.vstack((0.2 * (1-xi), 0.1 * (1-xi), params[-2,i] * \
                            (1-xi),params[-2,i] * (1-xi)))
            # Get ODE system and BCs with current params baked in
            odeSys_set = lambda x,y: odeSys(x,y,params[:,i])
            BC_set = lambda x,y: BC(x,y,params[:,i])
            # Solve BVP 
            soln = solve_bvp(odeSys_set,BC_set,xi,yi,tol=10**-2,max_nodes=10**6)
            # Calculate reaction rate across the thickness of the electrode 
            didx = -np.diff(soln.y[2,:])/np.diff(soln.x)
            # Append an arbitrary number to describe the overall heterogeneity 
            heterogeneity_factor.append((max(didx)-min(didx)) \
                                        / (sum(didx) / len(didx)))
        return np.array(heterogeneity_factor / max(heterogeneity_factor))
    else: # One set of params were passed to the function 
        # Get matrix of initial y values at each position node 
        y = np.vstack((0.2 * (1-x), 0.1 * (1-x), params[-2] * (1-x),params[-2] * (1-x)))
        # Get ODE system and BCs with current params baked in
        odeSys_set = lambda x,y: odeSys(x,y,params)
        BC_set = lambda x,y: BC(x,y,params)
        # Solve BVP
        soln = solve_bvp(odeSys_set,BC_set,x,y,tol=10**-4,max_nodes=10**6)
        # Calculate reaction rate across the thickness of the electrode 
        didx = -np.diff(soln.y[2,:])/np.diff(soln.x)
        # Calculate an arbitrary number to describe the overall heterogeneity 
        heterogeneity_factor = (max(didx)-min(didx)) / (sum(didx) / len(didx))
        # Calculate an arbitrary number to describe the relative heterogeneity 
        relative_heterogeneity = (max((didx[0],didx[-1])) / min((didx[0],didx[-1]))) * ((-1)**(didx[-1] > didx[0]))
        return heterogeneity_factor, relative_heterogeneity

# Set up the Morris Problem. Code from Prof. West lecture 
morris_problem = {
    # There are seven variables
    'num_vars': 7,
    # These are their names
    'names': ['sigma', 'kappa', 'a', 'i_0', 'T', 'i_tot', 'L'], 

    # These are their plausible ranges over which we'll move the variables
    'bounds': [[10**-4,10**-1],
               [10**-4, 10**-1],
               [10**5, 10**7],
               [10**-1, 10**1],
               [283, 303], 
               [0.5*10**-3,2*10**-3], 
               [0.5*10**-4,2*10**-4] 
              ],
    # I don't want to group any of these variables together
    'groups': None
    }

num_levels = 4
trajectories = int(1e1)
sample = ms.sample(morris_problem, trajectories, num_levels=num_levels)

# Run the sample through the model
output = solveSys(sample.T)

# Store the results for plotting of the analysis
Si = ma.analyze(morris_problem, 
                sample, 
                output, 
                print_to_console=False, 
                num_levels=num_levels)
print("{:20s} {:>7s} {:>7s} {:>7s}".format("Name", "mu", "mu_star", "sigma"))
for name, s1, st, mean in zip(morris_problem['names'], Si['mu'], Si['mu_star'], Si['sigma']):
    print("{:20s} {:=7.2f} {:=7.2f} {:=7.2f}".format(name, s1, st, mean))
    
fig, (ax1, ax2) = plt.subplots(1,2)
mp.horizontal_bar_plot(ax1, Si) #  param_dict={}
mp.covariance_plot(ax2, Si, {})

# Set up arrays of equally spaced ionic and electronic conductivities in 
# logarithmic space 
sigma_array = 10**np.linspace(-5,-1,40)
kappa_array = 10**np.linspace(-5,-1,40)

# Initialize matrices to store results 
heterogeneity_factor_array = np.zeros((np.size(sigma_array,0),np.size(kappa_array,0)))
relative_heterogeneity_array = np.zeros((np.size(sigma_array,0),np.size(kappa_array,0)))

# Solve system for all combinations of conductivities 
for s in tqdm(range(np.size(sigma_array,0))):
    for k in range(np.size(kappa_array,0)):
        heterogeneity_factor_array[s,k], relative_heterogeneity_array[s,k] = \
            solveSys([sigma_array[s],kappa_array[k],10**6,1,298,10**-3,10**-4])

# Get the log and normalize the results              
heterogeneity_factor_array = np.log10(heterogeneity_factor_array)
heterogeneity_factor_array = heterogeneity_factor_array \
                                / np.amax(heterogeneity_factor_array)
relative_heterogeneity_array = np.log10(np.absolute(relative_heterogeneity_array))\
                                * ((-1)**(relative_heterogeneity_array<-1))

# plt.figure()
# plt.contourf(np.log10(sigma_array), np.log10(kappa_array), heterogeneity_factor_array);
# plt.xlabel('Log Sigma [S/m]')
# plt.ylabel('Log Kappa [S/m]')

fig, ax = plt.subplots(constrained_layout=True)
CS = ax.contourf(np.log10(sigma_array), np.log10(kappa_array), heterogeneity_factor_array);
ax.set_xlabel('Log Sigma [S/m]')
ax.set_ylabel('Log Kappa [S/m]')
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel('Relative Heterogeneity')

fig, ax = plt.subplots(constrained_layout=True)
CS = ax.contourf(np.log10(sigma_array), np.log10(kappa_array), relative_heterogeneity_array);
ax.set_xlabel('Log Sigma [S/m]')
ax.set_ylabel('Log Kappa [S/m]')
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel('Relative Heterogeneity')