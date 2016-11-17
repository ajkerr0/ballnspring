"""


@author: Alex Kerr
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse.linalg as slinalg

def kappa(m, k, drivers, crossings, gamma=10., pfunc="vector", sparse=False):
    """Return the thermal conductivity of the mass system
    
    Arguments:
        m (array-like): 1D array of the masses of the system.
        k (array-like): 2D, symmetric [square] array of the spring constants of the system.
            Also known as the Hessian.  Indexed like m.  The dimensions of this array relative to m
            determines the number of degrees of freedom for the masses.
        drivers (array-like): 2D array of atomic indices driven, corresponding to 2 separate interfaces.
    Keywords:
        gamma (float): Drag coefficient in the calculation, applied to every driver uniformly.
        pfunc (str): Name of function that returns the driving power for each crossing interaction.
            'vector' uses a numpy implementation of the double sum.  'loop' uses brute-force for-loops
            from default python.  'record' is used for tracking values, for debugging purposes.  
            Default is 'vector'."""
    
    dim = k.shape[0]//m.shape[0]
    
    #standardize the driverList
    drivers = np.array(drivers)
    
    g = calculate_gamma_mat(dim, m.shape[0], gamma, drivers)
    
    m = np.diag(np.repeat(m,dim))
    
    val, vec = calculate_thermal_evec(k, g, m, sparse=sparse)
    
    coeff = calculate_coeff(val, vec, np.diag(m), np.diag(g), sparse=sparse)
         
    #initialize the thermal conductivity value
    kappa = 0.
    
    if pfunc.lower() == 'list':
        
        table = []
        
        for crossing in crossings:
            i,j = crossing
            kappa += calculate_power_list(i, j, dim, val, vec, coeff, k, drivers, table)
            
        return kappa, table
        
    else:
    
        if pfunc.lower() == 'vector':
            calculate_power = calculate_power_vector
        elif pfunc.lower() == 'loop':
            calculate_power = calculate_power_loop
                    
        for crossing in crossings:
            i,j = crossing
            kappa += calculate_power(i,j,dim, val, vec, coeff, k, drivers)
    
        return kappa
    
def calculate_power_loop(i,j, dim,val, vec, coeff, kMatrix, driverList):
    
    driver1 = driverList[1]    
    
    n = val.shape[0]//2
    
    kappa = 0.
    
    for idim in range(dim):
        for jdim in range(dim):
            for driver in driver1:
                term = 0.
                for sigma in range(2*n):
                    cosigma = 0.
                    for k in dim:
                        cosigma += coeff[sigma, dim*driver + k]
                    for tau in range(2*n):
                        cotau = 0.
                        for k in dim:
                            cotau += coeff[tau, dim*driver + k]
                        try:
                            term += kMatrix[dim*i + idim, dim*j + jdim]*(cosigma*cotau*(vec[:n,:][dim*i + idim ,sigma])*(
                                    vec[:n,:][dim*j + jdim,tau])*((val[sigma]-val[tau])/(val[sigma]+val[tau])))
                        except FloatingPointError:
                            print("Divergent term")
                kappa += term
            
    return kappa
    
def calculate_power_vector(i,j, dim,val, vec, coeff, kMatrix, driverList):
    
    #assuming same drag constant as other driven atom
    driver1 = driverList[1]
    
    n = val.shape[0]
    
    kappa = 0.
    
    val_sigma = np.tile(val, (n,1))
    val_tau = np.transpose(val_sigma)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        valterm = np.true_divide(val_sigma-val_tau,val_sigma+val_tau)
    valterm[~np.isfinite(valterm)] = 0.
    
    for idim in range(dim):
        for jdim in range(dim):
            
            term3 = np.tile(vec[dim*i + idim,:], (n,1))
            term4 = np.transpose(np.tile(vec[dim*j + jdim,:], (n,1)))
            
            for driver in driver1:
                
                dterm = np.zeros((coeff.shape[0],), dtype=np.complex128)
                for k in range(dim):
                    dterm += coeff[:, dim*driver + k]
    
                term1 = np.tile(dterm, (n,1))
                term2 = np.transpose(term1)
                termArr = term1*term2*term3*term4*valterm
                kappa += kMatrix[dim*i + idim, dim*j + jdim]*np.sum(termArr)
                
    return kappa
    
def calculate_power_list(i, j, dim, val, vec, coeff, kMatrix, driverList, table):
    
    #assuming same drag constant as other driven atom
    driver1 = driverList[1]
    
    n = val.shape[0]
    
    kappa = 0.
    
    val_sigma = np.tile(val, (n,1))
    val_tau = np.transpose(val_sigma)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        valterm = np.true_divide(val_sigma-val_tau,val_sigma+val_tau)
    valterm[~np.isfinite(valterm)] = 0.
    
    for idim in range(dim):
        for jdim in range(dim):
            
            term3 = np.tile(vec[dim*i + idim,:], (n,1))
            term4 = np.transpose(np.tile(vec[dim*j + jdim,:], (n,1)))
            
            for driver in driver1:
                
                dterm = np.zeros((coeff.shape[0],), dtype=np.complex128)
                for k in range(dim):
                    dterm += coeff[:, dim*driver + k]
    
                term1 = np.tile(dterm, (n,1))
                term2 = np.transpose(term1)
                termArr = kMatrix[dim*i + idim, dim*j + jdim]*term1*term2*term3*term4*valterm
                
                #add whole arrays to table
#                table.append(termArr)
                #add indices of top m values
                m = 3
                max_indices = np.argpartition(termArr.flatten(), -m)[-m:]
                max_indices = np.vstack(np.unravel_index(max_indices, termArr.shape)).T
                for sigma, tau in max_indices:
                    table.append([sigma, tau, 
                                  termArr[sigma, tau], kMatrix[dim*i + idim, dim*j + jdim],
                                  term1[sigma, tau], term2[sigma, tau], term3[sigma, tau], term4[sigma, tau],
                                  dim*i + idim, dim*j + jdim])
                
                kappa += np.sum(termArr)
                
    return kappa
    
def calculate_coeff(val, vec, mass, gamma, sparse=False):
    """Return the M x N Green's function coefficient matrix;
    N is the number of coordinates in the problem, M is the number of
    eigenmodes."""
    
    N = vec.shape[0]//2
    M = vec.shape[1]
    
    #need to determine coefficients in eigenfunction/vector expansion
    # need linear solver to solve equations from notes
    # AX = B where X is the matrix of expansion coefficients
    
    A = np.zeros((2*N, M), dtype=np.complex128)
    A[:N,:] = vec[:N,:]
    
    #adding mass and damping terms to A
    lambda_ = np.tile(val, (N,1))
    
    A[N:,:] = np.multiply(A[:N,:], np.tile(mass, (M,1)).T*lambda_ + np.tile(gamma, (M,1)).T)
    
    #now prep B
    B = np.concatenate((np.zeros((N,N)), np.eye(N)), axis=0)

    if sparse:
        return np.linalg.lstsq(A,B)[0]
    else:
        return np.linalg.solve(A,B)
    
def calculate_thermal_evec(K, G, M, sparse=False):
    
    N = M.shape[0]
    
    a = np.zeros((N,N))
    a = np.concatenate((a,np.eye(N)),axis=1)
    b = np.concatenate((K,G),axis=1)
    c = np.concatenate((a,b),axis=0)
    
    x = np.eye(N)
    x = np.concatenate((x,np.zeros((N,N))),axis=1)
    y = np.concatenate((np.zeros((N,N)),-M),axis=1)
    z = np.concatenate((x,y),axis=0)
    
    if sparse:
        return slinalg.eigs(c, M=z, k=200,) #which='SM')
    else:
        return linalg.eig(c,b=z,right=True)
    
def calculate_gamma_mat(dim, N, gamma, drivers):
    
    gmat = np.zeros((dim*N, dim*N))
    drivers = np.hstack(drivers)
    
    for driver in drivers:
        for i in range(dim):
            gmat[dim*driver + i, dim*driver + i] = gamma
        
    return gmat