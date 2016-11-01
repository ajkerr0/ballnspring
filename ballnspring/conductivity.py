"""


@author: Alex Kerr
"""

import numpy as np
import scipy.linalg as linalg

def kappa(m, k, drivers, crossings):
    """Return the thermal conductivity of the mass system
    
    Arguments:
        m (array-like): 1D array of the masses of the system.
        k (array-like): 2D, symmetric [square] array of the spring constants of the system.
            Also known as the Hessian.  Indexed like m.  The dimensions of this array relative to m
            determines the number of degrees of freedom for the masses.
        drivers (array-like): 2D array of atomic indices driven, corresponding to 2 separate interfaces."""
    
    #give each driver the same drag constant (Calculation gamma)
    gamma = 10.
    
    dim = len(k)//len(m)
    
    #standardize the driverList
    drivers = np.array(drivers)
    
    g = _calculate_gamma_mat(dim, len(m), gamma, drivers)
    
    m = np.diag(np.repeat(m,dim))
    
    val, vec = _calculate_thermal_evec(k, g, m)
    
    coeff = _calculate_coeff(val, vec, m, g)
         
    #initialize the thermal conductivity value
    kappa = 0.
    
#    mullenTable = []
    mullenTable = None
                
    for crossing in crossings:
        i,j = crossing
        kappa += _calculate_power(i,j,dim, val, vec, coeff, k, drivers, mullenTable)
#        kappa += _calculate_power_loop(i,j,val, vec, coeff, kMatrix, driverList, mullenTable)
    
#    inspect(mol, val, vec, k, crossings, mullenTable)

#    import pprint
#    pprint.pprint(mullenTable)
    print(kappa)
    return kappa
#    return kappa, mullenTable
    
def _calculate_power_loop(i,j, dim,val, vec, coeff, kMatrix, driverList, mullenTable):
    
    driver1 = driverList[1]    
    
    n = len(val)//2
    
    kappa = 0.
    
    for idim in range(dim):
        for jdim in range(dim):
            for driver in driver1:
                term = 0.
                for sigma in range(2*n):
                    cosigma = coeff[sigma, 3*driver + 1] + coeff[sigma, 3*driver +2] + coeff[sigma, 3*driver]
                    for tau in range(2*n):
                        cotau = coeff[tau, 3*driver] + coeff[tau, 3*driver+1] + coeff[tau, 3*driver+2]
                        try:
                            test= kMatrix[3*i + idim, 3*j + jdim]*(cosigma*cotau*(vec[:n,:][3*i + idim ,sigma])*(
                                    vec[:n,:][3*j + jdim,tau])*((val[sigma]-val[tau])/(val[sigma]+val[tau])))
                            if mullenTable is not None:
                                mullenTable.append(test)
                            term += test
                        except FloatingPointError:
                            print("Divergent term")
#                term *= kMatrix[3*i + idim, 3*j + jdim]
                kappa += term
            
    return kappa
    
def _calculate_power(i,j, dim,val, vec, coeff, kMatrix, driverList, mullenTable):
    
    #assuming same drag constant as other driven atom
    driver1 = driverList[1]
    
    n = len(val)
    
    kappa = 0.
    
    val_sigma = np.tile(val, (n,1))
    val_tau = np.transpose(val_sigma)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        valterm = np.true_divide(val_sigma-val_tau,val_sigma+val_tau)
    valterm[~np.isfinite(valterm)] = 0.
    
    for idim in range(dim):
        for jdim in range(dim):
            
            term3 = np.tile(vec[3*i + idim,:], (n,1))
            term4 = np.transpose(np.tile(vec[3*j + jdim,:], (n,1)))
            
            for driver in driver1:
    
                term1 = np.tile(coeff[:, 3*driver] + coeff[:, 3*driver+1] + coeff[:, 3*driver+2], (n,1))
                term2 = np.transpose(term1)
                
                termArr = kMatrix[3*i + idim, 3*j + jdim]*term1*term2*term3*term4*valterm
                if mullenTable is not None:
#                    mullenTable.append(kMatrix[3*i + idim, 3*j +jdim])
                #####
#                    mullenTable.append(termArr)
                ########
                    large_vals = np.where(np.absolute(termArr) > 250.)
##                    print(large_vals)
                    for x,y in zip(large_vals[0], large_vals[1]):
                        mullenTable.append([termArr[x, y], x, y, i, j])
#                        mullenTable.append([term1[x,y], term2[x,y], term3[x,y], term4[x,y], val_sigma[x,y], val_tau[x,y], valterm[x,y]])
#                        mullenTable.append(x)
#                term = kMatrix[3*i + idim, 3*j + jdim]*np.sum(term1*term2*term3*term4*valterm)
#                kappa += term
                kappa += np.sum(termArr)
                
    return kappa
    
def _calculate_coeff(val, vec, massMat, gMat):
    """Return the 2N x N Green's function coefficient matrix."""
    
    N = len(vec)//2
    
    #need to determine coefficients in eigenfunction/vector expansion
    # need linear solver to solve equations from notes
    # AX = B where X is the matrix of expansion coefficients
    
    A = np.zeros((2*N, 2*N), dtype=complex)
    A[:N,:] = vec[:N,:]

    #adding mass and damping terms to A
    lamda = np.tile(val, (N,1))

    A[N:,:] = np.multiply(A[:N,:], np.dot(massMat,lamda) + np.dot(gMat,np.ones((N,2*N))))
    
    #now prep B
    B = np.concatenate((np.zeros((N,N)), np.identity(N)), axis=0)

    return np.linalg.solve(A,B)
    
def _calculate_thermal_evec(K,G,M):
    
    N = len(M)
    
    a = np.zeros([N,N])
    a = np.concatenate((a,np.identity(N)),axis=1)
    b = np.concatenate((K,G),axis=1)
    c = np.concatenate((a,b),axis=0)
    
    x = np.identity(N)
    x = np.concatenate((x,np.zeros([N,N])),axis=1)
    y = np.concatenate((np.zeros([N,N]),-M),axis=1)
    z = np.concatenate((x,y),axis=0)
    
    w,vr = linalg.eig(c,b=z,right=True)
    
    return w,vr
    
def _calculate_gamma_mat(N,gamma, driverList):
    
    gmat = np.zeros((3*N, 3*N))
    driveList = np.hstack(driverList)
    
    for drive_atom in driveList:
        gmat[3*drive_atom  , 3*drive_atom  ] = gamma
        gmat[3*drive_atom+1, 3*drive_atom+1] = gamma
        gmat[3*drive_atom+2, 3*drive_atom+2] = gamma
        
    return gmat
    
def _calculate_ballandspring_k_mat(N,k0,nLists):
    """Return the Hessian of a linear chain of atoms assuming only nearest neighbor interactions."""
    
    KMatrix = np.zeros([3*N,3*N])
    
    for i,nList in enumerate(nLists):
        KMatrix[3*i  ,3*i  ] = k0*len(nList)
        KMatrix[3*i+1,3*i+1] = k0*len(nList)
        KMatrix[3*i+2,3*i+2] = k0*len(nList)
        for neighbor in nList:
            KMatrix[3*i  ,3*neighbor] = -k0
            KMatrix[3*i+1,3*neighbor+1] = -k0
            KMatrix[3*i+2,3*neighbor+2] = -k0
    
    return KMatrix