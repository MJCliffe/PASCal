import scipy
import numpy as np

def Round(var, dec):
    # Rounding the number to desired decimal places
    # round() is more accurate at rounding float numbers than np.round()
    #NewVar = np.zeros((var.shape[0], var.shape[1]))
    if var.ndim == 1:
        for i in range(var.shape[0]):
            var[i] = round(var[i], dec)
    if var.ndim == 2:
        for i in range(var.shape[0]):
            for j in range(var.shape[1]):
                var[i][j] = round(var[i][j], dec)
    return var

def Orthomat(latt):
    # Compute the corresponding change-of-basis transformation (square matrix M) in E = M x A
    # lattice numpy array of lattice parameters, a b c alpha beta gamma (Angstrom, degrees)
    alpha = latt[3]*(np.pi/180)
    beta = latt[4]*(np.pi/180)
    gamma = latt[5]*(np.pi/180)
    gaS = np.arccos((np.cos(alpha)*np.cos(beta)-np.cos(gamma))/(np.sin(alpha)*np.sin(beta)))
    m11 = 1/(latt[0]*np.sin(beta)*np.sin(gaS))
    m21 = np.cos(gaS)/(latt[1]*np.sin(alpha)*np.sin(gaS))
    m22 = 1/(latt[1]*np.sin(alpha))
    m31 = (np.cos(alpha)*np.cos(gaS)/np.sin(alpha)+np.cos(beta)/np.sin(beta))/(-1*latt[2]*np.sin(gaS))
    m32 = -1*np.cos(alpha)/(latt[2]*np.sin(alpha))
    m33 = 1/latt[2]
    return np.array([[m11, 0, 0], [m21, m22, 0], [m31, m32, m33]])

def DiaLinearStrain(LattPam, FiniteStrain, EulerianStrain):
    # Diagonolise the strain
    # lattice numpy array of lattice parameters, a b c alpha beta gamma (Angstrom, degrees)
    # type of strains; FiniteStrain(True or False), EulerianStrain (True or False)
    OutStrain = np.zeros((3, LattPam.shape[0]))
    OutVectors = np.zeros((3, 3, LattPam.shape[0]))
    for i in range(1, LattPam.shape[0]):
        if EulerianStrain == False:
            e = np.matmul(np.linalg.inv(Orthomat(LattPam[i])),Orthomat(LattPam[0]))-np.identity(3)
        elif EulerianStrain == True:
            e = np.identity(3)-np.matmul(np.linalg.inv(Orthomat(LattPam[0])),Orthomat(LattPam[i]))
        if FiniteStrain == False: Strain = (e+np.transpose(e))/2
        elif FiniteStrain == True: Strain = (e+np.transpose(e)+np.matmul(e, np.transpose(e)))/2
        values, vector = scipy.linalg.eigh(Strain, lower=False, driver="ev")
        OutStrain[:,i] = values[:]
        OutVectors[:,:,i] = vector[:]
    return OutStrain, OutVectors

def PrePRAXnCRAX(MedLattPam, MedEigenVector):
    # function used to calculate both the pricipal axes and the crystallographic axes
    # lattice parameters a b c al be ga at the median point  (Angstrom, degrees)
    # eigen vector at the median point
    e = Orthomat(MedLattPam)
    ep = np.matmul(e, MedEigenVector)
    trans = np.transpose(ep)
    return trans

def ProjOfXnOnUnitCell(trans):
    # calculate the projections of the principal axes onto each of the crystallographic axes, i.e. giving the [UVW] direction of each of the principal axis.
    # variable trans calcaluted using the function PrePRAXnCRAX()
    prax = np.zeros((3, 3))
    for i in range(3):
        prax[i,:] = trans[i,:]/((np.sum(trans[i,:]**2))**0.5)
    return prax

def CellVol(LattPam):
    # Calculate the unit-cell volume
    # lattice parameters a b c al be ga at the median point  (Angstrom, degrees)
    vol = np.zeros((LattPam.shape[0]))
    for i in range(0, LattPam.shape[0]):
        latt = LattPam[i]
        vol[i] = latt[0]*latt[1]*latt[2]*((1-np.cos(latt[3]*(np.pi/180))**2)-(np.cos(latt[4]*(np.pi/180))**2)
        -(np.cos(latt[5]*(np.pi/180))**2)+(2*np.cos(latt[3]*(np.pi/180))*np.cos(latt[4]*(np.pi/180))
        *np.cos(latt[5]*(np.pi/180))))**(0.5)
    return vol

def linear_func(x,a,b): ####NOT USED
    return a*x+b

def linfit(strain, TP,TPErr): ####NOT USED
    #Execute linear fit of strain to obtain alpha (gradient), This is also coefficient of thermal expansion
    #strain: numpy array 3 x data points, strain
    #TP: temperature/pressure/electrochem x data points, numpy array
    #TPErr: std err in TP
    #Output:
    #Alpha:
    #...
    Alpha = np.zeros(3)
    Offset = np.zeros(3)
    AlphaErr = np.zeros(3)
    OffsetErr = np.zeros(3)
    for i in range(3):
        popt,pcov = scipy.optimize.curve_fit(linear_func,TP,strain[i],sigma=TPErr)
        Alpha[i] = popt[0]
        Offset[i] = popt[1]
        AlphaErr[i] = np.sqrt(pcov[0,0])
        OffsetErr[i] = np.sqrt(pcov[1,1])
    return Alpha, Offset, AlphaErr, OffsetErr

def StrainFit(x, y, yErr, n):
    #least square
    # Calculate the gradient, y intercept and the error from the linear fit
    # x: input temperature or pressure data (K, MPa)
    # y: calculated strain (decimal)
    # yErr: errors in input temperature data (K)
    Alpha = np.zeros(n)
    Offset = np.zeros(n)
    AlphaErr = np.zeros(n)
    #OffsetErr = np.zeros(3)
    for i in range(n):
        Del = sum(np.power(yErr,-2))*sum(np.power((x/yErr),2))-(np.power(sum(x/np.power(yErr,2)),2))
        Offset[i] = (sum(np.power((x/yErr),2))*sum(y[i]/np.power(yErr,2))-sum(x/np.power(yErr, 2))*sum(x*y[i]/np.power(yErr,2)))/Del
        Alpha[i]= (sum(x*y[i]/np.power(yErr,2))*sum(1/np.power(yErr,2))-sum(x/np.power(yErr,2))*sum(y[i]/np.power(yErr,2)))/Del
        U = y[i]-(Offset[i]+Alpha[i]*np.array(x))
        sig_b =(np.power(sum(x),2)*sum(np.power(U,2))-2*len(x)*sum(x)*sum(np.power(U,2)*x) + np.power(len(x),2)*sum(np.power(U,2)*np.power(x,2)))/np.power((len(x)*sum(np.power(x,2))-np.power(sum(x),2)),2)
        AlphaErr[i] = np.sqrt(sig_b)
    return Alpha, Offset, AlphaErr

def EmpEq(TP, Epsilon0, Pc, lambdaP, Nu):
    # Empirical fit for pressure input data
    # TP: pressure data points, numpy array
    # Epsilon0: calculated strain (decimal)
    # lambdaP: gradient from the linear fit
    # Pc: critical pressure (GPa)
    # Nu: 0.5
    return Epsilon0+(lambdaP*((TP-Pc)**Nu))

def Comp(lambdaP, Nu, TP, Pc):
    # Calculate the compressibility from the derivative -de/dp
    # TP: pressure data points, numpy array
    # lambdaP: gradient from the linear fit
    # Pc: critical pressure (GPa)
    # Nu: 0.5
    return -lambdaP*Nu*((TP-Pc)**(Nu-1))

def CompErr(Pcov, Pc, Nu, lambdaP, TP): ### need to check again with FORTRAN
    # Calculate errors in compressibilities
    # TP: pressure data points, numpy array
    # lambdaP: gradient from the linear fit
    # Pc: critical pressure (GPa)
    # Nu: 0.5
    # Pcov: the estimated covariance of optimal values of the empirical parameters
    Jac = np.zeros((4, TP.shape[0])) #jacobian matrix
    KErr = np.zeros(TP.shape[0])
    for i in range(0, len(TP)):
        Jac[0][i] = 0
        Jac[1][i] = ((TP[i]-Pc)**(Nu-1))*Nu
        Jac[2][i] = -lambdaP*Nu*(Nu-1)*(TP[i]-Pc)**(Nu-2)
        Jac[3][i] = Nu*np.log(TP[i]-Pc+1)*(TP[i]-Pc)**(Nu-1)*lambdaP
        KErrPoint = 0
        for j in range(0, 4):
            for n in range(0, 4):
                KErrPoint = KErrPoint+Jac[j][i]*Jac[n][i]*Pcov[j][n]
        KErr[i] = KErrPoint**0.5
    return KErr

def Eta(V, V0):
    # Defining the parameter to be used in Birch-Murnaghan equations of state
    # V: unit-cell volume at a pressure point
    # V0: the zero pressure unit-cell volume
    return np.abs(V0/V)**(1/3)

def SecBM(V, V0, B):
    # The second-order Birch-Murnaghan fit corresponding the equation of state
    # V: unit-cell volume at a pressure point (Angstrom^3)
    # V0: the zero pressure unit-cell volume (Angstrom^3)
    # B: Bulk modulus (GPa)
    return (3/2)*B*(Eta(V, V0)**7-Eta(V, V0)**5)

def ThirdBM(V, V0, B0, Bprime):
    # The third-order Birch-Murnaghan fit corresponding the equation of state
    # V: unit-cell volume at a pressure point (Angstrom^3)
    # V0: the zero pressure unit-cell volume (Angstrom^3)
    # B0: Bulk modulus at zero pressure (GPa)
    # Bprime: pressure derivative of the bulk modulus (dimensionless)
    return 3/2*B0*(Eta(V, V0)**7-Eta(V, V0)**5)*(1+3/4*(Bprime-4)*(Eta(V, V0)**2-1))

def ThirdBMPc(V, V0, B0, Bprime, Pc):
    # The third-order Birch-Murnaghan fit corresponding the equation of state
    # V: unit-cell volume at a pressure point (Angstrom^3)
    # V0: the zero pressure unit-cell volume (Angstrom^3)
    # B0: Bulk modulus at zero pressure (GPa)
    # Bprime: pressure derivative of the bulk modulus (dimensionless)
    # Pc: critical pressure (GPa)
    return (Eta(V, V0)**5)*(Pc-1/2*((3*B0)-(5*Pc))*(1-(Eta(V, V0)**2))+(9/8)*B0*
    (Bprime-4+(35*Pc)/(9*B0))*(1-(Eta(V, V0)**2))**2)

def WrapperThirdBMPc(InpPc):
    # Allows ThirdBMPc to be fitted using curve_fit() with InpPc as a constant
    # InpPc: input critical pressure (GPa)
    def TempFunc(V, V0, B0, Bprime, Pc=InpPc):
        return ThirdBMPc(V, V0, B0, Bprime, Pc)
    return TempFunc


def CRAX(trans):
    # Compute crystallogrphic axes in crystallographic coordinate
    # variable trans calcaluted from the function PrePRAXnCRAX()
    crax = np.linalg.inv(trans)
    return crax

def NormCRAX(CalCrax, PrinComp):
    # Normalise the crystallogrphic axes for the indicatrix plot
    # CalCrax: calculated crystallogrphic axes
    # PrinComp: 3 principal components which are
    # coefficient of thermal expansion (MK^-1) or
    # median compressibilities (TPa^-1)
    NormCrax = np.zeros((3, 3))
    maxalpha = np.abs(max(PrinComp[0], PrinComp[1], PrinComp[2]))
    lens = np.zeros(3)
    for i in range(0,3):
        lenIn = 0
        for j in range(0,3):
            lenIn += CalCrax[i][j]**2
        lens[i] = lenIn**0.5
    maxlen = max(lens)

    for i in range(0,3): #normalise the axes
        NormCrax[i] = CalCrax[i]*maxalpha/maxlen
    return NormCrax

def Indicatrix(PrinComp):
    # Indicatrix plot for coefficient of thermal expansion (MK^-1) or median compressibilities (TPa^-1)
    # PrinComp: 3 principal components which are
    # coefficient of thermal expansion (MK^-1) or
    # median compressibilities (TPa^-1)
    theta, phi = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
    THETA, PHI = np.meshgrid(theta, phi)
    maxIn = np.amax(np.abs(PrinComp))
    R = PrinComp[0]*(np.sin(THETA)*np.cos(PHI))**2+ PrinComp[1]*(np.sin(THETA)*np.sin(PHI))**2+PrinComp[2]*(np.cos(THETA)**2)
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)
    return maxIn, R, X, Y, Z
