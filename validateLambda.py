from __future__ import division
internalCode = """
#declarations
import numpy as np

def spatialTospin(eriMO, nbf):
    #openfermion style transformation

    import numpy as np

    spin = np.zeros((2*nbf, 2*nbf, 2*nbf, 2*nbf))

    eriMO = eriMO.transpose(0,2,3,1)
    for p in range(nbf):
        for q in range(nbf):
            for r in range(nbf):
                for s in range(nbf):
                    #anti-spin
                    spin[2*p, 2*q+1, 2*r+1, 2*s], spin[2*p+1, 2*q, 2*r, 2*s+1] = [eriMO[p,q,r,s]] * 2
                    #syn-spin
                    spin[2*p, 2*q, 2*r, 2*s], spin[2*p+1, 2*q+1, 2*r+1, 2*s+1] = [eriMO[p,q,r,s]] * 2

    return spin

def gMOspin(e, c, eri, nbf):
    #construct MO spin eri

    import numpy as np

    def iEri(i,j,k,l):
        #index into the four-index eri integrals
        p = max(i*(i+1)/2 + j, j*(j+1)/2 + i)
        q = max(k*(k+1)/2 + l, l*(l+1)/2 + k)
        return  int(max(p*(p+1)/2 + q, q*(q+1)/2 + p))

    #get 4 index eri and spinblock to spin basis
    g = np.zeros((nbf,nbf,nbf,nbf))
    for i in range(nbf):
        for j in range(nbf):
            for k in range(nbf):
                for l in range(nbf):
                    g[i,j,k,l] = eri[iEri(i,j,k,l)]
 
    #eri to MO
    g = np.einsum('pQRS,pP->PQRS', np.einsum('pqRS,qQ->pQRS', np.einsum('pqrS,rR->pqRS', np.einsum('pqrs,sS->pqrS', \
    g, c, optimize=True), c, optimize=True), c, optimize=True), c, optimize=True)

    return g, np.kron(e, np.ones(2))



iterations = 50
tolerance = 1e-10

data = np.load(mints, allow_pickle=True)
fock = data['f']
c = data['c']
eri = data['i']
e = data['e']

charge, mole, base, nuclearRepulsion = data['m']

#orbital occupations
spinOrbitals = (fock.shape[0]) * 2
nsocc = sum(data['a'])
nsvir = spinOrbitals - nsocc

#check right molecule and basis
if (molecule != mole) or (basis != base):
    exit('wrong molecule or basis in harpy mints file')

#get one electron operators
h1 = np.dot(c.T, np.dot(data['j']+data['k'], c))
hcSpin = np.kron(h1, np.eye(2))
nbf = h1.shape[0]

#get fock in MO spin basis
cSpin = np.kron(c, np.eye(2))
fock = np.dot(cSpin.T, np.dot(np.kron(fock, np.eye(2)), cSpin))

#get two-electron repulsion integrals in MO basis
eriMO, eps = gMOspin(e, c, eri, spinOrbitals//2)
eriMOspin = spatialTospin(eriMO, spinOrbitals//2)
g = (np.einsum('ijkl', eriMOspin) - np.einsum('ijlk', eriMOspin)).transpose(0, 1, 3, 2)

#slices
n = np.newaxis
o = slice(None,nsocc)
v = slice(nsocc, None)

#D tensors
d_ai = 1.0 / (-eps[v, n] + eps[n, o])
d_abij = 1.0 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])
d_abcijk = 1.0 / (- eps[ v, n, n, n, n, n] - eps[n, v, n, n, n, n] - eps[n, n, v, n, n, n]
                   + eps[n, n, n, o, n, n] + eps[n, n, n, n, o, n] + eps[n, n, n, n, n, o] )

#HF energy
HFenergy = 1.0 * np.einsum('ii', fock[o, o]) -0.5 * np.einsum('ijij', g[o, o, o, o])
print('Hartree-Fock electronic energy ',HFenergy, '   Total energy ', HFenergy + float(nuclearRepulsion))

#amplitude initialisation
ts = np.zeros((nsvir, nsocc))
td = np.zeros((nsvir,nsvir,nsocc,nsocc))

#get initial cluster energy
lastCycleEnergy =  cc_energy(fock, g, o, v, t1=ts, t2=td, t3=None )

#dummies
triples = None

#iterations
print('\t  Cycle        Energy                 \u0394-Energy     ')
print('       ' + '='*52)

for cycle in range(iterations):

    #update amplitudes
    singles = cc_singles(fock, g, o, v, t1=ts, t2=td, t3=None) * d_ai + ts 
    doubles = cc_doubles(fock, g, o, v, t1=ts, t2=td, t3=None) * d_abij + td

    #recalculate energy
    cycleEnergy = cc_energy(fock, g, o, v, t1=singles, t2=doubles, t3=None)
    deltaEnergy = np.abs(lastCycleEnergy - cycleEnergy)

    #convergence test
    if deltaEnergy < tolerance:

        cycleEnergy =  cc_energy(fock, g, o, v, t1=singles, t2=doubles, t3=None) - HFenergy
        print('Final energy correction ', cycleEnergy)
        ts = singles
        td = doubles
        break
    else:
        ts = singles
        td = doubles
        lastCycleEnergy = cycleEnergy
        print("\t{: 5d}\t{: 5.15f}\t{: 5.15f}".format(cycle, lastCycleEnergy, deltaEnergy))
else:
    print("Did not converge")
    exit('cc failed')

#for lagrange amplitudes l are transposes of t
d_ia   = d_ai.transpose(1,0)
d_ijab = d_abij.transpose(2,3,0,1)

#initial values for lagrange amplitudes
ls = ts.transpose(1,0)
ld = td.transpose(2,3,0,1)

lastCycleEnergy = cc_lambda_lagrangian_energy(fock, g, o, v, t1=ts, t2=td, l1=ls, l2=ld)

#iterations
print('')
print('\t  Cycle        Energy               \u0394-Amplitudes           Pseudo-Energy       ')
print('       ' + '='*78)

for cycle in range(iterations):

    singlesResidual = cc_lambda_singles(fock, g, o, v, ts, td, ls, ld)
    doublesResidual = cc_lambda_doubles(fock, g, o, v, ts, td, ls, ld)

    lambdaResidual = np.linalg.norm(singlesResidual) + np.linalg.norm(doublesResidual)

    singles = singlesResidual * d_ia + ls
    doubles = doublesResidual * d_ijab + ld

    lambdaCycleEnergy = cc_lambda_lagrangian_energy(fock, g, o, v, ts, td, singles, doubles)
    pseudoEnergy = 0.25 * np.einsum('jiab,jiab', g[o, o, v, v], ld)

    energyDelta = np.abs(lastCycleEnergy - lambdaCycleEnergy)

    if energyDelta < 1e-10 and lambdaResidual < 1e-10:
        ls = singles
        ld = doubles
        lambdaCycleEnergy =  cc_lambda_lagrangian_energy(fock, g, o, v, ts, td, ls, ld) - HFenergy
        print('Final \u039B-energy correction ', lambdaCycleEnergy)
        print('Final \u039B-pseudo energy     ', pseudoEnergy)
        break
    else:
        ls = singles
        ld = doubles
        lastCycleEnergy = lambdaCycleEnergy
        print("\t{: 5d}\t{: 5.15f}\t{: 5.15f}\t{: 5.15f}".format(
            cycle, lastCycleEnergy,
            np.linalg.norm(cc_lambda_singles(fock, g, o, v, ts, td, ls, ld)) +
            np.linalg.norm(cc_lambda_doubles(fock, g, o, v, ts, td, ls, ld)),
            pseudoEnergy
        ))
else:
    print("Did not converge")
    exit('lambda failed')

#response density matrices
opdm = cc_oprdm(o, v, ts, td, ls, ld)
tpdm = cc_tprdm(o, v, ts, td, ls, ld)

if mole == 'h2o':
    rdmEnergy = np.einsum('ij,ij', hcSpin, opdm) + 0.25 * np.einsum('ijlk,ijlk',tpdm, g)- HFenergy
    if base == 'sto-3g': 
        print(np.isclose(-0.07068008709615015, cycleEnergy, 1e-8))
        print(np.isclose(-0.07068008881678622, lambdaCycleEnergy, 1e-8))
        print(np.isclose(-0.068888211452107, pseudoEnergy, 1e-8))
        print(np.isclose(np.trace(opdm), nsocc, 1e-8))
        print(np.isclose(-0.07068008881304877, rdmEnergy, 1e-8))
    if base == '6-31g': 
        print(np.isclose(-0.1494126881297717, cycleEnergy, 1e-8))
        print(np.isclose(-0.1494126821783368, lambdaCycleEnergy, 1e-8))
        print(np.isclose(-0.148311220517852, pseudoEnergy, 1e-8))
        print(np.isclose(np.trace(opdm), nsocc, 1e-8))
        print(np.isclose(-0.14941268217515358, rdmEnergy, 1e-8))

"""
#parse arguments
import sys

nargs = len(sys.argv)
if nargs == 1:
    mints = 'mints/h2o-sto-3g-mints.npz'
    molecule, basis = ['h2o', 'sto-3g']
    files = ['codes/ccsd.py', 'codes/ccsd_lambda.py', 'codes/cc_rdm.py']
elif nargs == 3:
    molecule, basis = sys.argv[1:3] 
    mints = 'mints/' + molecule + '-' + basis + '-mints.npz'
    files = ['codes/ccsd.py', 'codes/ccsd_lambda.py', 'codes/cc_rdm.py']
else:
    exit('command line error')

externalCode = """
global cc_energy, cc_singles, cc_doubles
"""

for f in files:
	input = open(f, 'r')
	externalCode += input.read()

data = {'mints' : mints, 'molecule' : molecule, 'basis' : basis}

exec(externalCode + internalCode,{}, data)
