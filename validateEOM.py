import os
internalCode = """
import numpy as np

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
 
    spinBlock = np.kron(np.eye(2), np.kron(np.eye(2), g).T)
    g = spinBlock.transpose(0,2,1,3) - spinBlock.transpose(0,2,3,1)

    #prepare orbital energies
    eps = np.concatenate((e,e), axis=0)
    C = np.block([
                 [c, np.zeros_like(c)],
                 [np.zeros_like(c), c]])
    C =C[:, eps.argsort()]
    eps = np.sort(eps)

    #eri to MO
    g = np.einsum('pQRS,pP->PQRS', np.einsum('pqRS,qQ->pQRS', np.einsum('pqrS,rR->pqRS', np.einsum('pqrs,sS->pqrS', \
    g, C, optimize=True), C, optimize=True), C, optimize=True), C, optimize=True)

    return g

iterations = 50
tolerance = 1e-10

data = np.load(mints, allow_pickle=True)
fock = data['f']
c = data['c']
eri = data['i']
e = data['e']

charge, mole, base, nuclearRepulsion = data['m']

#orbital occupations
nso = (fock.shape[0]) * 2
nsocc = sum(data['a'])
nsvir = nso - nsocc

#get fock in MO spin basis
cSpin = np.kron(c, np.eye(2))
fock = np.dot(cSpin.T, np.dot(np.kron(fock, np.eye(2)), cSpin))

#get two-electron repulsion integrals in MO basis
g = gMOspin(e, c, eri, nso//2)

#slices
n = np.newaxis
o = slice(None,nsocc)
v = slice(nsocc, None)

#D tensors
eps = np.kron(e, np.ones(2))

d_ai = 1.0 / (-eps[v, n] + eps[n, o])
d_abij = 1.0 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])
d_abcijk = 1.0 / (- eps[ v, n, n, n, n, n] - eps[n, v, n, n, n, n] - eps[n, n, v, n, n, n]
                   + eps[n, n, n, o, n, n] + eps[n, n, n, n, o, n] + eps[n, n, n, n, n, o] )

#HF energy
HFenergy = 1.0 * np.einsum('ii', fock[o, o]) -0.5 * np.einsum('ijij', g[o, o, o, o])

#amplitude initialisation
ts = np.zeros((nsvir, nsocc))
td = np.zeros((nsvir,nsvir,nsocc,nsocc))
tt = np.zeros((nsvir,nsvir,nsvir,nsocc,nsocc,nsocc))

#reciprocal D tensors
fock_d_ai = np.reciprocal(d_ai)
fock_d_abij = np.reciprocal(d_abij)
fock_d_abcijk = np.reciprocal(d_abcijk)

#get initial cluster energy
lastCycleEnergy =  cc_energy(fock, g, o, v, t1=ts, t2=td, t3=tt )

#dummies

triples = None

#iterations
for cycle in range(iterations):

    #update amplitudes
    singles = cc_singles(fock, g, o, v, t1=ts, t2=td, t3=tt) * d_ai + ts 
    doubles = cc_doubles(fock, g, o, v, t1=ts, t2=td, t3=tt) * d_abij + td

    #recalculate energy
    cycleEnergy = cc_energy(fock, g, o, v, t1=singles, t2=doubles, t3=triples )
    deltaEnergy = np.abs(lastCycleEnergy - cycleEnergy)

    #convergence test
    if deltaEnergy < tolerance:

        cycleEnergy =  cc_energy(fock, g, o, v, t1=singles, t2=doubles, t3=triples ) - HFenergy
        break
    else:
        ts = singles
        td = doubles
        lastCycleEnergy = cycleEnergy
else:
    print("Did not converge")
    exit('cc failed')

#we're using nso arrays in direct code
t1 = np.zeros((nso, nso))
t2 = np.zeros((nso, nso, nso, nso))
t1[v, o], t2[v,v,o,o] = [ts, td]

hss, hsd, hds, hdd = ee_r_eom(fock, g, [nsvir, nsocc], t1, t2)
eomMatrix = np.bmat([[hss,hsd],[hds,hdd]])

from scipy.linalg import eigvals

#post-process eigenvalues
eomEVal = eigvals(eomMatrix).real
eomEVal = np.sort(eomEVal)
eomEVal = eomEVal[eomEVal > 0.3]
eomEVal = np.around(eomEVal, 6)

uniqueValues, degeneracy = np.unique(eomEVal, return_counts=True)

#get singlets and triplets
singlets = []
triplets = []
for i, e in enumerate(uniqueValues):
    if degeneracy[i] == 1: singlets.append(e)
    if degeneracy[i] == 3: triplets.append(e)

print('Singlets->',singlets)
print('Triplets->',triplets)

#reference pyscf
if (molecule == 'h2') and (basis == '3-21g'):
    print(np.isclose(singlets, [0.58425642,1.12164943,1.17162132,1.51189885,1.87522426,2.18653002,2.22648573,2.85809643,3.29778787], 1e-5))
    print(np.isclose(triplets, [0.39882804,0.97280038,1.48474618,1.58875293,2.02686245,2.56343784], 1e-5))


"""
import sys

nargs = len(sys.argv)
if nargs == 1:
    mints = 'mints/h2-3-21g-mints.npz'
    method, molecule, basis = ['ee-r-eom', 'h2', '3-21g']
    files = ['codes/ccsd.py', 'codes/ee_r_eom.py']
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


