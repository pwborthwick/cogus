from __future__ import division
internalCode = """
#declarations
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
spinOrbitals = (fock.shape[0]) * 2
nsocc = sum(data['a'])
nsvir = spinOrbitals - nsocc

#check right molecule and basis
if (molecule != mole) or (basis != base):
    exit('wrong molecule or basis in harpy mints file')

#get fock in MO spin basis
cSpin = np.kron(c, np.eye(2))
fock = np.dot(cSpin.T, np.dot(np.kron(fock, np.eye(2)), cSpin))

#get two-electron repulsion integrals in MO basis
g = gMOspin(e, c, eri, spinOrbitals//2)

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
print('Hartree-Fock electronic energy ',HFenergy, '   Total energy ', HFenergy + float(nuclearRepulsion))

#amplitude initialisation
ts = np.zeros((nsvir, nsocc))
td = np.zeros((nsvir,nsvir,nsocc,nsocc))
tt = np.zeros((nsvir,nsvir,nsvir,nsocc,nsocc,nsocc))

#get initial cluster energy
lastCycleEnergy =  cc_energy(fock, g, o, v, t1=ts, t2=td, t3=tt )

#dummies
if method in ['ccd', 'lccd']: singles = None
if not method in ['ccsdt', 'ccsd_t', 'cc3']: triples = None

#iterations
for cycle in range(iterations):

    #update amplitudes
    if not method in ['ccd' , 'lccd'] : singles = cc_singles(fock, g, o, v, t1=ts, t2=td, t3=tt) * d_ai + ts 
    doubles = cc_doubles(fock, g, o, v, t1=ts, t2=td, t3=tt) * d_abij + td
    if method in ['ccsdt','ccsd_t','cc3']: triples = cc_triples(fock, g, o, v, t1=ts, t2=td, t3=tt) * d_abcijk + tt

    #recalculate energy
    cycleEnergy = cc_energy(fock, g, o, v, t1=singles, t2=doubles, t3=triples )
    deltaEnergy = np.abs(lastCycleEnergy - cycleEnergy)

    #convergence test
    if deltaEnergy < tolerance:

        cycleEnergy =  cc_energy(fock, g, o, v, t1=singles, t2=doubles, t3=triples ) - HFenergy
        print('Final energy correction ', cycleEnergy)
        break
    else:
        ts = singles
        td = doubles
        tt = triples
        lastCycleEnergy = cycleEnergy
        print("\tIteration {: 5d}\t{: 5.15f}\t{: 5.15f}".format(cycle, lastCycleEnergy, deltaEnergy))
else:
    print("Did not converge")
    exit('cc failed')

#handle post iteration perturbative triples
if method == 'ccsd_t':
    perturbativeTriples = cc_triples(fock, g, o, v, t1=singles, t2=doubles, t3=triples)
    triples = perturbativeTriples + fock_d_abcijk * triples
    triples = triples * d_abcijk 

    l1, l2 = [singles.transpose(1,0) ,doubles.transpose(2,3,0,1)]
    perturbationEnergy = cc_perturbation_energy(fock, g, o, v, l1, l2, triples)
    print('Perturbative energy correction ', perturbationEnergy)

if method in ['ccd','ccsd','ccsdt','ccsd_t']:
    if method == 'ccsd':
        if mole == 'h2o':
            if base == 'sto-3g': print(np.isclose(-0.07068008709615015, cycleEnergy, 1e-8))
            if base == '6-31g': print(np.isclose(-0.1494126881297717, cycleEnergy, 1e-8))
            if base == 'dz': print(np.isclose(-0.15985561935376325, cycleEnergy, 1e-8))
        elif mole == 'acetaldehyde':
            if base == 'sto-3g': print(np.isclose(-0.20760397451113022, cycleEnergy, 1e-8))
        elif mole == 'ch4':
            if base == 'sto-3g': print(np.isclose(-0.07833502718385432, cycleEnergy, 1e-8))
 
    if method == 'ccd':
        if (mole == 'h2o') and (base == 'sto-3g'): print(np.isclose(-0.07015048, cycleEnergy, 1e-8))

    if method == 'ccsdt':
        if (mole == 'h2o') and (base == 'sto-3g'): print(np.isclose(-0.07081280801921253, cycleEnergy, 1e-8))

    if method == 'ccsd_t':
        if (mole == 'h2o') and (base == 'sto-3g'): print(np.isclose(-9.987726961762642e-05, perturbationEnergy, 1e-8))
        if (mole == 'h2o') and (base == '6-31g'):  print(np.isclose(-0.0015985955089234949, perturbationEnergy, 1e-8))
        if (mole == 'ch4') and (base == 'sto-3g'): print(np.isclose(-0.00013627396947413217, perturbationEnergy, 1e-8))

elif method in ['lccd', 'lccsd']:
    if method == 'lccd':
        if (mole == 'h2o') and (base == 'sto-3g'): print(np.isclose(-0.07192916394222108, cycleEnergy, 1e-8))
    if method == 'lccsd':
        if (mole == 'h2o') and (base == 'sto-3g'): print(np.isclose(-0.07257658934412553, cycleEnergy, 1e-8))
elif method in ['cc2', 'cc3']:
    if method == 'cc2':
        if (mole == 'h2o') and (base == 'sto-3g'): print(np.isclose(-0.0493991397445086, cycleEnergy, 1e-8))
        if (mole == 'h2o') and (base == 'dz'): print(np.isclose(-0.15422460085373757, cycleEnergy, 1e-8))
    if method == 'cc3':
        if (mole == 'h2o') and (base == 'sto-3g'): print(np.isclose(-0.07077803146036388, cycleEnergy, 1e-8))
        if (mole == 'h2o') and (base == 'dz'): print(np.isclose(-0.16149308528945028, cycleEnergy, 1e-8))

"""
#parse arguments
import sys
nargs = len(sys.argv)
if nargs == 1:
    mints = 'mints/h2o-sto-3g-mints.npz'
    method, molecule, basis = ['ccsd', 'h2o', 'sto-3g']
    file = 'codes/' + method + '.py'
elif nargs == 4:
    method, molecule, basis = sys.argv[1:4] 
    mints = 'mints/' + molecule + '-' + basis + '-mints.npz'
    file = 'codes/' + method + '.py'
else:
    exit('command line error')

f = open(file, 'r')
externalCode = f.read()

data = {'mints' : mints, 'method' : method, 'molecule' : molecule, 'basis' : basis}

exec(externalCode + internalCode,{},data)