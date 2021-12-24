## Cluster Operator Generation Using Sympy
Sympy has a secondquant module which is an implementation of the second quantization formalism. [Josh Goings](https://github.com/jjgoings) alerted me to this automatic code generation program [pdaggerq](https://github.com/edeprince3/pdaggerq). I wanted to implement the same idea as pdaggerq but using the secondquant sympy module as an exercise.

There are eight main programs (at present) generateSymbols, generateMathML, generatePython, generateSympy, validateCoupledCluster and validateLambda.

**generateSymbols code**\
This will generate a 'pickled' file containing sympy symbolic data in binary form called *symbols.pkl*. The program is run from the command line as *python generateSymbols.py method code*. If method is *coupled-cluster* code can be one of - ccd, ccsd, ccsdt, ccsd_t, cc2, cc3, lccd, lccsd , for method equals *response-density* code can be one of - cc, ee, ip, ea. For method *cluster-lambda* code should be - sd. There is a method *equation-of-motion* which can run as eg *python generateSymbols.py equation-of-motion ee-r* for right-hand EE or eg *ip-l* for left-hand ionization potential. 

**generatemathML**\
This will produce from the *symbols.pkl* an *equations.html* file. This file will display the sympy equations in readable form.

**generatePython**\
This will produce from the *symbols.pkl* file a 'code'.py file containing python subroutines generated from the sympy symbolic code. The files produced are\
ccd.py, ccsd.py, ccsdt.py, ccsd_t.py, cc2.py, cc3.py, lccd.py and lccsd.py for coupled-cluster, cc_rdm.py for coupled-cluster response density matrices and ccsd_lambda.py for the CCSD &Lambda; equations. EOM is not implemented for code generation.

**generateText**\
This converts the binary *symbols.pkl* to a text file *symbols.txt* which is readable. It can be run as *python generateText.py sympy* which produces a file with the raw sympy output or *python generateText.py einsum* which produces a text file with the components of the einsum expressions. The einsum file contains lines like, ```'oooo,-1.00,'ai,ka,jl->ijkl','t1,l,kd','ij,kl'``` - [identifier, multiplier, einsum indices, einsum tensors, permutations].

**generateEOM.py**\
The generates code from a *symbols.txt* einsum file. The code file is named according to the header of the *symbols.txt*. There is a 'v' command line which will print to the consule the generated expression components in the form \[matrix block, factor, loops, einsum indices, tensors, target element, permutation\], for example ```['dd', '-1.00', 'e', 'ke,ej,ad,bc,il', 'fs,t1,kd,kd,kd', 'cdkl', 'ij']```

**validateCoupledCluster**\
This will run tests on the coupled-cluster codes producted by generatePython. It does this by 1) reading the generated amplitude code file, 2) concatenating that with an internal code object containing the code to run the amplitude subroutines, 3) the concatenated code is then exececuted via exec. So the program contains no coupled-cluster amplitude code itself. For CCD, test is H<sub>2</sub>O in STO-3G, for CCSD tests are H<sub>2</sub>O in STO-3G, 6-31G and DZ, methane and acetaldehyde in STO-3G ,  and for CCSDT test is H<sub>2</sub>O in STO-3G. Run from command line as *python validateCoupledCluster.py code molecule basis* or just *python validateCoupledCluster.py* to run *ccsd h2o sto-3g*. For LCCD and LCCSD test is H<sub>2</sub>O in STO-3G. For CC2 and CC3 test is H<sub>2</sub>O in STO-3G and DZ bases. There are also tests for H<sub>2</sub>O in STO-3G and 6-31g and methane in STO-3G for CCSD(T).

**validateLambda**\
This will run tests on the response-density and lambda codes producted by generatePython. Run as just *python validateLambda,py* to test  H<sub>2</sub>O in STO-3G or as *python validateLambda.py h2o 6-31g*. Tests against values generated using *pdaggerq* codes. The tested results are CCSD energy correction, &Lambda; Lagrange energy correction, &Lambda; pseudo energy correction, rdm energy and a check that trace oprdm is equal to the number of electrons. 

**validateEOM**\
This will test the ee_r_eom.py code. Run as just *python validateEOM.py*. This will read files *ee_r_eom.py* and *ccsd.py* and calculate the singlet and triplet EOM states for H<sub>2</sub>sub> in the 3-21g basis. The reference values are taken from *pyscf*.

There are 2 additional folders:\
**codes**\
This folder contains the python codes generated by *generatePython*, they are listed under generatePython above. The coupled-cluster files contain routines cc_energy (cc_perturbation_energy), cc_singles, cc_doubles and cc_triples. For cc_rdm.py the contained routines are cc_oprdm and cc_tprdm. For &Lambda; file the routines are cc_lagrangian_energy, cc_lambda_singles and cc_lambda_doubles. For EOM there is an ee_r_eom code file.

**mints**\
This folder contains harpy mints files h2-3-21g-mints.npz, acetaldehyde-sto-3g-mints.npz, ch4-sto-3g-mints.npz, h2o-sto-3g-mints.npz, h2o-6-31g-mints.npz, h2o-dz-mints.npz. These are compressed files containing details of a converged Hartree-Fock computation with keys **s**-overlap, **k**-kinetic, **j**-potential, **i**-tei, **f**-Fock, **d**-density, **c**-eigenvalues, **e**-eigenvalues, **E**-HF energy, **m**-[charge, molecule name, basis name, nuclear repulsion energy], **a**-list of atomic numbers.
These files allow post HF calculations to be performed without the need for a full HF program.

There are more detailed notes in the Jupyter notebook.

Below is a schema of the files involved. Rectangles are programs and ellipses are files, the modular nature means it is possible to write applications from generated files at different points in the data flow.

![image](https://user-images.githubusercontent.com/73105740/129344492-2e5e4092-135e-405d-8d12-d41f10b6feda.png)