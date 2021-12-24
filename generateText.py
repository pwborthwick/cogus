from __future__ import division
from sympy import *
from sympy.physics.secondquant import * 
import pickle
input = open('symbols.pkl', 'rb')

def parseCompound(mixture):
    #parse the compound expression and reduce to components

    if isinstance(mixture, Mul):
        compound = [mixture]
    elif isinstance(mixture, AntiSymmetricTensor): 
        compound = [mixture] 
    else:
        compound = mixture.args

    atoms = []
    for element in compound:

        #multiple component term
        if isinstance(element, Mul):
            subAtoms = []
            for atom in element.args:
                subAtoms.append(atom)
            atoms.append(subAtoms)

        #kronecker deltas
        elif isinstance(element, KroneckerDelta): 
            atoms.append([element]) 
        #single tensor term
        elif isinstance(element, AntiSymmetricTensor):
            atoms.append([element])
        else:
            print('No handler for class ', type(element),' ', element)

    return atoms

def getScripts(script):
    #reduce to alpha characters
    return str(script).replace('(','').replace(')','').replace(',','').replace('_','').replace(' ','')

def einsumComponents(method, atoms, type, level):
    #parse the components

    def commaSeperate(s):
        #comma delimit string
        return ''.join([c + ',' for c in s])[:-1]

    #write block header for density and density matrix slice

    #loop over all the atoms and process into single equation
    count = 0

    returnSymbols = []
    for atom in atoms:

        factor, tensorString, einsumString, kroneckerString = [''] * 4
        permutation = ''

        factor = '1.00'
        for a in atom:
            #process multiplicative factor
            if isinstance(a, Number):
                precision = 2 + (sign(Float(a))== -1)
                factor = str(Float(a,precision))

            #process tensors and einsum string
            if isinstance(a, AntiSymmetricTensor):
                symbolString = str(a.symbol)
                lowerIndices = getScripts(a.lower)
                upperIndices = getScripts(a.upper)

                #determine symbol and indices
                tensorString += symbolString + ','
                einsumString +=  upperIndices + lowerIndices + ','

            if isinstance(a, PermutationOperator):
                permutation += getScripts(a.args) + ','

            if isinstance(a, KroneckerDelta):
                kroneckerString = 'kd' 
                tensorString += kroneckerString + ','
                einsumString += getScripts(a.args) + ','
                kroneckerString = ''

            #einsum -> string
            if method == 'coupled-cluster':   targetIndices = {'E':'','Et':'','S':'ai','D':'abij','T':'abcijk'}
            if method == 'response-density': targetIndices = {'oo':'ij','ov':'ia','vo':'ai','vv':'ab', \
                                 'oooo':'ijkl','ooov':'ijka','oovo':'ijak','ovoo':'iakl','vooo':'aikl', \
                                 'vvvv':'abcd','vvvo':'abci','vvov':'abic','vovv':'aicd','ovvv':'iacd', \
                                 'oovv':'ijab','vvoo':'abij','ovov':'iajb','voov':'aijb','ovvo':'iabj','vovo':'aibj'}
            if method == 'cluster-lambda':   targetIndices = {'E':'','Et':'','S':'ia','D':'ijab','T':'ijkabc'}
            if method == 'equation-of-motion':   targetIndices = {'ss': 'ai','sd':'abij','ds':'ai','dd':'abij'}

            targetIndex = targetIndices[type]

        returnSymbols.append([factor, einsumString[:-1] + '->' + targetIndex, tensorString[:-1], permutation[:-1]])

    return returnSymbols

#read pickle file
[method, type, level, sections] = pickle.load(input)
sectionSymbols = pickle.load(input)

input.close()

import sys
mode = sys.argv[1]

#raw sympy output
if mode == 'sympy':
	output = open('symbols.txt', 'w')
	output.write('method = ' + method + '\n')
	output.write('type = ' + type + '\n')
	output.write('level = ' + level + '\n')
	output.write('sections = ' + sections + '\n')
	output.write('')

	for section in sectionSymbols.keys():
		output.write('\nsection ' + section + '\n')
		output.write(str(sectionSymbols[section]) + '\n')

	output.close()

#einsum style strings
elif mode == 'einsum':
    if not method in ['coupled-cluster', 'response-density', 'cluster-lambda', 'equation-of-motion']: exit('not implemented')
    output =open('symbols.txt','w')
    output.write(method + ',' + type.lower() + ',' + level.lower() + ',' + sections.lower() + '\n')

    #disassemble sympy script
    for section in sectionSymbols.keys():

        atoms = parseCompound(sectionSymbols[section])

        returnSymbols = einsumComponents(method, atoms, section, level)

        for out in returnSymbols:
            output.write(section + ',' + out[0] + ',\'' + out[1] + '\',\'' + out[2] + '\',\'' + out[3] + '\'\n')

    output.close()

