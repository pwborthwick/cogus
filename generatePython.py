from __future__ import division
from sympy import *
from sympy.physics.secondquant import * 
import datetime

def parseCompound(mixture):
    #parse the compound expression and atomise

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
    return str(script).replace('(','').replace(')','').replace(',','').replace('_','').replace(' ','')

def lagrangian_energy(f):
    #for lambda equations - the Lagrangian energy

    f.write('\ndef cc_lambda_lagrangian_energy(f, g, o, v, t1=None, t2=None, l1=None, l2=None):\n')
    f.write('\n    \'\'\'\n        COGUS generated level [SD] on ' + \
                datetime.datetime.now().strftime("%d %b %Y") + '   \n    \'\'\'\n')
    f.write('\n    from numpy import einsum\n')
    f.write('\n    T =  cc_energy(f, g, o, v, t1, t2, t3=None)\n')
    f.write('\n    T += einsum(\'ia,ai->\', l1, cc_singles(f, g, o, v, t1, t2, t3=None))\n')
    f.write('\n    T += einsum(\'ijab,abij->\', l2, cc_doubles(f, g, o, v, t1, t2, t3=None))\n')
    f.write('\n\n    return T\n\n')

def pythonCode(f, method, atoms, type, level):
    #parse the atoms

    def getSlice(arguements):
        #get the o,v slice of arguements

        slice = '['
        for ov in arguements:
            if ov in 'abcdef': slice += 'v,'
            if ov in 'ijklmn': slice += 'o,'

        return slice[:-1] + ']'

    def getSwapAxis(target, permutation):
        #returns swapaxis arguements 

        i = target.find(permutation)
        return str(i+1) + ', ' + str(i)

    def commaSeperate(s):
        #comma delimit string

        return ''.join([c + ',' for c in s])[:-1]

    #only write header once for each subroutine
    if type in ['E','S','D','T','Et']:
        f.write('\n    \'\'\'\n        COGUS generated level [' + level + '] on ' + \
                datetime.datetime.now().strftime("%d %b %Y") + '   \n    \'\'\'\n')
        f.write('    from numpy import einsum, swapaxes\n\n')
    elif type == 'oo':
        f.write('\n    \'\'\'\n        COGUS generated level [' + level + '] on ' + \
                datetime.datetime.now().strftime("%d %b %Y") + '   \n    \'\'\'\n')
        f.write('    from numpy import einsum, swapaxes, eye, zeros\n\n    ns = sum(t1.shape)\n    d = zeros((ns, ns))\n    kd = eye(ns)\n')
    elif type == 'oooo':
        f.write('\n    \'\'\'\n        COGUS generated level [' + level + '] on ' + \
                datetime.datetime.now().strftime("%d %b %Y") + '   \n    \'\'\'\n')
        f.write('    from numpy import einsum, swapaxes, eye, zeros\n\n    ns = sum(t1.shape)\n    d = zeros((ns, ns, ns, ns))\n    kd = eye(ns)\n')

    #write block header for density and density matrix slice
    if method == 'response-density':
        f.write('\n    # density matrix block [' + type + ']\n')
        matrix = 'd[' + commaSeperate(type) + '] '
        operator = matrix + '= '
        returnSymbol = 'd '

        #we can use symmetries to reduce the computation
        symmetryDictionary = {'oovo':['ooov','3, 2'],'vooo':['ovoo','1, 0'],'vvov':['vvvo','3, 2'],'ovvv':['vovv','1, 0'], \
                              'voov':['ovov','1, 0'],'vovo':['ovvo','1, 0']}
        if type in symmetryDictionary.keys():
            f.write('    d[' + commaSeperate(type) + '] -= d[' + \
                    commaSeperate(symmetryDictionary[type][0]) + '].swapaxes(' + symmetryDictionary[type][1] + ')\n\n')
            return returnSymbol
    else:
        returnSymbol = 'T'
        operator = returnSymbol + ' = '

    #loop over all the atoms and process into single equation
    count = 0

    for atom in atoms:
        factor, tensorString, einsumString, commentString, kroneckerString = [''] * 5
        permutation = []

        factor = '1.0000 * '
        for a in atom:
            #process multiplicative factor
            if isinstance(a, Number):
                precision = 4 + (sign(Float(a))== -1)
                factor = str(Float(a,precision)) + ' * '

            #process tensors and einsum string
            if isinstance(a, AntiSymmetricTensor):
                symbolString = str(a.symbol)
                lowerIndices = getScripts(a.lower)
                upperIndices = getScripts(a.upper)

                #determine which amplitude
                if symbolString in 'tl':  symbolString += str(len(a.upper))

                if symbolString != 'g':
                    commentString += symbolString + '(' + ','.join(upperIndices+lowerIndices) + ') '
                else:
                    commentString += '<' + ','.join(upperIndices) + '||' + ','.join(lowerIndices) + '> '


                #determine slice of g
                if symbolString in 'fg':
                    symbolString += getSlice(upperIndices + lowerIndices)
                tensorString += symbolString + ' ,'

                einsumString +=  upperIndices + lowerIndices + ','

            if isinstance(a, PermutationOperator):
                permutation.append(getScripts(a.args))

            if isinstance(a, KroneckerDelta):
                kroneckerString = 'kd' + getSlice(getScripts(a.args)[0] + getScripts(a.args)[1])
                tensorString += kroneckerString + ', '
                commentString += kroneckerString + ' '
                einsumString += getScripts(a.args) + ','
                kroneckerString = ''

        #einsum -> string
        if method == 'coupled-cluster':   targetIndices = {'E':'','Et':'','S':'ai','D':'abij','T':'abcijk'}
        if method == 'response-density': targetIndices = {'oo':'ij','ov':'ia','vo':'ai','vv':'ab', \
                             'oooo':'ijkl','ooov':'ijka','oovo':'ijak','ovoo':'iakl','vooo':'aikl', \
                             'vvvv':'abcd','vvvo':'abci','vvov':'abic','vovv':'aicd','ovvv':'iacd', \
                             'oovv':'ijab','vvoo':'abij','ovov':'iajb','voov':'aijb','ovvo':'iabj','vovo':'aibj'}
        if method == 'cluster-lambda':   targetIndices = {'E':'','Et':'','S':'ia','D':'ijab','T':'ijkabc'}

        targetIndex = targetIndices[type]

        #process for comment string
        permutationString = ''
        if permutation != []:
            for p in permutation:
                permutationString += 'P(' + ','.join(p) + ')'
        commentString = '\n    #  ' + factor + permutationString + ' ' + commentString 

        f.write(commentString)

        #process the permutations
        if permutation == []:
            f.write('\n    ' + operator + factor + 'einsum(\'' + einsumString[:-1] + '->' + targetIndex + \
                    '\' ,' + tensorString[:-2] + ', optimize=True)\n')
        else:
            f.write('\n    t = ' + factor + 'einsum(\'' + einsumString[:-1] + '->' + targetIndex + \
                    '\' ,' + tensorString[:-2] + ', optimize=True)\n')

        if len(permutation) == 1:
            f.write('    ' + operator + 't - t.swapaxes(' + getSwapAxis(targetIndex, permutation[0]) + ')\n')
        elif len(permutation) == 2:
            p, q = [getSwapAxis(targetIndex, permutation[0]),getSwapAxis(targetIndex, permutation[1]) ]
            f.write('    ' + operator + 't - t.swapaxes(' + p + ') - t.swapaxes(' + q + \
                    ') + t.swapaxes(' + p + ').swapaxes(' + q +')\n')

        #change operator to +=
        if method == 'response-density':
            operator = matrix + '+= '
        else:
            operator = returnSymbol + ' += '

    return returnSymbol

import pickle
input = open('symbols.pkl', 'rb')

[method, type, level, sections] = pickle.load(input)
sectionSymbols = pickle.load(input)

if not method in ['coupled-cluster', 'response-density', 'cluster-lambda']: exit('not implemented')

input.close()

typeDictionary = {'CD':'ccd','CSD':'ccsd','CSDT':'ccsdt','CSDt':'ccsd_t','LLD':'lccd','LLSD':'lccsd','A2':'cc2','A3':'cc3', \
                  'DCC':'cc_rdm','DIP':'ip_rdm','DEA':'ea_rdm','DEE':'ee_rdm','GSD':'ccsd_lambda'}
sectionsDirectory = {'CD':'E+D','CSD':'E+S+D','CSDT':'E+S+D+T','CSDt':'E+S+D+T','A2':'E+S+D','A3':'E+S+D+T','LLD':'E+D','LLSD':'E+S+D', \
                     'DCC':'1+2','DEE':'1+2','DIP':'1+2','DEA':'1+2', \
                     'GSD':'E+S+D'}

name = typeDictionary[type+level]
f = open(name + '.py', 'w')

#write file header with occupancy information
f.write('\n\'\'\'s - spin orbitals, o - occupied spin orbitals, v - virtual spin orbitals\n' +\
        't1[v,o] - singles amplitudes, t2[v,v,o,o] - doubles amplitudes\n' + \
        'l1[o,v] - singles lambda, ' + \
        'l2[o,o,v,v] - doubles lambda\n\'\'\'\n\n')

#subroutine headers - only density blocks oo and oooo trigger header
if method in ['coupled-cluster', 'response-density']:
    title = 'def cc'
    subTitle = {'E':'_energy(f, g, o, v, t1=None, t2=None, t3=None):\n',  \
                'S':'_singles(f, g, o, v, t1=None, t2=None, t3=None):\n', \
                'D':'_doubles(f, g, o, v, t1=None, t2=None, t3=None):\n', \
                'T':'_triples(f, g, o, v, t1=None, t2=None, t3=None):\n', \
                'Et':'_perturbation_energy(f, g, o, v, l1=None, l2=None, t3=None):\n', \
                'oo':'_oprdm(o, v, t1=None, t2=None, l1=None, l2=None):\n', 'ov':'','vo':'','vv':'', \
                'oooo':'_tprdm(o, v, t1=None, t2=None, l1=None, l2=None):\n', 'ooov':'','oovo':'', \
                'ovoo':'','vooo':'','vvvv':'','vvvo':'','vvov':'','vovv':'','ovvv':'','oovv':'','vvoo':'','ovov':'', \
                'vovo':'','voov':'','ovvo':''}
elif method == 'cluster-lambda':
    title = 'def cc_lambda'
    subTitle = {'S':'_singles(f, g, o, v, t1=None, t2=None, l1=None, l2=None):\n', \
                'D':'_doubles(f, g, o, v, t1=None, t2=None, l1=None, l2=None):\n'}

#loop over the quantities needed for output
for compound in sectionSymbols.items():

    atoms = parseCompound(compound[1])

    #write subroutine def:
    if (method == 'cluster-lambda') and (compound[0] == 'S'): lagrangian_energy(f)
    if compound[0] in ['E','S','D','T','Et','oo', 'oooo']: 
        f.write(title + subTitle[compound[0]])

    returnSymbol = pythonCode(f, method, atoms, compound[0], level)

    #return for subroutines
    if compound[0] in ['E','S','D','T','Et','vv','vovo']:
        f.write('\n    return ' + returnSymbol + '\n\n')


f.close()
