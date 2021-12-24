import numpy as np

import sys
import csv

def generateEOM(einsum, tensors, index):

    #labels
    fermi = {'o':'ijkl', 'v':'abcd', 'ro':'mnp', 'rv':'efq'}

    #strip target index
    i = einsum.find('-')
    einsum = einsum[:i]

    #get repeated indices
    def indexCount(s):
        #character count in string

        count = {}
        for keys in s:
            count[keys] = count.get(keys, 0) + 1
        count.pop(',')

        return count

    counts = indexCount(einsum)

    #get deltas, if index in rvector is not repeated
    deltas = ''
    tEigenvectors = einsum.split(',')[tensors.split(',').index('r')]

    for s in range(len(tEigenvectors)):
        if counts[tEigenvectors[s]] == 1:
            deltas +=  ',' + tEigenvectors[s] + index[s] 
            tensors += ',kd'

    #substitute repeated indices
    for s in einsum:

        if (s in counts.keys()) and (counts[s] != 1):
            if s in fermi['o']: 
                einsum = einsum.replace(s, fermi['ro'][0])
                fermi['ro'] = fermi['ro'][1:]
                counts[s] -= 1
            if s in fermi['v']: 
                einsum = einsum.replace(s, fermi['rv'][0])
                fermi['rv'] = fermi['rv'][1:]
                counts[s] -= 1

    #get the current r-vector labels
    tEigenvectors = einsum.split(',')[tensors.split(',').index('r')]

    #substitute reference r-vector into current r-vector
    for i in range(len(index)) :
        einsum = einsum.replace(tEigenvectors[i], index[i])

    tEinsum = einsum.split(',')[tensors.split(',').index('r')]
    tIndex = (einsum + deltas).replace(tEinsum, '').replace(',,',',')
    tensors = tensors.replace(',r','').replace('f','fs')

    #determine repeated loops indices
    counts = indexCount(einsum)
    loops = ''
    for s in counts.keys():
        if (counts[s] != 1) and (s not in index):
            loops += s 
    

    #rationalise loops
    if not 'mn' in loops:  tIndex, loops = [tIndex.replace('n','m'), loops.replace('n','m')]
    if not 'ef' in loops:  tIndex, loops = [tIndex.replace('f','e'), loops.replace('f','e')]
    if not 'm' in loops:  tIndex, loops = [tIndex.replace('n','m'), loops.replace('n','m')]
    if not 'e' in loops:  tIndex, loops = [tIndex.replace('f','e'), loops.replace('f','e')]
    loops = ''.join(sorted(loops))

    return loops, tIndex, tensors, tEinsum

def eomCodeGenerator(definition, eomList):
    #write eom code

    #output file
    f = open(definition + '.py', 'w')

    def tensorExpression(e):
        #create the tensor product string

        if float(e[1]) == 1: exp = ''
        elif float(e[1]) == -1: exp = '-'
        else: exp = e[1] + '*'
        tensors = e[4].split(',')
        indices = e[3].split(',')

        for i in range(len(tensors)):
            exp += tensors[i] + '[' + ''.join([c + ',' for c in indices[i]])[:-1] + ']*'

        return exp[:-1]

    def getSign(expression):
        #get the sign of the permutation term

        if '-' in expression:
            return expression.replace('-','+')
        if '+' in expression:
            return expression.replace('+','-')
        return '-' + expression

    tab = '    '
    tabLevel = 0
    t = tab*tabLevel

    #subroutine header
    f.write('def ' + definition + '(fs, g, occupancy, t1=None, t2=None):\n')

    codeBlocks = {}
    codeBlocks['ss'] = """
    from numpy import  zeros, eye

    nsvir, nsocc = occupancy
    nso = nsvir + nsocc
    nrot = nsvir*nsocc

    kd  = eye((nrot))

    #[ss]-block
    hss = zeros((nrot, nrot))
    row = 0

    for i in range(nsocc):
        for a in range(nsocc, nso):

            col = 0
            for k in range(nsocc):
                for c in range(nsocc, nso):
"""
    codeBlocks['sd'] = """
    #[sd]-block
    hsd = zeros((nrot, nrot*nrot))
    row = 0

    for i in range(nsocc):
        for a in range(nsocc, nso):

            col = 0
            for k in range(nsocc):
                for c in range(nsocc, nso):
                    for l in range(nsocc):
                        for d in range(nsocc, nso):
"""
    codeBlocks['ds'] = """
    #[ds]-block
    hds = zeros((nrot*nrot, nrot))
    row = 0

    for i in range(nsocc):
        for a in range(nsocc, nso):
            for j in range(nsocc):
                for b in range(nsocc, nso):

                    col = 0
                    for k in range(nsocc):
                        for c in range(nsocc, nso):
"""
    codeBlocks['dd'] = """
    #[dd]-block
    hdd = zeros((nrot*nrot, nrot*nrot))
    row = 0

    for i in range(nsocc):
        for a in range(nsocc, nso):
            for j in range(nsocc):
                for b in range(nsocc, nso):

                    col = 0
                    for k in range(nsocc):
                        for c in range(nsocc, nso):
                            for l in range(nsocc):
                                for d in range(nsocc, nso):
"""
    tab = '    '

    #get the blocks expression
    matrix = {'ss':'hss[row,col] += ','sd':'hsd[row,col] += ','ds':'hds[row,col] += ','dd':'hdd[row,col] += '}

    for section in matrix.keys():
        
        #print pre-defined code 
        f.write(codeBlocks[section])

        block = [x for x in eomList if x[0] ==  section]
        block.sort(key = lambda i: (i[2], len(i[2])))

        #compute loop indents
        loops = ['', 'e', 'ef', 'em', 'efm', 'emn', 'm', 'mn']

        initialTab = {'ss':5, 'sd':7,'ds':7,'dd':9}
        tabLevel = initialTab[section]

        for loop in loops:
            loopEinsum = [x for x in block if x[2] == loop]

            if loopEinsum != []:
                if loop == 'e': f.write(tab*(tabLevel+len(loop)-1) + 'for e in range(nsocc, nso):\n')
                if loop == 'm': f.write(tab*(tabLevel+len(loop)-1) + 'for m in range(nsocc):\n')
                if loop == 'ef': f.write(tab*(tabLevel+len(loop)-1) + 'for f in range(nsocc, nso):\n')
                if loop == 'em': f.write(tab*(tabLevel+len(loop)-1) + 'for m in range(nsocc):\n')
                if loop == 'mn': f.write(tab*(tabLevel+len(loop)-1) + 'for n in range(nsocc):\n')
                if loop == 'efm': f.write(tab*(tabLevel+len(loop)-1) + 'for f in range(nsocc, nso):\n')
                if loop == 'emn': f.write(tab*(tabLevel+len(loop)-1) + 'for n in range(nsocc):\n')
                for ein in loopEinsum:
                    tensors = tensorExpression(ein)
                    #handle permutations
                    if ein[6] != '':
                        if len(ein[6]) == 2: 
                            sign = getSign(tensors)
                            tensors += sign.replace(ein[6][0], 'P').replace(ein[6][1], 'Q').replace('P', ein[6][1]).replace('Q', ein[6][0])
                            f.write(tab*(tabLevel+(len(loop))) + matrix[section] + tensors + '\n')
                        if len(ein[6]) == 5:
                            sign = getSign(tensors)
                            p, q = ein[6].split(',')
                            tensorPermutate = tensors + sign.replace(p[0], 'P').replace(p[1], 'Q').replace('P', p[1]).replace('Q', p[0])
                            f.write(tab*(tabLevel+len(loop)) + matrix[section] + tensorPermutate + '  \\\n')
                            tensorPermutate = sign.replace(q[0], 'P').replace(q[1], 'Q').replace('P', q[1]).replace('Q', q[0]) 
                            if not tensors[0] in ['+','-']: tensors = '+' + tensors
                            tensorPermutate += tensors.replace(p[0], 'P').replace(p[1], 'Q').replace('P', p[1]).replace('Q', p[0]) \
                            .replace(q[0], 'P').replace(q[1], 'Q').replace('P', q[1]).replace('Q', q[0])  
                            f.write(tab*(tabLevel+(len(loop)+4)) +  tensorPermutate+ '\n')
                    else:
                        f.write(tab*(tabLevel+len(loop)) + matrix[section] + tensors + '\n')

        f.write('\n')
        counterTab = {'ss':[3, 5],'sd':[3,7],'ds':[5,7],'dd':[5,9]}
        f.write(tab*counterTab[section][1] + 'col += 1\n')
        f.write(tab*counterTab[section][0] + 'row += 1\n\n')

    f.write('\n')
    f.write(tab + 'return hss, hsd, hds, hdd')

    f.close()

#get arguements
args = ''
for arg in sys.argv:
    args += arg

verbose = ('v' in args)

#read symbols file in einsum format
sectionSymbols = []
with open('symbols.txt', 'r') as f:
    reader = csv.reader(f, quotechar='\'', delimiter=',') 
    for row in reader:
        sectionSymbols.append(row)

#output if requested method and level
method, type, level, blocks = sectionSymbols[0]
sectionSymbols.pop(0)

if verbose:
    print('Method [', method, ']')
    print('Type[', level[:2], ']')
    print('Eigenvectors[', level[-1], ']')

#def name
definition = level[:2] + '_' + level[-1] + '_eom'

#target indices of H array
targetVectors = {'ss': 'ck','sd':'dclk','ds':'ck','dd':'dclk'}
outputVectors = {'ss': 'ck','sd':'cdkl','ds':'ck','dd':'cdkl'}

#loop over matrix blocks
sections = blocks.split('+')
codeSymbols = []
for section in sections:

    block = [x for x in sectionSymbols if x[0] == section]

    for b in block:
        loop, einsum, tensors, target = generateEOM(b[2], b[3], targetVectors[section])
        codeSymbols.append([section,b[1],loop, einsum, tensors, outputVectors[section] , b[4]])

        if verbose: print(codeSymbols[-1])

eomCodeGenerator(definition, codeSymbols)

