from __future__ import division
from sympy import *
from sympy.physics.secondquant import * 

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

def lagrangeEnergyTable(file):
    #special handler for Lagrange energy in lambda

    file.write('\n\t<p style=\'font-size : 12px;\'>\n\t\t<math>\n\t\t\t<mtable columnalign=\'left\' rowlines=\'solid\' columnlines=\'solid\' frame=\'solid\'>')
    file.write('\n\t\t<mtr><mtd></mtd><mtd><mn>CCSD<sub>E</sub></mn></mtd></mtr>')
    file.write('\n\t\t<mtr><mtd></mtd><mtd><mmultiscripts><mn>CCSD<sub>S</sub></mn><mi>m</mi><mi>e</mi></mmultiscripts><mmultiscripts> \
                      <mn>&nbsp;l</mn><mi>e</mi><mi>m</mi></mmultiscripts></mtd><mtd></mtd></mtr>')
    file.write('\n\t\t<mtr><mtd></mtd><mtd><mmultiscripts><mn>CCSD<sub>D</sub></mn><mi>mn</mi><mi>ef</mi></mmultiscripts><mmultiscripts> \
                      <mn>&nbsp;l</mn><mi>ef</mi><mi>mn</mi></mmultiscripts></mtd><mtd></mtd></mtr>')

def mathMLtable(f, atoms):
    #parse the atoms in mathML tables 

    f.write('\n\t<p style=\'font-size : 12px;\'>\n\t\t<math>\n\t\t\t<mtable columnalign=\'left\' rowlines=\'solid\' columnlines=\'solid\' frame=\'solid\'>')

    #loop over all the atoms and process into single equation
    count = 0

    for atom in atoms:
        factor, tensor, permutation = ['', '', '']

        for a in atom:

            if isinstance(a, Number):
                factor = str(a)

            if isinstance(a, KroneckerDelta):                         
                tensor += '<mn>&delta;</mn><mi>' + getScripts(a.args) + '</mi>'

            if isinstance(a, AntiSymmetricTensor):
                tensor += '<mmultiscripts><mn>&nbsp;' + str(a.symbol) + '</mn><mi>' + getScripts(a.lower) + \
                     '</mi><mi>' + getScripts(a.upper) + '</mi></mmultiscripts>'

            if isinstance(a, PermutationOperator):
                permutation +=  '<mi>' + getScripts(a.args) + '&nbsp;</mi>'

        script = '\n\t\t\t\t<mtr><mtd>' + factor + '</mtd><mtd>' + tensor + '</mtd><mtd>' + permutation + '</mtd></mtr>'

        #for display remove amplitude degree identification
        script = script.replace('t1','t').replace('t2','t').replace('t3','t').replace('l1','l').replace('l2','l')
        f.write(script)

    f.write('\n\t\t\t</mtable>\n\t\t</math></p>')

    return

import pickle
input = open('symbols.pkl', 'rb')

[method, type, level, sections] = pickle.load(input)
sectionSymbols = pickle.load(input)
input.close()

typeDictionary = {'CD':'ccd','CSD':'ccsd','CSDT':'ccsdt','CSDt':'ccsd(t)','LLD':'lccd','LLSD':'lccsd','A2':'cc2','A3':'cc3', \
                  'DCC':'coupled-cluster response density','DIP':'ionization potential response density','DEA':'electron affinity response density', \
                  'DEE':'electron excitation response density','GSD':'ccsd-lambda', \
                  'EIPR':'equation of motion ionization potential','EEAR':'equation of motion electron affinity', \
                  'EEER':'equation of motion neutral excitation', 'EIPL':'equation of motion ionization potential', \
                  'EEAL':'equation of motion electron affinity',  'EEEL':'equation of motion neutral excitation'}

f = open('equations.html', 'w')
f.write('<!DOCTYPE html>\n<html lang="en">\n\t<body>\n\t<h6>' + typeDictionary[type+level] + '</h6>')

#table descriptions
title = '<p style=\'font-size : 12px;\'>' + method + ' '
if method in ['coupled-cluster','cluster-lambda']:
    subTitle = {'E':'energy</p>', 'Et':'perturbative triples energy</p>','S':'singles amplitudes</p>','D':'doubles amplitudes</p>', \
                'T':'triples amplitudes</p>'}
elif method in ['response-density', 'equation-of-motion']:
    subTitle = {}
    for key in sectionSymbols.keys():
        subTitle[key] = '[' + key + '] block'

#loop over the quantities for output
for compound in sectionSymbols.items():

    #Lambda lagrange energy
    if (method == 'cluster-lambda') and (compound[0] == 'S'): 
        f.write(title + 'Lagrangian energy')
        lagrangeEnergyTable(f)

    atoms = parseCompound(compound[1])
    f.write(title + subTitle[compound[0]])

    mathMLtable(f, atoms) 

f.write('\n\t</body>\n</html>')
f.close()
