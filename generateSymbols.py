from __future__ import division
from sympy import *
from sympy.physics.secondquant import * 

def amplitudes(amplitudeType, amplitudeLevel):
    '''
    operator amplitudes
    '''

    def lagrangeOperators(level):
        #Lagrange multipliers

        if 'S' in level:
            #Lagrange singles amplitudes
            i = symbols('i', below_fermi=True, cls=Dummy)
            a = symbols('a', above_fermi=True, cls=Dummy)
            l1 = AntiSymmetricTensor('l1', (i,), (a,)) * NO(Fd(i)*F(a))

        if 'D' in level:
            #Lagrange doubles amplitudes
            i, j = symbols('i,j', below_fermi=True, cls=Dummy)
            a, b = symbols('a,b', above_fermi=True, cls=Dummy)
            l2 = Rational(1, 4) * AntiSymmetricTensor('l2', (i, j), (a, b)) *  NO(Fd(i)*F(a)*Fd(j)*F(b))

        if level == 'S':    return l1
        if level == 'D':    return l2
        if level == 'SD':   return l1+l2

    def clusterOperators(level):
    #compute the cluster singles, doubles and triples amplitudes 

        if 'S' in level:
            #Cluster singles amplitudes
            i = symbols('i', below_fermi=True, cls=Dummy)
            a = symbols('a', above_fermi=True, cls=Dummy)
            t1 = AntiSymmetricTensor('t1', (a,), (i,)) * NO(Fd(a)*F(i))

        if 'D' in level:
            #Cluster doubles amplitudes
            i, j = symbols('i,j', below_fermi=True, cls=Dummy)
            a, b = symbols('a,b', above_fermi=True, cls=Dummy)
            t2 = Rational(1, 4) * AntiSymmetricTensor('t2', (a, b), (i, j)) * NO(Fd(a)*F(i)*Fd(b)*F(j))

        if 'T' in level:
            #Cluster triples amplitudes
            i, j, k = symbols('i:k', below_fermi=True, cls=Dummy)
            a, b, c = symbols('a:c', above_fermi=True, cls=Dummy)
            t3 = Rational(1, 36) * AntiSymmetricTensor('t3', (a, b, c), (i, j, k)) * NO(Fd(a)*F(i)*Fd(b)*F(j)*Fd(c)*F(k))

        if level == 'S':   return t1
        if level == 'D':   return t2
        if level == 'T':   return t3
        if level == 'SD':  return t1 + t2
        if level == 'SDT': return t1 + t2 + t3

    def excitationOperators(level):

        i, j, k = symbols('i,j,k' ,below_fermi=True, cls=Dummy)
        a ,b, c = symbols('a:c' ,above_fermi=True, cls=Dummy)   

        if level == 'IP':
            return [0, AntiSymmetricTensor('r',(),(i,))*F(i), Rational(1, 2)*AntiSymmetricTensor('r',(a,),(j,k))*Fd(a)*F(k)*F(j)]
        elif level == 'EA':
            return [0, AntiSymmetricTensor('r',(a,),())*Fd(a), Rational(1, 2)*AntiSymmetricTensor('r',(b,c),(i,))*Fd(b)*Fd(c)*F(i)]
        elif level == 'EE':
            return [AntiSymmetricTensor('r0',(),()), AntiSymmetricTensor('r',(a,),(i,))*Fd(a)*F(i), \
                    Rational(1, 4)*AntiSymmetricTensor('r',(b,c),(j,k))*Fd(b)*Fd(c)*F(k)*F(j) ]
        elif level == 'CC':
            return [1, 0, 0]

    def deExcitationOperators(level):

        i, j, k = symbols('i:k' ,below_fermi=True, cls=Dummy)
        a ,b, c = symbols('a:c' ,above_fermi=True, cls=Dummy)   

        if level == 'IP':
            return [0, AntiSymmetricTensor('l',(i,),())*Fd(i), Rational(1, 2)*AntiSymmetricTensor('l',(j,k),(a,))*Fd(j)*Fd(k)*F(a)]  
        elif level == 'EA':
            return [0, AntiSymmetricTensor('l',(),(a,))*F(a), Rational(1, 2)*AntiSymmetricTensor('l',(i,),(b,c))*Fd(i)*F(c)*F(b)]
        elif level == 'EE':
            return [0, AntiSymmetricTensor('l',(i,),(a,))*Fd(i)*F(a), Rational(1, 4)*AntiSymmetricTensor('l',(j,k),(b,c))*Fd(j)*Fd(k)*F(c)*F(b)]
        elif level == 'CC':
            return [1, AntiSymmetricTensor('l',(i,),(a,))*Fd(i)*F(a), Rational(1, 4)*AntiSymmetricTensor('l',(j,k),(b,c))*Fd(j)*Fd(k)*F(c)*F(b)]

    if amplitudeType == 'lagrange': 
        return lagrangeOperators(amplitudeLevel)
    elif amplitudeType == 'cluster': 
        return clusterOperators(amplitudeLevel)
    elif amplitudeType == 'excitation':
        return excitationOperators(amplitudeLevel)
    elif amplitudeType == 'de-excitation':
        return deExcitationOperators(amplitudeLevel)

def hamiltonian(hamilitonianType, h=None, type=None, level=None):
    '''
    Hamiltonian related entities
    '''

    def bakerCampbellHausdorff(h, type, level):
        #Baker-Campbell-Hausdorff

        symbols = {'above': 'defg','below': 'lmno', 'general': 'pqrst' }

        #commutator bracket
        commutatorBracket = Commutator

        #define symbolic array for BCH terms
        bch = zeros(5)
        bch[0] = h

        #order is 0-4 for full coupled-cluster and 0-1 for linear cc
        orderDictionary = {'C':4, 'L':1, 'A':4, 'G':4}
        order = orderDictionary[type]

        for i in range(order):
            t  = amplitudes('cluster',level)
            bch[i+1] = wicks(commutatorBracket(bch[i], t))
            bch[i+1] = evaluate_deltas(bch[i+1])          
            bch[i+1] = substitute_dummies(bch[i+1])

        #BCH expansion
        BCH = bch[0] + bch[1] + bch[2]/2 + bch[3]/6 + bch[4]/24 

        #tidy up and compact
        BCH = BCH.expand()
        BCH = evaluate_deltas(BCH)
        BCH = substitute_dummies(BCH, new_indices=True, pretty_indices=symbols)

        return BCH

    def hfGroundStateEnergy():
        #get Hartree-Fock electronic energy

        p, q, r, s = symbols('p,q,r,s', cls=Dummy)                                      

        h = AntiSymmetricTensor('f', (p,), (q,))*Fd(p)*F(q)  + \
            Rational(1, 4)*AntiSymmetricTensor('g', (p, q), (r, s))*Fd(p)*Fd(q)*F(r)*F(s)                                                                                                                                              
                                                                                        
        hf = wicks(h, keep_only_fully_contracted=True, simplify_kronecker_deltas=True)                                   
        hf = substitute_dummies(hf, new_indices=True, pretty_indices={'below':  'ijklmno','above':  'abcde', 'general': 'pqrs'})

        return hf


    if hamilitonianType == 'bch':  
        return bakerCampbellHausdorff(h, type, level)
    elif hamilitonianType == 'hf': 
        return hfGroundStateEnergy()
    elif hamilitonianType == 'h':
        p, q, r, s = symbols('p:s', cls=Dummy)
        f = AntiSymmetricTensor('f', (p, ), (q, )) * NO(Fd(p)*F(q))
        v = Rational(1, 4)*AntiSymmetricTensor('g', (p, q), (r, s)) * NO(Fd(p)*Fd(q)*F(s)*F(r))
        return f, v



def kernel(method, type, level, sections):
    '''
    wicks generation
    '''

    def coupledCluster(type, level, sections):

        def perturbativeEnergy(level):
            #compute the energy contribution from perturbative triples

            cc =   Commutator(v, amplitudes('cluster','T'))
            leftOperators = amplitudes('lagrange','S') + amplitudes('lagrange','D')
            w = wicks(leftOperators*cc, simplify_kronecker_deltas=True, keep_only_fully_contracted=True)
            symbolRules = {'below':'ijklmno', 'above': 'abcdef', 'general':'pqrstu'}

            return substitute_dummies(w, new_indices=True, pretty_indices=symbolRules)

        #CC2,3 is essentially CCSD,T and SDt is SD
        perturbativeTriples = False
        if level == '2': level = 'SD'
        if level == '3': level = 'SDT'
        if level == 'SDt': 
            level = 'SD'
            perturbativeTriples = True
        if level in ['LD', 'LSD'] : level = level[1:]

        #define indices for occupied, virtual and general orbitals
        p, q, r, s = symbols('p:s', cls=Dummy)
        i, j, k = symbols('i, j, k', below_fermi = True)
        a, b, c = symbols('a, b, c', above_fermi = True)

        #get similarity-transformed Hamiltonian
        f, v = hamiltonian('h')
        st = hamiltonian('bch', f+v, type, level)

        #prepare mixture dictionary
        sectionSymbols = {}
        if 'E' in sections:
            #Cluster energy
            w = wicks(st, simplify_kronecker_deltas=True, keep_only_fully_contracted=True)
            symbolRules = {'below':'ijklmno', 'above': 'abcdef', 'general':'pqrstu'}

            sectionSymbols['E'] = substitute_dummies(w, new_indices=True, pretty_indices=symbolRules)
            sectionSymbols['E'] = hamiltonian('hf') + sectionSymbols['E']
            if perturbativeTriples: sectionSymbols['Et'] = perturbativeEnergy(level)

        if 'S' in sections:
            #Cluster singles amplitude
            w = wicks(Fd(i)*F(a)*st, simplify_kronecker_deltas=True, keep_only_fully_contracted=True)
            symbolRules = {'below':'jklmno', 'above': 'bcdef', 'general':'pqrstu'}

            sectionSymbols['S'] = substitute_dummies(w, new_indices=True, pretty_indices=symbolRules)

        if 'D' in sections:
            #Cluster doubles amplitude
            if (type == 'A') and (level == 'SD'): 
                st = hamiltonian('bch', f, type, 'D') + hamiltonian('bch', v, type, 'S')
            w = wicks(Fd(i)*Fd(j)*F(b)*F(a)*st, simplify_kronecker_deltas=True, keep_only_fully_contracted=True)
            symbolRules = {'below':'klmno', 'above': 'cdef', 'general':'pqrstu'}
            td = substitute_dummies(w, new_indices=True, pretty_indices=symbolRules)
            p = [PermutationOperator(i,j), PermutationOperator(a,b)]

            sectionSymbols['D'] = simplify_index_permutations(td, p)

        if 'T' in sections:
            #Cluster triples amplitude
            if (type == 'A') and (level == 'SDT'):
                st = hamiltonian('bch', f, type, 'SDT') + hamiltonian('bch', v, type, 'S') + v +  \
                     Commutator(v, amplitudes('cluster','D')) + \
                     Commutator(Commutator(v, amplitudes('cluster','S')),  amplitudes('cluster','D')) + \
                     Rational(1,2)*Commutator(Commutator(Commutator(v, amplitudes('cluster','S')), amplitudes('cluster','S')),  amplitudes('cluster','D')) + \
                     Rational(1,6)*Commutator(Commutator(Commutator(Commutator(v, amplitudes('cluster','S')), amplitudes('cluster','S')), amplitudes('cluster','S')),  amplitudes('cluster','D'))
            if perturbativeTriples: 
                st = hamiltonian('bch', f, type, 'T') + Commutator(v,  amplitudes('cluster','D'))

            w = wicks(Fd(i)*Fd(j)*Fd(k)*F(c)*F(b)*F(a)*st, simplify_kronecker_deltas=True, keep_only_fully_contracted=True)
            symbolRules = {'below':'lmno', 'above': 'def', 'general':'pqrstu'}
            tt = substitute_dummies(w, new_indices=True, pretty_indices=symbolRules)
            p = [PermutationOperator(i,j), PermutationOperator(a,b),PermutationOperator(j,k), PermutationOperator(b,c)]
            
            sectionSymbols['T'] = simplify_index_permutations(tt, p)

        return sectionSymbols

    def responseDensity(type, level, sections):
        #electron response density matrices

        R = sum(amplitudes('excitation',level))
        L = sum(amplitudes('de-excitation',level))

        sectionSymbols = {}
        if '1' in sections:

            i,j = symbols('i,j' , below_fermi=True)
            a,b = symbols('a,b' , above_fermi=True)

            st = hamiltonian('bch', Fd(i)*F(j),'C','SD')
            oo = wicks(L*st*R , keep_only_fully_contracted=True, simplify_kronecker_deltas = True)
            sectionSymbols['oo'] = substitute_dummies(oo,new_indices=True, pretty_indices={'below':  'klmno','above':  'abcde'})

            st = hamiltonian('bch', Fd(i)*F(a),'C','SD')
            ov = wicks(L*st*R , keep_only_fully_contracted=True, simplify_kronecker_deltas = True)
            sectionSymbols['ov'] = substitute_dummies(ov,new_indices=True, pretty_indices={'below':  'jklmn','above':  'bcdef'})
            
            st = hamiltonian('bch', Fd(a)*F(i),'C','SD')
            vo = wicks(L*st*R , keep_only_fully_contracted=True, simplify_kronecker_deltas = True)
            sectionSymbols['vo'] = substitute_dummies(vo,new_indices=True, pretty_indices={'below':  'jklmn','above':  'bcdef'})
           
            st = hamiltonian('bch', Fd(a)*F(b),'C','SD')
            vv = wicks(L*st*R , keep_only_fully_contracted=True, simplify_kronecker_deltas = True)
            sectionSymbols['vv'] = substitute_dummies(vv,new_indices=True, pretty_indices={'below':  'ijklm','above':  'cdefg'})

        if '2' in sections:

            i,j,k,l = symbols('i:l' , below_fermi=True)
            a,b,c,d = symbols('a:d' , above_fermi=True)
            p = [PermutationOperator(i,j), PermutationOperator(a,b)]

            st = hamiltonian('bch', Fd(i)*Fd(j)*F(l)*F(k), 'C', 'SD')
            oooo = wicks(L*st*R , keep_only_fully_contracted=True, simplify_kronecker_deltas = True)
            oooo = simplify_index_permutations(oooo, [PermutationOperator(i,j), PermutationOperator(k,l)])
            sectionSymbols['oooo'] = substitute_dummies(oooo,new_indices=True, pretty_indices={'below':  'mnop','above':  'abcde'})  

            st = hamiltonian('bch', Fd(i)*Fd(j)*F(a)*F(k), 'C', 'SD')
            ooov = wicks(L*st*R , keep_only_fully_contracted=True, simplify_kronecker_deltas = True)
            ooov = simplify_index_permutations(ooov, [PermutationOperator(i,j)])
            sectionSymbols['ooov'] = substitute_dummies(ooov,new_indices=True, pretty_indices={'below':  'lmnop','above':  'bcde'})  

            sectionSymbols['oovo'] = -ooov   

            st = hamiltonian('bch', Fd(i)*Fd(a)*F(l)*F(k), 'C', 'SD')
            ovoo = wicks(L*st*R , keep_only_fully_contracted=True, simplify_kronecker_deltas = True)
            ovoo = simplify_index_permutations(ovoo, [PermutationOperator(k,l)])
            sectionSymbols['ovoo'] = substitute_dummies(ovoo,new_indices=True, pretty_indices={'below':  'mnop','above':  'bcde'})

            sectionSymbols['vooo'] = -ovoo 

            st = hamiltonian('bch', Fd(a)*Fd(b)*F(d)*F(c), 'C', 'SD')
            vvvv = wicks(L*st*R , keep_only_fully_contracted=True, simplify_kronecker_deltas = True)
            vvvv = simplify_index_permutations(vvvv, [PermutationOperator(a,b), PermutationOperator(c,d)])
            sectionSymbols['vvvv'] = substitute_dummies(vvvv,new_indices=True, pretty_indices={'below':  'ijklmn','above':  'efgh'})
  
            st = hamiltonian('bch', Fd(a)*Fd(b)*F(i)*F(c), 'C', 'SD')
            vvvo = wicks(L*st*R , keep_only_fully_contracted=True, simplify_kronecker_deltas = True)
            vvvo = simplify_index_permutations(vvvo, [PermutationOperator(a,b)])
            sectionSymbols['vvvo'] = substitute_dummies(vvvo,new_indices=True, pretty_indices={'below':  'jklmn','above':  'efgh'})

            sectionSymbols['vvov'] = -vvvo   

            st = hamiltonian('bch', Fd(a)*Fd(i)*F(d)*F(c), 'C', 'SD')
            vovv = wicks(L*st*R , keep_only_fully_contracted=True, simplify_kronecker_deltas = True)
            vovv = simplify_index_permutations(vovv, [PermutationOperator(c,d)])
            sectionSymbols['vovv'] = substitute_dummies(vovv,new_indices=True, pretty_indices={'below':  'jklmn','above':  'befgh'})

            sectionSymbols['ovvv'] = -vovv  

            st = hamiltonian('bch', Fd(i)*Fd(j)*F(b)*F(a), 'C', 'SD')
            oovv = wicks(L*st*R , keep_only_fully_contracted=True, simplify_kronecker_deltas = True)
            oovv = simplify_index_permutations(oovv, [PermutationOperator(i,j), PermutationOperator(a,b)])
            sectionSymbols['oovv'] = substitute_dummies(oovv,new_indices=True, pretty_indices={'below':  'klmn','above':  'cdefg'})

            st = hamiltonian('bch', Fd(a)*Fd(b)*F(j)*F(i), 'C', 'SD')
            vvoo = wicks(L*st*R , keep_only_fully_contracted=True, simplify_kronecker_deltas = True)
            vvoo = simplify_index_permutations(vvoo, [PermutationOperator(i,j), PermutationOperator(a,b)])
            sectionSymbols['vvoo'] = substitute_dummies(vvoo,new_indices=True, pretty_indices={'below':  'klmn','above':  'defg'})

            st = hamiltonian('bch', Fd(i)*Fd(a)*F(b)*F(j), 'C', 'SD')
            ovov = wicks(L*st*R , keep_only_fully_contracted=True, simplify_kronecker_deltas = True)
            ovov = simplify_index_permutations(ovov, [])
            sectionSymbols['ovov'] = substitute_dummies(ovov,new_indices=True, pretty_indices={'below':  'klmn','above':  'defg'})

            sectionSymbols['voov'] = -ovov

            st = hamiltonian('bch', Fd(i)*Fd(a)*F(j)*F(b), 'C', 'SD')
            ovvo = wicks(L*st*R , keep_only_fully_contracted=True, simplify_kronecker_deltas = True)
            ovvo = simplify_index_permutations(ovvo, [PermutationOperator(a,b)])
            sectionSymbols['ovvo'] = substitute_dummies(ovvo,new_indices=True, pretty_indices={'below':  'klmn','above':  'defg'})

            sectionSymbols['vovo'] = -ovvo

        return sectionSymbols

    def clusterLambda(type, level, sections):
        #coupled-cluster operators

        #define indices for occupied, virtual and general orbitals
        p, q, r, s = symbols('p:s', cls=Dummy)
        i, j, k = symbols('i, j, k', below_fermi = True)
        a, b, c = symbols('a, b, c', above_fermi= True)

        #get similarity-transformed Hamiltonian
        f, v = hamiltonian('h')
        h = f + v
        st = hamiltonian('bch', h, type, level)

        #prepare mixture dictionary
        sectionSymbols = {}
        if 'E' in sections:
            #Lambda energy
            w = wicks(st, simplify_kronecker_deltas=True, keep_only_fully_contracted=True)
            symbolRules = {'below':'ijklmno', 'above': 'abcdef', 'general':'pqrstu'}

            sectionSymbols['E'] = substitute_dummies(w, new_indices=True, pretty_indices=symbolRules)
            sectionSymbols['E'] = hamiltonian('hf') + sectionSymbols['E']

        if 'S' in sections:
            #Lambda singles amplitude
            st = hamiltonian('bch', h*Fd(a)*F(i), type, level)
            w = wicks(st, simplify_kronecker_deltas=True, keep_only_fully_contracted=True)

            st = hamiltonian('bch', Commutator(h ,Fd(a)*F(i)), type, level)
            leftOperators = amplitudes('lagrange','S') + amplitudes('lagrange','D')
            w += wicks(leftOperators*st, simplify_kronecker_deltas=True, keep_only_fully_contracted=True)

            symbolRules = {'below':'jklmno', 'above': 'bcdef', 'general':'pqrstu'}
            sectionSymbols['S'] = substitute_dummies(w, new_indices=True, pretty_indices=symbolRules)

        if 'D' in sections:
            #Lambda doubles amplitude
            st = hamiltonian('bch', h*Fd(a)*Fd(b)*F(j)*F(i), type, level)
            w = wicks(st, simplify_kronecker_deltas=True, keep_only_fully_contracted=True)

            st = hamiltonian('bch', Commutator(h,Fd(a)*Fd(b)*F(j)*F(i)), type, level)
            leftOperators = amplitudes('lagrange','S') + amplitudes('lagrange','D')
            w += wicks(leftOperators*st, simplify_kronecker_deltas=True, keep_only_fully_contracted=True)

            symbolRules = {'below':'klmno', 'above': 'cdef', 'general':'pqrstu'}
            ld = substitute_dummies(w, new_indices=True, pretty_indices=symbolRules)

            p = [PermutationOperator(i,j), PermutationOperator(a,b)]
            sectionSymbols['D'] = simplify_index_permutations(ld, p)

        return sectionSymbols

    def equationMotion(type, level, sections):
        #equation of motion equations

        #define indices for occupied, virtual and general orbitals
        p, q, r, s = symbols('p:s', cls=Dummy)
        i, j, k = symbols('i, j, k', below_fermi = True)
        a, b, c = symbols('a, b, c', above_fermi= True)

        eigenvector = level[-1]
        level = level[:2]
        
        #get similarity-transformed Hamiltonian
        f, v = hamiltonian('h')
        h = f + v
        st = hamiltonian('bch', h, 'C', 'SD')

        #prepare mixture dictionary
        sectionSymbols = {}

        if eigenvector == 'R':

            #excitation amplitudes
            R = amplitudes('excitation', level)

            if level == 'IP': qOperators, qSymbols = [Fd(i), {'below': 'jklmno','above': 'abcdefg'}]
            if level == 'EA': qOperators, qSymbols = [F(a), {'below': 'ijklmno','above': 'bcdefg'}]
            if level == 'EE': qOperators, qSymbols = [Fd(i)*F(a), {'below': 'jklmno','above': 'bcdefg'}]

            if 'ss' in sections:
                #singles-singles block
                ss = evaluate_deltas(wicks(qOperators*Commutator(st,R[1]) , keep_only_fully_contracted=True))
                ss = substitute_dummies(ss, new_indices=True, pretty_indices= qSymbols)
                p = [PermutationOperator(i,j), PermutationOperator(a,b)]

                sectionSymbols['ss'] = simplify_index_permutations(ss, p)

            if 'sd' in sections:
                #singles-doubles block
                sd = evaluate_deltas(wicks(qOperators*Commutator(st,R[2]) , keep_only_fully_contracted=True))
                sd = substitute_dummies(sd, new_indices=True, pretty_indices= qSymbols)
                p = [PermutationOperator(i,j), PermutationOperator(a,b)]

                sectionSymbols['sd'] = simplify_index_permutations(sd, p)

            if level == 'IP': qOperators, qSymbols = [Fd(i)*Fd(j)*F(a), {'below': 'klmno','above': 'bcdefg'}]
            if level == 'EA': qOperators, qSymbols = [Fd(i)*F(b)*F(a), {'below': 'jklmno','above': 'cdefg'}]
            if level == 'EE': qOperators, qSymbols = [Fd(i)*Fd(j)*F(b)*F(a), {'below': 'klmno','above': 'cdefg'}]

            if 'ds' in sections:
                #doubles-singles block
                ds = evaluate_deltas(wicks(qOperators*Commutator(st,R[1]) , keep_only_fully_contracted=True))
                ds = substitute_dummies(ds, new_indices=True, pretty_indices= qSymbols)
                p = [PermutationOperator(i,j), PermutationOperator(a,b)]

                sectionSymbols['ds'] = simplify_index_permutations(ds, p)

            if 'dd' in sections:
                #doubles-doubles block'
                dd = evaluate_deltas(wicks(qOperators*Commutator(st,R[2]) , keep_only_fully_contracted=True))
                dd = substitute_dummies(dd, new_indices=True, pretty_indices= qSymbols)
                p = [PermutationOperator(i,j), PermutationOperator(a,b)]

                sectionSymbols['dd'] = simplify_index_permutations(dd, p)

        if eigenvector == 'L':

            #excitation amplitudes
            L = amplitudes('de-excitation', level)

            if level == 'IP': qOperators, qSymbols = [F(i), {'below': 'jklmno','above': 'abcdefg'}]
            if level == 'EA': qOperators, qSymbols = [Fd(a), {'below': 'ijklmno','above': 'bcdefg'}]
            if level == 'EE': qOperators, qSymbols = [Fd(a)*F(i), {'below': 'jklmno','above': 'bcdefg'}]

            #cc energy
            stEnergy = evaluate_deltas(wicks(st , keep_only_fully_contracted=True)) 

            if 'ss' in sections:
                #singles-singles block
                ss = evaluate_deltas(wicks(L[1]*(st-stEnergy)*qOperators , keep_only_fully_contracted=True))
                ss = substitute_dummies(ss, new_indices=True, pretty_indices= qSymbols)
                p = [PermutationOperator(i,j), PermutationOperator(a,b)]

                sectionSymbols['ss'] = simplify_index_permutations(ss, p)

            if 'sd' in sections:
                #singles-doubles block
                sd = evaluate_deltas(wicks((L[2]*st)*qOperators , keep_only_fully_contracted=True))
                sd = substitute_dummies(sd, new_indices=True, pretty_indices= qSymbols)
                p = [PermutationOperator(i,j), PermutationOperator(a,b)]

                sectionSymbols['sd'] = simplify_index_permutations(sd, p)

            if level == 'IP': qOperators, qSymbols = [Fd(a)*F(j)*F(a), {'below': 'klmno','above': 'bcdefg'}]
            if level == 'EA': qOperators, qSymbols = [Fd(a)*F(b)*F(i), {'below': 'jklmno','above': 'cdefg'}]
            if level == 'EE': qOperators, qSymbols = [Fd(a)*Fd(b)*F(j)*F(i), {'below': 'klmno','above': 'cdefg'}]

            if 'ds' in sections:
                #doubles-singles block
                ds = evaluate_deltas(wicks((L[1]*st)*qOperators , keep_only_fully_contracted=True))
                ds = substitute_dummies(ds, new_indices=True, pretty_indices= qSymbols)
                p = [PermutationOperator(i,j), PermutationOperator(a,b)]

                sectionSymbols['ds'] = simplify_index_permutations(ds, p)

            if 'dd' in sections:
                #doubles-doubles block'
                dd = evaluate_deltas(wicks(L[2]*(st-stEnergy)*qOperators , keep_only_fully_contracted=True))
                dd = substitute_dummies(dd, new_indices=True, pretty_indices= qSymbols)
                p = [PermutationOperator(i,j), PermutationOperator(a,b)]

                sectionSymbols['dd'] = simplify_index_permutations(dd, p)

        return sectionSymbols
 
    #return symbol strings
    if method == 'coupled-cluster':      return coupledCluster(type, level, sections)
    if method == 'response-density':     return responseDensity(type, level, sections)
    if method == 'cluster-lambda':       return clusterLambda(type, level, sections)
    if method == 'equation-of-motion':   return equationMotion(type, level, sections)


import sys

method, level = sys.argv[1:3]

typeDictionary = {'coupled-cluster': {'ccd': ['C','D'],'ccsd':['C','SD'],'ccsdt':['C','SDT'],'ccsd_t':['C','SDt'], \
                                     'cc2':['A','SD'],'cc3':['A','SDT'],'lccd':['L','D'],'lccsd':['L','SD']} , \
                  'response-density': {'cc':['D','CC'],'ee':['D','EE'],'ip':['D','IP'],'ea':['D','EA']}, \
                  'cluster-lambda':{'sd':['G','SD']}, \
                  'equation-of-motion': {'ip-r':['E','IPR'], 'ea-r':['E','EAR'], 'ee-r':['E','EER'], \
                                         'ip-l':['E','IPL'], 'ea-l':['E','EAL'], 'ee-l':['E','EEL']}}

sectionsDirectory = {'CD':'E+D','CSD':'E+S+D','CSDT':'E+S+D+T','CSDt':'E+S+D+T','A2':'E+S+D','A3':'E+S+D+T','LLD':'E+D','LLSD':'E+S+D', \
                     'DCC':'1+2','DEE':'1+2','DIP':'1+2','DEA':'1+2', \
                     'GSD':'S+D','EIPR':'ss+sd+ds+dd','EEAR':'ss+sd+ds+dd','EEER':'ss+sd+ds+dd', \
                     'EIPL':'ss+sd+ds+dd','EEAL':'ss+sd+ds+dd','EEEL':'ss+sd+ds+dd'}

if method in ['coupled-cluster', 'response-density', 'cluster-lambda', 'equation-of-motion']:
    try:
        type, level = typeDictionary[method][level]
        sections = sectionsDirectory[type+level]
    except:
        print('Key error [', level, '] not implemented')

    definition = [method , type, level, sections]
    symbols = kernel(method, type, level, sections)

    #save symbolic data in pickle HIGHEST_PROTOCOL format
    import pickle
    output = open('symbols.pkl', 'wb')
    pickle.dump(definition, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(symbols, output, 0)
    output.close()