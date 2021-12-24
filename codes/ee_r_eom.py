def ee_r_eom(fs, g, occupancy, t1=None, t2=None):

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
                    hss[row,col] += fs[a,c]*kd[i,k]
                    hss[row,col] += g[a,k,i,c]
                    hss[row,col] += -fs[k,i]*kd[a,c]
                    for e in range(nsocc, nso):
                        hss[row,col] += -fs[k,e]*t1[e,i]*kd[a,c]
                        hss[row,col] += -g[a,k,c,e]*t1[e,i]
                        for m in range(nsocc):
                            hss[row,col] += g[a,m,c,e]*t1[e,m]*kd[i,k]
                            hss[row,col] += g[k,m,c,e]*t2[a,e,i,m]
                            hss[row,col] += -g[k,m,i,e]*t1[e,m]*kd[a,c]
                            hss[row,col] += -g[k,m,c,e]*t1[a,m]*t1[e,i]
                            for f in range(nsocc, nso):
                                hss[row,col] += -0.500*g[k,m,e,f]*t2[e,f,i,m]*kd[a,c]
                                hss[row,col] += g[k,m,e,f]*t1[e,m]*t1[f,i]*kd[a,c]
                            for n in range(nsocc):
                                hss[row,col] += -0.500*g[m,n,c,e]*t2[a,e,m,n]*kd[i,k]
                                hss[row,col] += g[m,n,c,e]*t1[a,n]*t1[e,m]*kd[i,k]
                    for m in range(nsocc):
                        hss[row,col] += g[k,m,i,c]*t1[a,m]
                        hss[row,col] += -fs[m,c]*t1[a,m]*kd[i,k]

                    col += 1
            row += 1


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
                            hsd[row,col] += fs[k,c]*kd[a,d]*kd[i,l]
                            hsd[row,col] += 0.50*g[a,k,d,c]*kd[i,l]
                            hsd[row,col] += -0.500*g[l,k,i,c]*kd[a,d]
                            for e in range(nsocc, nso):
                                hsd[row,col] += 0.50*g[l,k,c,e]*t1[e,i]*kd[a,d]
                                for m in range(nsocc):
                                    hsd[row,col] += g[k,m,c,e]*t1[e,m]*kd[a,d]*kd[i,l]
                            for m in range(nsocc):
                                hsd[row,col] += 0.50*g[k,m,d,c]*t1[a,m]*kd[i,l]

                            col += 1
            row += 1


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
                            hds[row,col] += g[a,b,i,c]*kd[j,k]-g[a,b,j,c]*kd[i,k]
                            hds[row,col] += -g[a,k,i,j]*kd[b,c]+g[b,k,i,j]*kd[a,c]
                            for e in range(nsocc, nso):
                                hds[row,col] += fs[k,e]*t2[b,e,i,j]*kd[a,c]-fs[k,e]*t2[a,e,i,j]*kd[b,c]
                                hds[row,col] += g[a,b,c,e]*t1[e,j]*kd[i,k]-g[a,b,c,e]*t1[e,i]*kd[j,k]
                                hds[row,col] += g[a,k,c,e]*t2[b,e,i,j]-g[b,k,c,e]*t2[a,e,i,j]
                                hds[row,col] += -g[a,k,i,e]*t1[e,j]*kd[b,c]+g[b,k,i,e]*t1[e,j]*kd[a,c]  \
                                                +g[a,k,j,e]*t1[e,i]*kd[b,c]-g[b,k,j,e]*t1[e,i]*kd[a,c]
                                for f in range(nsocc, nso):
                                    hds[row,col] += -0.500*g[a,k,e,f]*t2[e,f,i,j]*kd[b,c]+0.500*g[b,k,e,f]*t2[e,f,i,j]*kd[a,c]
                                    hds[row,col] += -g[a,k,e,f]*t1[e,i]*t1[f,j]*kd[b,c]+g[b,k,e,f]*t1[e,i]*t1[f,j]*kd[a,c]
                                for m in range(nsocc):
                                    hds[row,col] += g[a,m,c,e]*t2[b,e,j,m]*kd[i,k]-g[b,m,c,e]*t2[a,e,j,m]*kd[i,k]  \
                                                    -g[a,m,c,e]*t2[b,e,i,m]*kd[j,k]+g[b,m,c,e]*t2[a,e,i,m]*kd[j,k]
                                    hds[row,col] += g[k,m,c,e]*t1[a,m]*t2[b,e,i,j]-g[k,m,c,e]*t1[b,m]*t2[a,e,i,j]
                                    hds[row,col] += g[k,m,c,e]*t1[e,i]*t2[a,b,j,m]-g[k,m,c,e]*t1[e,j]*t2[a,b,i,m]
                                    hds[row,col] += -g[k,m,i,e]*t2[b,e,j,m]*kd[a,c]+g[k,m,i,e]*t2[a,e,j,m]*kd[b,c]  \
                                                    +g[k,m,j,e]*t2[b,e,i,m]*kd[a,c]-g[k,m,j,e]*t2[a,e,i,m]*kd[b,c]
                                    hds[row,col] += g[k,m,i,e]*t1[b,m]*t1[e,j]*kd[a,c]-g[k,m,i,e]*t1[a,m]*t1[e,j]*kd[b,c]  \
                                                    -g[k,m,j,e]*t1[b,m]*t1[e,i]*kd[a,c]+g[k,m,j,e]*t1[a,m]*t1[e,i]*kd[b,c]
                                    hds[row,col] += -g[a,m,c,e]*t1[b,m]*t1[e,j]*kd[i,k]+g[b,m,c,e]*t1[a,m]*t1[e,j]*kd[i,k]  \
                                                    +g[a,m,c,e]*t1[b,m]*t1[e,i]*kd[j,k]-g[b,m,c,e]*t1[a,m]*t1[e,i]*kd[j,k]
                                    for f in range(nsocc, nso):
                                        hds[row,col] += 0.50*g[k,m,e,f]*t1[b,m]*t2[e,f,i,j]*kd[a,c]-0.50*g[k,m,e,f]*t1[a,m]*t2[e,f,i,j]*kd[b,c]
                                        hds[row,col] += -g[k,m,e,f]*t1[e,m]*t2[b,f,i,j]*kd[a,c]+g[k,m,e,f]*t1[e,m]*t2[a,f,i,j]*kd[b,c]
                                        hds[row,col] += g[k,m,e,f]*t1[b,m]*t1[e,i]*t1[f,j]*kd[a,c]-g[k,m,e,f]*t1[a,m]*t1[e,i]*t1[f,j]*kd[b,c]
                                        hds[row,col] += -g[k,m,e,f]*t1[e,i]*t2[b,f,j,m]*kd[a,c]+g[k,m,e,f]*t1[e,i]*t2[a,f,j,m]*kd[b,c]  \
                                                        +g[k,m,e,f]*t1[e,j]*t2[b,f,i,m]*kd[a,c]-g[k,m,e,f]*t1[e,j]*t2[a,f,i,m]*kd[b,c]
                                    for n in range(nsocc):
                                        hds[row,col] += 0.50*g[m,n,c,e]*t1[e,j]*t2[a,b,m,n]*kd[i,k]-0.50*g[m,n,c,e]*t1[e,i]*t2[a,b,m,n]*kd[j,k]
                                        hds[row,col] += -g[m,n,c,e]*t1[e,m]*t2[a,b,j,n]*kd[i,k]+g[m,n,c,e]*t1[e,m]*t2[a,b,i,n]*kd[j,k]
                                        hds[row,col] += g[m,n,c,e]*t1[a,m]*t1[b,n]*t1[e,j]*kd[i,k]-g[m,n,c,e]*t1[a,m]*t1[b,n]*t1[e,i]*kd[j,k]
                                        hds[row,col] += -g[m,n,c,e]*t1[a,m]*t2[b,e,j,n]*kd[i,k]+g[m,n,c,e]*t1[b,m]*t2[a,e,j,n]*kd[i,k]  \
                                                        +g[m,n,c,e]*t1[a,m]*t2[b,e,i,n]*kd[j,k]-g[m,n,c,e]*t1[b,m]*t2[a,e,i,n]*kd[j,k]
                            for m in range(nsocc):
                                hds[row,col] += fs[m,c]*t2[a,b,j,m]*kd[i,k]-fs[m,c]*t2[a,b,i,m]*kd[j,k]
                                hds[row,col] += g[k,m,i,j]*t1[b,m]*kd[a,c]-g[k,m,i,j]*t1[a,m]*kd[b,c]
                                hds[row,col] += -g[k,m,i,c]*t2[a,b,j,m]+g[k,m,j,c]*t2[a,b,i,m]
                                hds[row,col] += -g[a,m,i,c]*t1[b,m]*kd[j,k]+g[b,m,i,c]*t1[a,m]*kd[j,k]  \
                                                +g[a,m,j,c]*t1[b,m]*kd[i,k]-g[b,m,j,c]*t1[a,m]*kd[i,k]
                                for n in range(nsocc):
                                    hds[row,col] += 0.50*g[m,n,i,c]*t2[a,b,m,n]*kd[j,k]-0.50*g[m,n,j,c]*t2[a,b,m,n]*kd[i,k]
                                    hds[row,col] += g[m,n,i,c]*t1[a,m]*t1[b,n]*kd[j,k]-g[m,n,j,c]*t1[a,m]*t1[b,n]*kd[i,k]

                            col += 1
                    row += 1


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
                                    hdd[row,col] += 0.50*g[a,b,d,c]*kd[i,l]*kd[j,k]
                                    hdd[row,col] += 0.50*g[l,k,i,j]*kd[a,d]*kd[b,c]
                                    hdd[row,col] += fs[k,i]*kd[a,d]*kd[b,c]*kd[j,l]-fs[k,j]*kd[a,d]*kd[b,c]*kd[i,l]
                                    hdd[row,col] += -fs[a,c]*kd[b,d]*kd[i,l]*kd[j,k]+fs[b,c]*kd[a,d]*kd[i,l]*kd[j,k]
                                    hdd[row,col] += g[a,k,i,c]*kd[b,d]*kd[j,l]-g[b,k,i,c]*kd[a,d]*kd[j,l]  \
                                                    -g[a,k,j,c]*kd[b,d]*kd[i,l]+g[b,k,j,c]*kd[a,d]*kd[i,l]
                                    for e in range(nsocc, nso):
                                        hdd[row,col] += 0.50*g[l,k,i,e]*t1[e,j]*kd[a,d]*kd[b,c]-0.50*g[l,k,j,e]*t1[e,i]*kd[a,d]*kd[b,c]
                                        hdd[row,col] += -fs[k,e]*t1[e,j]*kd[a,d]*kd[b,c]*kd[i,l]+fs[k,e]*t1[e,i]*kd[a,d]*kd[b,c]*kd[j,l]
                                        hdd[row,col] += -0.500*g[l,k,c,e]*t2[b,e,i,j]*kd[a,d]+0.500*g[l,k,c,e]*t2[a,e,i,j]*kd[b,d]
                                        hdd[row,col] += g[a,k,c,e]*t1[e,j]*kd[b,d]*kd[i,l]-g[b,k,c,e]*t1[e,j]*kd[a,d]*kd[i,l]  \
                                                        -g[a,k,c,e]*t1[e,i]*kd[b,d]*kd[j,l]+g[b,k,c,e]*t1[e,i]*kd[a,d]*kd[j,l]
                                        for f in range(nsocc, nso):
                                            hdd[row,col] += 0.25*g[l,k,e,f]*t2[e,f,i,j]*kd[a,d]*kd[b,c]
                                            hdd[row,col] += 0.50*g[l,k,e,f]*t1[e,i]*t1[f,j]*kd[a,d]*kd[b,c]
                                        for m in range(nsocc):
                                            hdd[row,col] += g[k,m,i,e]*t1[e,m]*kd[a,d]*kd[b,c]*kd[j,l]-g[k,m,j,e]*t1[e,m]*kd[a,d]*kd[b,c]*kd[i,l]
                                            hdd[row,col] += -g[a,m,c,e]*t1[e,m]*kd[b,d]*kd[i,l]*kd[j,k]+g[b,m,c,e]*t1[e,m]*kd[a,d]*kd[i,l]*kd[j,k]
                                            hdd[row,col] += g[k,m,c,e]*t2[b,e,j,m]*kd[a,d]*kd[i,l]-g[k,m,c,e]*t2[a,e,j,m]*kd[b,d]*kd[i,l]  \
                                                            -g[k,m,c,e]*t2[b,e,i,m]*kd[a,d]*kd[j,l]+g[k,m,c,e]*t2[a,e,i,m]*kd[b,d]*kd[j,l]
                                            hdd[row,col] += -g[k,m,c,e]*t1[b,m]*t1[e,j]*kd[a,d]*kd[i,l]+g[k,m,c,e]*t1[a,m]*t1[e,j]*kd[b,d]*kd[i,l]  \
                                                            +g[k,m,c,e]*t1[b,m]*t1[e,i]*kd[a,d]*kd[j,l]-g[k,m,c,e]*t1[a,m]*t1[e,i]*kd[b,d]*kd[j,l]
                                            for f in range(nsocc, nso):
                                                hdd[row,col] += -0.500*g[k,m,e,f]*t2[e,f,j,m]*kd[a,d]*kd[b,c]*kd[i,l]+0.500*g[k,m,e,f]*t2[e,f,i,m]*kd[a,d]*kd[b,c]*kd[j,l]
                                                hdd[row,col] += g[k,m,e,f]*t1[e,m]*t1[f,j]*kd[a,d]*kd[b,c]*kd[i,l]-g[k,m,e,f]*t1[e,m]*t1[f,i]*kd[a,d]*kd[b,c]*kd[j,l]
                                            for n in range(nsocc):
                                                hdd[row,col] += -0.500*g[m,n,c,e]*t2[b,e,m,n]*kd[a,d]*kd[i,l]*kd[j,k]+0.500*g[m,n,c,e]*t2[a,e,m,n]*kd[b,d]*kd[i,l]*kd[j,k]
                                                hdd[row,col] += g[m,n,c,e]*t1[b,n]*t1[e,m]*kd[a,d]*kd[i,l]*kd[j,k]-g[m,n,c,e]*t1[a,n]*t1[e,m]*kd[b,d]*kd[i,l]*kd[j,k]
                                    for m in range(nsocc):
                                        hdd[row,col] += -fs[m,c]*t1[b,m]*kd[a,d]*kd[i,l]*kd[j,k]+fs[m,c]*t1[a,m]*kd[b,d]*kd[i,l]*kd[j,k]
                                        hdd[row,col] += -0.500*g[a,m,d,c]*t1[b,m]*kd[i,l]*kd[j,k]+0.500*g[b,m,d,c]*t1[a,m]*kd[i,l]*kd[j,k]
                                        hdd[row,col] += -0.500*g[k,m,d,c]*t2[a,b,j,m]*kd[i,l]+0.500*g[k,m,d,c]*t2[a,b,i,m]*kd[j,l]
                                        hdd[row,col] += -g[k,m,i,c]*t1[b,m]*kd[a,d]*kd[j,l]+g[k,m,i,c]*t1[a,m]*kd[b,d]*kd[j,l]  \
                                                        +g[k,m,j,c]*t1[b,m]*kd[a,d]*kd[i,l]-g[k,m,j,c]*t1[a,m]*kd[b,d]*kd[i,l]
                                        for n in range(nsocc):
                                            hdd[row,col] += 0.25*g[m,n,d,c]*t2[a,b,m,n]*kd[i,l]*kd[j,k]
                                            hdd[row,col] += 0.50*g[m,n,d,c]*t1[a,m]*t1[b,n]*kd[i,l]*kd[j,k]

                                    col += 1
                    row += 1


    return hss, hsd, hds, hdd