from pyscf.gto import M
from pyscf.scf import RHF
import numpy as np
import basis_set_exchange as bse
import re
import sys

atoms="G,H,He,Li,Be,B,C,N,O,F,Ne,Na,Mg,Al"
atoms=atoms.split(',')
pcxabse={('Li', 'Be'): 0.000597429639372038,
 ('Li', 'B'): 0.007357073732691788,
 ('Li', 'C'): 0.01904677594460935,
 ('Li', 'N'): 0.042939880749266024,
 ('Li', 'O'): 0.10165371325324202,
 ('Li', 'F'): 0.23516350418471177,
 ('Li', 'Ne'): 0.5031076757715311,
 ('Be', 'Li'): 0.005854676279352766,
 ('Be', 'B'): 0.031150052578613696,
 ('Be', 'C'): 0.036659868407134866,
 ('Be', 'N'): 0.030695583033050866,
 ('Be', 'O'): 0.04401342345613557,
 ('Be', 'F'): 0.08149265277130269,
 ('Be', 'Ne'): 0.16501763564878047,
 ('B', 'Li'): 0.08226335998251866,
 ('B', 'Be'): 0.29063899911544944,
 ('B', 'C'): 0.0004283153812849605,
 ('B', 'N'): 0.0034839360923086815,
 ('B', 'O'): 0.010612357946030215,
 ('B', 'F'): 0.02432775370101581,
 ('B', 'Ne'): 0.049767861647552536,
 ('C', 'Li'): 0.10276570874058866,
 ('C', 'Be'): 0.3057919604366326,
 ('C', 'B'): 0.004704187808904692,
 ('C', 'N'): 0.0004869952092718677,
 ('C', 'O'): 0.003256895259795556,
 ('C', 'F'): 0.009786823062597705,
 ('C', 'Ne'): 0.02247150838257994,
 ('N', 'Li'): 0.1281082199350223,
 ('N', 'Be'): 0.32848909085522315,
 ('N', 'B'): 0.023096966385320883,
 ('N', 'C'): 0.004133047217614205,
 ('N', 'O'): 0.0005674468915941588,
 ('N', 'F'): 0.0034924036055485885,
 ('N', 'Ne'): 0.010066474631855726,
 ('O', 'Li'): 0.15822889521008587,
 ('O', 'Be'): 0.3590144940877078,
 ('O', 'B'): 0.5206220939767476,
 ('O', 'C'): 0.01888254727820282,
 ('O', 'N'): 0.003269512295844379,
 ('O', 'F'): 0.0007991081521083743,
 ('O', 'Ne'): 0.00403529998192198,
 ('F', 'Li'): 0.20353606389520706,
 ('F', 'Be'): 0.4114599289680232,
 ('F', 'B'): 0.552025228953962,
 ('F', 'C'): 0.05504855401603237,
 ('F', 'N'): 0.015786319118880954,
 ('F', 'O'): 0.0029667638878976277,
 ('F', 'Ne'): 0.000901918755943143,
 ('Ne', 'Li'): 0.2624751236425453,
 ('Ne', 'Be'): 0.48816657598033864,
 ('Ne', 'B'): 0.6064508152824715,
 ('Ne', 'C'): 0.8045156955800863,
 ('Ne', 'N'): 0.04724663703657228,
 ('Ne', 'O'): 0.015091877997491565,
 ('Ne', 'F'): 0.0029921855481092052}

def abse_atom(ref,targ,bs="pcX-2"):
    if ref==targ: return 0
    if bs=="pcX-2": return -pcxabse[(ref,targ)]
    spin=(atoms.index(targ))%2
    T=M(atom='{} 0 0 0'.format(targ),spin=spin,\
             basis=bse.get_basis(bs,fmt="nwchem",elements=[atoms.index(targ)]),verbose=0)
    TatR=M( atom='{} 0 0 0'.format(targ),spin=spin,\
             basis=bse.get_basis(bs,fmt="nwchem",elements=[atoms.index(ref)]),verbose=0)
    eT=RHF(T).kernel()
    mf=RHF(TatR)
    eTatR=mf.scf(dm0=mf.init_guess_by_1e())
    return eT-eTatR

def absec(ref,targ,bs="pcX-2"):
    reflist= re.sub( r"([A-Z])", r" \1", ref).split()
    targlist= re.sub( r"([A-Z])", r" \1", targ).split()
    if len(reflist) != len(targlist):
        print(reflist,targlist,"reference and target lengths do not match!", sys.exc_info()[0])
        raise 
    bsae=0
    for i in range(len(reflist)):
        bsae+=abse_atom(reflist[i],targlist[i],bs)
    return (bsae)