
def rothess(h):
    hr=np.zeros_like(h)
    ridx={0:0,1:1,2:3,3:4,4:2}
    for i in range(5):
        for j in range(5):
            hr[i,j]=r.apply(r.apply(h[ridx[i],ridx[j]]).T).T
    return hr
def rotgrad(g):
    b=r.apply(g)
    b[[2,3,4]]=b[[3,4,2]]
    return b 
