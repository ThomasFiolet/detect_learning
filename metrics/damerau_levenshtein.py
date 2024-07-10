import numpy as np

def damerau_levenshtein(chain0, chain1):

    len0 = len(chain0)
    len1 = len(chain1)

    d = np.zeros((len0 + 1, len1 + 1))

    for i in range(0, len0 + 1):
        d[i,0] = i
    for j in range(0, len1 + 1):
        d[0,j] = j

    for i in range(0, len0):
        for j in range(0, len1):

            
            if chain0[i] == chain1[j]: subcost = 0
            else: subcost = 1

            d[i+1,j+1] = min(d[i,j+1]+1, d[i+1,j]+1, d[i,j]+subcost)

            if i>1 and j>1 and chain0[i] == chain1[j-1] and chain0[i-1] == chain1[j]:

                
                d[i+1,j+1] = min(d[i+1,j+1], d[i-1,j-1] + subcost)

    return d[len0,len1]
