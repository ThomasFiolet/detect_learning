def iter_extract(iterator, idx):
    for i, v in enumerate(iterator):
        #Need or for strings and int
        if i is idx or i == idx : return v

def indx_extract(iterator, el):
    for i, v in enumerate(iterator):
        #Need or for strings and int
        if v is el or v == el : return i