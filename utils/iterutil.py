def iter_extract(iterator, idx):
    for i, v in enumerate(iterator):
        if i is idx : return v

def indx_extract(iterator, el):
    for i, v in enumerate(iterator):
        if v is el : return i