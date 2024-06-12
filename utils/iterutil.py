def iter_extract(iterator, idx):
    for i, v in enumerate(iterator):
        if i is idx : return v