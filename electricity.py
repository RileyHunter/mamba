import numpy as np

def get_data(decimals=3):
    data = [str(round(v, decimals)) for v in np.loadtxt('reads_81888.txt')]

    counts = {}
    max_count = 0
    max_token = None
    for i in data:
        counts[i] = (1 if i not in counts else counts[i]) + 1
        if counts[i] > max_count:
            max_count = counts[i]
            max_token = i
    
    print(f'Modal value is {max_token} ({counts[max_token]})')
    # Unique characters
    vals = sorted(list(set(data)))
    print(','.join(vals))
    vocab_size = len(vals)
    print(vocab_size)

    #Tokenizers
    ftoi = {ch:i for i,ch in enumerate(vals)}
    itof = {i:ch for i,ch in enumerate(vals)}
    encode = lambda xx: [ftoi[x] for x in xx]
    decode = lambda xx: ','.join([itof[x] for x in xx])
    return data, encode, decode, vocab_size