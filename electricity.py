import numpy as np

def get_data():
    data = np.loadtxt('reads_81888.txt')

    # Unique characters
    vals = sorted(list(set(data)))
    print(','.join(str(v) for v in vals))
    vocab_size = len(vals)
    print(vocab_size)

    #Tokenizers
    encode = lambda xx: xx
    decode = lambda xx: ','.join(str(x) for x in xx)
    return data, encode, decode, vocab_size