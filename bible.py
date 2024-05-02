def get_data():
    with open("shakespeare.txt", "r") as f:
      text = f.read()

    # Unique characters
    chars = sorted(list(set(text)))
    print(''.join(chars))
    vocab_size = len(chars)
    print(vocab_size)

    #Tokenizers
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    encode = lambda xx: [stoi[x] for x in xx]
    decode = lambda xx: ''.join([itos[x] for x in xx])
    encode("Hello!")
    print(decode(encode("Hello!")))
    return text, encode, decode, vocab_size