import random
import torch
import torch.nn.functional as F

pokemon = open("pokemon.txt", "r").read().splitlines()

# == build vocabulary ==
chars = sorted(list(set(''.join(pokemon))))
string_token_map = { s:i+1 for i,s in enumerate(chars)}
string_token_map['#'] = 0
token_string_map = { i:s for s,i in string_token_map.items()}

# == build datasets ==
BLOCK_SIZE = 3 # context length: how many chrs it takes to predict next one
SEED = 1337

def build_dataset(words):
    X, Y = [], []
    for word in words:
        context = [0] * BLOCK_SIZE
        for character in word + '#': # word + end_token
            t = string_token_map[character] # t(oken)
            X.append(context)
            Y.append(t)

            context = context[1:] + [t] # crop and append (its a window)
    
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

random.seed(SEED)
random.shuffle(pokemon)

# split into training, dev, and testing
n1, n2 = int(0.8*len(pokemon)), int(0.9*len(pokemon))
X_training, Y_training = build_dataset(pokemon[:n1])
X_dev, Y_dev = build_dataset(pokemon[n1:n2])
X_test, X_test = build_dataset(pokemon[n2:])

# == make the model ==
g = torch.Generator().manual_seed(SEED)
C = torch.randn((len(chars)+1, 10), generator=g)
W1 = torch.randn((30, 200), generator=g)
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, len(chars)+1), generator=g)
b2 = torch.randn(len(chars)+1, generator=g)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    # 4 some reason its False by default,,, idk y
    p.requires_grad = True

print(f"number of parameters in total: {sum(p.nelement() for p in parameters)}")

