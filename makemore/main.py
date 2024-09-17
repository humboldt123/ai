import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

# only contains pokemon up to gen V, none of the fake new ones 🥱
pokemon = open('pokemon.txt', 'r').read().splitlines()

# its é,  , ', ., -, 2 and ♂ if you were wondering (total: 33 chars long)
chars = sorted(list(set(''.join(pokemon))))

str_to_token = {s : i+1 for i,s in enumerate(chars)} # start at 1 to reserve [0] for ('→')
str_to_token['→'] = 0 # token for starting char is 0

token_to_str = {i : s for s,i in str_to_token.items()}

# +1 for the special <start> and <end> tokens :3
# STEP 1: Create an (empty) map of each char pairing and its likelihood
N = torch.zeros((len(chars) + 1, len(chars) + 1), dtype=torch.int32)

# training set of bigrams
xs, ys = [], []
for p in pokemon:
    chs = ['→'] + list(p) + ['→']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = str_to_token[ch1]
        ix2 = str_to_token[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)

num = xs.nelement()
print('number of examples: ', num)

# initialize the 'network'
g = torch.Generator().manual_seed(32767)
W = torch.randn((len(chars) + 1, len(chars) + 1), generator=g, requires_grad=True)  # awesome mega matrix of weights
# 👆 worth noting `requires_grad` is False by default, but it needs to be true to support backpropogation

for k in range(999):
    xenc = F.one_hot(xs, num_classes=len(chars) + 1).float() # embeddings kinda
    logits = xenc @ W # dot product !!!
    counts = logits.exp() # counts (same as N) (exp stands for exponentiation)

    probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()

    print(loss)

    # backward pass
    W.grad = None # set to zero the gradient
    loss.backward()

    # update
    W.data += -50 * W.grad


for i in range(5):

    out = []
    ix = 0
    while True:

        xenc = F.one_hot(torch.tensor([ix]), num_classes=(len(chars) + 1)).float()
        logits = xenc @ W # predict log-counts
        counts = logits.exp() # counts, equivalent to N
        p = counts / counts.sum(1, keepdims=True) # probabilities for next character
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if ix == 0:
            break
        out.append(token_to_str[ix])
    print(''.join(out))