import numpy as np

base_dict = {'A':0,'C':1,'G':2,'U':3}
kidera = np.array([
    [0.61,-1.52,-0.23,0.39,-0.17,0.59,0.02,-0.44,-1.03,-1.12],
    [1.10,0.59,1.58,0.84,0.26,-0.44,-0.06,0.10,-0.45,0.93],
    [0.46,1.02,-0.25,0.36,1.16,0.48,-1.12,-0.36,-0.31,-0.02],
    [0.70,0.15,1.10,-0.33,-0.63,1.19,0.27,0.10,-1.59,-1.19],
])

def encode_sequence(seq, max_len=101):
    one_hot = np.zeros((max_len,4), dtype=np.float32)
    prop = np.zeros((max_len,10), dtype=np.float32)
    for i,base in enumerate(seq[:max_len]):
        idx = base_dict.get(base, None)
        if idx is not None:
            one_hot[i,idx] = 1.
            prop[i] = kidera[idx]
    return np.concatenate([one_hot, prop], axis=1)