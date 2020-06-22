import numpy as np
import torch
# label embedding using for rcv1
def label_embedding_rcv1():
    label_emb = np.zeros((102, 128))
    with open('label_embedding/rcv1_edge_128.emb', 'r') as f:
        for index, i in enumerate(f.readlines()):
            if index == 0:
                continue
            i = i.rstrip('\n')
            n = i.split(' ')[0]
            content = i.split(' ')[1:]
            label_emb[int(n)] = [float(value) for value in content]

    label_emb[-1:] = np.random.randn(1, 128)
    label_emb = torch.from_numpy(label_emb).float()
    return label_emb

# label embedding using for eurlex4k
def label_embedding_eurlex4k():
    label_emb = np.zeros((3714, 128))
    with open('label_embedding/eurlex4k_edge_128.emb', 'r') as f:
        for index, i in enumerate(f.readlines()):
            if index == 0:
                continue
            i = i.rstrip('\n')
            n = i.split(' ')[0]
            content = i.split(' ')[1:]
            label_emb[int(n)] = [float(value) for value in content]

    label_emb = torch.from_numpy(label_emb).float()
    return label_emb

# label embedding using for wiki10_31k
def label_embedding_wiki10_31k():
    label_emb = np.zeros((28139, 128))
    with open('label_embedding/wiki10_31k_edge_128.emb', 'r') as f:
        for index, i in enumerate(f.readlines()):
            if index == 0:
                continue
            i = i.rstrip('\n')
            n = i.split(' ')[0]
            content = i.split(' ')[1:]
            label_emb[int(n)] = [float(value) for value in content]

    label_emb = torch.from_numpy(label_emb).float()
    return label_emb

# label embedding using for ydata
def label_embedding_ydata():
    label_emb = np.zeros((414, 128))
    with open('label_embedding/ydata_edge_128.emb', 'r') as f:
        for index, i in enumerate(f.readlines()):
            if index == 0:
                continue
            i = i.rstrip('\n')
            n = i.split(' ')[0]
            content = i.split(' ')[1:]
            label_emb[int(n)] = [float(value) for value in content]
    # label_emb[-13:] = np.random.randn(13, 128)
    label_emb = torch.from_numpy(label_emb).float()
    return label_emb