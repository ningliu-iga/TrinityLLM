import debugpy
import socket
import glob
import pandas as pd
from typing import List
from rdkit import Chem
import torch


def getipaddress():
    return socket.gethostbyname(socket.getfqdn())


def debug():
    print("Waiting for debugger to connect")
    if (
        socket.getfqdn().startswith("dcc")
        or socket.getfqdn().startswith("mol")
        or socket.getfqdn().startswith("ccc")
    ):
        debugpy.listen(address=(getipaddress(), 3000))
        debugpy.wait_for_client()
    debugpy.breakpoint()


class ListDataset:
    def __init__(self, seqs):
        self.seqs = seqs

    def __getitem__(self, index):
        return self.seqs[index]

    def __len__(self):
        return len(self.seqs)


def transform_single_embedding_to_multiple(smiles_z_map):
    """Transforms an embedding map of the format smi->embedding to
    smi-> {"canonical_embeddings":embedding}. This function exists
    as a compatibility layer

    Args:
        smiles_z_map ([type]): [description]
    """
    retval = dict()
    for key in smiles_z_map:
        retval[key] = {"canonical_embeddings": smiles_z_map[key]}
    return retval


def normalize_smiles(smi, canonical, isomeric):
    normalized = Chem.MolToSmiles(
        Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
    )
    return normalized


def get_all_proteins(affinity_dir: str):
    files = glob.glob(affinity_dir + "/*.csv")
    all_proteins = []
    print(files)
    for file in files:
        df = pd.read_csv(file)
        all_proteins.extend(df["protein"].tolist())
    return set(all_proteins)


def append_to_file(filename, line):
    with open(filename, "a") as f:
        f.write(line + "\n")


def write_to_file(filename, line):
    with open(filename, "w") as f:
        f.write(line + "\n")


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)