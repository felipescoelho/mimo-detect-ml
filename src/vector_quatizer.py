"""vector_quatizer.py

Script for the vector quatizer methods.

luizfelipe.coelho@smt.ufrj.br
Sep 25, 2024
"""


import torch


def mpgbp(x: torch.Tensor, M_max: int, P: int, device: str, epsilon=1e-6):
    """
    Method to perform the Matching Pursuits with Generalized Bit Plane
    (MPGBP) algortihm for vector quantization.
    
    Args
    ----
    x : torch.Tensor
        input vector
    M_max : int
        maximum number of SPT in vector
    P : int
        number of non zero elements in codeword components
    device : str
        device where we compute the tensors
    epsilon : float (default=1e-12)
        approximation threshold
    """


    x_hat = torch.zeros((len(x),), device=device)
    residue = x.detach().clone()
    M = 0  # Number of SPT
    while M < M_max:
        codeword = torch.zeros((len(x),), device=device)
        idx_sort = torch.argsort(-torch.abs(residue))
        for idx in range(P):
            idx = idx_sort[idx].item()
            codeword[idx] = torch.sign(residue[idx])
        codeword_normalized = codeword/torch.norm(codeword, p=2)
        ip = torch.dot(residue, codeword_normalized)/torch.norm(codeword, p=2)
        powers = -torch.ceil(torch.log2(3/(4*ip)))
        spt = 2**powers
        residue -= spt*codeword
        x_hat += spt*codeword
        M = spt_count(x_hat, device)
        if torch.norm(residue, p=2) <= epsilon:
            break
    
    return x_hat


def spt_count(x_hat: torch.Tensor, device: str):
    """
    Method to count the number of signed powers of two in approximation.

    Args
    ----
    x_hat : torch.Tensor
        approximated input vector.
    device : str
        device where we compute the tensors
    """


    K = len(x_hat)
    count = torch.zeros((K,), device=device)
    for idx in range(K):
        residue = torch.abs(x_hat[idx])
        while residue != 0:
            bitplane = torch.floor(torch.log2(residue))
            residue = torch.abs(residue - 2**bitplane)
            count[idx] += 1
    
    return torch.sum(count)
