import torch
import torch.nn as nn

# if __name__ == '__main__':
#     ln = nn.LayerNorm(10)
#     print(dir(ln))

#     print(f"weight shape: {ln.weight.shape}")
#     print(f"bias shape: {ln.bias.shape}")

#     # NLP example
#     print('NLP')
#     N, T, embedding_dim = 5, 20, 10
#     ln = nn.LayerNorm(embedding_dim)
#     print(f"weight shape: {ln.weight.shape}")
#     print(f"bias shape: {ln.bias.shape}")

#     # CV example
#     print('CV')
#     N, C, H, W = 5, 3, 10, 10
#     ln = nn.LayerNorm([C, H, W])
#     print(f"weight shape: {ln.weight.shape}")
#     print(f"bias shape: {ln.bias.shape}")

#     # CV BN example
#     print('CV BN')
#     N, C, H, W = 5, 3, 10, 10
#     bn = nn.BatchNorm2d(C)
#     print(f"weight shape: {bn.weight.shape}")
#     print(f"bias shape: {bn.bias.shape}")

if __name__ == '__main__':
    N, T, embedding_dim = 5, 20, 10
    ln = nn.LayerNorm(embedding_dim)
    x = torch.randn((N, T, embedding_dim))
    y = ln(x)
    print(x.shape)
    print(y.shape)

    mean = x[0][0].mean()
    var = torch.sum(torch.pow(x[0][0] - mean, 2)) / embedding_dim
    print(mean, var)
    y_hat = (x[0][0] - mean) / torch.sqrt(var + 1e-05)
    print(y[0][0])
    print(y_hat)

    # print(ln.weight)
    # print(ln.bias)