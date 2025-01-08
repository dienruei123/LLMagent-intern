import torch.nn as nn
import torch

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
print(input, target)
output = loss(input, target)
print (output)
output.backward()

a = torch.tensor([0.4, 0.5, 0.6, 0.7])
b = torch.tensor([1, 0, 1, 1])
c = torch.abs(a-b)
print(c)
d = c[c < 0.5]
print(d)

a = torch.tensor([[2, 0, 2, 0], [3, 1, 3, 1]])
print(a.shape)
d = torch.max(a, dim=0)
print(d.values)