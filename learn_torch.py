import torch
import torch.nn as nn
from torch.autograd import Variable

output = Variable(torch.FloatTensor([0,0,0,1])).view(1, -1)
target = Variable(torch.LongTensor([3]))
print(output.size())
criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)
print(loss)