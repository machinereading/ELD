import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pickle
from .utils import DataLoad
import time

device = torch.device('cuda')
batch_size = 128

class EntityEmbedding(nn.Module):
    
    def __init__(self, entity_dim):
        super(EntityEmbedding, self).__init__()
        
        self.entity_dim = entity_dim
        self.z = nn.Embedding(self.entity_dim, 300, max_norm=1.0).to(device)
        self.gamma = 0.1
      
    def forward(self, x, margins, expects):
        emb = self.z(x).view(margins.size()[0], 1, -1)
        h = torch.bmm(emb, margins)
        h = F.relu(self.gamma - h)
        J = torch.bmm(expects, h.view(margins.size()[0], -1, 1))
        J = J.view(-1, J.size()[0])
        return J


def train():
    D = DataLoad()
    ent2idx = list(D.ent2idx.items())
    entity_dim = len(ent2idx)
    model = EntityEmbedding(entity_dim)
    
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.3)

    checkpoint = torch.load('../model/pytorch/entity_embedding/ent_emb_checkpoint_09.ckpt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
    model.train()

    total_batch = (entity_dim // batch_size) + 1 
    for epoch in range(5):
        for i in range(total_batch):
            time1 = time.time()
            target = ent2idx[i*batch_size:(i+1)*batch_size]
            margins_list, expects_list = D.get_next_batch(target)
            
            data = torch.LongTensor([t[1] for t in target]).view(-1, len(target)).to(device)    
            margins_list = torch.FloatTensor(margins_list).view(len(target), 300, -1).to(device)
            expects_list = torch.FloatTensor(expects_list).view(len(target), -1, 100).to(device)

            output = model(data, margins_list, expects_list)

            loss = torch.norm(output)
            time2 = time.time()
            print("{}/{} step of {}/{} batch has loss {:.4f} in {:.5f} seconds".format(epoch + 10, 10, i + 1, total_batch, loss.item(), time2 - time1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, '../model/pytorch/entity_embedding/ent_emb_checkpoint_{}.ckpt'.format(str(epoch).zfill(2)))

def main():
    # train()
    D = DataLoad()
    ent2idx = list(D.ent2idx.items())
    entity_dim = len(ent2idx)
    model = EntityEmbedding(entity_dim)
    checkpoint = torch.load('../model/pytorch/entity_embedding/ent_emb_checkpoint_04.ckpt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    z = nn.Parameter(model.z.weight)
    z = z.cpu().detach().numpy()
    np.save('../data/wiki/entity_embedding.npy', z)

    
if __name__ == "__main__":
    main()
