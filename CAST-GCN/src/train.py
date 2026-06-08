import torch
from tqdm import tqdm
import yaml


from loss.regulizer_loss import temporal_smoothness
from data.negative_sampling import NegativeSampler
from data.dataset import TemporalEdgeDataset
from model.tgn import TGN
from loss.link_loss import LinkPredictionLoss







def train(edge_file, num_nodes, config):
    device = torch.device(config['training']['device'])


    dataset = TemporalEdgeDataset(edge_file)
    neg_sampler = NegativeSampler(num_nodes)


    model = TGN(num_nodes, config['model']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    criterion = LinkPredictionLoss()


    batch_size = config['training']['batch_size']
    lambda_smooth = config['regularization']['lambda_smooth']


    for epoch in range(config['training']['epochs']):
        total_loss = 0.0
        pbar = tqdm(range(0, len(dataset), batch_size), desc=f"Epoch {epoch}")


        for start in pbar:
            u, v, t = dataset.get_batch(start, batch_size)
            u, v, t = u.to(device), v.to(device), t.to(device)


            h_u, h_v, msg_u, msg_v = model(u, v, t)

            z_u = model.encoder(h_u)
            z_v = model.encoder(h_v)

            neg_v = neg_sampler.sample(len(u)).to(device)
            z_neg = model.encoder(model.memory.get(neg_v))

            # normalize (prevents collapse)
            z_u = torch.nn.functional.normalize(z_u, dim=1)
            z_v = torch.nn.functional.normalize(z_v, dim=1)
            z_neg = torch.nn.functional.normalize(z_neg, dim=1)

            link_loss = criterion(z_u, z_v, z_neg)

            smooth_loss = temporal_smoothness(h_u, h_u + msg_u)

            loss = link_loss + lambda_smooth * smooth_loss
            loss.backward()
            optimizer.step()

            model.memory.update(u, 0.9 * h_u + 0.1 * msg_u, t.max())
            model.memory.update(v, 0.9 * h_v + 0.1 * msg_v, t.max())
            



            


        print(f"Epoch {epoch} | Avg Loss: {total_loss / len(dataset):.6f}")




if __name__ == '__main__':
    with open('configs/default.yaml') as f:
        config = yaml.safe_load(f)
    
    print(config)


    train(
    edge_file='processed/askubuntu.pt',
    num_nodes=137517,
    config=config
    )