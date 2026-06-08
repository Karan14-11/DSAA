import torch
import pandas as pd


# Input CSV must have columns: u, v, t


df = pd.read_csv('./scripts/data/raw/sx-superuser-a2q.txt',header=None, sep=' ')
print(df.head())
torch.save({
'u': torch.tensor(df[0].values),
'v': torch.tensor(df[1].values),
't': torch.tensor(df[2].values)
}, './processed/train.pt')



data = torch.load("processed/superuser.pt")

u = data['u']
v = data['v']

nodes = torch.unique(torch.cat([u, v]))
node_map = {int(n): i for i, n in enumerate(nodes.tolist())}

u_new = torch.tensor([node_map[int(x)] for x in u])
v_new = torch.tensor([node_map[int(x)] for x in v])

torch.save({
    'u': u_new,
    'v': v_new,
    't': data['t'],
    'num_nodes': len(nodes)
}, "processed/superuser_compact.pt")