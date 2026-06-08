import torch
import matplotlib.pyplot as plt

# Load first 50 snapshots of Reddit
reddit_counts = []
for i in range(50):
    data = torch.load(f"./processed/reddit/snap_{i}.pt")
    reddit_counts.append(data.edge_index.size(1) // 2) # Divide by 2 (undirected)

# Load first 50 snapshots of Citation
citation_counts = []
for i in range(50):
    data = torch.load(f"./processed/citation/snap_{i}.pt")
    citation_counts.append(data.edge_index.size(1) // 2)

# Plot
plt.figure(figsize=(10, 5))
# plt.plot(reddit_counts, label="Reddit (Social)", color="red")
plt.plot(citation_counts, label="Citation (HepPh)", color="blue")
plt.title("Edge Count Evolution: Windowed vs Cumulative")
plt.xlabel("Time Step")
plt.ylabel("Number of Edges")
plt.legend()
plt.savefig("data_verification_citiation.png")
print("Verification plot saved to data_verification.png")