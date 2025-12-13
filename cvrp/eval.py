import torch
from cvrp.data import generate_cvrp_batch, route_length
from cvrp.model import CVRPModel

@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load("checkpoints/best.pt", map_location=device)

    n_customers = ckpt["n_customers"]
    model = CVRPModel(embed_dim=128, n_heads=8, n_layers=3).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    coords, demands, cap = generate_cvrp_batch(1024, n_customers, device=device)
    path, _ = model(coords, demands, cap, decode_type="greedy")
    dist = route_length(coords, path).mean().item()
    print("mean greedy distance:", dist)

if __name__ == "__main__":
    main()
