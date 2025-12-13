import os
import torch
from tqdm import tqdm
from cvrp.data import generate_cvrp_batch, route_length
from cvrp.model import CVRPModel

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # ====== 하이퍼파라미터 ======
    n_customers = 20
    batch_size  = 512
    steps       = 3000      # 더 올리면 성능 상승
    lr          = 1e-4

    model = CVRPModel(embed_dim=128, n_heads=8, n_layers=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)
    best = 1e9

    model.train()
    pbar = tqdm(range(1, steps + 1))
    for step in pbar:
        coords, demands, cap = generate_cvrp_batch(
            batch_size=batch_size, n_customers=n_customers,
            cap_range=(20, 50), demand_range=(1, 9),
            device=device
        )

        # sample policy
        path_s, logp = model(coords, demands, cap, decode_type="sampling")
        dist_s = route_length(coords, path_s)

        # baseline: greedy (같은 모델로 greedy 추론)
        with torch.no_grad():
            path_g, _ = model(coords, demands, cap, decode_type="greedy")
            dist_g = route_length(coords, path_g)

        # minimize expected distance: loss = (dist - baseline) * logp
        advantage = (dist_s - dist_g).detach()
        loss = (advantage * logp).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        mean_s = dist_s.mean().item()
        mean_g = dist_g.mean().item()
        pbar.set_description(f"step {step} | sample {mean_s:.4f} | greedy {mean_g:.4f} | loss {loss.item():.4f}")

        # checkpoint
        if step % 200 == 0:
            ckpt = {
                "model": model.state_dict(),
                "n_customers": n_customers
            }
            torch.save(ckpt, "checkpoints/last.pt")

        if mean_g < best:
            best = mean_g
            ckpt = {
                "model": model.state_dict(),
                "n_customers": n_customers
            }
            torch.save(ckpt, "checkpoints/best.pt")

    print("done. best greedy:", best)

if __name__ == "__main__":
    train()

