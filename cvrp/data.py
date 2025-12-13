import torch

def generate_cvrp_batch(batch_size: int,
                        n_customers: int,
                        cap_range=(20, 50),
                        demand_range=(1, 9),
                        device="cpu"):
    """
    Returns:
      coords: (B, N+1, 2)  depot=0 + customers
      demands: (B, N+1)    raw demand (depot demand=0)
      capacity: (B,)       raw capacity
    """
    B = batch_size
    N = n_customers + 1

    # depot + customers coordinates in [0,1]
    coords = torch.rand(B, N, 2, device=device)
    # depot 위치를 조금 안정적으로 (원하면 주석 처리 가능)
    coords[:, 0, :] = 0.5

    capacity = torch.randint(cap_range[0], cap_range[1] + 1, (B,), device=device).float()

    # demands (depot = 0)
    demands = torch.zeros(B, N, device=device)
    # 각 배치별 capacity를 고려해서 demand가 capacity를 넘지 않게
    # demand_range 최대가 capacity보다 큰 경우를 방지
    max_d = min(demand_range[1], cap_range[0] - 1)  # 최소 capacity 기준으로 안전
    max_d = max(max_d, demand_range[0])
    demands[:, 1:] = torch.randint(demand_range[0], max_d + 1, (B, n_customers), device=device).float()

    return coords, demands, capacity


@torch.no_grad()
def route_length(coords: torch.Tensor, path: torch.Tensor):
    """
    coords: (B, N, 2)
    path:   (B, L) indices, includes depot=0 and intermediate depot returns
    """
    B, N, _ = coords.shape
    ordered = coords.gather(1, path.unsqueeze(-1).expand(B, path.size(1), 2))
    shifted = torch.roll(ordered, shifts=-1, dims=1)
    seg = ((ordered - shifted) ** 2).sum(-1).sqrt()
    return seg.sum(-1)  # (B,)


def split_trips(path_1d):
    """
    path_1d: list[int], e.g. [0,4,2,0,1,3,0]
    return: list[list[int]] trips excluding consecutive zeros
    """
    trips = []
    cur = []
    for i, v in enumerate(path_1d):
        if i == 0:
            cur = [v]
            continue
        cur.append(v)
        if v == 0 and len(cur) > 1:
            # end of a trip
            if not (len(cur) == 2 and cur[0] == 0 and cur[1] == 0):
                trips.append(cur)
            cur = [0]
    # 마지막이 depot으로 끝나지 않는 경우 (안전)
    if len(cur) > 1 and cur[-1] != 0:
        cur.append(0)
        trips.append(cur)
    return trips
