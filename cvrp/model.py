import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CVRPModel(nn.Module):
    """
    Transformer Encoder + Pointer-style Decoder
    - nodes: depot(0) + customers(1..N)
    - features: (x, y, demand_ratio)
    - state: current node, remaining_capacity_ratio, visited
    """
    def __init__(self, embed_dim=128, n_heads=8, n_layers=3):
        super().__init__()
        self.embed = nn.Linear(3, embed_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # pointer logits
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # capacity context
        self.cap_embed = nn.Linear(1, embed_dim)

        # combine (current, global, cap) -> query
        self.ctx_proj = nn.Linear(embed_dim * 3, embed_dim)

    def forward(self, coords, demands_raw, capacity_raw,
                decode_type="sampling", max_steps=None):
        """
        coords: (B, N, 2)
        demands_raw: (B, N)  depot demand=0
        capacity_raw: (B,)   raw capacity (varies)
        returns:
          path: (B, L) indices, L = 1 + max_steps (starts with depot 0)
          logp_sum: (B,) sum log prob over sampled actions (RL용)
        """
        device = coords.device
        B, N, _ = coords.shape
        cap = capacity_raw.view(B, 1).float()

        demands = demands_raw.float()
        demand_ratio = demands / cap  # (B, N), depot=0

        # node features
        x = torch.cat([coords, demand_ratio.unsqueeze(-1)], dim=-1)  # (B,N,3)

        h = self.encoder(self.embed(x))  # (B,N,E)
        global_ctx = h.mean(dim=1)       # (B,E)

        # decode state
        current = torch.zeros(B, dtype=torch.long, device=device)  # start at depot
        visited = torch.zeros(B, N, dtype=torch.bool, device=device)
        visited[:, 0] = False
        remaining = torch.ones(B, 1, device=device)  # remaining capacity ratio (start full)

        done = torch.zeros(B, dtype=torch.bool, device=device)

        # max steps: 최악(고객을 한 명씩 배송)일 때 0->c->0 반복
        if max_steps is None:
            max_steps = 2 * (N - 1) + 1

        actions = []
        logps = []

        sqrt_d = math.sqrt(h.size(-1))

        for _ in range(max_steps):
            cur_h = h[torch.arange(B, device=device), current]  # (B,E)
            cap_h = self.cap_embed(remaining)                   # (B,E)
            q_in = torch.cat([cur_h, global_ctx, cap_h], dim=-1)
            q = self.q_proj(self.ctx_proj(q_in))                # (B,E)
            k = self.k_proj(h)                                  # (B,N,E)

            logits = (k * q.unsqueeze(1)).sum(-1) / sqrt_d       # (B,N)

            # infeasible mask: visited OR demand > remaining
            infeasible = visited.clone()
            infeasible |= (demand_ratio > remaining)            # depot demand=0 -> OK

            # all served?
            all_served = visited[:, 1:].all(dim=1)               # (B,)

            # 규칙 1) 다 배송했는데 depot이 아니면 depot 강제
            need_return = all_served & (current != 0) & (~done)
            force_depot = need_return.clone()

            # 규칙 2) depot에 있는데 나갈 수 있는 고객이 있으면 depot 선택 막기(무한 depot 루프 방지)
            at_depot = (current == 0) & (~done)
            feasible_customers = (~visited[:, 1:]) & (demand_ratio[:, 1:] <= remaining)
            has_feasible = feasible_customers.any(dim=1)
            mask_depot = at_depot & has_feasible
            infeasible[mask_depot, 0] = True

            # 규칙 3) depot이 아닌데 갈 수 있는 고객이 없으면 depot 강제
            not_depot = (current != 0) & (~done)
            no_feasible = (~feasible_customers.any(dim=1))
            force_depot |= (not_depot & no_feasible)

            # 규칙 4) done이면 depot 유지
            force_depot |= done

            masked_logits = logits.masked_fill(infeasible, -1e9)
            probs = F.softmax(masked_logits, dim=-1)

            if decode_type == "greedy":
                sel = probs.argmax(dim=-1)
                sel_logp = torch.log(probs.gather(1, sel.view(B,1)).squeeze(1) + 1e-12)
            else:
                dist = torch.distributions.Categorical(probs=probs)
                sel = dist.sample()
                sel_logp = dist.log_prob(sel)

            # force depot override (logp=0 처리)
            sel = torch.where(force_depot, torch.zeros_like(sel), sel)
            sel_logp = torch.where(force_depot, torch.zeros_like(sel_logp), sel_logp)

            actions.append(sel)
            logps.append(sel_logp)

            # update
            is_depot = (sel == 0)

            # remaining capacity ratio reset / subtract
            take = demand_ratio.gather(1, sel.view(B,1))  # (B,1)
            remaining = torch.where(is_depot.view(B,1), torch.ones_like(remaining), remaining - take)

            # mark visited customers
            visited[torch.arange(B, device=device), sel] |= (~is_depot)

            current = sel

            # done if all served and at depot
            done = done | (all_served & (current == 0))

        actions = torch.stack(actions, dim=1)             # (B, max_steps)
        logps = torch.stack(logps, dim=1).sum(dim=1)      # (B,)
        path = torch.cat([torch.zeros(B,1, dtype=torch.long, device=device), actions], dim=1)
        return path, logps
