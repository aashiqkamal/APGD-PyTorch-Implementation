#Attack wrappers for APGD for bs=1 
import torch 
import DataManagerPytorch as DMP
import torchvision


# project into Linf ball around x_orig  ———
def project_linf(x_adv, x_orig, eps):
    return torch.max(
        torch.min(x_adv, x_orig + eps),
        x_orig - eps
    )

# compute Vx f(model, x, y) 
def get_grad(model, x, y, loss_fn):
    x_req = x.clone().detach().requires_grad_(True)
    logits = model(x_req)
    loss   = loss_fn(logits, y)
    grad   = torch.autograd.grad(loss, x_req)[0]
    return grad

# Main APGD wrapper
def APGDNativePytorch(device, dataLoader, model, eps_max, num_steps, alpha=0.75, rho=0.75, clip_min=-1.0, clip_max=1.0, random_start=False):

    # 1) Eval mode
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    # 2) Metadata
    N = len(dataLoader.dataset)
    C, H, W = DMP.GetOutputShape(dataLoader)

    # 3) allocate
    x_adv_all = torch.zeros(N, C, H, W)
    y_all     = torch.zeros(N, dtype=torch.long)

    # 4) Output index
    idx_out = 0

    # 5) Checkpoint schedule W
    #   p0=0, p1=0.22, then pj+1 = pj + max(pj-pj-1-0.03, 0.06)
    W = [0]
    p_prev2, p_prev1 = 0.0, 0.22
    W.append(int(p_prev1 * num_steps))
    while W[-1] < num_steps:
        delta = max(p_prev1 - p_prev2 - 0.03, 0.06)
        p_next = p_prev1 + delta
        w_next = int(p_next * num_steps)
        W.append(w_next)
        p_prev2, p_prev1 = p_prev1, p_next

    # 6) Loop over samples
    for x_clean, y in dataLoader:
        bs = x_clean.size(0)
        if bs != 1:
            raise ValueError("This APGDNativePytorch requires batch_size=1")
        x_clean = x_clean.to(device)
        y = y.to(device)

        # 6b) Random start x(0)
        if random_start:
            delta = torch.empty_like(x_clean).uniform_(-eps_max, eps_max)
            x_k = torch.clamp(x_clean + delta, clip_min, clip_max).detach()
        else:
            x_k = x_clean.clone().detach()

        # 6c) First step --> x(1) line 3: first step with scalar eta 
        eta = 2 * eps_max
        grad = get_grad(model, x_k, y, loss_fn)
        z_next = torch.clamp(x_k + eta * grad.sign(), clip_min, clip_max)
        x_next = project_linf(z_next, x_clean, eps_max).detach()

        # 6d) Initialize best trackers (lines 4,5: pick (x_max, f_max) per-sample)
        f_x0 = loss_fn(model(x_clean), y)
        f_x1 = loss_fn(model(x_next), y)
        if f_x1 > f_x0:
            f_max   = f_x1
            x_max   = x_next.clone()
        else:
            f_max   = f_x0 
            x_max   = x_clean.clone()

        x_prev = x_k.clone()
        x_k = x_next.clone()
        improvement = 0
        checkpoint_ptr = 1
        prev_eta = eta
        prev_f_max = f_max
 
        # 6e) Main PGD loop k = 1…num_steps-1 (line 6 - line 17) 
        for k in range(1, num_steps):
            # gradient step --> z(k+1)
            grad   = get_grad(model, x_k, y, loss_fn)
            z_next = torch.clamp(x_k + eta * grad.sign(), clip_min, clip_max)
            z_next = project_linf(z_next, x_clean, eps_max)

            # momentum update --> x(k+1)
            x_next = x_k + alpha * (z_next - x_k) + (1 - alpha) * (x_k - x_prev)
            x_next = project_linf(x_next, x_clean, eps_max)
            x_next = torch.clamp(x_next, clip_min, clip_max).detach()

            # track improvement and  best
            f_k    = loss_fn(model(x_k), y)
            f_next = loss_fn(model(x_next), y)
            if f_next > f_k:
                improvement += 1
            if f_next > f_max:
                f_max   = f_next 
                x_max   = x_next.clone()
            # checkpoint logic
            if k  == W[checkpoint_ptr]:
                interval = W[checkpoint_ptr] - W[checkpoint_ptr - 1]
                cond1 = improvement < rho * interval
                cond2 = (eta == prev_eta) and (f_max == prev_f_max)
                if cond1 or cond2:
                    eta   = eta / 2.0
                    x_k   = x_max.clone()
                    x_prev = x_max.clone()
                improvement = 0
                prev_eta    = eta
                prev_f_max  = f_max
                checkpoint_ptr += 1

            # shift for next iter
            x_prev = x_k.clone()
            x_k    = x_next.clone()

        # 6f) Save best for this sample
        x_adv_all[idx_out] = x_max.cpu()
        y_all[idx_out]     = y.cpu()
        idx_out += 1

        # 6g) Free GPU memory
        torch.cuda.empty_cache()

    # 8) Pack into DataLoader
    advLoader = DMP.TensorToDataLoader(
        x_adv_all, y_all,
        transforms=None,
        batchSize=dataLoader.batch_size,
        randomizer=None
    )

    # 9) Return
    return advLoader
