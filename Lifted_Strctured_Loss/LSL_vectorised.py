def lifted_structured_loss(projection, target, alpha):
    D = torch.square(torch.cdist(projection, projection))
    target_long = target.long()
    classes = torch.unique(target_long)
    bin_counts = torch.bincount(target_long)
    P = torch.zeros_like(classes)
    for idx, c in enumerate(classes):
        P[idx] = bin_counts[c.item()]
    p = [torch.where(target_long == c)[0] for c in classes]
    A = [D[i][:, i] for i in p]
    n = [torch.where(target_long != c)[0] for c in classes]
    B = [torch.sum(torch.exp(alpha - D[i.view(-1, 1), j.view(1, -1)]), dim=1) for i, j in zip(p, n)]
    C = [torch.sum(torch.exp(alpha - D[i.view(-1, 1), j.view(1, -1)]), dim=1) for i, j in zip(p, n)]
    loss = sum((torch.sum(torch.square(torch.maximum(a + torch.log(b + c), torch.tensor(0.0, dtype=torch.float64)))) / (2.0 * p_count) for a, b, c, p_count in zip(A, B, C, P)))
    return loss / len(target)
