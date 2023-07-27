def lifted_structured_loss(projection, target, alpha):
    D = torch.square(torch.cdist(projection, projection))
    loss = 0
    for c in torch.unique(target):
        P = torch.sum(target == c)
        p = torch.where(target == c)[0]
        n = torch.where(target != c)[0]
        i = torch.repeat_interleave(p, len(p))
        j = p.repeat(len(p))
        A = D[i, j]
        B = torch.sum(torch.exp(alpha - D[i.view(-1, 1), n.view(1, -1)]), dim=1)
        C = torch.sum(torch.exp(alpha - D[j.view(-1, 1), n.view(1, -1)]), dim=1)
        loss = loss + torch.sum(torch.square(torch.maximum(A + torch.log(B + C), torch.tensor(0)))) / (2 * P)
    return loss / len(target)
