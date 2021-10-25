
def reg_covariance_loop(basis, weights, mean, label, scaler, null_val = 0.0):
    """
    basis : [b, f, n, 1]
    weight: [f, 12]
    mean  : [b, 1, n, 12], normalized, types: zero mean, batch mean, prediction
    label : [b, 1, n, 12], denormalized
    """
    x = basis.transpose(1, 3) # basis, b,t(=1),n,f
    s = x.size()
    batch_size = s[0]
    n = s[2]
    f = s[3]
    x = x.view(-1,n,f)        # b,1,n,f -> b,n,f
    label = label - mean      # b,1,n,f
    y = label.view(-1, n, 12) # b,1,n,12 -> b,n,12
    m = torch.mean(weights)
    var = torch.mean(torch.pow(weights - m,2))
    cov_loss = []
    for i in range(0, batch_size):
        b_1 = x[i] # [n,f]
        y_1 = y[i] # [n,12]
        y_1_normalized = scaler.transform(y_1) # [n,12]
        for j in range(0, batch_size):
            # basis covariance
            b_2 = x[j] # [n,f]
            y_2 = y[j] # [n,12]
            cov = var * torch.mm(b_1, torch.transpose(b_2, 0, 1)) # [n,n]
            
            # mask
            mat = torch.mm(y_1, torch.transpose(y_2, 0, 1))
            mask =  (mat != null_val)
            mask =  mask.float()
            mask /= torch.mean((mask))
            mask =  torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

            # empirical covariance
            y_2_normalized = scaler.transform(y_2) # [n,12]
            emp_cov = torch.mm(y_1_normalized, torch.transpose(y_2_normalized, 0, 1)) # [n,n]
            
            # loss
            loss = torch.pow(cov - emp_cov, 2) # [n,n]
            loss = loss * mask                 # [n,n]
            loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss) # [n,n]
            cov_loss.append(torch.mean(loss))
            return torch.mean(torch.Tensor(cov_loss))
