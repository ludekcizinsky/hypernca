import torch


def hidden_tokenize(w1, b1, w2):
    """
    w1: [batch_size, fc_dim, chn * 4]
    b1: [batch_size, fc_dim]
    w2: [batch_size, chn, fc_dim]

    out: [batch_size, num_tokens, token_dim]
    num_tokens = fc_dim
    token_dim = chn * 5 + 1
    """
    W = torch.cat([w1, w2.permute(0, 2, 1), b1[:, :, None]], dim=2)
    return W


def hidden_untokenize(W):
    """
    W: [batch_size, num_tokens, token_dim]
    num_tokens = fc_dim
    token_dim = chn * 5 + 1

    returns:
        w1: [batch_size, fc_dim, chn * 4]
        b1: [batch_size, fc_dim]
        w2: [batch_size, chn, fc_dim]
    """
    b, _, token_dim = W.shape
    chn = (token_dim - 1) // 5
    w1 = W[:, :, : chn * 4]
    w2 = W[:, :, chn * 4 : chn * 5].permute(0, 2, 1)
    b1 = W[:, :, -1]

    return w1, b1, w2


def mixed_tokenize(w1, b1, w2):
    """
    w1: [batch_size, fc_dim, chn * 4]
    b1: [batch_size, fc_dim]
    w2: [batch_size, chn, fc_dim]

    out: [batch_size, num_tokens, token_dim]
    num_tokens = chn * 5 + 1
    token_dim = fc_dim
    """
    # (B, token_dim,n_tokens) -> (B, n_tokens, token_dim)
    return hidden_tokenize(w1, b1, w2).permute(0, 2, 1)


def mixed_untokenize(W):
    """
    W: [batch_size, num_tokens, token_dim]
    num_tokens = chn * 5 + 1
    token_dim = fc_dim

    returns:
        w1: [batch_size, fc_dim, chn * 4]
        b1: [batch_size, fc_dim]
        w2: [batch_size, chn, fc_dim]
    """
    return hidden_untokenize(W.permute(0, 2, 1))


if __name__ == "__main__":
    import torch

    B = 2
    w1 = torch.randn(B, 96, 48)
    b1 = torch.randn(B, 96)
    w2 = torch.randn(B, 12, 96)

    W = hidden_tokenize(w1, b1, w2)

    print("W shape:", W.shape)
    print(W.permute(0, 2, 1).shape)


    input_mean = torch.mean(W, dim=0, keepdim=True)
    input_std = torch.std(W, dim=0, keepdim=True)
    print("input_mean shape:", input_mean.shape)
    print("input_std shape:", input_std.shape)