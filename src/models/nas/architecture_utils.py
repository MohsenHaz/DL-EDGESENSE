"""Architecture manipulation utilities."""
def count_flops(model, input_shape=(1, 3, 224, 224)):
    try:
        from thop import profile
        import torch
        inp = torch.randn(*input_shape)
        flops, _ = profile(model, inputs=(inp,))
        return flops
    except ImportError:
        return None
