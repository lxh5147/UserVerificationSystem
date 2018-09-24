import tensorflow as tf


def flip(x, dim):
    '''
    Reverse the elements along the given dim. a[i] <--a[n-i]
    :param x: tensor to be flipped
    :param dim: int scale tensor or int, along which to flip the tensor
    :return: tensor with the same shape x
    '''
    tf.reshape
    x.reshape()
    x.flip()
    xsize = x.shape()
    dim = x.ndim + dim if dim < 0 else dim
    x = x.reshape()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)
