# Xavier

最早来自2010年的论文:《Understanding the difficulty of training deep feedforward neural networks》，在2016年开始流行

常见于数据初始化代码且配合后面的Relu
```py
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)
```

https://zhuanlan.zhihu.com/p/22028079
