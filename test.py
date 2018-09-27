def test(**kwargs):
    p = kwargs['p']
    print(p)


def test2(x=1, q=2, **kwargs):
    print(q)


param = {'p': 2, 'q': 3}

test2(x=1, **param)

# remove 8k files

path = './data/train'

