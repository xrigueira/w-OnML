from river import compose

x = {'a': 0, 'b': 0, 'c': 0}

print(type(x.keys()))

print(compose.Select(tuple(x.keys())))