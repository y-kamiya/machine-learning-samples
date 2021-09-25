


def f(output, kernel, stride):
    return (output - 1) * stride + kernel

layer2 = f(f(1,4,1), 4, 1)
print(layer2)

layer3 = f(layer2, 4, 2)
print(layer3)

layer5 = f(f(f(layer2, 4,2), 4,2), 4,2)
print(layer5)

layer7 = f(f(layer5,4,2), 4,2)
print(layer7)

