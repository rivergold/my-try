class A:
    def __init__(self):
        self.a = 10

    def forward(self, x):
        return self.a + x


class B:
    def __init__(self):
        self.a = A()

        def func(self, x):
            return 2 * self.a + x

        print(type(func))
        self.a.forward = func.__get__(self.a)


b = B()
res = b.a.forward(1)
print(res)


def func(x):
    return x


print(dir(func))
