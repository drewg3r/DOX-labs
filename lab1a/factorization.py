def isqrt(n):
    x = n
    y = (x + n // x) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def fermat(n):
    a = isqrt(n)
    b2 = a*a - n
    b = isqrt(n)
    count = 0
    while b*b != b2:
        a = a + 1
        b2 = a*a - n
        b = isqrt(b2)
        count += 1
    p = a+b
    q = a-b
    assert n == p * q
    return [p, q]


def cycle_fermat(n):
    result = fermat(n)
    if 1 in result:
        result.remove(1)
        return result
    else:
        new_result = []
        for i in result:
            res = cycle_fermat(i)
            for j in res:
                new_result.append(j)
        return new_result
