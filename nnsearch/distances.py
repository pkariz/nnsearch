

def minkowski(x, y, p=2):
    """
    Returns minkowski distance between points x and y.
    :param x: first vector
    :param y: second vector
    :param p: defining Lp-norm, default is euclidean distance, -1 is for L_inf-norm
    :return: distance between x and y using Lp-norm
    """
    if p == -1:
        #L_inf
        return max(abs(x-y))
    if p < 1.0:
        raise ValueError("Invalid minkowski parameter p, must be >= 1 or -1!")
    res = 0.0
    for i in range(len(x)):
        res += (max(x[i], y[i]) - min(x[i], y[i]))**p
    return res ** (1/float(p))


def edit_distance(a, b):
    """Returns edit distance between two strings.
    :param a: first string
    :param b: second string
    :return: edit distance between a and b
    """
    if len(b) < len(a):
        b, a = a, b
    if len(a) == 0:
        return len(b)

    last = range(len(a) + 1)
    for i, cur_b in enumerate(b):
        cur = [i + 1]
        for j, cur_a in enumerate(a):
            cur.append(min(last[j] + (cur_b != cur_a), last[j + 1] + 1, cur[j] + 1))
        last = cur

    return last[-1]