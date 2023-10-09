
def is_subsequence(a, b):
    b_it = iter(b)
    count = 0
    try:
        for a_val in a:
            if a_val in b:
                count += 1
                while next(b_it) != a_val:
                    pass

    except StopIteration:
        return False, 0

    if count == 0:
        return False, 0
    else:
        return True, count/len(a)