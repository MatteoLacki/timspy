def ranges(min_idx, max_idx, step):
    if step >= max_idx - min_idx:
        yield min_idx, max_idx
    else:
        it = iter(range(min_idx, max_idx, step))
        i_ = next(it)
        for _i in it:
            yield (i_, _i)
            i_ = _i
        yield (_i, max_idx)