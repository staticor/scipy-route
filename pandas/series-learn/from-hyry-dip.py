# coding: utf-8

def C(*args, **kw):
    from itertools import izip_longest
    gap = kw.pop("gap", 5)
    results = []
    for item in args:
       if isinstance(item, tuple):
           results.append("\n".join(unicode(row) for row in item).split("\n"))

    for i, item in enumerate(results):
       width = max(len(row) for row in item)
       results[i].insert(1, "-"*width)
       results[i] = [row.ljust(width+gap) for row in item]
    col_widths = [max(map(len, column)) for column in results]

    for row in izip_longest(*results):
       row = [x if x is not None else (" " * col_widths[i]) for i, x in enumerate(row)]
       print "".join(row)


import pandas as pd
s = pd.Series([1,2,3,4,5], index=["a","b","c","d","e"])
print s.index
print s.values

print s[2]
C((u"s[1:3]", s[1:3]), (u"s['b':'d']", s['b':'d']))

print list(s.iteritems())
index = s.index
print index.get_loc('c')

