#!/usr/bin/env python3
import sys
from pylab import *

# definitions of interpolation and AP score from http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf
def interpolate(y, x, z):
    return max((i for i, j in zip(y, x) if j >= z), default=0)

def APscore(precision, recall):
    return (1/11) * sum(interpolate(precision, recall, r) for r in arange(0,1.1,0.1))


scores = []
for filename in sys.argv[1:]:
    if not filename.endswith("csv"): continue
    print("reading "+filename)
    with open(filename, "r") as f:
        lines = f.readlines()
    table = [[float(i) for i in row.split(",")] for row in lines]
    table = [row for row in table if all(i>=0 for i in row) and not all(i==0 for i in row)]
#    assert len(table) >= 2, "no plotting with less than 2 points"

    precision = [i[0] for i in table]
    recall = [i[1] for i in table]

    plot(recall, precision)
    xlabel('recall')
    ylabel('precision')
    title('PR curve')

    savefig(filename.replace("csv", "png"))
    close()
    aps = APscore(precision, recall)
    print("AP score: "+str(aps))
    scores.append(aps)

print("summary:")
print("min:  {:2.5f}".format(min(scores)))
print("max:  {:2.5f}".format(max(scores)))
print("avrg: {:2.5f}".format(sum(scores)/len(scores)))
print("mean: {:2.5f}".format(sorted(scores)[len(scores)//2]))


