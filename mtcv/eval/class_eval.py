import numpy as np


def g_mean1(recall,precision):
    """
    Calculate G-mean 1 metric, \sqrt (Recall * Precision)
    """
    return np.sqrt(recall*precision)

def f_score(recall,precision,beta=1,weigh_r=None,weight_p=None):
    """
    Calculate F_score metric.
    """
    if weigh_r and weight_p:
        if not weigh_r+weight_p ==1:
            raise ValueError("Recall weight + Precision weight must equal 1.")
        beta = weigh_r/weight_p
    f=(1+beta)*recall*precision/(recall+precision)
    return f

a=[73.1 ,66.4]
b=[70 ,72.2]
c=[66.6 , 80.1]
d=[75.2 , 49.2]
e=[70 , 79.7]

print("g-mean")
print(g_mean1(*a))
print(g_mean1(*b))
print(g_mean1(*c))
print(g_mean1(*d))
print(g_mean1(*e))

print("f1-score")
print(f_score(*a))
print(f_score(*b))
print(f_score(*c))
print(f_score(*d))
print(f_score(*e))