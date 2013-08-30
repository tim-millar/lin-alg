#!/usr/bin/env python

def dict2list(dct, keylist):
    """
    Input: a dict and a list of keys
    Output: a list of values
    """
    return [dct[i] for i in keylist]

def list2dict(L, keylist):
    """
    Input: a list of values and a list of keys
    Output: a dict
    """
    return {x:y for (x,y) in list(zip(keylist,L))}

def listrange2dict(L):
    """
    Input: a list
    Output: a dictionary that, for i = 0, 1, 2, . . . , len(L), maps i to L[i]
    """
    return list2dict(L,range(len(L)))

