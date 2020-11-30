"""
Machine Learning - Homework 1
Author: Giuseppe Sensolini Arra', 1661198
November 2020    
"""

import numpy as np
import math
import json

def print_asm(data,index):
    print("\nData["+str(index)+"]['lista_asm']:")
    for elem in data[index]["lista_asm"]:
        print(elem)
    print("-------------")

def dict_sum(d1,d2):
    for key in d1:
        d2[key]+=d1[key]
    return d2

def dict_truncate(d,dec):
    new_d = d
    for key in d:
        new_d[key] = math.floor(d[key]*(10**dec))/(10**dec)
    return new_d

def init_feature_dict():
    d = {
        "instr":0, 
        "mov":0,
        "cmp":0,
        "arithmetic":0,
        "bitwise":0,
        "call":0,
        "jump":0,
        "shift":0,
        "float":0,
    }
    return d

def av_features(d,n):
    if n==0:
        return 0
    else:
        for key in d:
            if key!="instr": d[key] = 100*(d[key]/d["instr"])
        d["instr"]/=n
        return d

def feature_extraction(data, blind=False):
    """Given a dataset exract the features of interest

    Args:
        data: dataset
        blind (bool, optional): if true the dataset is considered blind. Defaults to False.

    Returns:
        fncts: nxm matrix (n = dataset_length, m = number_of_features)
        labels: ground truth, contains semantic for each entry of the dataset
        header: contains feature names
        classes: contains class names
    """

    print("extracting asm lists... ")
    for elem in data:
        elem["lista_asm"] = elem["lista_asm"].split("'")
        while ", " in elem["lista_asm"]: elem["lista_asm"].remove(", ")
        elem["lista_asm"] = elem["lista_asm"][1:len(elem["lista_asm"])-1]
    print("done.")

    # famility of assembly operations
    arithmetic_ops = ["add", "sub", "mul", "div"]
    bitwise_ops = ["not", "and", "or", "xor"]
    jump_ops = ["jmp","je","jz","jne","jnz","js","jg","jnle","jge","jnl","jl,jnge","jle","jng","ja","jnbe","jae","jnb","jb","jnae","jbe","jna"]
    shift_ops = ["shr", "shl", "sal", "sar"]
    float_ops = ["ss","sd"]

    # count features
    for fnct in data:
        fnct["features"] = init_feature_dict()
        fnct["features"]["instr"] = len(fnct["lista_asm"])
        for instr in fnct["lista_asm"]:
            op = instr.split(" ")[0]
            if op=="mov": fnct["features"]["mov"]+=1
            elif op=="cmp" or op=="test":
                fnct["features"]["cmp"]+=1
            elif op in arithmetic_ops:
                fnct["features"]["arithmetic"]+=1
            elif op in bitwise_ops:
                fnct["features"]["bitwise"]+=1
                if op=="xor": fnct["features"]["bitwise"]+=1
            elif op=="call":
                fnct["features"]["call"]+=1
            elif op in jump_ops:
                fnct["features"]["jump"]+=1
            elif op in shift_ops:
                fnct["features"]["shift"]+=1
            elif op[len(op)-2:len(op)] in float_ops:
                fnct["features"]["float"]+=1
            fnct["features"]["float"] += 0.5*(instr.count("xmm"))

    # prepare data for the decision tree
    header = list(data[0]["features"].keys())
    classes = ["string","math","encryption","sort"]
    fncts, labels = [],[]
    if blind:
        for fnct in data:
            fncts.append( list(fnct["features"].values()) )
        return fncts
    else:
        # A useful print to better understand classes' features
        features_averaging(data)
        for fnct in data:
            fncts.append( list(fnct["features"].values()) )
            if fnct["semantic"]==classes[0]: labels.append(0)
            elif fnct["semantic"]==classes[1]: labels.append(1)
            elif fnct["semantic"]==classes[2]: labels.append(2)
            elif fnct["semantic"]==classes[3]: labels.append(3)
        return fncts, labels, header, classes

def features_averaging(data):
    """
    compute the average distibution of features over classes.
    note:It has to be used only after features_extraction()
    
    Args:
        data : dataset (in a suitable format)
    """

    print("semantic features averaging... ")
    features_string = init_feature_dict()
    features_encryption = init_feature_dict()
    features_math = init_feature_dict()
    features_sort = init_feature_dict()
    num_string = num_encryption = num_math = num_sort = 0

    for fnct in data:
        if fnct["semantic"]=="string":
            features_string = dict_sum(fnct["features"], features_string)           
            num_string+=1
        elif fnct["semantic"]=="math":
            features_math = dict_sum(fnct["features"], features_math)
            num_math+=1
        elif fnct["semantic"]=="encryption":
            features_encryption = dict_sum(fnct["features"], features_encryption)
            num_encryption+=1
        elif fnct["semantic"]=="sort":
            features_sort = dict_sum(fnct["features"], features_sort)
            num_sort+=1
            
    print("done.\n")
    print("dataset size: " + str(len(data)))
    print("string functions: " + str(num_string))
    print("math functions: " + str(num_math))
    print("encryption functions: " + str(num_encryption))
    print("sorting function: " + str(num_sort) + "\n")

    av_string = av_features(features_string, num_string)
    av_sort = av_features(features_sort, num_sort)
    av_math = av_features(features_math, num_math )
    av_encryption = av_features(features_encryption, num_encryption)
    print("Average features distribution:")
    print("  string:     "+str(dict_truncate(av_string,2)))
    print("  math:       "+str(dict_truncate(av_math,2)))
    print("  encryption: "+str(dict_truncate(av_encryption,2)))
    print("  sort:       "+str(dict_truncate(av_sort,2)) + "\n")

def blind_comparison(pred1, pred2):
    """
    Compare two blind predictions computed using two different methods or hyperparameters

    Args:
        pred1: blind prediction
        pred2: another blind prediction

    Returns:
        [float, float]: [error, accuracy]
    """
    if len(pred1)!=len(pred2):
        print("error: input lengths are not compatible!")
        print("len(p1)=" + str(len(pred1)))
        print("len(p2)=" + str(len(pred2)))
        return 0, 0
    else:
        n = len(pred1)
        e = 0
        for i in range(0,n):
            if pred1[i]!=pred2[i]: e+=1

    error = math.floor((e/n)*10000)/10000
    accuracy = 1-error
    print("classification differences: " + str(e) + "/" + str(n))
    print("error: " + str(error))
    print("accuracy: " + str(accuracy))
    return error, accuracy


