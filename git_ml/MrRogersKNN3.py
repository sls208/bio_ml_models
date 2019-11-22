#!/usr/bin/env python

import sys,os
import pyspark
import numpy as np

if len(sys.argv) != 5:
    sys.stderr.write("Need training labels, training data, test data, and output file name\n")
    sys.exit(1)

sc = pyspark.SparkContext.getOrCreate()
sc.setLogLevel("WARN")
    
#extract the labels of the training set
lfile = sys.argv[1] #"train_labels.txt" #sys.argv[1]
labels = dict()
for line in sc.textFile(lfile).collect():
    (name,val) = line.rstrip().split()
    labels[name] = val
    

###########
def tanimoto(testInfo, trainInfo):
    return len(set(testInfo).intersection(set(trainInfo))) / float(len(set(testInfo).union(set(trainInfo))))

def Convert(lst): 
    res_dct = {lst[i][0]: lst[i][1] for i in range(0, len(lst), 1)} 
    return res_dct 

###########
trainData = sc.textFile(sys.argv[2]) #"small_train" #sys.arg[2]
testfile = sys.argv[3]#"small_test" #sys.argv[3]
testData = sc.textFile(testfile)

testnamez = sc.textFile(testfile).map(lambda line: line.split()[0]).distinct().collect()

print ("PARTITIONS:",sc.textFile(testfile).getNumPartitions())

NThreads = sc.defaultParallelism
print(NThreads)

lines= trainData.map(lambda x: x.split())
testLines = testData.map(lambda x: x.split())

# lines = trainData.map(str.split)
# testLines = testData.map(str.split)

trainNonZeros = lines.filter(lambda x: x[2] != '0')
testNonZeros = testLines.filter(lambda x: x[2] != '0')

trainLocs = trainNonZeros.map(lambda x: (x[0], {int(x[1])}))
testLocs = testNonZeros.map(lambda x: (x[0], {int(x[1])}))

trainLocGroups = trainLocs.reduceByKey(lambda x, y: x.union(y), NThreads)
testLocGroups = testLocs.reduceByKey(lambda x, y: x.union(y), NThreads)

# trainLocGroups = trainLocs.groupByKey(NThreads)
# testLocGroups = testLocs.groupByKey(NThreads)

# at this point this will do the cartesian with lists so maybe include a 
# conversion from list to set in tanimoto before doing the calculations

cart = testLocGroups.cartesian(trainLocGroups)

doit = cart.map(lambda x: (x[0][0], (x[1][0], tanimoto(x[0][1], x[1][1]))))

K = 3
def seqOp(top, pair): #merge single record value (pair) with aggregate value (top)
    if len(top) < K or pair[1] > top[-1][1]:
        top.append(pair)
        top.sort(key=lambda kv: -kv[1])
        top = top[:K]
    return top

def combOp(a,b): #merge two aggregated values
    a += b
    a.sort(key=lambda kv: -kv[1])
    a = a[:K]
    return a

take3 = doit.aggregateByKey(list(), seqOp, combOp)

OMG3 = take3.map(lambda x: (x[0], x[1][1]) if labels[x[1][1][0]] == labels[x[1][2][0]] else (x[0], x[1][0]))

#omg = doit.reduceByKey(lambda x,y: x if x[1] > y[1] else y)

zomg3 = OMG3.collect()

finalDict = Convert(zomg3)

out = open(sys.argv[4],'w') #out = open("ass2out",'w') #out = open(sys.argv[4],'w')
#a = list(labels.values())
for name in testnamez:
    out.write('%s %s\n' % (name,labels[finalDict[name][0]]))


