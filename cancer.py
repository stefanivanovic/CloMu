import matplotlib.pyplot as plt
import copy

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.optim import Optimizer

import sys
import glob
import os
import numpy as np
import scipy




#name = './data/tracerx_lung.txt'
#file1 = open(name, 'r')
#data = file1.readlines()

#for a in range(100):
#    print (data[a][:-1])
#quit()



def loadnpz(name, allow_pickle=False):
    if allow_pickle:
        data = np.load(name, allow_pickle=True)
    else:
        data = np.load(name)
    data = data.f.arr_0
    return data


class MutationModel(nn.Module):
    def __init__(self, M):
        super(MutationModel, self).__init__()

        self.M = M
        L = 5
        #L = 10




        #L2 = 2
        #self.conv1 = torch.nn.Conv1d(14, mN1_1, 1)
        self.nonlin = torch.tanh

        #self.lin0 = torch.nn.Linear(L, L)

        #print (M)
        #quit()

        self.lin1 = torch.nn.Linear(M+20, L)
        self.lin2 = torch.nn.Linear(L, M)

        #self.linI = torch.nn.Linear(1, 1)

        ####self.linM = torch.nn.Linear(L, L)

        #self.linM1 = torch.nn.Linear(L1, L2)
        #self.linM2 = torch.nn.Linear(L2, L1)

        #self.linP = torch.nn.Linear(1, 20)

        #self.linSum = torch.nn.Linear(2, L)
        #self.linBaseline = torch.nn.Linear(L, 1)


    def forward(self, x):


        #print (x.shape)

        #x = self.lin1(x)
        #x = self.lin2(x)


        xSum = torch.sum(x, dim=1)#.reshape((-1, 1))
        xSum2 = torch.zeros((x.shape[0], 20))
        xSum2[np.arange(x.shape[0]), xSum.long()] = 1

        #x = x * 0

        x = torch.cat((x, xSum2), dim=1)

        x = self.lin1(x)

        x1 = x[:, 0].repeat_interleave(self.M).reshape((x.shape[0], self.M) )

        ####x = self.linM(x)
        ####x = self.nonlin(x)


        x = self.nonlin(x)

        xNP = x.data.numpy()

        #x = self.nonlin(x)

        #plt.plot(xNP)
        #plt.scatter(xNP[:, 2], xNP[:, 4])
        #plt.show()
        #quit()

        #x = self.linM(x)
        #x = self.nonlin(x)
        #x = self.linM2(x)
        #x = self.nonlin(x)
        x = self.lin2(x)

        #x = x * 0

        x = x + x1

        #shape1 = x.shape


        return x, xNP


class MutationModel2(nn.Module):
    def __init__(self, M1, M2):
        super(MutationModel2, self).__init__()

        self.M = M2
        L = 5
        self.nonlin = torch.tanh
        self.L = L

        self.M1 = M1

        #self.lin1 = torch.nn.Linear(M2+20, L)
        self.lin1 = torch.nn.Linear(M1+20, L)
        self.lin2 = torch.nn.Linear(L, M2)

        #self.linM = torch.nn.Linear(L, L)


        self.linB1 = torch.nn.Linear(M1 * M2, L)
        self.linB2 = torch.nn.Linear(L, L)

        self.matrix = torch.nn.Parameter(torch.rand(M1, M2) * 0.2)


    def forward(self, x, x2, giveMatrix=False):

        if giveMatrix:
            return self.matrix
        else:

            xSum = torch.sum(x, dim=1)#.reshape((-1, 1))
            xSum2 = torch.zeros((x.shape[0], 20))
            xSum2[np.arange(x.shape[0]), xSum.long()] = 1


            M1 = self.M1
            M2 = self.M
            L = self.L
            x2 = x2.reshape((1, M1 * M2))
            x2 = self.linB1(x2)
            #x2 = self.nonlin(x2)
            #x2 = self.linB2(x2)
            x2 = x2.repeat_interleave(x.shape[0]).reshape((L, x.shape[0])).T

            x = torch.cat((x, xSum2), dim=1)

            x = self.lin1(x)

            x1 = x[:, 0].repeat_interleave(self.M).reshape((x.shape[0], self.M) )

            x = x + x2

            x = self.nonlin(x)


            #x = self.linM(x)
            #x = self.nonlin(x)


            xNP = x.data.numpy()
            x = self.lin2(x)

            x = x + x1
            return x, xNP


class MatrixModel(nn.Module):
    def __init__(self, M1, M2):
        super(MatrixModel, self).__init__()

        #self.matrix = torch.nn.Parameter(torch.rand(M1, M2) * 0.1)
        self.matrix = torch.nn.Parameter(torch.rand(M1, M2) * 0.2)


    def forward(self):

        return self.matrix



class ClusterProbModel(nn.Module):
    def __init__(self, M1, M2):
        super(ClusterProbModel, self).__init__()
        L = 20
        self.M1 = M1
        self.M2 = M2
        self.nonlin = torch.tanh
        self.lin1 = torch.nn.Linear(M1 * M2, M1 * M2)
        #self.lin1 = torch.nn.Linear(M1 * M2, L)
        #self.lin2 = torch.nn.Linear(L, M1 * M2)

        self.matrix = torch.nn.Parameter(torch.rand(M1, M2) * 0.2)

    def forward(self, x):

        M1, M2 = self.M1, self.M2

        '''
        x = self.lin1(x)
        x = self.nonlin(x)

        x = x * 0

        x = self.lin2(x)
        x = x.reshape((M1, M2))
        x = torch.softmax(x, axis=1)
        '''

        x = self.lin1(x)
        #x = self.nonlin(x)

        #x = torch.zeros((1, 20))
        #x = x * 0

        #x = self.lin2(x)
        x = x.reshape((M1, M2))

        #x[0, 0] = x[0, 0] + 1
        #x[1, 0] = x[1, 0] + 1
        #x[2, 1] = x[2, 1] + 1
        #x[3, 1] = x[3, 1] + 1

        x = torch.softmax(x, axis=1)


        #x = self.matrix
        #x = torch.softmax(x, axis=1)

        return x


def addFromLog(array0):

    array = np.array(array0)
    array_max = np.max(array, axis=0)
    for a in range(0, array.shape[0]):
        array[a] = array[a] - array_max
    array = np.exp(array)
    array = np.sum(array, axis=0)
    array = np.log(array)
    array = array + array_max

    return array

def doRECAPplot(name, doCluster=False):

    import numpy as np #idk why this was needed
    allSaveVals = np.load('./dataNew/allSave_' + name + '.npy')
    #print (allSaveVals.shape)
    saveData = []
    for a in range(0, len(allSaveVals)):
        dataNow = allSaveVals[a]
        ar1 = dataNow[2:102]
        ar2 = dataNow[102:202]

        ar1 = ar1[ar1 != '0.0']
        ar2 = ar2[ar2 != '0.0']

        incorrect = 0
        for b in range(0, len(ar1)):
            if ar1[b] != ar2[b]:
                incorrect += 1

        accuracy = 1 - (incorrect / len(ar1))

        incorrect2 = 0
        for b in range(0, len(np.unique(ar1))):
            if np.unique(ar1)[b] != np.unique(ar2)[b]:
                incorrect2 += 1


        kTrue = np.unique(ar1).shape[0]
        kPred = np.unique(ar2).shape[0]

        #print (dataNow[0])

        mVal = int(dataNow[0].split('m')[1])

        #if mVal != 12:
        #    print (incorrect2)
        #    if incorrect2 != 0:
        #        print ("Issue")
        #        quit()

        dataNew = [dataNow[0], dataNow[1], accuracy, mVal, kTrue, kPred]

        saveData.append(np.copy(np.array(dataNew)))

    saveData = np.array(saveData)

    #quit()

    import os, sys, glob
    import math
    import numpy as np
    import pandas as pd
    #%matplotlib inline
    #%config InlineBackend.figure_format = 'svg'
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    pd.set_option('display.max_columns', None)
    sns.set_style('whitegrid')
    mpl.rc('text', usetex=True)
    sns.set_context("notebook", font_scale=1.5)


    # Read in results file for all methods
    df = pd.read_csv("./data/results.tsv", sep="\t")
    # Set output results folder
    #orf = "./simulations/pdfs/"


    # Identifying the selected k for each instance
    # For Revolver and Hintra, we only import the k those methods selected

    df["selected_k"] = [1 if (x =='HINTRA' or x =='REVOLVER') else 0 for x in df['method']]

    assert df['selected_k'].sum() == len(df[(df["method"] == 'HINTRA')].index) + len(df[(df["method"] == 'REVOLVER')].index)


    # Functions to help us select k for RECAP

    def smooth(pairs):
        smooth_pairs = []
        pairs.sort(key=lambda elem: elem[0])
        low_d = pairs[0][1]
        for k,d in pairs:
            low_d = min(low_d, d)
            smooth_pairs.append((k,low_d))

        return smooth_pairs

    def find_k(pairs, p_threshold, a_threshold):
        pairs.sort(key=lambda elem: elem[0], reverse=True)
        prev_d = pairs[0][1]
        for k,d in pairs[1:]:
            a_change = d-prev_d
            p_change = a_change/(prev_d*1.0+0.0000001)
            if (p_change >= p_threshold) and (a_change >= a_threshold):
                return k+1
            prev_d = d

        return 1

    def select_k(df, pt, at):

        # Find unique list of instances, methods, and true k
        instance_df = df.drop_duplicates(['instance','method', 'true_k'])[['instance','method', 'true_k']]
        instance_df = instance_df[instance_df['method'].isin(['RECAP-r10', 'RECAP-r50', 'RECAP-r100'])]

        assert len(instance_df.index) == 1800

        # Iterate over these instances
        for index, row in instance_df.iterrows():

            # Extract instance information
            instance = row['instance']
            method = row['method']
            true_k = row['true_k']

            # Subset data to this instance
            subset = df[(df["instance"] == instance) & (df["method"] == method) & (df["true_k"] == true_k)]

            # Compute k for this instance
            pairs = []

            for index, row in subset.iterrows():
                pairs.append((row['inferred_k'], row['PC_dist']))

            smooth_pairs = smooth(pairs)
            selected_k = find_k(smooth_pairs, pt, at)

            # Fill in df with this selected k
            df.loc[(df["instance"] == instance) & (df["method"] == method) & (df["true_k"] == true_k) & (df["inferred_k"] == selected_k), 'selected_k'] = 1

        return df





    # Fill in selected k for RECAP with given percentage and absolute thresholds for finding the elbow
    pt = .05
    at = .5

    df = select_k(df, pt, at)

    #df.append(np.zeros(21).astype(str))
    #df.append(list(np.zeros(21).astype(str)))

    keys1 = np.array(df.keys())

    argAccuracy = np.argwhere(keys1 == 'selection_accuracy')[0, 0]
    argMethod = np.argwhere(keys1 == 'method')[0, 0]
    argSelectedK = np.argwhere(keys1 == 'selected_k')[0, 0]
    argTrueK = np.argwhere(keys1 == 'true_k')[0, 0]



    '''
    dataForDf = []
    for a in range(0, len(saveData)):

        #ar1 = np.zeros(21).astype(str)
        ar1 = list(np.zeros(21))
        ar1[argAccuracy] = saveData[a, 2]
        ar1[argMethod] = 'Stefan'
        ar1[argSelectedK] = 1
        ar1[argTrueK] = 1
        #ar1 = list(ar1)

        dataForDf.append(ar1)
    '''

    #dataDF = pd.DataFrame(dataForDf, columns=df.keys())


    df = df[df["selected_k"] == 1]

    df_vals = df.values

    #dataForDf = np.array(dataForDf)
    #dataForDf = np.zeros((len(saveData), len(keys1)))
    dataForDf = df_vals[:len(saveData)]

    df_vals = np.concatenate((dataForDf, df_vals), axis=0)



    df = pd.DataFrame(df_vals, columns=df.keys())



    dataForDf = []
    for a in range(0, len(saveData)):

        #print (int(saveData[a, 4]))

        df.loc[a, 'selected_k'] = 1
        df.loc[a, 'selection_accuracy'] = float(saveData[a, 2])
        df.loc[a, 'true_k'] = int(saveData[a, 4])
        df.loc[a, 'inferred_k'] = int(saveData[a, 5])
        df.loc[a, 'method'] = 'Stefan'
        df.loc[a, 'm'] = int(saveData[a, 3])



    df = df[df["m"] == 5]

    #0.986


    #print (np.mean(df[df['method'] == "RECAP-r50"]['selection_accuracy']))
    #print (np.mean(df[df['method'] == "Stefan"]['selection_accuracy']))
    #quit()

    random1 = np.random.normal(size = df[df['method'] == "RECAP-r50"]['selection_accuracy'].shape[0]) * 0.001
    random2 = np.random.normal(size = df[df['method'] == "RECAP-r50"]['selection_accuracy'].shape[0]) * 0.001

    y1 = df[df['method'] == "RECAP-r50"]['selection_accuracy']
    y2 = df[df['method'] == "Stefan"]['selection_accuracy']

    import scipy
    from scipy import stats

    #print (scipy.stats.pearsonr(y1, y2))

    #plt.scatter(y1 + random1, y2 + random2)
    #plt.xlabel('RECAP')
    #plt.ylabel('Stefan')
    #plt.scatter()
    #plt.show()
    #quit()




    #df = df.append(dataDF, ignore_index=True)
    #df = df.append(df, ignore_index=True)

    print (df.keys())


    #assert df[df["method"] == 'RECAP-r10']['selected_k'].sum() == 600
    #assert df[df["method"] == 'RECAP-r50']['selected_k'].sum() == 600
    #assert df[df["method"] == 'RECAP-r100']['selected_k'].sum() == 600






    #methods = ["REVOLVER", "RECAP-r50"]
    #methods = ["REVOLVER", "RECAP-r50", "HINTRA"]
    methods = ["REVOLVER", "RECAP-r50", 'Stefan']

    #print (df[df["selected_k"] == 1])
    # Model condition all


    if doCluster:
        sns.stripplot(data=df, x="true_k",
                  y="inferred_k", hue="method",
                  hue_order=methods,
                  alpha=.4, dodge=True, linewidth=1, jitter=.1,)
        sns.boxplot(data=df, x="true_k",
                    y="inferred_k", hue="method",
                    hue_order=methods, showfliers=False)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.gca().legend(handles[0:len(methods)], labels[0:len(methods)])
        #plt.gca().set_title("all model conditions")

        #['M5_m5', 'M12_m7', 'M12_m12']
        if name == 'M5_m5':
            plt.gca().set_title("$|\Sigma| = 5$, $5$ mutations per cluster")
        if name == 'M12_m7':
            plt.gca().set_title("$|\Sigma| = 12$, $7$ mutations per cluster")
        if name == 'M12_m12':
            plt.gca().set_title("$|\Sigma| = 12$, $12$ mutations per cluster")
        #plt.gca().set_title("$|\Sigma| = 5$, $5$ mutations per cluster")
        plt.gca().set_xlabel("simulated number $k^*$ of clusters")
        plt.gca().set_ylabel("inferred number $k$ of clusters")
        plt.gca().set_ylim((-0.05,8.5))
        plt.gcf().set_size_inches(7, 5.5)
        plt.show()

    else:

        sns.stripplot(data=df, x="true_k",
                      y="selection_accuracy", hue="method",
                      hue_order=methods,
                      alpha=.4, dodge=True, linewidth=1, jitter=.1,)
        sns.boxplot(data=df, x="true_k",
                    y="selection_accuracy", hue="method",
                    hue_order=methods,
                    showfliers=False)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.gca().legend(handles[0:len(methods)], labels[0:len(methods)])

        if name == 'M5_m5':
            plt.gca().set_title("$|\Sigma| = 5$, $5$ mutations per cluster")
        if name == 'M12_m7':
            plt.gca().set_title("$|\Sigma| = 12$, $7$ mutations per cluster")
        if name == 'M12_m12':
            plt.gca().set_title("$|\Sigma| = 12$, $12$ mutations per cluster")


        #plt.gca().set_title("all model conditions")
        #plt.gca().set_title("$|\Sigma| = 12$, $7$ mutations per cluster")
        plt.gca().set_xlabel("simulated number $k^*$ of clusters")
        plt.gca().set_ylabel("fraction of correctly selected trees")
        #plt.gca().set_ylim((-0.05,1.05))
        plt.gca().set_ylim((0.55,1.05))
        plt.gcf().set_size_inches(7, 5.5)
        #plt.savefig(orf+"selection_accuracy_all.pdf")
        plt.show()




def doChoice(x):
    #x = x.data.numpy() + 1e-8

    #print (x.shape)
    #totalProb = np.sum(x, axis=1)
    #totalProb = totalProb.repeat(x.shape[1]).reshape(x.shape)
    #print (np.min(totalProb))
    #x = x / totalProb
    x = np.cumsum(x, axis=1)

    #print (x[:5])

    randVal = np.random.random(size=x.shape[0])
    randVal = randVal.repeat(x.shape[1]).reshape(x.shape)

    #print (randVal[:5])

    #'''
    x = randVal - x
    x2 = np.zeros(x.shape)
    x2[x > 0] = 1
    x2[x <= 0] = 0
    x2[:, -1] = 0

    #print (x[:5])

    x2 = np.sum(x2, axis=1)
    #'''

    #print (x2[:10])


    '''
    #x = randVal - x
    x[x > 0] = 1
    x[x <= 0.1] = 0
    x[:, -1] = 0
    x = np.sum(x, axis=1)
    #'''

    #print (x[:10])
    #quit()
    #return x
    return x2



def saveLungCancer(doSolution=False):
    dataList = []
    patientIdx = 0
    #for filename in ['./breast_Razavi.txt']:
    #for filename in ['./data/M5_m5/simulations_input/s0_k1_n50_M5_m5_simulations.txt']:
    #for filename in ['./data/M5_m5/simulations_input/s0_k2_n100_M5_m5_simulations.txt']:

    fileIn = './data/lungData/tracerx_lung.txt'

    for filename in [fileIn]:
        with open(filename) as f:

            if doSolution:
                notPatient = True
                while notPatient:
                    line = f.readline().rstrip("\n")
                    if line[-8:] == 'patients':
                        notPatient = False

            else:
                line = f.readline().rstrip("\n")

            #print (line)

            numPatient = int(line.split()[0])

            for loop1 in range(0, numPatient):
                #print (loop1)
                #for loop1 in range(0, 3):
                treeList = []

                if doSolution:
                    line = f.readline().rstrip("\n")

                line = f.readline().rstrip("\n")



                if line == "":
                    continue

                numTrees = int(line.split()[0])

                if doSolution:
                    numTrees = 1


                #if numTrees > 10000:
                #    continue
                #if numTrees == 0:
                #    continue



                #print (numTrees)

                line2 = line
                ok = True
                for treeIdx in range(numTrees):
                    #print ("Base Tree")
                    line = f.readline().rstrip("\n")

                    assert 'edges' in line

                    numEdges = int(line.split()[0])

                    #print (numEdges)

                    #print (numEdges)
                    if numEdges == 0:
                        ok = False
                        continue
                    #if treeIdx == 0:
                    #    print (line2, "for patient", os.path.basename(filename).rstrip(".trees"))

                    #print (numEdges + 1, "#edges, tree", treeIdx)
                    tree = []
                    vertices = set([])
                    inner_vertices = set([])
                    for edgeIdx in range(numEdges):
                        line = f.readline().rstrip("\n")

                        #print (line)
                        s = line.split()
                        #print (s)

                        s1 = s[0].split(';')[-1]
                        s2 = s[1].split(';')

                        #print (s1)
                        #print (s2)


                        tree.append([s1, s2[0]])
                        vertices.add(tree[-1][0])
                        vertices.add(tree[-1][1])
                        inner_vertices.add(tree[-1][1])

                        for c in range(len(s2) - 1):

                            tree.append([s2[c], s2[c+1]])

                            vertices.add(tree[-1][0])
                            vertices.add(tree[-1][1])
                            inner_vertices.add(tree[-1][1])


                        #tree.append([s[0].split(":")[-1], s[1].split(":")[-1]])

                    #print (tree)

                    if len(set.difference(vertices, inner_vertices)) != 1:
                        print (vertices)
                        print (inner_vertices)
                        print (set.difference(vertices, inner_vertices))
                    assert len(set.difference(vertices, inner_vertices)) == 1
                    root = list(set.difference(vertices, inner_vertices))[0]
                    #for edge in tree:
                    #    print (edge[0], edge[1])
                    #print ("GL", root)

                    treeList.append(tree)
                if ok:
                    patientIdx += 1


                dataList.append(treeList)


    #####np.save('./treeData.npy', dataList)
    #####np.save('./treeDataSim2.npy', dataList)

    dataList1 = np.array(dataList, dtype=object)

    np.save('./data/lungData/processed.npy', dataList1)


#saveLungCancer()
#quit()


def transformManualData():

    def doUniqueEdge(tree1):
        tree1 = np.array(tree1)
        edges2 = []
        for b in range(0, tree1.shape[0]):
            edges2.append(str(tree1[b][0]) + ':' + str(tree1[b][1]) )
        edges2 = np.array(edges2)
        edges2, index = np.unique(edges2, return_index=True)
        tree1 = tree1[index]
        return tree1


    #print ("Hi")
    #data = np.loadtxt('./data/manualCan1.txt', dtype=str)

    file1 = open('./data/manualCan3.txt', 'r')
    data1 = file1.readlines()
    file2 = open('./data/manualCan2.txt', 'r')
    data2 = file2.readlines()
    file3 = open('./data/manualCan1.txt', 'r')
    data3 = file3.readlines()
    data1 = data1 + data2 + data3

    fullTrees = []
    patientsDone = []
    treeNow = []
    patientTrees = []
    treeDict = {}
    treeNum = -1
    doingData = False
    toAddTree = False
    isNumBefore = False
    for a in range(0, len(data1)):
        string1 = data1[a]
        string1 = string1.replace('\n', '')
        string1 = string1.replace(']', '')
        string1 = string1.replace('[', '')
        string1 = string1.replace(' ', '')
        string1 = string1.replace('’', '')
        string1 = string1.replace("'", '')
        string1 = string1.replace("‘", '')


        #print (string1)

        isData = False
        isNum = False
        isSpace = False
        if ',' in string1:
            isData = True
        try:
            int(string1)
            isNum = True
            doingData = False
        except:
            True

        if string1 == '':
            isSpace = True
            doingData = False

        #print (string1)


        if (isData == False) and (isNum == False) and (isSpace == False):
            print ("ERROR")
            quit()


        if isNum:
            treeNum = int(string1)
            if not (treeNum in patientsDone):
                toAddTree = True
                patientsDone.append(treeNum)
            else:
                toAddTree = False

        #print (patientsDone)
        #print (toAddTree)

        if toAddTree:


            if isData:
                if doingData == False:
                    #Add tree!!
                    if a != 1:
                        #print (treeNow)

                        treeNow = doUniqueEdge(treeNow)

                        patientTrees.append(copy.deepcopy(treeNow))
                        treeNow = []
                        treeDict = {}

                doingData = True
                locData = string1.split(',')
                #print (locData)
                for b in range(0, len(locData) - 1):

                    if not (locData[b+1] in treeDict.keys()):
                        treeDict[locData[b+1]] = locData[b]
                    else:
                        if treeDict[locData[b+1]] != locData[b]:
                            print ("Issue")
                            print (treeDict[locData[b+1]])
                            print (locData[b+1], locData[b])
                            quit()


                    edge = [locData[b], locData[b+1]]
                    treeNow.append(copy.copy(edge))

        #if len(patientsDone) == 1:
        #    print (string1)

        if isNumBefore:
            #if a != 0:
            #print (patientTrees)
            #quit()
            if False:#len(patientsDone) == 2:
                print ("Hi", treeNum)
                print (a, len(fullTrees))

                print ('patientTrees ', len(patientTrees))
                print (np.unique(np.array(patientTrees[0])))
                print (np.unique(np.array(patientTrees[1])))
                print (np.unique(np.array(patientTrees[2])))
                #print (np.unique(np.array(patientTrees[0])))
                #print (np.unique(np.array(patientTrees[1])))
                #quit()
                print ('')
                print ('')
                print ('')
                #print
            fullTrees.append(copy.deepcopy(patientTrees))
            patientTrees = []

        if toAddTree and (a != 0):
            isNumBefore = isNum

        #if a == len(data1) - 1:
        #    print ("Info")
        #    print (string1)

    patientTrees.append(copy.deepcopy(treeNow))
    fullTrees.append(copy.deepcopy(patientTrees))

    #print (len(fullTrees))
    #print (len(patientsDone))
    #quit()
    numberOfTrees = 0

    mutAll1 = np.array([])
    treeNumAll = np.array([])
    for a in range(0, len(fullTrees)):

        mutAll0 = np.array([])
        N1 = len(fullTrees[a])
        numberOfTrees += N1
        mutFull = []
        for b in range(0, N1):
            #print (fullTrees[a][b])
            #quit()
            mutations = []
            for c in range(0, len(fullTrees[a][b])):
                mutations.append(fullTrees[a][b][c][0])
                mutations.append(fullTrees[a][b][c][1])
            mutations = np.array(np.unique(mutations))
            mutAll0 = np.concatenate((mutAll0, mutations ))
            mutFull.append(np.copy(mutations))

        for b in range(0, N1):
            for c in range(0, N1):
                s1, s2, s3 = mutFull[b].shape[0], mutFull[c].shape[0], np.intersect1d(mutFull[b], mutFull[c]).shape[0]
                if (s3 != s1) or (s3 != s2):

                    print (a, patientsDone[a])
                    #print (b, c)
                    print (s1, s2, s3)
                    print (mutFull[b][np.isin(mutFull[b], mutFull[c]) == False])
                    print (mutFull[c][np.isin(mutFull[c], mutFull[b]) == False])
                    print ("Issue2")
                    quit()

        #mutAll0 = np.unique(mutAll0)
        if mutAll0.shape[0] == 0:
            print (a)
            print ("Issue3")
            quit()

        if 'Root' not in mutAll0:
            print ("Issue4")
            quit()
        mutAll1 = np.concatenate((mutAll1, mutAll0))
        treeNumAll = np.concatenate((treeNumAll, np.zeros(mutAll0.shape[0]) + patientsDone[a] ))

    #unique1, indices1, counts1 = np.unique(mutAll1, return_counts=True, return_index=True)
    #for a in range(0, unique1.shape[0]):
    #    print (unique1[a], counts1[a], treeNumAll[indices1[a]])
    patientsDone = np.array(patientsDone).astype(int)
    fullTrees2 = []
    for a in range(0, len(patientsDone)):
        #print (a+1)
        #print ( np.argwhere(patientsDone == (a+1) ))
        #print (np.unique(patientsDone))
        arg1 = np.argwhere(patientsDone == (a+1) )[0, 0]
        fullTrees2.append(copy.deepcopy(fullTrees[a]))

    print (numberOfTrees)

    np.save('./data/manualCancer.npy', fullTrees2)
    #np.savez_compressed('./data/manualCan_InitialProcessNum.npz', patientsDone)

#transformManualData()
#quit()



def saveTreeList(fileIn, fileOut, doSolution=False):
    dataList = []
    patientIdx = 0
    #for filename in ['./breast_Razavi.txt']:
    #for filename in ['./data/M5_m5/simulations_input/s0_k1_n50_M5_m5_simulations.txt']:
    #for filename in ['./data/M5_m5/simulations_input/s0_k2_n100_M5_m5_simulations.txt']:
    for filename in [fileIn]:
        with open(filename) as f:

            if doSolution:
                notPatient = True
                while notPatient:
                    line = f.readline().rstrip("\n")
                    if line[-8:] == 'patients':
                        notPatient = False

            else:
                line = f.readline().rstrip("\n")

            #print (line)

            numPatient = int(line.split()[0])

            for loop1 in range(0, numPatient):
                #print (loop1)
                #for loop1 in range(0, 3):
                treeList = []

                if doSolution:
                    line = f.readline().rstrip("\n")

                line = f.readline().rstrip("\n")

                #print (line)

                #quit()

                #print ("Base Line")
                #print (line)

                #if loop1:
                #    print (line)
                #    quit()

                if line == "":
                    continue

                numTrees = int(line.split()[0])

                if doSolution:
                    numTrees = 1


                #if numTrees > 10000:
                #    continue
                #if numTrees == 0:
                #    continue



                #print (numTrees)

                line2 = line
                ok = True
                for treeIdx in range(numTrees):
                    #print ("Base Tree")
                    line = f.readline().rstrip("\n")

                    assert 'edges' in line

                    numEdges = int(line.split()[0])

                    #print (numEdges)

                    #print (numEdges)
                    if numEdges == 0:
                        ok = False
                        continue
                    #if treeIdx == 0:
                    #    print (line2, "for patient", os.path.basename(filename).rstrip(".trees"))

                    #print (numEdges + 1, "#edges, tree", treeIdx)
                    tree = []
                    vertices = set([])
                    inner_vertices = set([])
                    for edgeIdx in range(numEdges):
                        line = f.readline().rstrip("\n")
                        #print (line)
                        s = line.split()
                        tree.append([s[0].split(":")[-1], s[1].split(":")[-1]])
                        vertices.add(tree[-1][0])
                        vertices.add(tree[-1][1])
                        inner_vertices.add(tree[-1][1])

                    if len(set.difference(vertices, inner_vertices)) != 1:
                        print (vertices)
                        print (inner_vertices)
                        print (set.difference(vertices, inner_vertices))
                    assert len(set.difference(vertices, inner_vertices)) == 1
                    root = list(set.difference(vertices, inner_vertices))[0]
                    #for edge in tree:
                    #    print (edge[0], edge[1])
                    #print ("GL", root)

                    treeList.append(tree)
                if ok:
                    patientIdx += 1


                dataList.append(treeList)


    #####np.save('./treeData.npy', dataList)
    #####np.save('./treeDataSim2.npy', dataList)

    dataList1 = np.array(dataList, dtype=object)

    np.save(fileOut, dataList1)







#saveTreeList('./data/M5_m5/simulations_input/s4_k1_n50_M5_m5_simulations.txt', './data/breastCancer.npy')
#saveTreeList('./data/breast_Razavi.txt', './data/breastCancer.npy')
#quit()

def saveTreeListSim():
    import os

    names1 = ['M5_m5', 'M12_m7', 'M12_m12']
    #names1 = ['M12_m12']
    #names1 = ['M5_m5']
    for name in names1:
        #['simulations_solution', '.DS_Store', 'simulations_input']
        arr = os.listdir('./data/' + name + '/simulations_input')
        for name2 in arr:

            #print ('./data/' + name + '/simulations_solution/' + name2)
            #quit()
            fileIn = './data/' + name + '/simulations_input/' + name2
            fileOut = './data/p_' + name + '/simulations_input/' + name2
            saveTreeList(fileIn, fileOut)
            quit()

            #fileIn = './data/' + name + '/simulations_solution/' + name2
            #fileOut = './data/p_' + name + '/simulations_solution/' + name2
            #saveTreeList(fileIn, fileOut, doSolution=True)


#saveTreeListSim()
#quit()


def saveAllTrees(N):

    L = 101
    trees = np.zeros((1, N+1, 3)).astype(int)
    trees[:] = L - 1

    for a in range(N):

        #print ("A")
        size1 = a+1
        trees2 = trees.reshape((trees.size,)).repeat(N*size1)
        trees2 = trees2.reshape((trees.shape[0], trees.shape[1], trees.shape[2], N, size1 ))

        for b in range(size1):
            trees2[:, size1, 0, :, b] = np.copy(trees2[:, b, 1, :, 0])
            trees2[:, size1, 2, :, b] = b
        for b in range(N):
            #print (b)
            trees2[:, size1, 1, b, :] = b
            #trees2[:, size1, 1, :, b] = trees2[:, a, 0, :, b]

        trees2 = np.swapaxes(trees2, 1, 3)
        trees2 = np.swapaxes(trees2, 2, 4)

        trees2 = trees2.reshape((trees.shape[0] * N * size1, trees.shape[1], trees.shape[2] ))


        #Eliminate duplicate mutations
        tree_end = np.copy(trees2[:, :a+2, 1])
        tree_end = np.sort(tree_end, axis=1)
        tree_end = tree_end[:, 1:] - tree_end[:, :-1]
        tree_end = np.min(np.abs(tree_end), axis=1)
        trees2 = trees2[tree_end != 0]






        #Reducing to unique trees
        trees2_edgenum = (trees2[:, :, 0] * L) + trees2[:, :, 1]
        trees2_edgenum = np.sort(trees2_edgenum, axis=1)

        trees2_str = trees2_edgenum.astype(str)
        trees2_str_sep = np.zeros(trees2_str.shape).astype(str)
        trees2_str_sep[:] = ':'

        trees2_str = np.char.add(trees2_str, trees2_str_sep)

        trees2_str_full = np.zeros(trees2_str.shape[0]).astype(str)
        trees2_str_full[:] = ''
        for b in range(trees2_str.shape[1]):
            trees2_str_full = np.char.add(trees2_str_full, np.copy(trees2_str[:, b]))

        _, uniqueArgs = np.unique(trees2_str_full, return_index=True)
        trees2 = trees2[uniqueArgs]

        trees = np.copy(trees2)

    trees = trees[:, 1:]

    clones = np.zeros((trees.shape[0], N+1, N))

    for a in range(N):

        clonesNow = clones[np.arange(trees.shape[0]), trees[:, a, 2] ]
        clonesNow[np.arange(trees.shape[0]), trees[:, a, 1]] = 1
        clones[:, a+1] = np.copy(clonesNow)

    clones = clones[:, 1:]

    print (clones.shape[0])

    np.savez_compressed('./data/allTrees/edges/' + str(N) + '.npz', trees)
    np.savez_compressed('./data/allTrees/clones/' + str(N) + '.npz', clones)

    #allTrees

#saveAllTrees(7)
#quit()

def bulkFrequencyPossible(freqs, M):

    clones = loadnpz('./data/allTrees/clones/' + str(M) + '.npz')
    #clones_sum = np.sum(clones, axis=1)
    #clones_sum_diff = clones_sum[:, 1:] - clones_sum[:, :-1]
    #clones_sum_diff = np.min(clones_sum_diff, axis=1)
    clones_0 = np.copy(clones)
    clones = np.swapaxes(clones, 1, 2)
    clones_shape = clones.shape

    #print (clones)


    freq_shape = freqs.shape

    freqs = freqs.reshape((freq_shape[0]*freq_shape[1], freq_shape[2]))
    freqs = freqs.T


    clones = np.linalg.inv(clones)

    clones = clones.reshape((clones_shape[0] * clones_shape[1], clones_shape[2]))

    mixture = np.matmul(clones, freqs)


    mixture = mixture.reshape((clones_shape[0], clones_shape[1], freq_shape[0], freq_shape[1]))
    #mixture = mixture.reshape((clones_shape[0], clones_shape[1], freq_shape[1] * freq_shape[0]))



    mixture_sum = np.sum(np.copy(mixture), axis=1)
    mixture_sum = np.max(mixture_sum, axis=2).T

    mixture = np.min(mixture, axis=1)
    mixture = np.min(mixture, axis=2).T

    #print (mixture)
    #print (mixture_sum)

    #ep = 1e-4 #This is required otherwise sometimes there are issues when mixture_sum == 1 exactly, for example.
    ep = 1e-6
    argsGood = np.argwhere(np.logical_and(mixture >= 0 - ep, mixture_sum <= 1 + ep))

    #print (argsGood[:, 0])
    #print (clones_0[argsGood[:, 1]])

    sampleInverse = np.copy(argsGood[:, 0])

    trees = loadnpz('./data/allTrees/edges/' + str(M) + '.npz')
    trees = trees[argsGood[:, 1]][:, :, :2]

    #print (clones_0[argsGood[2, 1]])
    #print (clones_0[argsGood[3, 1]])
    #quit()

    return trees, sampleInverse


def testingBulkFreq():

    M = 3
    S = 2
    N = 2
    freqs = np.zeros((N, S, M))

    freqs[0, 0, 0] = 0.9
    freqs[0, 0, 1] = 0.4
    freqs[0, 0, 2] = 0.3

    freqs[0, 1, 0] = 0.9
    freqs[0, 1, 1] = 0.3
    freqs[0, 1, 2] = 0.4

    freqs[1, 0, 0] = 0.7
    freqs[1, 0, 1] = 0.8
    freqs[1, 0, 2] = 0.9

    bulkFrequencyPossible(freqs, M)
    quit()


def simulationBulkFrequency(clones, M, S):

    clones_sum = np.sum(clones, axis=1)
    clones_argsort = np.argsort(clones_sum, axis=1)
    clones_argsort = clones_argsort[:, -1::-1]
    allArgs = np.argwhere(clones > -100)
    allArgs[:, 2] = clones_argsort[allArgs[:, 0], allArgs[:, 2]]


    clones_flat = clones[allArgs[:, 0], allArgs[:, 1], allArgs[:, 2]]
    clones = clones_flat.reshape(clones.shape)

    clones = clones[:, 1:, :M]

    N = clones.shape[0]

    clones_shape = clones.shape
    clones = clones.reshape((clones.size,))
    clones = clones.repeat(S)
    clones = clones.reshape((clones_shape[0], clones_shape[1] * clones_shape[2], S))
    clones = np.swapaxes(clones, 1, 2)
    clones = clones.reshape((clones_shape[0] * S, clones_shape[1], clones_shape[2]))


    freqs = np.random.random((clones.shape[0], clones.shape[1]))
    freqs_sum = np.sum(freqs, axis=1).repeat(clones.shape[1]).reshape(freqs.shape)
    freqs = freqs / freqs_sum
    freqs = freqs.reshape((freqs.size,)).repeat(clones.shape[2]).reshape(clones.shape)
    freqs = np.sum(freqs * clones, axis=1)

    freqs = freqs.reshape((N, S, M))

    trees, sampleInverse = bulkFrequencyPossible(freqs, M)

    sampleInverse_reshape = sampleInverse.repeat(trees.shape[1]).repeat(trees.shape[2]).reshape(trees.shape)

    trees_copy = np.copy(trees)
    trees[trees == 100] = -1
    trees = clones_argsort[sampleInverse_reshape, trees]
    trees[trees_copy == 100] = 100


    return trees, sampleInverse

def multisizeBulkFrequency(clones, treeSizes, S):

    uniqueSizes = np.unique(treeSizes)

    maxSize = int(np.max(uniqueSizes))
    fullTree = np.zeros((0, maxSize, 2))
    fullSampleInverse = np.zeros(0)

    for a in range(len(uniqueSizes)):

        argSize = np.argwhere(uniqueSizes[a] == treeSizes)[:, 0]

        clonesNow = clones[argSize]

        treesNow, sampleInverseNow = simulationBulkFrequency(clonesNow, uniqueSizes[a], S)

        treesNow2 = np.zeros((treesNow.shape[0], maxSize, 2))
        treesNow2[:] = 101
        treesNow2[:, :treesNow.shape[1], :] = treesNow
        fullTree = np.concatenate((fullTree, np.copy(treesNow2)))

        #print ("A")
        #print (treesNow2[0])

        #if fullSampleInverse.shape[0] > 0:
        #    sampleInverseNow = sampleInverseNow + 1 + int(np.max(fullSampleInverse))

        sampleInverseNow = argSize[sampleInverseNow.astype(int)]

        fullSampleInverse = np.concatenate((fullSampleInverse, np.copy(sampleInverseNow)))

    fullSampleInverse_argsort = np.argsort(fullSampleInverse)

    fullTree = fullTree[fullSampleInverse_argsort]
    fullSampleInverse = fullSampleInverse[fullSampleInverse_argsort]


    return fullTree, fullSampleInverse


def makePartSimulation(probabilityMatrix, mutationTypeMatrix, mutationTypeMatrix_extended, N, M, K, treeSize, maxLogProb=1000, useRequirement=False, reqMatrix=False):



    #N = 100
    clones = np.zeros((N, treeSize+1, M))

    edges = np.zeros((N, treeSize+1, 2)).astype(int)
    edges[:] = M

    for a in range(0, treeSize):

        treePos = a + 1

        clonesNow = clones[:, :treePos]
        clonesNow = clonesNow.reshape((N*treePos, M))


        #if a == 1:
        #    print ("A")
        #    print (clonesNow[1])

        clonesNow = np.matmul(clonesNow, mutationTypeMatrix_extended)
        clonesNow[:, K] = 1
        clonesNow[clonesNow > 1] = 1


        if useRequirement:

            clonesNow = np.matmul(clonesNow, reqMatrix)
            clonesNow[clonesNow < 1] = 0
            clonesNow[clonesNow >= 1] = 1

            #if a == 1:
            #    print (clonesNow[1])

        clonesNow = np.matmul(clonesNow, probabilityMatrix)
        clonesNow[clonesNow > maxLogProb] = maxLogProb
        clonesNow = np.matmul(clonesNow, mutationTypeMatrix.T)

        #print (clonesNow[0])
        #quit()

        #if a == 1:
        #    print (clonesNow[1])
            #quit()

        #plt.plot(clonesNow[0])
        #plt.show()

        clonesNow = clonesNow.reshape((N, treePos*M))

        clonesNow_max = np.max(clonesNow, axis=1)
        clonesNow_max = clonesNow_max.repeat(clonesNow.shape[1]).reshape(clonesNow.shape)
        clonesNow = clonesNow - clonesNow_max
        clonesNow = np.exp(clonesNow)

        clonesNow = clonesNow.reshape((N, treePos, M))
        for b in range(a):
            for c in range(a+1):
                clonesNow[np.arange(N), c, edges[:, b+1, 1]] = 0
        clonesNow = clonesNow.reshape((N, treePos*M))

        clonesNow_sum = np.sum(clonesNow, axis=1)
        clonesNow_sum = clonesNow_sum.repeat(clonesNow.shape[1]).reshape(clonesNow.shape)
        clonesNow = clonesNow / clonesNow_sum

        #plt.plot(clonesNow[0])
        #plt.show()


        choicePoint = doChoice(clonesNow)


        clonePoint = (choicePoint // M).astype(int)
        mutPoint = (choicePoint % M).astype(int)

        #print ('MUT', mutPoint[0])

        #edges[:, a+1, 2] = np.copy(clonePoint)
        edges[:, a+1, 1] = np.copy(mutPoint)
        edges[:, a+1, 0] = np.copy(edges[np.arange(N), clonePoint, 1])

        cloneSelect = np.copy(clones[np.arange(N), clonePoint])
        cloneSelect[np.arange(N), mutPoint] = 1
        clones[:, a+1] = np.copy(cloneSelect)

        #print (clones[0])
        #quit()


    #print (edges[0])
    #print (edges[1])
    #print (edges[2])
    #print (np.unique(edges[1:, 1], return_counts=True))
    #quit()

    #print (edges[5])
    #quit()
    edges = edges[:, 1:]

    return edges, clones


def makeSimulation():




    S = 3

    M = 10
    K = 6

    #mutationType = np.random.randint(50, size=M)
    #mutationType[mutationType >= K] = K - 1

    mutationType = np.arange(M)
    mutationType[mutationType >= K] = K - 1

    mutationTypeMatrix_extended = np.zeros((M, K+1))
    mutationTypeMatrix_extended[np.arange(M), mutationType] = 1
    #mutationTypeMatrix_extended[:, K] = 1

    mutationTypeMatrix = np.copy(mutationTypeMatrix_extended[:, :-1])


    probabilityMatrix = np.random.randint(2, size=(K+1) *  K).reshape((K+1, K))
    probabilityMatrix = (probabilityMatrix - 0.5) * 6


    probabilityMatrix[K] = 0
    probabilityMatrix[K-1, :] = 0
    probabilityMatrix[:, K-1] = 0



    #probabilityMatrix = np.zeros((K+1, K))
    #probabilityMatrix[:, :3] = 10

    #probabilityMatrix[:, :K-1] = probabilityMatrix[:, :K-1] + 4

    treeSize = min(5, M)
    N = 1000


    edges, clones = makePartSimulation(probabilityMatrix, mutationTypeMatrix, mutationTypeMatrix_extended, N, M, K, treeSize)


    np.savez_compressed('./data/specialSim/temp_trees_0.npz', edges)
    np.savez_compressed('./data/specialSim/temp_mutationType_0.npz', mutationType)
    np.savez_compressed('./data/specialSim/temp_prob_0.npz', probabilityMatrix)

    trees, sampleInverse = simulationBulkFrequency(clones, treeSize, S)

    #print (N)
    #print (trees.shape)
    #quit()

    np.savez_compressed('./data/specialSim/temp_bulkTrees_0.npz', trees)
    np.savez_compressed('./data/specialSim/temp_bulkSample_0.npz', sampleInverse)


#makeSimulation()
#quit()



def makeOccurSimulation():

    for a in range(2, 20):

        #T0 = default
        #T3 = randomly 5 to 7 tree length

        print (a)

        #S = 3
        S = 5

        M = 10
        K = 6

        #mutationType = np.random.randint(50, size=M)
        #mutationType[mutationType >= K] = K - 1

        mutationType = np.arange(M)
        mutationType[mutationType >= K] = K - 1

        mutationTypeMatrix_extended = np.zeros((M, K+1))
        mutationTypeMatrix_extended[np.arange(M), mutationType] = 1
        #mutationTypeMatrix_extended[:, K] = 1

        mutationTypeMatrix = np.copy(mutationTypeMatrix_extended[:, :-1])


        probabilityMatrix = np.random.randint(2, size=(K+1) *  K).reshape((K+1, K))
        probabilityMatrix = probabilityMatrix * np.log(11)


        probabilityMatrix[K] = 0
        probabilityMatrix[K-1, :] = 0
        probabilityMatrix[:, K-1] = 0



        #probabilityMatrix = np.zeros((K+1, K))
        #probabilityMatrix[:, :3] = 10

        #probabilityMatrix[:, :K-1] = probabilityMatrix[:, :K-1] + 4

        treeSize = min(7, M)
        N = 1000
        #N = 100


        edges, clones = makePartSimulation(probabilityMatrix, mutationTypeMatrix, mutationTypeMatrix_extended, N, M, K, treeSize)

        if True:

            treeSizes = np.random.randint(3, size=edges.shape[0]) + 5

            for b in range(len(treeSizes)):
                size1 = treeSizes[b]
                edges[b, size1:] = M

            #print (treeSizes.shape)
            #print (treeSizes[:10])
            #quit()


        #print (edges[0])

        np.savez_compressed('./data/specialSim/dataSets/T_4_R_' + str(a) + '_treeSizes.npz', treeSizes)
        np.savez_compressed('./data/specialSim/dataSets/T_4_R_' + str(a) + '_trees.npz', edges)
        np.savez_compressed('./data/specialSim/dataSets/T_4_R_' + str(a) + '_mutationType.npz', mutationType)
        np.savez_compressed('./data/specialSim/dataSets/T_4_R_' + str(a) + '_prob.npz', probabilityMatrix)

        #trees, sampleInverse = simulationBulkFrequency(clones, treeSize, S)
        trees, sampleInverse = multisizeBulkFrequency(clones, treeSizes, S)

        print (trees.shape)
        #quit()


        trees[trees == 100] = M
        trees[trees == 101] = M + 1

        np.savez_compressed('./data/specialSim/dataSets/T_4_R_' + str(a) + '_bulkTrees.npz', trees)
        np.savez_compressed('./data/specialSim/dataSets/T_4_R_' + str(a) + '_bulkSample.npz', sampleInverse)

        #quit()

#makeOccurSimulation()
#quit()




def makePathwaySimulation():

    for saveNum in range(100):

        #int1 = np.random.randint(100)
        #print (int1)
        #np.random.seed(int1)


        #print (a)

        #S = 3
        S = 5

        M = 20
        #K = 4
        #L = 3

        #pathways = [[  [0, 1], [2, 3]  ]]

        #pathways = [[  [0], [1, 2], [3, 4]  ]]
        #pathways = [[  [0, 1, 2], [3, 4], [5, 6, 7]  ]]

        #pathways = [[  [0], [1, 2], [3]  ], [  [4, 5], [6], [7]  ]]

        #pathways = [[  [0, 1], [2, 3]  ] ]


        #Forgot to allow into pathway

        pathways = []

        #np.random.seed(6)

        sizes = np.random.randint(3, size=6) + 1

        pathwayNum = np.random.randint(4) + 1
        if pathwayNum > 2:
            pathwayNum = 2

        #print (pathwayNum)

        count1 = 0
        for b in range(pathwayNum):
            pathways.append([])
            for c in range(3):
                size1 = sizes[(b * 3) + c]
                pathways[b].append([])
                for d in range(size1):
                    pathways[b][c].append(count1 + d)
                count1 += size1

        #print (sizes)
        #print (pathways)
        #quit()





        #print (sizes)
        #quit()

        #pathways = [[  [0, 1], [2, 3] ], [  [4, 5], [6]  ]]

        #pathways = [[  [0], [1]  ], [  [2], [3]  ]]
        #pathways = [[  [4, 5], [6], [7]  ]]
        #pathways = [[  [0, 1], [6], [7]  ]]

        #mutationType = np.random.randint(50, size=M)
        #mutationType[mutationType >= K] = K - 1

        mutationType = (np.zeros(M) - 1).astype(int)
        #mutationType[mutationType >= L] = L - 1

        reducedPathways = []
        c = 0
        for a in range(len(pathways)):
            reducedPathways.append([])
            for b in range(len(pathways[a])):
                reducedPathways[a].append(c)
                mutationType[np.array(pathways[a][b]).astype(int)] = c
                c += 1

        K = c + 1

        mutationType[mutationType == -1] = K - 1

        propertyRequirement = np.zeros((K+1, K+1))
        propertyRequirement[:, K] = 1

        probabilityMatrix = np.zeros((K+1, K))

        Amplify = 21# * 100

        probabilityMatrix[K, reducedPathways[0][0]] = np.log(6)
        if len(reducedPathways) >= 2:
            probabilityMatrix[K, reducedPathways[1][0]] = np.log(6)

        #probabilityMatrix[K, reducedPathways[0][0]] = np.log(Amplify / ( len(reducedPathways) * 2 ) )
        #probabilityMatrix[K, reducedPathways[1][0]] = np.log(Amplify / ( len(reducedPathways) * 2 ) )

        #print (probabilityMatrix)
        #quit()



        for a in range(len(reducedPathways)):
            pathway1 = np.array(reducedPathways[a]).astype(int)
            for b in range(len(pathway1) - 1):
                in1 = pathway1[:b+1]
                out1 = pathway1[b+1]
                outRem1 = pathway1[b]

                propertyRequirement[in1, out1] = (1 / in1.shape[0]) + 1e-2

                #probabilityMatrix[out1, out1] = np.log(11)
                if True:#a == 1: #TODO REMOVE!
                    #print (in1)
                    #print (out1)
                    probabilityMatrix[out1, out1] = np.log(Amplify) #* (b + 1)
                    probabilityMatrix[out1, outRem1] = -1 * np.log(Amplify) #* (b + 1)

        #print (mutationType)



        #print (propertyRequirement)
        #quit()


        mutationTypeMatrix_extended = np.zeros((M, K+1))
        mutationTypeMatrix_extended[np.arange(M), mutationType] = 1
        #mutationTypeMatrix_extended[:, K] = 1

        mutationTypeMatrix = np.copy(mutationTypeMatrix_extended[:, :-1])


        treeSize = min(7, M)
        N = 1000


        edges, clones = makePartSimulation(probabilityMatrix, mutationTypeMatrix, mutationTypeMatrix_extended, N, M, K, treeSize, maxLogProb=np.log(10000), useRequirement=True, reqMatrix=propertyRequirement)

        if True:

            treeSizes = np.random.randint(3, size=edges.shape[0]) + 5

            for b in range(len(treeSizes)):
                size1 = treeSizes[b]
                edges[b, size1:] = M


        pathways_save = np.array(pathways, dtype=object)
        np.savez_compressed('./data/specialSim/dataSets/T_6_R_' + str(saveNum) + '_pathway.npz', pathways_save)


        np.savez_compressed('./data/specialSim/dataSets/T_6_R_' + str(saveNum) + '_treeSizes.npz', treeSizes) #Was 4 instead of 5. saved over at least 3

        np.savez_compressed('./data/specialSim/dataSets/T_6_R_' + str(saveNum) + '_trees.npz', edges)
        np.savez_compressed('./data/specialSim/dataSets/T_6_R_' + str(saveNum) + '_mutationType.npz', mutationType)
        np.savez_compressed('./data/specialSim/dataSets/T_6_R_' + str(saveNum) + '_prob.npz', probabilityMatrix)

        #trees, sampleInverse = simulationBulkFrequency(clones, treeSize, S)
        trees, sampleInverse = multisizeBulkFrequency(clones, treeSizes, S)

        trees[trees == 100] = M
        trees[trees == 101] = M + 1

        np.savez_compressed('./data/specialSim/dataSets/T_6_R_' + str(saveNum) + '_bulkTrees.npz', trees)
        np.savez_compressed('./data/specialSim/dataSets/T_6_R_' + str(saveNum) + '_bulkSample.npz', sampleInverse)


        print (saveNum)
        print (trees.shape)
        #quit()

#makePathwaySimulation()
#quit()



def makeLatentSimulation():

    for a in range(1):


        print (a)

        S = 3

        M = 10
        K = 3

        #mutationType = np.random.randint(50, size=M)
        #mutationType[mutationType >= K] = K - 1

        mutationType = np.arange(M)
        mutationType[mutationType >= K] = K - 1

        mutationTypeMatrix_extended = np.zeros((M, K+1))
        mutationTypeMatrix_extended[np.arange(M), mutationType] = 1
        #mutationTypeMatrix_extended[:, K] = 1

        mutationTypeMatrix = np.copy(mutationTypeMatrix_extended[:, :-1])


        #probabilityMatrix = np.random.randint(2, size=(K+1) *  K).reshape((K+1, K))
        #probabilityMatrix = probabilityMatrix * np.log(11)

        probabilityMatrix = np.zeros((K+1, K))
        probabilityMatrix[:, 0] = probabilityMatrix[:, 0] + np.log(10)
        probabilityMatrix[1, :] = probabilityMatrix[1, :] + np.log(10)


        treeSize = min(5, M)
        N = 2000


        edges, clones = makePartSimulation(probabilityMatrix, mutationTypeMatrix, mutationTypeMatrix_extended, N, M, K, treeSize)


        np.savez_compressed('./data/specialSim/dataSets/T_2_R_' + str(a) + '_trees.npz', edges)
        np.savez_compressed('./data/specialSim/dataSets/T_2_R_' + str(a) + '_mutationType.npz', mutationType)
        np.savez_compressed('./data/specialSim/dataSets/T_2_R_' + str(a) + '_prob.npz', probabilityMatrix)

        trees, sampleInverse = simulationBulkFrequency(clones, treeSize, S)

        np.savez_compressed('./data/specialSim/dataSets/T_2_R_' + str(a) + '_bulkTrees.npz', trees)
        np.savez_compressed('./data/specialSim/dataSets/T_2_R_' + str(a) + '_bulkSample.npz', sampleInverse)


#makeLatentSimulation()
#quit()


def processTreeData(maxM, fileIn):

    #treeData = np.load('./data/treeData.npy', allow_pickle=True)

    #treeData = np.load('./data/treeDataSim2.npy', allow_pickle=True)
    #treeData = np.load('./data/manualCancer.npy', allow_pickle=True)

    treeData = np.load(fileIn, allow_pickle=True)


    MVal = 100

    sampleInverse = np.zeros(100000)
    treeLength = np.zeros(100000)
    newTrees = np.zeros((100000, maxM, 2)).astype(str)
    lastName = 'ZZZZZZZZZZZZZZZZ'
    firstName = 'ZZZZZZZZZZ'
    newTrees[:] = lastName

    count1 = 0
    for a in range(0, len(treeData)):
        treeList = treeData[a]
        treeList = np.array(treeList)

        if treeList.shape[1] <= maxM:
            size1 = treeList.shape[0]

            newTrees[count1:count1+size1, :treeList.shape[1]] = treeList
            treeLength[count1:count1+size1] = treeList.shape[1]
            sampleInverse[count1:count1+size1] = a
            count1 += size1



    newTrees = newTrees[:count1]
    newTrees[newTrees == 'Root'] = 'GL'
    if ('0' in newTrees) and not ('GL' in newTrees):
        newTrees[newTrees == '0'] = firstName
    else:
        newTrees[newTrees == 'GL'] = firstName
    treeLength = treeLength[:count1]
    sampleInverse = sampleInverse[:count1]
    shape1 = newTrees.shape
    newTrees = newTrees.reshape((newTrees.size,))


    #uniqueMutation =  np.unique(newTrees)
    #for name in uniqueMutation:
    #    name1 = name.split('_')[0]
    #    #print (name, name1)
    #    newTrees[newTrees == name] = name1
    #quit()


    uniqueMutation, newTrees = np.unique(newTrees, return_inverse=True)

    uniqueMutation2 = []
    for name in uniqueMutation:
        name1 = name.split('_')[0]
        uniqueMutation2.append(name1)
    uniqueMutation2 = np.array(uniqueMutation2)
    uniqueMutation2, mutationCategory = np.unique(uniqueMutation2, return_inverse=True)

    if fileIn == './data/manualCancer.npy':
        np.save('./data/mutationNames.npy', uniqueMutation)

        np.save('./data/categoryNames.npy', uniqueMutation2)
    if fileIn == './data/breastCancer.npy':
        #print ("Hi")
        #print (len(uniqueMutation))
        #np.save('./data/mutationNamesBreast.npy', uniqueMutation)
        #np.save('./data/mutationNamesBreastLarge.npy', uniqueMutation)
        True

    newTrees = newTrees.reshape(shape1)
    M = uniqueMutation.shape[0] - 2

    if (lastName in uniqueMutation) and (lastName != uniqueMutation[-1]):
        print ("Error in Mutation Name")
        quit()


    return newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M

#processTreeData(5, './data/manualCancer.npy')
#quit()


def trainGroupModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave=False, baselineSave=False, usePurity=False, adjustProbability=False, trainSet=False, unknownRoot=False):



    doTrainSet = not (type(trainSet) == type(False))

    N1 = newTrees.shape[0]
    N2 = int(np.max(sampleInverse) + 1)

    M2 = np.unique(mutationCategory).shape[0]


    if doTrainSet:
        #trainSet = np.argwhere(np.isin(sampleInverse, trainSet))[:, 0]
        testSet = np.argwhere(np.isin(np.arange(N2), trainSet) == False)[:, 0]

        trainSet2 = np.argwhere(np.isin(sampleInverse, trainSet))[:, 0]


    #model = MutationModel(M)
    model = MutationModel(M2)
    #model = torch.load('./Models/savedModel.pt')

    #model = torch.load('./Models/savedModel24.pt')

    #N1 = 10000

    nPrint = 100
    #if adjustProbability:
        #learningRate = 1e1
    #    learningRate = 1e0
        #learningRate = 1e-1#-1 #1 #2
    #else:
    #learningRate = 1e-1#-2 #1
    #learningRate = 2e0

    #learningRate = 1e0
    #learningRate = 5e-1
    learningRate = 1e0
    #learningRate = 1e1

    optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)
    #learningRate = 1e-2#-1
    #optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)

    if adjustProbability or True:
        baseLine = np.ones(N1) * 0
        #baseLine = np.load('./Models/baseline1.npy')
        baseN = 10

    accuracies = []


    for iter in range(0, 20000):#301): #3000
        doPrint = False
        if iter % nPrint == 0:
            doPrint = True

        if doPrint:
            print (iter)



        Edges = np.zeros((N1, maxM+1, 2))
        Edges[:, 0, 1] = M
        #clones =  torch.zeros((N1, maxM+1, M))
        clones =  torch.zeros((N1, maxM+1, M2))


        edgesRemaining = np.copy(newTrees)
        edgesRemainingGroup = mutationCategory[edgesRemaining.astype(int)]

        #print (newTrees[0])

        edgesRemainingNum = (edgesRemaining[:, :, 0] * (M + 2)) + edgesRemainingGroup[:, :, 1]

        #edgesRemaining = (edgesRemaining[:, :, 0] * (M + 2)) + edgesRemaining[:, :, 1]

        #print (edgesRemaining[0])
        #quit()

        probLog1 = torch.zeros(N1)
        probLog2 = torch.zeros(N1)

        for a in range(0, maxM):

            argsLength = np.argwhere(treeLength >= (a + 1))[:, 0]

            M1 = a + 1
            counter = np.arange(N1)

            #if doPrint:
            #    print ("clones")
            #    print (clones[0, :M1])
            #    print (clones[1, :M1])

            clones1 = clones[:, :M1].reshape((N1 * M1, M2))
            output, _ = model(clones1)
            output = output.reshape((N1, M1 * M2))
            output = torch.softmax(output, dim=1)

            newStart = Edges[:, :M1, 1].repeat(M2).reshape((N1, M1 * M2))
            newStartClone = np.arange(M1).repeat(N1*M2).reshape((M1, N1, M2))
            newStartClone = np.swapaxes(newStartClone, 0, 1).reshape((N1, M1 * M2))

            newEnd = np.arange(M2).repeat(N1*M1).reshape((M2, N1*M1)).T.reshape((N1, M1 * M2))

            edgeNums = (newStart * (M + 2)) + newEnd

            validEndMask = np.zeros((N1, M1 * M2))
            for b in range(0, N1):
                validEndMask[b, np.isin(edgeNums[b], edgesRemainingNum[b])] = 1
                #print (validEndMask[b, np.isin(edgeNums[b], edgesRemainingNum[b])])
            #quit()


            output2 = output * torch.tensor(validEndMask).float()
            output2_sum = torch.sum(output2, dim=1).repeat_interleave(M1*M2).reshape((N1, M1*M2))
            output2 = output2 / output2_sum

            choiceNow = doChoice(output2.data.numpy()).astype(int)
            #print (clones[0, a])

            printNum = 10
            #print (output[printNum])
            #print (output2[printNum])


            sampleProbability = output2[counter, choiceNow]
            theoryProbability = output[counter, choiceNow]
            edgeChoice = edgeNums[counter, choiceNow]
            newStartClone = newStartClone[counter, choiceNow]

            edgeChoice_end_individual = np.zeros(N1)

            #print ("A ----------------------------------------------")

            for b in range(0, N1):
                #print (edgeChoice[b])
                #print (edgesRemainingNum[b])
                argIn1 = np.argwhere(edgesRemainingNum[b] == edgeChoice[b])
                if argIn1.shape[0] != 0:
                    argIn1 = argIn1[0, 0]
                    edgesRemainingNum[b, argIn1] = (M + 2)**2
                    argIn1 = edgesRemaining[b, argIn1, 1]
                    edgeChoice_end_individual[b] = argIn1


            edgeChoice_start = edgeChoice // (M + 2)
            edgeChoice_end = edgeChoice % (M + 2)


            clones[counter, a+1] = clones[counter, newStartClone].clone()
            clones[counter, a+1, edgeChoice_end] = clones[counter, a+1, edgeChoice_end] + 1
            #clones[counter, a+1, edgeChoice_end] = 1

            Edges[:, M1, 0] = edgeChoice_start
            Edges[:, M1, 1] = edgeChoice_end_individual

            #print (newStartClone[printNum], edgeChoice_end[printNum])

            #probLog1[argsLength] += torch.log(theoryProbability[argsLength]+1e-6)
            #probLog2[argsLength] += torch.log(sampleProbability[argsLength]+1e-6)
            probLog1[argsLength] += torch.log(theoryProbability[argsLength]+1e-12)
            probLog2[argsLength] += torch.log(sampleProbability[argsLength]+1e-12)

            #print (theoryProbability[printNum])

            #print (printNum in argsLength)


            #if doPrint:
            #    print ('theory ', theoryProbability[0])
            #    print ('theory ', theoryProbability[1])


        #print (sampleInverse[:10])

        '''
        if doPrint:
            print (newTrees[printNum])
            #print (Edges[0])
            #print (Edges[1])
            #print ('theory ', torch.exp(probLog1)[printNum])
            print ('theory ', torch.exp(probLog1)[:10])
            print (np.load('./Models/baseline1.npy')[:10])
            #print ('theory ', torch.exp(probLog1)[1])
            #print ('sample ', torch.exp(probLog2)[0])
            #print ('sample ', torch.exp(probLog2)[1])

        quit()
        #'''

        if False:
            baseLine = np.load('./Models/baseline1.npy')

            #plt.plot(np.log(patientProb))
            plt.hist(baseLine, bins=100)
            #plt.plot(np.exp(probLog1_np))
            plt.show()
            #quit()
        #quit()


        if adjustProbability or True:

            probLog1_np = probLog1.data.numpy()
            probLog2_np = probLog2.data.numpy()


            baseLine = baseLine * ((baseN - 1) / baseN)
            baseLine = baseLine + ((1 / baseN) * np.exp(probLog1_np - probLog2_np)   )


        if adjustProbability:
            baseLineLog = np.log(baseLine)


            #loss_array = probLog1 / (torch.exp( torch.tensor(probLog2_np).float() ) + 1e-10)
            loss_array = probLog1 / (torch.exp(probLog2.detach()) + 1e-10)
            loss_array = loss_array / maxM

            sampleUnique, sampleIndex = np.unique(sampleInverse, return_index=True)

            prob_adjustment = np.zeros(sampleInverse.shape[0])

            baseLineMean = np.zeros(int(np.max(sampleInverse) + 1)) + 1

            for b in range(0, sampleIndex.shape[0]):
                start1 = sampleIndex[b]
                if b == sampleIndex.shape[0] - 1:
                    end1 = N1
                else:
                    end1 = sampleIndex[b+1]

                argsLocal = np.arange(end1 - start1) + start1
                localProb = probLog1_np[argsLocal]
                localBaseline = baseLineLog[argsLocal]
                #maxLogProb = max(np.max(localBaseline), np.max(localProb))
                maxLogProb = np.max(localBaseline)
                localProb = localProb - maxLogProb
                localBaseline = localBaseline - maxLogProb

                #localProb_0 = np.copy(localProb)

                localProb = np.exp(localProb) / (np.sum(np.exp(localBaseline)) + 1e-5)

                #if np.max(localProb) > 1:
                #    print ('Hi')
                #    print (np.exp(localProb+maxLogProb))
                #    print (np.exp(localBaseline+maxLogProb))
                #    quit()

                prob_adjustment[argsLocal] = np.copy(localProb)

                #baseLineMean[b] = np.sum(baseLine[argsLocal])
                baseLineMean[int(sampleUnique[b])] = np.sum(baseLine[argsLocal])

            #plt.plot(prob_adjustment)
            #plt.show()
            #quit()

            ##loss_array = loss_array * torch.tensor(prob_adjustment)
            loss_array = loss_array * torch.tensor(prob_adjustment)

            loss_array = loss_array[trainSet2]

            #plt.hist(prob_adjustment, bins=100)#, range=(0, 0.1))
            #plt.show()
            #plt.hist(np.log(baseLineMean), bins=100)#, range=(0, 0.1))
            #plt.show()
            #quit()




            #print (newTrees.shape)
            #print (baseLineMean.shape)
            #print (sampleInverse.shape)
            #print (np.max(sampleInverse))

            #print (baseLineMean.shape)
            #print (np.max(trainSet))
            #print (np.max(testSet))
            #print (trainSet)
            #print (testSet)
            #print (sampleInverse[trainSet2])
            #print (np.intersect1d(trainSet, testSet))
            #quit()

            score_train = np.mean(np.log(baseLineMean[trainSet] + 1e-6))
            score_test = np.mean(np.log(baseLineMean[testSet] + 1e-6))

        else:

            #loss_array = torch.exp( (probLog1 - probLog2.detach() ) / torch.tensor(treeLength).float()   )
            #loss_array = torch.exp( (probLog1 - probLog2.detach() ) / 1   )
            #loss_array = torch.exp( (probLog1 - probLog2.detach() ) / 5   )
            loss_array = torch.exp( probLog1 - probLog2.detach() )
            loss_array = loss_array[trainSet2]

        loss = -1 * torch.mean(loss_array)





        #'''
        regularization = 0
        numHigh = 0
        numAll = 0
        c1 = 0
        for param in model.parameters():
            if c1 in [0, 2, 3]:
                #regularization = regularization + torch.sum(torch.abs(param))
                #regularization = regularization + torch.sum( torch.abs(param) - ( 0.9 * torch.relu( torch.abs(param) - 0.2 ))       )
                regularization = regularization + torch.sum( torch.abs(param) - ( 0.9 * torch.relu( torch.abs(param) - 0.1 ))       )
                #regularization = regularization + torch.sum( torch.abs(param) - ( 0.8 * torch.relu( torch.abs(param) - 0.05 ))       )
                #regularization = regularization + (torch.sum( 1 - torch.exp(-torch.abs(param) * 10)   ) * 0.1)
                numHigh += np.argwhere(np.abs(np.abs(param.data.numpy()) < 0.01)).shape[0]
                numAll += np.argwhere(np.abs(np.abs(param.data.numpy()) > -1)).shape[0]
                #numAll += param.size
            c1 += 1
        #regularization = regularization * 0.02#0.0001#
        regularization = regularization * 0.02
        #regularization = regularization * 0.05
        #regularization = regularization * 0.002

        loss = loss + regularization
        #'''

        if doPrint:


            #TODO edit sampleInverse

            print ("A")
            print (np.mean(baseLine))
            print (score_train, score_test)
            #print (torch.sum( torch.exp(probLog1[sampleInverse[trainSet]])  ) / trainSet.shape[0]  )
            #print (torch.sum( torch.exp(probLog1[sampleInverse[testSet]])  ) / testSet.shape[0] )
            print (loss)
            #print (Trees[np.sum(Trees[:, 2], axis=1) == 2, 1:][:10])

            if baselineSave and fileSave:
                torch.save(model, fileSave)
                np.save(baselineSave, baseLine)
            #else:
            #    torch.save(model, './Models/savedModel3.pt')
            #    np.save('./Models/baseline3.npy', baseLine)




        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def trainModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave=False, baselineSave=False, usePurity=False, adjustProbability=False, trainSet=False, unknownRoot=False):

    excludeSameMut = True

    doTrainSet = not (type(trainSet) == type(False))

    N1 = newTrees.shape[0]
    N2 = int(np.max(sampleInverse) + 1)

    #M2 = np.unique(mutationCategory).shape[0]


    if doTrainSet:
        #trainSet = np.argwhere(np.isin(sampleInverse, trainSet))[:, 0]
        testSet = np.argwhere(np.isin(np.arange(N2), trainSet) == False)[:, 0]

        trainSet2 = np.argwhere(np.isin(sampleInverse, trainSet))[:, 0]


    model = MutationModel(M)
    #model = torch.load('./Models/savedModel23.pt')

    #N1 = 10000

    nPrint = 1#00


    #learningRate = 1e0

    #learningRate = 1e0 #Typically used May 22
    learningRate = 1e1
    #learningRate = 1e2

    optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)
    #learningRate = 1e-1
    #learningRate = 1e-2
    #learningRate = 1e-3
    #optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)

    if adjustProbability or True:
        baseLine = np.zeros(N1) #+ 0.1
        #baseLine = np.load('./Models/baseline1.npy')
        #baseN = 100
        baseN = 10

    accuracies = []

    recordBase = np.zeros((100000, N1))
    recordSamp = np.zeros((100000, N1))


    for iter in range(0, 1000):#301): #3000


        #if True:
        #    baseN = (min(iter, 1000) + 10)

        doPrint = False
        if iter % nPrint == 0:
            doPrint = True

        if doPrint:
            print (iter)



        Edges = np.zeros((N1, maxM+1, 2))
        Edges[:, 0, 1] = M
        clones =  torch.zeros((N1, maxM+1, M))
        edgesRemaining = np.copy(newTrees)
        #print (newTrees[0])

        edgesRemaining = (edgesRemaining[:, :, 0] * (M + 2)) + edgesRemaining[:, :, 1]

        #print (edgesRemaining[0])
        #quit()

        probLog1 = torch.zeros(N1)
        probLog2 = torch.zeros(N1)

        for a in range(0, maxM):

            argsLength = np.argwhere(treeLength >= (a + 1))[:, 0]

            if argsLength.shape[0] != 0:

                #print (argsLength)

                M1 = a + 1
                counter = np.arange(N1)

                #if doPrint:
                #    print ("clones")
                #    print (clones[0, :M1])
                #    print (clones[1, :M1])

                clones1 = clones[:, :M1].reshape((N1 * M1, M))
                output, _ = model(clones1)
                output = output.reshape((N1, M1 * M))
                output = torch.softmax(output, dim=1)

                newStart = Edges[:, :M1, 1].repeat(M).reshape((N1, M1 * M))
                newStartClone = np.arange(M1).repeat(N1*M).reshape((M1, N1, M))
                newStartClone = np.swapaxes(newStartClone, 0, 1).reshape((N1, M1 * M))

                newEnd = np.arange(M).repeat(N1*M1).reshape((M, N1*M1)).T.reshape((N1, M1 * M))

                edgeNums = (newStart * (M + 2)) + newEnd

                if excludeSameMut:
                    notAlreadyUsedMask = np.zeros((N1, M1 * M))
                    for b in range(0, N1):


                        notAlreadyUsedMask[b, np.isin(newEnd[b], Edges[b, :M1, 1]) == False]


                        notAlreadyUsedMask[b, np.isin(newEnd[b], Edges[b, :M1, 1]) == False] = 1

                    output = output * torch.tensor(notAlreadyUsedMask).float()
                    output_sum = torch.sum(output, dim=1).repeat_interleave(M1*M).reshape((N1, M1*M))
                    output = output / output_sum



                validEndMask = np.zeros((N1, M1 * M))
                for b in range(0, N1):
                    validEndMask[b, np.isin(edgeNums[b], edgesRemaining[b])] = 1

                output2 = output * torch.tensor(validEndMask).float()
                output2_sum = torch.sum(output2, dim=1).repeat_interleave(M1*M).reshape((N1, M1*M))
                output2 = output2 / output2_sum

                choiceNow = doChoice(output2.data.numpy()).astype(int)
                #print (clones[0, a])

                printNum = 10
                #print (output[printNum])
                #print (output2[printNum])


                sampleProbability = output2[counter, choiceNow]


                #print (sampleProbability[5:10])
                #quit()

                theoryProbability = output[counter, choiceNow]
                edgeChoice = edgeNums[counter, choiceNow]
                newStartClone = newStartClone[counter, choiceNow]

                for b in range(0, N1):
                    argsNotRemaining = np.argwhere(edgesRemaining[b] == edgeChoice[b])[:, 0]
                    edgesRemaining[b, argsNotRemaining] = (M + 2) ** 2

                edgeChoice_start = edgeChoice // (M + 2)
                edgeChoice_end = edgeChoice % (M + 2)


                clones[counter, a+1] = clones[counter, newStartClone].clone()
                clones[counter, a+1, edgeChoice_end] = clones[counter, a+1, edgeChoice_end] + 1
                #clones[counter, a+1, edgeChoice_end] = 1

                Edges[:, M1, 0] = edgeChoice_start
                Edges[:, M1, 1] = edgeChoice_end

                #print (newStartClone[printNum], edgeChoice_end[printNum])

                #probLog1[argsLength] += torch.log(theoryProbability[argsLength]+1e-6)
                #probLog2[argsLength] += torch.log(sampleProbability[argsLength]+1e-6)
                probLog1[argsLength] += torch.log(theoryProbability[argsLength]+1e-12)
                probLog2[argsLength] += torch.log(sampleProbability[argsLength]+1e-12)

                #print (probLog2[:10])

                #print (theoryProbability[printNum])

                #print (printNum in argsLength)


                #if doPrint:
                #    print ('theory ', theoryProbability[0])
                #    print ('theory ', theoryProbability[1])

                #print (torch.min(probLog1))

                #print (torch.mean(probLog1))
                #print (torch.mean(probLog2))


        #quit()



        #print (sampleInverse[:10])




        if adjustProbability or True:

            probLog1_np = probLog1.data.numpy()
            probLog2_np = probLog2.data.numpy()


            baseLine = baseLine * ((baseN - 1) / baseN)
            baseLine = baseLine + ((1 / baseN) * np.exp(probLog1_np - probLog2_np)   )

            recordBase[iter] = np.copy(probLog1_np)
            recordSamp[iter] = np.copy(probLog2_np)

            #baseLine = baseLine + np.log( ((baseN - 1) / baseN) )
            #adjustTerm = probLog1_np - probLog2_np + np.log( (1 / baseN)  )
            #baseLine = addFromLog(np.array([baseLine, adjustTerm]))



        if adjustProbability:
            baseLineLog = np.log(baseLine)
            #baseLineLog = np.copy(baseLine)


            #print (probLog2[:10])

            #loss_array = probLog1 / (torch.exp( torch.tensor(probLog2_np).float() ) + 1e-10)
            loss_array = probLog1 / (torch.exp(probLog2.detach()) + 1e-10)

            loss_array = loss_array / maxM


            sampleUnique, sampleIndex = np.unique(sampleInverse, return_index=True)

            prob_adjustment = np.zeros(sampleInverse.shape[0])

            baseLineMean = np.zeros(int(np.max(sampleInverse) + 1)) + 1

            for b in range(0, sampleIndex.shape[0]):
                start1 = sampleIndex[b]
                if b == sampleIndex.shape[0] - 1:
                    end1 = N1
                else:
                    end1 = sampleIndex[b+1]

                argsLocal = np.arange(end1 - start1) + start1
                localProb = probLog1_np[argsLocal]
                localBaseline = baseLineLog[argsLocal]
                #maxLogProb = max(np.max(localBaseline), np.max(localProb))
                maxLogProb = np.max(localBaseline)
                localProb = localProb - maxLogProb
                localBaseline = localBaseline - maxLogProb

                #localProb_0 = np.copy(localProb)

                localProb = np.exp(localProb) / (np.sum(np.exp(localBaseline)) + 1e-5)

                #if np.max(localProb) > 1:
                #    print ('Hi')
                #    print (np.exp(localProb+maxLogProb))
                #    print (np.exp(localBaseline+maxLogProb))
                #    quit()

                prob_adjustment[argsLocal] = np.copy(localProb)

                #baseLineMean[b] = np.sum(baseLine[argsLocal])


                baseLineMean[int(sampleUnique[b])] = np.sum(baseLine[argsLocal])
                #baseLineMean[int(sampleUnique[b])] = np.sum(np.exp(baseLine[argsLocal]))


            ##loss_array = loss_array * torch.tensor(prob_adjustment)
            loss_array = loss_array * torch.tensor(prob_adjustment)

            loss_array = loss_array[trainSet2]



            score_train = np.mean(np.log(baseLineMean[trainSet] + 1e-20))
            score_test = np.mean(np.log(baseLineMean[testSet] + 1e-20))

            #score_train = np.mean(np.log(baseLineMean[trainSet] + 1e-10))
            #score_test = np.mean(np.log(baseLineMean[testSet] + 1e-10))

            #score_train = np.mean(np.log(baseLineMean[trainSet] + 1e-8))
            #score_test = np.mean(np.log(baseLineMean[testSet] + 1e-8))

            #score_train = np.mean(np.log(baseLineMean[trainSet] + 1e-6))
            #score_test = np.mean(np.log(baseLineMean[testSet] + 1e-6))

            #print ("A")

        else:

            #loss_array = torch.exp( (probLog1 - probLog2.detach() ) / torch.tensor(treeLength).float()   )
            #loss_array = torch.exp( (probLog1 - probLog2.detach() ) / 1   )
            #loss_array = torch.exp( (probLog1 - probLog2.detach() ) / 5   )
            loss_array = torch.exp( probLog1 - probLog2.detach() )
            loss_array = loss_array[trainSet2]

        #quit()


        loss = -1 * torch.mean(loss_array)





        #'''
        regularization = 0
        numHigh = 0
        numAll = 0
        c1 = 0
        for param in model.parameters():
            if c1 in [0, 2, 3]:
                #regularization = regularization + torch.sum(torch.abs(param))
                regularization = regularization + torch.sum( torch.abs(param) - ( 0.9 * torch.relu( torch.abs(param) - 0.1 ))       )
                numHigh += np.argwhere(np.abs(np.abs(param.data.numpy()) < 0.01)).shape[0]
                numAll += np.argwhere(np.abs(np.abs(param.data.numpy()) > -1)).shape[0]
                #numAll += param.size
            c1 += 1

        #regularization = regularization * 0.0001
        regularization = regularization * 0.0002 #Best for breast cancer
        #regularization = regularization * 0.0005
        #regularization = regularization * 0.001
        #regularization = regularization * 0.002 #Used for our occurance simulation as well
        #regularization = regularization * 0.02
        #regularization = regularization * 0.1

        loss = loss + regularization
        #'''

        #print (loss)
        #quit()

        if doPrint:


            #TODO edit sampleInverse

            print ("A")
            print (np.mean(baseLine))
            print (score_train, score_test)
            #print (torch.sum( torch.exp(probLog1[sampleInverse[trainSet]])  ) / trainSet.shape[0]  )
            #print (torch.sum( torch.exp(probLog1[sampleInverse[testSet]])  ) / testSet.shape[0] )
            print (loss)
            #print (Trees[np.sum(Trees[:, 2], axis=1) == 2, 1:][:10])

            if baselineSave and fileSave:
                torch.save(model, fileSave)
                np.save(baselineSave, baseLine)
            #else:
            #    torch.save(model, './Models/savedModel3.pt')
            #    np.save('./Models/baseline3.npy', baseLine)

            #if iter % 100 == 0:
            #    plt.imshow(recordBase[:iter+1, :100].T)
            #    plt.show()

        if iter == 500:

            optimizer = torch.optim.SGD(model.parameters(), lr = 1e0)



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def trainRealData(dataName, maxM=10, trainPer=0.666):

    #maxM = 10
    #maxM = 5

    if dataName == 'manual':
        maxM = 10
        newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, './dataNew/manualCancer.npy')
    if dataName == 'breast':
        maxM = 9
        newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, './dataNew/breastCancer.npy')
    #newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, './data/lungData/processed.npy')


    #_, sampleIndex = np.unique(sampleInverse, return_index=True)
    #mut1, counts1 = np.unique(newTrees[sampleIndex], return_counts=True)
    #mutationCategory[counts1 <= 40] = 0
    #_, mutationCategory = np.unique(mutationCategory, return_inverse=True)


    rng = np.random.RandomState(2)
    #rng = np.random.RandomState(3)

    #np.random.seed(1)
    N2 = int(np.max(sampleInverse)+1)
    #trainSet = np.random.permutation(N2)
    trainSet = rng.permutation(N2)

    N3 = int(np.floor(trainPer * N2))

    #trainSet = trainSet[:(2*N2//3)]
    trainSet = trainSet[:N3]

    #print (newTrees[0])
    #print (newTrees[20])
    #quit()
    #print (treeLength[0])

    #trainClusterModel(newTrees, sampleInverse, treeLength, M, maxM, adjustProbability=True, trainSet=trainSet, unknownRoot=True)
    #####trainModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave='./Models/savedModel16.pt', baselineSave='./Models/baseline16.npy', adjustProbability=True, trainSet=trainSet, unknownRoot=True)
     #23
    if dataName == 'manual':
        trainGroupModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave='./Models/savedModel_manual.pt', baselineSave='./Models/baseline_manual.npy', adjustProbability=True, trainSet=trainSet, unknownRoot=True)
    else:
        trainModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave='./Models/savedModel_breast.pt', baselineSave='./Models/baseline_breast.npy', adjustProbability=True, trainSet=trainSet, unknownRoot=True)
        True
    #Was 3, not 4 April 29

#trainRealData('breast')
#quit()




#maxM = 10
#newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, './data/manualCancer.npy')
#categoryName = np.load('./data/categoryNames.npy')[:-2]
#mutationName = np.load('./data/mutationNames.npy')[:-2]
#_, sampleIndex = np.unique(sampleInverse, return_index=True)
#newTrees = newTrees[sampleIndex]




def doProportionAnalysis():


    #import scipy.stats.sem
    from scipy.stats import sem

    'NPM1'
    ar1 = [[30.7, 62.1], [69.8, 21.8], [14.0, 29.1], [23.7, 66.5], [19.9, 66.3],
           [24.0, 16.4], [12.0, 16.3], [12.0, 16.3], [21.7, 13.5], [13.7, 21.4], [18.1, 62.0], [27.7, 20.3],
           [17.1, 72.0], [12.6, 23.5], [23.8, 67.9], [20.7, 64.9], [2.5, 9.3], [14.4, 76.6], [20.5, 67.2],
           [7.9, 29.9], [19.6, 40.4], [32.8, 33.8], [2.9, 5.1], [15.3, 37.4], [18.1, 50.2], [22.4, 68.1]]
    'ASXL1'
    ar2 = [[29.2, 45.0], [27.3, 55.3], [6.8, 26.3], [19.1, 48.1], [24.2, 26.9], [0.9, 3.5], [12.0, 26.1], [9.9, 80.9]]
    'DNMT3A'
    ar3 = [[3.7, 6.9], [8.4, 34.4], [16.5, 37.4], [6.6, 19.9], [4.2, 13.7],
           [3.5, 23.5], [12.1, 15.6], [15.1, 52.6], [6.3, 19.8], [2.2, 6.9]]
    'NRAS'
    ar4 = [[4.2, 1.0], [23.2, 13.2], [23.2, 13.3], [18.6, 17.4], [18.6, 6.6], [16.4, 40.5],
           [16.3, 71.7], [17.3, 12.1], [17.3, 9.0], [17.3, 9.0], [34.9, 34.0], [13.5, 8.9],
           [20.3, 14.5], [17.0, 13.7], [18.0, 19.5], [26.3, 21.3], [4.6, 17.1], [5.4, 5.0],
           [5.4, 10.6], [23.5, 27.6], [27.6, 5.9],  [33.5, 17.8], [33.5, 5.8], [48.1, 7.6],
           [48.1, 6.3], [17.4, 13.4], [54.0, 6.5], [29.9, 5.2], [29.9, 3.8], [27.6, 5.6],
           [11.1, 12.9], [56.6, 4.7], [33.8, 34.0], [6.6, 38.7], [42.6, 3.2], [19.9, 15.5],
           [6.9, 44.0], [44.0, 8.0], [44.0, 8.8], [80.9, 6.6] ]
    'FLT3'
    ar5 = [[3.2, 15.9], [4.2, 0.7], [25.5, 9.9], [30.7, 10.6], [14.9, 8.6],
           [14.9, 1.7], [45.4, 15.1], [28.4, 1.6], [13.5, 15.2], [21.7, 9.3],
           [8.3, 66.1], [36.4, 3.1], [23.5, 4.8], [23.5, 10.3], [5.8, 20.7],
           [29.9, 28.7], [50.8, 43.9], [44.0, 7.5], [44.0, 6.1]]
    'IDH1'
    ar6 = [[4.2, 0.0], [1.0, 0.1], [42.4, 13.5], [46.4, 4.0], [14.5, 13.8],
           [16.7, 23.0], [1.7, 5.4], [1.7, 5.9], [4.3, 15.6], [3.5, 0.9],
           [15.6, 27.1], [4.2, 15.3], [21.9, 5.6], [1.4, 9.9]]
    'PTPN11'
    ar7 = [[70.3, 17.2], [45.4, 22.3], [18.6, 22.0], [19.5, 12.8], [21.4, 32.4],
            [5.4, 1.7], [5.9, 7.0], [7.0, 0.8], [33.8, 5.6], [33.5, 4.9],
            [11.1, 21.9], [11.1, 4.2], [5.1, 40.6], [5.1, 33.3], [5.1, 1.4], [16.0, 33.8], [19.9, 19.3] ]
    'FLT3-ITD'
    ar8 = [[4.2, 70.3], [25.5, 12.3], [23.2, 17.7], [14.9, 69.6], [26.3, 21.0],
            [16.3, 0.0], [28.4, 28.6], [44.7, 4.3], [4.5, 6.3], [42.4, 17.1],
            [18.0, 13.9], [8.3, 11.3], [32.4, 21.1], [5.4, 21.9], [27.6, 6.5],
            [3.1, 9.3], [46.6, 7.9], [4.7, 14.4], [29.9, 17.4], [40.2, 14.2],
            [27.2, 24.9], [49.0, 10.1], [5.1, 6.6], [37.4, 25.0], [22.0, 62.9]]


    mutNames = ['NPM1', 'ASXL1', 'DNMT3A', 'NRAS', 'FLT3', 'IDH1', 'PTPN11', 'FLT3-ITD']


    ar1 = np.array(ar1) + 1
    ar2 = np.array(ar2) + 1
    ar3 = np.array(ar3) + 1
    ar4 = np.array(ar4) + 1
    ar5 = np.array(ar5) + 1
    ar6 = np.array(ar6) + 1
    ar7 = np.array(ar7) + 1
    ar8 = np.array(ar8) + 1

    arList = [ar1, ar2, ar3, ar4, ar5, ar6, ar7, ar8]

    for a in range(len(arList)):
        #print (a)
        print (mutNames[a])

        mean1 = np.mean(np.log(arList[a][:, 1] / arList[a][:, 0]))
        sig1 = scipy.stats.sem(np.log(arList[a][:, 1] / arList[a][:, 0]))

        print (str(mean1)[:5] + str(' +- ') + str(sig1)[:5])

        #print (np.mean(np.log(arList[a][:, 1] / arList[a][:, 0])))
        #print (scipy.stats.sem(np.log(arList[a][:, 1] / arList[a][:, 0])))


#doProportionAnalysis()
#quit()


def trainNewSimulations(T, N):
    #T = 6
    #N = 100

    for a in range(0, N):


        maxM = 5

        if T == 0:
            M = 10
        if T == 1:
            M = 20

        if T in [3, 4]:
            M = 10
            maxM = 7

        if T in [5, 6]:
            M = 20
            maxM = 7


        newTrees = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_bulkTrees.npz')
        newTrees = newTrees.astype(int)
        #newTrees[newTrees == 100] = M
        sampleInverse = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_bulkSample.npz').astype(int)
        mutationCategory = ''
        #treeLength = (np.zeros(newTrees.shape[0]) + maxM).astype(int)

        treeLength = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_treeSizes.npz')
        treeLength = treeLength[sampleInverse]

        #treeLength  = (treeLength * 0) + 5

        _, sampleIndex = np.unique(sampleInverse, return_index=True)

        #print (newTrees[sampleIndex][:10])
        #print (treeLength[sampleIndex][:10])
        #quit()



        rng = np.random.RandomState(2)

        N2 = int(np.max(sampleInverse)+1)
        #trainSet = np.random.permutation(N2)
        trainSet = rng.permutation(N2)

        #trainSet = np.arange(N2)
        trainSet = trainSet[:N2//2]



        modelFile = './dataNew/specialSim/results/T_' + str(T) + '_R_' + str(a) + '_model.pt'
        baselineFile = './dataNew/specialSim/results/T_' + str(T) + '_R_' + str(a) + '_baseline.pt'

        trainModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave=modelFile, baselineSave=baselineFile, adjustProbability=True, trainSet=trainSet, unknownRoot=True)


#trainNewSimulations(4, 20)
#quit()

def testOccurSimulations(T, N):
    '''
    0
    [[80  0]
     [ 0 10]]
    1
    [[160   0]
     [  0  20]]
    2
    [[239   0]
     [  0  31]]
    3
    [[318   0]
     [  0  42]]
    4
    [[399   0]
     [  0  51]]
    5
    [[477   0]
     [  0  63]]
    6
    [[559   0]
     [  0  71]]
    7
    [[638   1]
     [  0  81]]
    8
    [[719   1]
     [  0  90]]
    9
    [[796   1]
     [  0 103]]
    10
    [[876   1]
     [  0 113]]
    11
    [[958   2]
     [  0 120]]
    12
    [[1040    2]
     [   0  128]]
    13
    [[1120    2]
     [   0  138]]
    14
    [[1200    2]
     [   0  148]]
    15
    [[1282    2]
     [   0  156]]
    16
    [[1362    2]
     [   0  166]]
    17
    [[1440    2]
     [   0  178]]
    18
    [[1523    2]
     [   0  185]]
    19
    [[1600    2]
     [   0  198]]
    '''
    #T = 4
    #T = 0
    #N = 100

    #N = 20

    M = 10


    categories = np.zeros((2, 2)).astype(int)
    #categories = np.array([[1600, 2], [0, 198]]).astype(int)

    #for a in range(20, N):
    for a in range(0, N):
        #print (a)
        probabilityMatrix = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_prob.npz')

        probabilityMatrix[np.arange(probabilityMatrix.shape[1]), np.arange(probabilityMatrix.shape[1])] = 0

        prob_true = np.zeros((M, M))
        prob_true[:5, :5] = np.copy(probabilityMatrix[:5, :5])
        prob_true[prob_true > 0.01] = 1



        #fig, axs = plt.subplots(1, 2)
        #axs[0].imshow(probabilityMatrix[:5, :5])


        #plt.imshow(probabilityMatrix)
        #plt.show()

        modelFile = './dataNew/specialSim/results/T_' + str(T) + '_R_' + str(a) + '_model.pt'
        model = torch.load(modelFile)


        X = torch.zeros((M, M))
        X[np.arange(M), np.arange(M)] = 1

        output, _ = model(X)

        output_np = output.data.numpy()
        output_np = output_np - np.mean(output_np)

        #output_np = output_np - (np.max(output_np) / 3)

        if False:

            output_np[np.arange(M), np.arange(M)] = -10
            output_np_flat = output_np.reshape((output_np.size,))

            num_high = int(np.sum(prob_true))

            cutOff = np.sort(output_np_flat)[-num_high]

            #print (output_np)
            #print (cutOff)
            #quit()

            output_np_bool = np.copy(output_np) - cutOff
            output_np_bool[output_np_bool >= 0] = 1
            output_np_bool[output_np_bool < 0] = 0

            #plt.imshow(output_np_bool)
            #plt.show()



        if True:
            output_np = output_np - (np.max(output_np) * 0.4)
            output_np[np.arange(M), np.arange(M)] = 0
            output_np_bool = np.copy(output_np)
            output_np_bool[output_np_bool > 0] = 1
            output_np_bool[output_np_bool < 0] = 0


        #axs[1].imshow(output_np)

        #axs[1].imshow(output_np_bool[:5, :5])

        #plt.imshow(output_np)


        for b in range(output_np.shape[0]):
            for c in range(output_np.shape[1]):
                if b != c:
                    categories[int(prob_true[b, c]), int(output_np_bool[b, c])] += 1

    #print (categories)

    print ('True Positives: ' + str(categories[1, 1]))
    print ('True Negatives: ' + str(categories[0, 0]))
    print ('False Positives: ' + str(categories[0, 1]))
    print ('False Negatives: ' + str(categories[1, 0]))

    #print (np.sum(categories) // 90)

        #plt.show()












def savePathwaySimulationPredictions():

    def pathwayTheoryProb(pathway, prob_assume):

        probLog = 0
        pathwayLength = len(pathway)
        for a in range(0, pathwayLength):
            pathwaySet = pathway[a]
            if len(pathwaySet) == 0:
                #probLog += -1000
                True
            else:
                prob = np.sum(prob_assume[np.array(pathwaySet)])
                probLog += np.log(prob)

        return probLog

    def pathwayRealProb(pathway, prob2_Adj_3D):

        subset = prob2_Adj_3D

        if min(min(len(pathway[0]), len(pathway[1])), len(pathway[2]) ) == 0:
            subset_sum = np.log(1e-50)
        else:

            subset = subset[np.array(pathway[0])]
            subset = subset[:, np.array(pathway[1])]
            subset = subset[:, :, np.array(pathway[2])]

            subset_max = np.max(subset)
            subset = subset - subset_max
            subset_sum = np.sum(np.exp(subset))
            subset_sum = np.log(subset_sum+1e-50)
            subset_sum = subset_sum + subset_max

        return subset_sum



    def evaluatePathway(pathway, prob2_Adj_3D, prob_assume, includeProb=False):

        probTheory = pathwayTheoryProb(pathway, prob_assume)
        probReal = pathwayRealProb(pathway, prob2_Adj_3D)

        probDiff = probReal - probTheory


        #score = (probReal * 0.1) + probDiff


        #score = probDiff - (0.1 * (np.abs(probDiff) ** 2))

        #score = (probReal * 0.03) + probDiff

        #score = (probReal * 0.2) + probDiff

        #score = (probReal * 0.35) + probDiff

        #score = (probReal * 0.4) + probDiff

        score = (probReal * 0.4) + probDiff

        #score = (probReal * 0.2) + probDiff
        #score = (probReal * 0.2) + probDiff
        #score = probReal - (0.8 * probTheory)
        if includeProb:
            return score, probReal, probDiff
        else:
            return score



    def singleModifyPathway(doAdd, step, position, pathway, prob2_Adj_3D, prob_assume):

        pathway2 = copy.deepcopy(pathway)
        set1 = pathway2[step]
        set1 = np.array(set1)

        if doAdd:
            set1 = np.concatenate((set1, np.array([position])))
            set1 = np.sort(set1)
        else:
            set1 = set1[set1 != position]

        #pathway2[step] = set1.astype(int)
        pathway2[step] = set1.astype(int)

        score= evaluatePathway(pathway2, prob2_Adj_3D, prob_assume)

        return score, pathway2

    def stepModifyPathway(doAdd, step, pathway, superPathway, prob2_Adj_3D, prob_assume):

        M = prob_assume.shape[0]

        set1 = copy.deepcopy(pathway[step])
        if doAdd or (len(set1) > 1):
            set2 = copy.deepcopy(superPathway[step])
            set3 = set2[np.isin(set2, set1) != doAdd]

            pathways = []
            scores = []
            for position in set3:
                score, pathway2 = singleModifyPathway(doAdd, step, position, pathway, prob2_Adj_3D, prob_assume)

                pathways.append(copy.deepcopy(pathway2))
                scores.append(score)

            return pathways, scores

        else:

            return [], []


    def iterOptimizePathway(doAddList, stepList, pathway, superPathway, prob2_Adj_3D, prob_assume):

        score = evaluatePathway(pathway, prob2_Adj_3D, prob_assume)
        pathway2 = copy.deepcopy(pathway)

        pathways2, scores2 = [pathway2], [score]
        for doAdd in doAddList:
            for step in stepList:
                pathways, scores = stepModifyPathway(doAdd, step, pathway, superPathway, prob2_Adj_3D, prob_assume)
                pathways2 = pathways2 + pathways
                scores2 = scores2 + scores





        for step in stepList:
            set1 = [-1] + list(range(M))
            for pos_add in set1:

                pathway2 = copy.deepcopy(pathway)

                if pos_add == -1:
                    pathway2[step] = list(np.arange(M))
                else:
                    pathway2[step] = [pos_add]

                score = evaluatePathway(pathway2, prob2_Adj_3D, prob_assume)

                pathways2.append(copy.deepcopy(pathway2))
                scores2.append(copy.copy(score))






        bestOne = np.argmax(np.array(scores2))
        bestPathway = pathways2[bestOne]
        bestScore = np.max(scores2)

        return bestPathway, bestScore


    def iterOptimizePathway2(doAddList, stepList, pathway, superPathway, prob2_Adj_3D, prob_assume):

        M = prob_assume.shape[0]

        score = evaluatePathway(pathway, prob2_Adj_3D, prob_assume)
        pathway2 = copy.deepcopy(pathway)

        pathways2, scores2 = [pathway2], [score]

        for step in stepList:

            set1 = copy.deepcopy(pathway[step])
            set2 = copy.deepcopy(superPathway[step])

            set_add = set2[np.isin(set2, set1) != True]
            set_rem = set2[np.isin(set2, set1) != False]

            set_add = [-1] + list(set_add)
            set_rem = [-1] + list(set_rem)


            '''
            for pos_add in set_add:
                for pos_rem in set_rem:

                    pathway2 = copy.deepcopy(pathway)

                    if pos_add >= 0:
                        score, pathway2 = singleModifyPathway(True, step, pos_add, pathway2, prob2_Adj_3D, prob_assume)

                    if pos_rem >= 0:
                        #print ("T2")
                        #print (pathway2)
                        #print (pos_rem, step)
                        score, pathway2 = singleModifyPathway(False, step, pos_rem, pathway2, prob2_Adj_3D, prob_assume)


                    pathways2.append(copy.deepcopy(pathway2))
                    scores2.append(copy.copy(score))
            '''


            for pos_add in set_add:
                pathway2 = copy.deepcopy(pathway)

                if pos_add >= 0:
                    score, pathway2 = singleModifyPathway(True, step, pos_add, pathway2, prob2_Adj_3D, prob_assume)

                pathways2.append(copy.deepcopy(pathway2))
                scores2.append(copy.copy(score))

            for pos_rem in set_rem:
                pathway2 = copy.deepcopy(pathway)

                if pos_rem >= 0:
                    score, pathway2 = singleModifyPathway(False, step, pos_rem, pathway2, prob2_Adj_3D, prob_assume)

                pathways2.append(copy.deepcopy(pathway2))
                scores2.append(copy.copy(score))

        for step in stepList:
            set1 = [-1] + list(range(M))
            for pos_add in set1:

                pathway2 = copy.deepcopy(pathway)

                if pos_add == -1:
                    pathway2[step] = list(np.arange(M))
                else:
                    pathway2[step] = [pos_add]

                #print (pathway2)

                score = evaluatePathway(pathway2, prob2_Adj_3D, prob_assume)

                pathways2.append(copy.deepcopy(pathway2))
                scores2.append(copy.copy(score))

        #quit()

        bestOne = np.argmax(np.array(scores2))
        bestPathway = pathways2[bestOne]
        bestScore = np.max(scores2)

        return bestPathway, bestScore


    def singleOptimizePathway(doAddList, stepList, pathway, superPathway, prob_Adj_3D, prob_assume):

        pathway2 = copy.deepcopy(pathway)
        bestScoreBefore = -10000
        notDone = True
        while notDone:
            #pathway2, bestScore = iterOptimizePathway(doAddList, stepList, pathway2, superPathway, prob_Adj_3D, prob_assume)
            pathway2, bestScore = iterOptimizePathway2(doAddList, stepList, pathway2, superPathway, prob_Adj_3D, prob_assume)

            #print ("B")
            #print (pathway2)
            #print (bestScore)



            if bestScoreBefore == bestScore:
                notDone = False
            bestScoreBefore = bestScore

        #print (evaluatePathway(pathway2, prob2_Adj_3D, prob_assume, includeProb=True))

        return pathway2



    def removePathFromProb(prob, pathway):

        set1, set2, set3 = np.array(pathway[0]), np.array(pathway[1]), np.array(pathway[2])
        M = prob.shape[0]
        modMask = np.zeros(prob.shape)
        modMask[np.arange(M)[np.isin(np.arange(M), set1) == False]] = 1
        modMask[:, np.arange(M)[np.isin(np.arange(M), set2) == False]] = 1
        modMask[:, :, np.arange(M)[np.isin(np.arange(M), set3) == False]] = 1

        prob[modMask == 0] = -1000

        return prob


    def doMax(ar1):

        if ar1.size == 0:
            #print ("A")
            return 0
        else:
            #print ("B")
            return np.max(ar1)



    def doProbabilityFind(M, model, argsInteresting, mutationName):


        #X0 = torch.zeros((M*M, M))
        X0 = torch.zeros((1, M))
        X1 = torch.zeros((M, M))
        X2 = torch.zeros((M*M, M))

        arange0 = np.arange(M)
        arange1 = np.arange(M*M)

        X1[arange0, arange0] = 1

        X2[arange1, arange1 % M] = X2[arange1, arange1 % M] + 1
        X2[arange1, arange1 // M] = X2[arange1, arange1 // M] + 1

        pred0, _ = model(X0)
        pred1, xLatent1 = model(X1)
        pred2, _ = model(X2)


        pred2 = pred2.reshape((M, M, M))
        pred1[arange0, arange0] = -1000
        pred2[arange0, arange0, :] = -1000
        pred2[:, arange0, arange0] = -1000
        pred2[arange0, :, arange0] = -1000
        pred2 = pred2.reshape((M * M, M))



        #print (xLatent1.shape)

        #plt.plot(xLatent1)
        #plt.show()
        #quit()

        if True:
            prob0 = torch.softmax(pred0, dim=1)
            prob1 = torch.softmax(pred1, dim=1)
            prob2 = torch.softmax(pred2, dim=1)

            #plt.imshow(prob1.data.numpy())
            #plt.show()

            prob0 = prob0.data.numpy()
            prob1 = prob1.data.numpy()
            prob2 = prob2.data.numpy()
            prob0_Adj = np.log(np.copy(prob0) + 1e-10)
            outputProb0 = np.log(prob0[0] + 1e-10)
            outputProb0 = outputProb0.repeat(M).reshape((M, M))
            prob1_Adj = outputProb0 + np.log(prob1 + 1e-10)


            #plt.imshow(np.exp(prob1_Adj))
            #plt.show()


            outputProb1 = prob1_Adj.repeat(M).reshape((M*M, M))
            prob2_Adj = outputProb1 + np.log(prob2 + 1e-10)
        else:

            prob0 = pred0.data.numpy() * -1
            prob1 = pred1.data.numpy() * -1
            prob2 = pred2.data.numpy() * -1
            prob0_Adj = np.copy(prob0)
            outputProb0 = prob0[0]
            outputProb0 = outputProb0.repeat(M).reshape((M, M))
            prob1_Adj = addFromLog([outputProb0, prob1])
            outputProb1 = prob1_Adj.repeat(M).reshape((M*M, M))
            prob2_Adj = addFromLog([outputProb1, prob2])
            prob2_Adj = prob2_Adj * -1


        #print (np.sum(np.exp(prob2_Adj)))
        #quit()

        prob2_Adj_3D = prob2_Adj.reshape((M, M, M))


        #plt.imshow(np.sum(prob2_Adj_3D, axis=0))
        #plt.show()
        #quit()

        ###############prob2_Adj_3D = np.exp(prob2_Adj_3D_log)
        #prob2_Adj_3D[arange0, arange0, :] = -1000
        #prob2_Adj_3D[arange0, :, arange0] = -1000
        #prob2_Adj_3D[:, arange0, arange0] = -1000
        #################prob2_Adj_3D = prob2_Adj_3D / np.exp(prob2_Adj_3D)
        prob2_Adj_3D = prob2_Adj_3D - np.log(np.sum(np.exp(prob2_Adj_3D)))



        if False:
            argsBoring = np.arange(M)[np.isin(np.arange(M), argsInteresting) == False]
            for b in range(3):

                sizeBefore = list(prob2_Adj_3D.shape)

                sizeBefore[b] = argsInteresting.shape[0] + 1

                prob2_Adj_3D_new = np.zeros((sizeBefore[0], sizeBefore[1], sizeBefore[2]))

                if b == 0:
                    prob2_Adj_3D_new[:-1, :, :] = np.copy(prob2_Adj_3D[argsInteresting, :, :])
                    #max1 = np.max(prob2_Adj_3D[argsBoring])
                    max1 = doMax(prob2_Adj_3D[argsBoring])
                    prob2_Adj_3D_new[-1, :, :] = np.log(np.sum(np.exp(prob2_Adj_3D[argsBoring, :, :] - max1), axis=b)+1e-20) + max1
                if b == 1:
                    prob2_Adj_3D_new[:, :-1, :] = np.copy(prob2_Adj_3D[:, argsInteresting, :])
                    #max1 = np.max(prob2_Adj_3D[:, argsBoring])
                    max1 = doMax(prob2_Adj_3D[:, argsBoring])
                    prob2_Adj_3D_new[:, -1, :] = np.log(np.sum(np.exp(prob2_Adj_3D[:, argsBoring, :] - max1), axis=b)+1e-20) + max1
                if b == 2:
                    prob2_Adj_3D_new[:, :, :-1] = np.copy(prob2_Adj_3D[:, :, argsInteresting])
                    #max1 = np.max(prob2_Adj_3D[:, :, argsBoring])
                    max1 = doMax(prob2_Adj_3D[:, :, argsBoring])
                    prob2_Adj_3D_new[:, :, -1] = np.log(np.sum(np.exp(prob2_Adj_3D[:, :, argsBoring] - max1), axis=b)+1e-20) + max1

                prob2_Adj_3D = np.copy(prob2_Adj_3D_new)

            mutationName = np.concatenate((mutationName[argsInteresting], np.array(['Generic'])))


        #if argsBoring.size > 0:
        M = argsInteresting.shape[0] #+ 1
        #else:
        #    M = argsInteresting.shape[0]





        prob2_Adj = prob2_Adj_3D.reshape((M*M, M))



        prob2_sum0 = np.sum(np.sum(np.exp(prob2_Adj_3D), axis=0), axis=0)
        prob2_sum1 = np.sum(np.sum(np.exp(prob2_Adj_3D), axis=1), axis=1)
        prob2_sum2 = np.sum(np.sum(np.exp(prob2_Adj_3D), axis=0), axis=1)
        prob2_Assume = (prob2_sum0 + prob2_sum1 + prob2_sum2) / 3


        return prob2_Assume, prob2_Adj, prob2_Adj_3D, mutationName









    #32

    T = 1
    #T = 5
    #T = 6

    #Rnow = 4
    for Rnow in range(32):

        print (Rnow)

        pathways_true = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(Rnow) + '_pathway.npz', allow_pickle=True)

        modelFile = './dataNew/specialSim/results/T_' + str(T) + '_R_' + str(Rnow) + '_model.pt'
        model = torch.load(modelFile)

        #mutationName = np.load('./data/mutationNamesBreast.npy')[:-2]
        #argsInteresting = np.load('./data/interestingMutations.npy')


        M = 20
        argsInteresting = np.arange(M)
        mutationName = np.arange(M)

        prob2_Assume, prob2_Adj, prob2_Adj_3D, mutationName = doProbabilityFind(M, model, argsInteresting, mutationName)

        pathway = [np.arange(M-1), np.arange(M-1), np.arange(M-1)]
        #pathway = [[], np.arange(M-1), np.arange(M-1)]
        #pathway = [[], [] , np.arange(M-1)]

        superPathway = [np.arange(M-1), np.arange(M-1), np.arange(M-1)]

        #pathway = [np.arange(M), np.arange(M), np.arange(M)]

        prob2_Adj_3D_mod = np.copy(prob2_Adj_3D)

        #quit()


        predictedPathways = []
        pathwayScoreList = []


        for a in range(0, 4):

            #score = evaluatePathway(pathway, prob2_Adj_3D, prob2_Assume)
            doAdd = True
            step = 0
            position = 0
            doAddList = [True, False]
            stepList = [0, 1, 2]
            #score, pathway2 = stepModifyPathway(doAdd, step, pathway, superPathway, prob2_Adj_3D, prob2_Assume)
            #score, pathway2 = stepModifyPathway(doAdd, step, pathway, superPathway, prob2_Adj_3D, prob2_Assume)


            pathway2 = singleOptimizePathway(doAddList, stepList, pathway, superPathway, prob2_Adj_3D_mod, prob2_Assume)

            predictedPathways.append(copy.copy(pathway2))

            pathwayScores = evaluatePathway(pathway2, prob2_Adj_3D_mod, prob2_Assume, includeProb=True)

            pathwayScoreList.append(copy.copy( [ pathwayScores[0], pathwayScores[1], pathwayScores[2]  ]  ))



            #prob2_Adj_3D_mod = removePathFromProb(prob2_Adj_3D_mod, pathway2)

            pathway2_full = np.concatenate((pathway2[0], pathway2[1], pathway2[2])).astype(int)
            prob2_Adj_3D_mod[pathway2_full] = -1000
            prob2_Adj_3D_mod[:, pathway2_full] = -1000
            prob2_Adj_3D_mod[:, :, pathway2_full] = -1000


            #print (pathway2)
            #print (pathwayScores)

        predictedPathways = np.array(predictedPathways, dtype=object)

        np.save('./dataNew/specialSim/results/T_' + str(T) + '_R_' + str(Rnow) + '_predictedPathway.npy', predictedPathways)
        np.save('./dataNew/specialSim/results/T_' + str(T) + '_R_' + str(Rnow) + '_pathwayScore.npy', pathwayScoreList)



#savePathwaySimulationPredictions()
#quit()


def testPathwaySimulation():

    numberPathwaysErrors = 0

    pathwaySetError = np.zeros((2, 2))

    pathwayError = 0
    mutationsUsed = 0
    mutationTotal = 0

    T = 1
    #T = 5
    #T = 6

    for Rnow in range(0, 30):

        M = 20

        pathways_true = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(Rnow) + '_pathway.npz', allow_pickle=True)

        predictedPathways = np.load('./dataNew/specialSim/results/T_' + str(T) + '_R_' + str(Rnow) + '_predictedPathway.npy', allow_pickle=True)

        pathwayScores = np.load('./dataNew/specialSim/results/T_' + str(T) + '_R_' + str(Rnow) + '_pathwayScore.npy')
        pathwayScores = np.array(pathwayScores)
        bestArgs = np.argwhere(pathwayScores[:, 0] > 0)[:, 0]

        if len(bestArgs) != len(pathways_true):

            numberPathwaysErrors += 1
            print (pathways_true)
            print (Rnow)
            quit()

        pathwaySets_true = []

        for b in range(len(pathways_true)):
            pathwaySets_true.append([])
            for c in range(len(pathways_true[b])):
                pathwaySets_true[b] = pathwaySets_true[b] + list(pathways_true[b][c])

        pathwaySets_pred = []

        for b in range(len(bestArgs)):
            pathwaySets_pred.append([])
            for c in range(len(predictedPathways[b])):
                pathwaySets_pred[b] = pathwaySets_pred[b] + list(predictedPathways[b][c])


        overlapScores = np.zeros((len(pathwaySets_true), len(pathwaySets_pred) ))
        overlapScoresBoth = np.zeros((len(pathwaySets_true), len(pathwaySets_pred), 2 ))
        for b in range(len(pathwaySets_true)):
            for c in range(len(pathwaySets_pred)):

                set_true = np.array(pathwaySets_true[b])
                set_pred = np.array(pathwaySets_pred[c])

                #overlapScore = 2 * np.intersect1d(set_true, set_pred).shape[0] / (set_true.shape[0] + set_pred.shape[0])

                overlapScore = (set_true.shape[0] + set_pred.shape[0]) - (2 * np.intersect1d(set_true, set_pred).shape[0])

                #print ("B")
                #print ( set_true.shape[0] -    np.intersect1d(set_true, set_pred).shape[0] )
                #print ( set_pred.shape[0] -    np.intersect1d(set_true, set_pred).shape[0] )

                overlapScores[b][c] = overlapScore

                overlapScoresBoth[b][c][0] = set_true.shape[0] -    np.intersect1d(set_true, set_pred).shape[0]
                overlapScoresBoth[b][c][1] = set_pred.shape[0] -    np.intersect1d(set_true, set_pred).shape[0]

        overlapBestArg = np.argsort(overlapScores, axis=0)[0, :]
        overlapBest = np.min(overlapScores, axis=0)


        overlapScoresBoth = overlapScoresBoth[overlapBestArg, np.arange(overlapBestArg.shape[0])]
        #print (overlapScoresBoth)

        predictedPathways_copy = []
        for b in range(len(overlapBestArg)):
            predictedPathways_copy.append(copy.copy(  predictedPathways[overlapBestArg[b]]   ))
        predictedPathways = predictedPathways_copy

        #if Rnow == 16:
        #    print (pathways_true)
        #    print (predictedPathways)
        #    quit()


        if len(bestArgs) == len(pathways_true):
            for b in range(len(predictedPathways)):

                set_true = np.array(pathwaySets_true[b])
                set_pred = np.array(pathwaySets_pred[overlapBestArg[b]])

                #print (set_true)
                #print (set_pred)

                truePos = np.intersect1d(set_true, set_pred).shape[0]
                falsePos = set_pred.shape[0] - truePos
                falseNeg = set_true.shape[0] - truePos
                trueNeg = M - truePos - falsePos - falseNeg



                pathwaySetError[0][0] += truePos
                pathwaySetError[0][1] += falsePos
                pathwaySetError[1][0] += falseNeg
                pathwaySetError[1][1] += trueNeg

                #print (pathwaySetError)


            mutationErrorSet = []
            for b in range(len(predictedPathways)):
                for c in range(len(predictedPathways[b])):
                    set_true = np.array(pathways_true[b][c])
                    set_pred = np.array(predictedPathways[b][c])

                    mutationsUsed += set_true.shape[0]

                    mutationErrorSet = mutationErrorSet + list(set_true[np.isin(set_true, set_pred) == False])
                    mutationErrorSet = mutationErrorSet + list(set_pred[np.isin(set_pred, set_true) == False])
            mutationErrorSet = np.array(mutationErrorSet)

            #print (mutationErrorSet)
            #print (mutationErrorSet.shape)
            mutationErrorSet = np.unique(mutationErrorSet)

            pathwayError += mutationErrorSet.shape[0]
            mutationTotal += 20


            #print ((Rnow+1) * M, mutationsUsed, pathwayError)

            #print ('numberPathwaysErrors', numberPathwaysErrors)
            #quit()

    print ("Number of Mutations: " + str( (Rnow+1) * M ))
    print ("Mutations Used In Pathways : " + str( mutationsUsed ))
    print ("Number Errors: " + str( pathwayError ))


#testPathwaySimulation()
#quit()




def sampleModelUnrestrict():

    M = 4
    maxM = 4
    N1 = 100


    model = torch.load('./Models/savedModel5.pt')


    #doPrint = False
    #if iter % nPrint == 0:
    #    doPrint = True

    #if doPrint:
    #    print (iter)



    Edges = np.zeros((N1, maxM+1, 2))
    Edges[:, 0, 1] = M
    clones =  torch.zeros((N1, maxM+1, M))

    probLog1 = torch.zeros(N1)
    probLog2 = torch.zeros(N1)

    for a in range(0, maxM):

        #argsLength = np.argwhere(treeLength >= (a + 1))[:, 0]

        M1 = a + 1
        counter = np.arange(N1)

        clones1 = clones[:, :M1].reshape((N1 * M1, M))
        #print (clones1.shape)
        output, _ = model(clones1)
        output = output.reshape((N1, M1 * M))
        output = torch.softmax(output, dim=1)

        newStart = Edges[:, :M1, 1].repeat(M).reshape((N1, M1 * M))
        newStartClone = np.arange(M1).repeat(N1*M).reshape((M1, N1, M))
        newStartClone = np.swapaxes(newStartClone, 0, 1).reshape((N1, M1 * M))

        newEnd = np.arange(M).repeat(N1*M1).reshape((M, N1*M1)).T.reshape((N1, M1 * M))

        edgeNums = (newStart * (M + 2)) + newEnd



        choiceNow = doChoice(output.data.numpy()).astype(int)
        #print (clones[0, a])

        printNum = 0
        #print (output[printNum])
        #quit()
        #print (output2[printNum])

        theoryProbability = output[counter, choiceNow]
        edgeChoice = edgeNums[counter, choiceNow]
        newStartClone = newStartClone[counter, choiceNow]


        edgeChoice_start = edgeChoice // (M + 2)
        edgeChoice_end = edgeChoice % (M + 2)


        clones[counter, a+1] = clones[counter, newStartClone].clone()
        clones[counter, a+1, edgeChoice_end] = 1

        Edges[:, M1, 0] = edgeChoice_start
        Edges[:, M1, 1] = edgeChoice_end

        #print (newStartClone[printNum], edgeChoice_end[printNum])

        #probLog1[argsLength] += torch.log(theoryProbability[argsLength]+1e-6)
        #probLog2[argsLength] += torch.log(sampleProbability[argsLength]+1e-6)
        #probLog1[argsLength] += torch.log(theoryProbability[argsLength]+1e-12)
        probLog1 += torch.log(theoryProbability+1e-12)

    Edges = Edges + 1
    Edges[Edges == 5] = 0

    count1 = 0
    a = 0
    while count1 <= 10:
        vals = []
        for b in range(0, Edges[a].shape[0]):
            vals.append( str(Edges[a][b][0]) + ':' + str(Edges[a][b][1]))
        vals = np.array(vals)
        #print (a)
        #print (Edges[a, 1:])
        if vals.shape[0] == np.unique(vals).shape[0]:
            print (a)
            print (Edges[a, 1:])
            count1 += 1
        a += 1

def trainSimulationModels(name):
    import os

    #names1 = ['M5_m5', 'M12_m7', 'M12_m12']
    #names1 = ['M5_m5']
    #names1 = ['M12_m7']
    #names1 = ['M12_m12']

    #for name in names1:
    #['simulations_solution', '.DS_Store', 'simulations_input']
    arr = os.listdir('./dataNew/' + name + '/simulations_input')
    #arr = arr[:7] #Done with 12.
    for name2 in arr:

        name2 = name2[:-4]

        trueCluster = int(name2.split('k')[1].split('_')[0])
        if True:#trueCluster == 5:
            #print (trueCluster)
            #quit()

            print (name2)

            fileIn = './dataNew/p_' + name + '/simulations_input/' + name2 + '.txt.npy'
            fileSave = './dataNew/p_' + name + '/models/' + name2
            baselineSave = './dataNew/p_' + name + '/baselines/' + name2


            maxM = 5
            #maxM = 7
            #maxM = 12
            newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, fileIn)
            N2 = int(np.max(sampleInverse)+1)
            trainSet = np.random.permutation(N2)

            #newTrees = newTrees[:, :-1]
            #M = M - 1
            #quit()


            trainSet = trainSet[:N2//2]

            #print (newTrees.shape)
            #print (np.max(sampleInverse))
            #print (sampleInverse.shape)
            #quit()

            #newTrees = newTrees + 1
            #newTrees[newTrees == 5] = 0

            #print (newTrees[0])
            #print (newTrees[100])
            #print (newTrees[200])

            #quit()

            trainModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave=fileSave, baselineSave=baselineSave, adjustProbability=True, trainSet=trainSet)




def evaluateSimulations(name):

    def makeTreeChoice(treeProbs, index, sampleInverse):

        treeChoice = np.zeros(index.shape[0]).astype(int)
        isRequired = np.zeros(index.shape[0]).astype(int)
        probabilityChangeLog = np.zeros(index.shape[0])

        for a in range(0, index.shape[0]):
            start1 = index[a]
            if a == (index.shape[0] - 1):
                end1 = sampleInverse.shape[0]
            else:
                end1 = index[a+1]
            argsLocal = np.arange(end1 - start1) + start1
            argsLocal = argsLocal.astype(int)
            localTreeProb = treeProbs[argsLocal]
            treeMax = np.argmax(localTreeProb) + start1
            treeChoice[a] = treeMax

            if (np.sum(localTreeProb) - np.max(localTreeProb)) <= 0.05:
                isRequired[a] = 2
            if (np.sum(localTreeProb) - np.max(localTreeProb)) == 0:
                isRequired[a] = 1
                probabilityChangeLog[a] = 1000
            else:
                sortedProb = np.sort(localTreeProb)
                #print ( np.log( sortedProb[-1] + 1e-5 ) - np.log( sortedProb[-2] + 1e-5 ))
                probabilityChangeLog[a] = np.log( sortedProb[-1] + 1e-5 ) - np.log( sortedProb[-2] + 1e-5 )


        return treeChoice, isRequired, probabilityChangeLog

    def treeToString(choiceTrees):
        choiceNums = (  choiceTrees[:, :, 0] * (maxM + 2) ) + choiceTrees[:, :, 1]
        choiceString = np.zeros(choiceNums.shape[0]).astype(str)
        for a in range(0, choiceNums.shape[0]):
            #choiceString[a] = str(choiceNums[a, 0]) + ':' + str(choiceNums[a, 1]) + ':' + str(choiceNums[a, 2]) + ':' + str(choiceNums[a, 3])
            choiceString1 = str(choiceNums[a, 0])
            for b in range(1, choiceNums.shape[1]):
                choiceString1 = choiceString1 + ':' + str(choiceNums[a, b])
            choiceString[a] = choiceString1

        return choiceString



    import os

    allSaveVals = []

    doCluster = True

    #names1 = ['M5_m5', 'M12_m7', 'M12_m12']
    #names1 = ['M12_m12']

    clusterNums = [[], []]
    #names1 = ['M5_m5']
    #names1 = ['M12_m7']
    #names1 = ['M12_m12']
    #names1 = ['M5_m5', 'M12_m7', 'M12_m12']
    accuracies = []
    #for name in names1:
    #['simulations_solution', '.DS_Store', 'simulations_input']
    arr = os.listdir('./dataNew/' + name + '/simulations_input')
    for name2 in arr:#[9:]:

        print (name2)

        name2 = name2[:-4]


        if doCluster:
            trueCluster = int(name2.split('k')[1].split('_')[0])
            clusterNums[0].append(trueCluster)
            Ncluster = 1

        #if True:#trueCluster == 5:
        try:
            fileIn = './dataNew/p_' + name + '/simulations_input/' + name2 + '.txt.npy'
            fileSave = './dataNew/p_' + name + '/models/' + name2
            baselineSave = './dataNew/p_' + name + '/baselines/' + name2

            if name == 'M5_m5':
                maxM = 5
            elif name == 'M12_m7':
                maxM = 7
            else:
                maxM = 12

            newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, fileIn)

            N2 = int(np.max(sampleInverse)+1)
            trainSet = np.random.permutation(N2)
            #trainSet = trainSet[:N2//2]

            _, index = np.unique(sampleInverse, return_index=True)

            treeProbs = np.load(baselineSave + '.npy')
            treeChoice = np.zeros(index.shape[0]).astype(int)

            for a in range(0, index.shape[0]):
                start1 = index[a]
                if a == (index.shape[0] - 1):
                    end1 = sampleInverse.shape[0]
                else:
                    end1 = index[a+1]
                argsLocal = np.arange(end1 - start1) + start1
                argsLocal = argsLocal.astype(int)
                localTreeProb = treeProbs[argsLocal]
                treeMax = np.argmax(localTreeProb) + start1
                treeChoice[a] = treeMax

                #if a == 2:
                #    print (argsLocal)
                #    print (newTrees[argsLocal[0]])
                #    print (newTrees[argsLocal[1]])


            if doCluster:

                treeString = treeToString(newTrees)

                treeProbs2 = np.copy(treeProbs)



                if True:
                    treeChoice, isRequired, probabilityChangeLog = makeTreeChoice(treeProbs2, index, sampleInverse)
                    choiceTrees = newTrees[treeChoice]
                    choiceString = treeToString(choiceTrees)
                    treeProbs2[np.isin(treeString, choiceString) == False] = 0.0


                currentCluster = 100000
                iter = 0
                while (currentCluster > Ncluster) and (iter < 1000):
                    treeChoice, isRequired, probabilityChangeLog = makeTreeChoice(treeProbs2, index, sampleInverse)
                    choiceTrees = newTrees[treeChoice]


                    choiceString = treeToString(choiceTrees)

                    probabilityChangeLog2 = np.zeros(probabilityChangeLog.shape)
                    choiceStringUnique = np.unique(choiceString)
                    for b in range(0, choiceStringUnique.shape[0]):
                        argsInString = np.argwhere(choiceString == choiceStringUnique[b])[:, 0]
                        #print (argsInString)
                        probabilityChangeLog2[argsInString] = np.sum(probabilityChangeLog[argsInString])

                    #print (probabilityChangeLog)
                    #print (probabilityChangeLog2)
                    #plt.hist(probabilityChangeLog2, bins=100)
                    #plt.show()
                    #quit()


                    #argsDeletable = np.argwhere(np.isin(choiceString, choiceString[isRequired != 0]) == False)[:, 0]
                    #if argsDeletable.shape[0] == 0:
                    argsDeletable = np.argwhere(np.isin(choiceString, choiceString[isRequired == 1]) == False)[:, 0]
                    choiceString_deletable = choiceString[argsDeletable]
                    probabilityChangeLog2_del = probabilityChangeLog2[argsDeletable]


                    #choiceString_deletable = choiceString[np.isin(choiceString, choiceString[isRequired == 1]) == False]

                    if choiceString_deletable.shape[0] == 0:
                        iter = 1000
                    else:
                        #uniqueString, countString = np.unique(choiceString_deletable, return_counts=True)
                        #deleteString = uniqueString[np.argmin(countString)]
                        #print (np.min(probabilityChangeLog2_del))

                        if np.min(probabilityChangeLog2_del) < 40:#np.min(probabilityChangeLog2_del) < 20:
                            deleteString = choiceString_deletable[np.argmin(probabilityChangeLog2_del)]
                            treeProbs2[treeString == deleteString] = 0
                        else:
                            iter = 1000

                        iter += 1

                    currentCluster = np.unique(choiceString).shape[0]

                    #print (currentCluster)

                #print ("Good")
                #quit()

                clusterNums[1].append(currentCluster)







            #treeProbs2 = np.zeros(treeProbs.shape[0])
            #treeProbs2[treeChoice] = 0.3

            choiceTrees = newTrees[treeChoice]

            solutionFile = './dataNew/p_' + name + '/simulations_solution/' + name2 + '.txt.npy'

            #solutionTrees = np.load(solutionFile)
            #solutionTrees = solutionTrees[:, 0]

            solutionTrees2, sampleInverse_, mutationCategory, treeLength_, uniqueMutation_, M_ = processTreeData(maxM, solutionFile)

            solutionTrees2 = solutionTrees2[:, :-1]
            choiceTrees = choiceTrees[:, :-1]

            #print (solutionTrees2[2])
            #print (choiceTrees[2])

            choiceNums = (  choiceTrees[:, :, 0] * (maxM + 2) ) + choiceTrees[:, :, 1]
            solutionNums = (  solutionTrees2[:, :, 0] * (maxM + 2) ) + solutionTrees2[:, :, 1]


            choiceString = treeToString(choiceTrees)
            solutionString = treeToString(solutionTrees2)
            #choiceString = np.zeros(choiceNums.shape[0]).astype(str)
            #solutionString = np.zeros(choiceNums.shape[0]).astype(str)
            #for a in range(0, choiceNums.shape[0]):
            #    choiceString[a] = str(choiceNums[a, 0]) + ':' + str(choiceNums[a, 1]) + ':' + str(choiceNums[a, 2]) + ':' + str(choiceNums[a, 3])
            #    solutionString[a] = str(solutionNums[a, 0]) + ':' + str(solutionNums[a, 1]) + ':' + str(solutionNums[a, 2]) + ':' + str(solutionNums[a, 3])

            #print (np.unique(choiceString, return_counts=True))
            #print (np.unique(solutionString, return_counts=True))
            #quit()

            '''
            choiceNums = np.sort(choiceNums, axis=1)
            solutionNums = np.sort(solutionNums, axis=1)

            diff = np.sum(np.abs(choiceNums - solutionNums), axis=1).astype(int)

            argsValid = np.argwhere(diff == 0)[:, 0]
            argsBad = np.argwhere(diff != 0)[:, 0]

            #print (argsValid.shape)
            #quit()

            accuracy = argsValid.shape[0] / diff.shape[0]

            print ('accuracy: ' + str(accuracy))

            accuracies.append(accuracy)

            #np.save('./data/accuracy_M5_UnknownCluster4.npy', accuracies)
            #np.save('./data/clusterNumberPrediction4.npy', clusterNums)

            #quit()
            '''


            valSave = np.zeros(202).astype(str)
            valSave[0] = name
            valSave[1] = name2
            valSave[2:2 + len(solutionString)] = solutionString
            valSave[102:102 + len(choiceString)] = choiceString

            allSaveVals.append(valSave)

            print (len(allSaveVals))
            print (name)
            np.save('./dataNew/allSave_' + name + '.npy', allSaveVals)




        except:
            True
            quit()
            #quit()

#trainSimulationModels('M5_m5')
#evaluateSimulations('M5_m5')
#quit()

def fastAllArgwhere(ar):
    ar_argsort = np.argsort(ar)
    ar1 = ar[ar_argsort]
    _, indicesStart = np.unique(ar1, return_index=True)
    _, indicesEnd = np.unique(ar1[-1::-1], return_index=True) #This is probably needless and can be found from indicesStart
    indicesEnd = ar1.shape[0] - indicesEnd - 1
    return ar_argsort, indicesStart, indicesEnd

def easySubsetArgwhere(A, B):
    argsInB = np.argwhere(np.isin(A, B))[:, 0]
    A = A[argsInB]
    A_argsort = np.argsort(A)
    A = A[A_argsort]
    _, indicesStart = np.unique(A, return_index=True)
    indicesEnd = np.copy(indicesStart)
    indicesEnd[:-1] = indicesEnd[1:]
    indicesEnd[-1] = A.shape[0]

    A_unique = np.unique(A)

    places = []

    c = 0
    for a in range(0, B.shape[0]):
        if B[a] in A_unique:
            place = argsInB[A_argsort[indicesStart[c]:indicesEnd[c]]]
            places.append(place)
            c += 1
        else:
            places.append(np.array([]))

    #for a in range(0, indicesStart.shape[0]):
    #    place = argsInB[A_argsort[indicesStart[a]:indicesEnd[a]]]
    #    places.append(place)

    return places


class PurityModel(nn.Module):
    def __init__(self):
        super(PurityModel, self).__init__()

        self.linP = torch.nn.Linear(1, 20)


    def forward(self):

        x = self.linP(torch.zeros((1, 1)))
        prob = torch.softmax(x, dim=1)

        return prob


def analyzeModel(modelName):

    print ("analyzeModel")

    if modelName == 'manual':
        model = torch.load('./Models/savedModel_manual.pt')
        mutationName = np.load('./data/categoryNames.npy')[:-2]
        M = 24
        latentMin = 0.1

    if modelName == 'breast':
        model = torch.load('./Models/savedModel_breast.pt')
        mutationName = np.load('./data/mutationNamesBreastLarge.npy')[:-2]
        M = 406
        #M = 365
        latentMin = 0.01



    #model = torch.load('./Models/savedModel25.pt')
    #model = torch.load('./Models/savedModel26.pt')
    ###model = torch.load('./Models/savedModel27.pt')
    #mutationName = np.load('./data/mutationNames.npy')[:-2]
    #mutationName = np.load('./data/mutationNamesBreast.npy')[:-2]
    #mutationName = np.load('./data/mutationNamesBreastLarge.npy')[:-2]
    #mutationName = np.load('./data/categoryNames.npy')[:-2]



    #model = torch.load('./data/specialSim/results/T_' + str(0) + '_R_' + str(0) + '_model.pt')

    #mutationName = np.arange(9).astype(str)




    #mutationType = loadnpz('./data/specialSim/dataSets/T_' + str(1) + '_R_' + str(0) + '_mutationType.npz')
    #probabilityMatrix = loadnpz('./data/specialSim/dataSets/T_' + str(1) + '_R_' + str(0) + '_prob.npz')

    #plt.plot(mutationType)
    #plt.show()
    #plt.imshow(probabilityMatrix)
    #plt.show()




    #M = 5
    #M = 365
    #M = 406
    #M = 10
    #X = torch.zeros((5, M))

    X = torch.zeros((M, M))
    X[np.arange(M), np.arange(M)] = 1


    #X = torch.zeros((M*M, M))
    #arange1 = np.arange(M*M)
    #X[arange1, arange1 % M] = X[arange1, arange1 % M] + 1
    #X[arange1, arange1 // M] = X[arange1, arange1 // M] + 1



    pred, xNP = model(X)

    for a in range(0, 5):
        xNP[:, a] = xNP[:, a] - np.median(xNP[:, a])

    latentSize = np.max(np.abs(xNP), axis=1)



    #argsInteresting = np.argwhere(latentSize > 0.01)[:, 0]
    #argsInteresting = np.argwhere(latentSize > 0.1)[:, 0]
    argsInteresting = np.argwhere(latentSize > latentMin)[:, 0]
    np.save('./data/interestingMutations.npy', argsInteresting)


    if False:

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=8, random_state=0).fit(xNP[argsInteresting])
        labels = np.array(kmeans.labels_)

        plt.plot(xNP[argsInteresting][np.argsort(labels)])
        plt.plot((labels[np.argsort(labels)] / 4) - 1)
        plt.show()
        #kmeans.predict([[0, 0], [12, 3]])
        #kmeans.cluster_centers_

        argsInteresting = argsInteresting[np.argsort(labels)]

    if True:
        #plt.plot(xNP[argsInteresting][np.argsort(xNP[argsInteresting, 0])])
        plt.plot(xNP)
        plt.title("Mutation Properties")
        plt.xlabel("Mutation Number")
        plt.ylabel("Latent Variable Value")

        argsHigh = np.argwhere(latentSize > 0.15)[:, 0]

        for i in argsHigh:
            name = mutationName[i]

            delt1 = np.max(xNP) / 100
            max1 = np.max(np.abs(xNP[i])) + (delt1 * 4)
            sign1 = np.sign(xNP[i][np.argmax(np.abs(xNP[i]))] )
            max1 = (max1  * sign1) - (delt1 * 3)

            plt.annotate(name, (i -  (M / 40), max1    ))
            #plt.annotate(name, (i -  (M / 20), np.max(xNP[i]) + (np.max(xNP) / 100)    ))


        #plt.plot(xNP[np.argsort(np.sum(np.abs(xNP), axis=1) )])
        plt.savefig('./images/LatentPlot_1.png')
        plt.show()
        #quit()

    if False:
        #plt.plot(xNP[argsInteresting][np.argsort(xNP[argsInteresting, 0])])
        plt.plot(xNP[argsInteresting])
        plt.title("Mutation Properties")
        plt.xlabel("Mutation Number")
        plt.ylabel("Latent Variable Value")

        argsHigh = np.argwhere(latentSize[argsInteresting] > 0.1)[:, 0]

        for i in argsHigh:
            print (i)
            name = mutationName[argsInteresting[i]]

            delt1 = np.max(xNP) / 100
            max1 = np.max(np.abs(xNP[argsInteresting[i]])) + (delt1 * 4)
            sign1 = np.sign(xNP[argsInteresting[i]][np.argmax(np.abs(xNP[argsInteresting[i]]))] )
            max1 = (max1  * sign1) - (delt1 * 3)

            if i == 0:
                plt.annotate(name, (i, max1     ))
            else:
                plt.annotate(name, (i - 0.5, max1     ))
            #plt.annotate(name, (i - 0.5, np.max(xNP[argsInteresting[i]]) + (np.max(xNP) / 100)    ))


        #plt.plot(xNP[np.argsort(np.sum(np.abs(xNP), axis=1) )])
        plt.savefig('./images/SomeLatentPlot_1.png')
        plt.show()
        #quit()

    #quit()


    pred_np = pred.data.numpy()

    pred2 = pred.reshape((1, -1))

    prob = torch.softmax(pred, dim=1)
    prob2 = torch.softmax(pred2, dim=1)
    prob2 = prob2.reshape(prob.shape)


    #prob1 = prob[0].data.numpy()
    prob_np = prob.data.numpy()
    prob2_np = prob2.data.numpy()


    prob2_sum = np.sum(prob2_np, axis=1)


    #fitnessChart = np.sum(prob2_np, axis=1).reshape((M, M))
    #plt.imshow(fitnessChart)
    #plt.show()

    #argsHigh = np.argwhere(prob2_sum > np.median(prob2_sum) * 2)[:, 0]
    argsHigh = np.argwhere(prob2_sum > np.median(prob2_sum) * 1.5)[:, 0]

    if True:
        plt.plot(prob2_sum)
        plt.ylabel('Relative Fitness')
        plt.xlabel('Mutation Number')
        for i in argsHigh:
            name = mutationName[i]
            #plt.annotate(name, (i -  (M / 20), prob2_sum[i] + (np.max(prob2_sum) / 100)    ))
            plt.annotate(name, (i , prob2_sum[i] + (np.max(prob2_sum) / 100)    ))
        plt.savefig('./images/fitnessPlot_1.jpg')
        plt.show()

    #plt.imshow(prob2_sum.reshape(M, M))
    #plt.show()
    #quit()

    #argsInteresting = np.argwhere(np.sum(prob_np, axis=0) > np.mean(np.sum(prob_np, axis=0)))[:, 0]

    #np.save('./data/interestingMutations.npy', argsInteresting)

    prob_np_inter = prob_np[argsInteresting][:, argsInteresting]

    prob_np_adj = np.copy(prob_np)
    prob_np_adj[np.arange(prob_np_adj.shape[0]), np.arange(prob_np_adj.shape[0])] = 0
    for a in range(len(prob_np_adj)):
        prob_np_adj[a] = prob_np_adj[a] / np.sum(prob_np_adj[a])

    prob_np_adj_inter = prob_np_adj[argsInteresting][:, argsInteresting]


    if False:


        plt.imshow(prob_np)
        #plt.imshow(pred_np)
        plt.ylabel('Existing Mutation')
        plt.xlabel("New Mutation")
        plt.colorbar()
        plt.savefig('./images/fullOccurancePlot_1.jpg')
        plt.show()





    if False:

        plt.imshow(prob_np_adj_inter)
        plt.xlabel("New Mutation")
        plt.ylabel('Existing Mutation')
        plt.colorbar()

        #plt.savefig('./images/occurancePlot_1.jpg')
        plt.show()



    if True:
        fig, ax = plt.subplots(1,1)

        plt.imshow(prob_np_inter)
        img = ax.imshow(prob_np_inter)
        plt.xlabel("New Mutation")
        plt.ylabel('Existing Mutation')
        plt.colorbar()
        ax.set_yticks(np.arange(argsInteresting.shape[0]))
        ax.set_yticklabels(mutationName[argsInteresting])

        ax.set_xticks(np.arange(argsInteresting.shape[0]))
        ax.set_xticklabels(mutationName[argsInteresting])


        plt.xticks(rotation = 90)
        plt.savefig('./images/occurancePlot_1.jpg')
        plt.show()




    if False:
        fig, ax = plt.subplots(1,1)

        prob_np_inter_adj = np.copy(prob_np_inter)

        #cutOff = 0.02
        cutOff = 0

        for a in range(prob_np_inter_adj.shape[1]):
            mean1 = np.mean(prob_np_inter_adj[:, a])

            #prob_np_inter_adj[:, a] = prob_np_inter_adj[:, a] - mean1
            prob_np_inter_adj[:, a] = (prob_np_inter_adj[:, a] / mean1)# - 1


            #prob_np_inter_adj[prob_np_inter_adj[:, a] < -1 * cutOff , a] = -1
            #prob_np_inter_adj[prob_np_inter_adj[:, a] >  1 * cutOff , a] =  1
            #prob_np_inter_adj[np.abs(prob_np_inter_adj[:, a]) < cutOff , a] =  0


        intername = mutationName[argsInteresting]

        #nameCheck = [['NRAS', 'PTPN11'], ['FLT3', 'NRAS'], ['NPM1', 'PTPN11'], ['KRAS', 'NRAS'], ['KRAS', 'PTPN11'], ['FLT3', 'KRAS'], ['FLT3', 'NPM1']]
        nameCheck = [['NRAS', 'PTPN11'], ['FLT3', 'NRAS'], ['NPM1', 'PTPN11'], ['KRAS', 'NRAS'], ['KRAS', 'PTPN11'], ['FLT3', 'KRAS'], ['FLT3', 'NPM1'], ['KRAS', 'NPM1'], ['FLT3', 'PTPN11'], ['NPM1', 'NRAS']]


        for a in range(len(nameCheck)):
            pair = nameCheck[a]

            name1 = np.argwhere(intername == pair[0])[0, 0]
            name2 = np.argwhere(intername == pair[1])[0, 0]

            print (prob_np_inter_adj[name1, name2], prob_np_inter_adj[name2, name1])


        prob_flat = prob_np_inter_adj.reshape((prob_np_inter_adj.size,))
        prob_argsort0 = np.argsort(prob_flat)
        prob_argsort = np.array([prob_argsort0 // prob_np_inter_adj.shape[0], prob_argsort0 % prob_np_inter_adj.shape[0]]).astype(int).T

        print (prob_flat[prob_argsort0][-15:-5])
        print (prob_flat[prob_argsort0][5:15])

        print (intername[prob_argsort[-15:-5, 0]][-1::-1])
        print (intername[prob_argsort[-15:-5, 1]][-1::-1])
        print (intername[prob_argsort[6:16, 0]])
        print (intername[prob_argsort[6:16, 1]])



        plt.imshow(prob_np_inter_adj)
        img = ax.imshow(prob_np_inter_adj)
        plt.xlabel("New Mutation")
        plt.ylabel('Existing Mutation')
        plt.colorbar()
        ax.set_yticks(np.arange(argsInteresting.shape[0]))
        ax.set_yticklabels(mutationName[argsInteresting])

        ax.set_xticks(np.arange(argsInteresting.shape[0]))
        ax.set_xticklabels(mutationName[argsInteresting])


        plt.xticks(rotation = 90)
        plt.savefig('./images/occurancePlot_1.jpg')
        plt.show()
    quit()


    #plt.plot( (autoPred*0) - 1 )
    #plt.plot(autoPred)
    #plt.show()

    #plt.plot(pred2[:5].T)
    #plt.plot(pred2[1])
    #plt.show()


    #quit()

    argsBad = np.argwhere(autoPred < -2)[:, 0]

    #print (mutationName[argsBad])
    #plt.plot(xNP[argsBad])
    #plt.show()
    #quit()


    #'NPM1_p.L287fs' #Common
    #'NRAS_p.G12D' #Common


    #arg1 = np.argwhere(mutationName == 'IDH2_p.R140Q')[0][0] #'FLT3-ITD'
    #'IDH2_p.R140Q' 'IDH2_p.R172K'
    arg2 = np.argwhere(mutationName == 'DNMT3A_p.R882H')[0][0]
    #arg1 = np.argwhere(np.isin(mutationName, np.array(['KRAS_p.D33E', 'KRAS_p.G12A', 'KRAS_p.G12D', 'KRAS_p.G12R', 'KRAS_p.G12S', 'KRAS_p.G60D', 'KRAS_p.V14I']) ))[:, 0]
    #arg1 = np.argwhere(np.isin(mutationName, np.array(['NRAS_p.G12A', 'NRAS_p.G12D', 'NRAS_p.G12R', 'NRAS_p.G12S', 'NRAS_p.G12V',
    #                                                   'NRAS_p.G13D', 'NRAS_p.G13R', 'NRAS_p.G13V', 'NRAS_p.G60E', 'NRAS_p.Q61H', 'NRAS_p.Q61K']) ))[:, 0]
    arg1 = np.argwhere(np.isin(mutationName, np.array(['IDH1_p.R132C', 'IDH1_p.R132G', 'IDH1_p.R132H', 'IDH2_p.R140Q', 'IDH2_p.R172K']) ))[:, 0]



    plt.plot(xNP[arg1])
    plt.show()
    quit()

    #arg1, arg2 = arg2, arg1
    #arg1 = arg1[3]

    hotCode = np.zeros(prob_np.shape[0])
    hotCode[arg2] = 10



    #plt.plot(np.mean(prob_np, axis=0))
    #plt.plot(prob_np[arg1])
    #plt.plot(np.mean(pred_np, axis=0))
    #plt.plot(pred_np[arg1])
    #plt.plot(hotCode)
    #plt.show()


    quit()


    diff = (np.abs(prob1 - np.median(prob1))) / np.median(prob1)
    argsGood = np.argwhere(diff > 0.01)[:, 0]

    #print (argsGood)

    prob1 = prob1 / np.mean(prob1)

    argSorted = np.argsort(prob1)[-1::-1]


    #plt.plot(prob1[argsGood])
    #plt.plot(prob2[:11, argsGood].T)
    plt.plot(prob2[:3, :].T)
    #plt.plot(prob1)
    plt.show()
    quit()


#analyzeModel('manual')
#analyzeModel('breast')
#quit()




def analyzePathway():

    makeLength2 = True
    #makeLength2 = False

    def pathwayTheoryProb(pathway, prob_assume):

        #makeLength2 = True

        probLog = 0
        pathwayLength = len(pathway)

        if makeLength2:
            pathwayLength = 2

        for a in range(0, pathwayLength):
            pathwaySet = pathway[a]
            if len(pathwaySet) == 0:
                #probLog += -1000
                True
            else:
                prob = np.sum(prob_assume[np.array(pathwaySet)])
                probLog += np.log(prob)

        return probLog

    def pathwayRealProb(pathway, prob2_Adj_3D):

        #makeLength2 = True

        subset = prob2_Adj_3D

        if min(min(len(pathway[0]), len(pathway[1])), len(pathway[2]) ) == 0:
            subset_sum = np.log(1e-50)
        else:

            subset = subset[np.array(pathway[0])]
            subset = subset[:, np.array(pathway[1])]

            if not makeLength2:
                subset = subset[:, :, np.array(pathway[2])]

            subset_max = np.max(subset)
            subset = subset - subset_max
            subset_sum = np.sum(np.exp(subset))
            subset_sum = np.log(subset_sum+1e-50)
            subset_sum = subset_sum + subset_max

        return subset_sum



    def evaluatePathway(pathway, prob2_Adj_3D, prob_assume, includeProb=False):

        probTheory = pathwayTheoryProb(pathway, prob_assume)
        probReal = pathwayRealProb(pathway, prob2_Adj_3D)

        probDiff = probReal - probTheory


        #if probReal > np.log(0.1):
        #    probReal = np.log(0.1)

        if probReal > np.log(0.05): #Used for breast cancer data
            probReal = np.log(0.05)  #Used for breast cancer data

        #score = (probReal * 0.1) + probDiff


        #score = probDiff - (0.1 * (np.abs(probDiff) ** 2))

        #score = (probReal * 0.05) + probDiff
        #score = (probReal * 0.2) + probDiff #Used for breast cancer data

        #score = (probReal * 0.1) + probDiff

        score = (probReal * 0.2) + probDiff

        #score = (probReal * 0.05) + probDiff

        #score = (probReal * 0.2) + probDiff
        #score = (probReal * 0.2) + probDiff
        #score = probReal - (0.8 * probTheory)
        if includeProb:
            return score, probReal, probDiff
        else:
            return score



    def singleModifyPathway(doAdd, step, position, pathway, prob2_Adj_3D, prob_assume):

        pathway2 = copy.deepcopy(pathway)
        set1 = pathway2[step]
        set1 = np.array(set1)

        if doAdd:
            set1 = np.concatenate((set1, np.array([position])))
            set1 = np.sort(set1)
        else:
            set1 = set1[set1 != position]

        #pathway2[step] = set1.astype(int)
        pathway2[step] = set1.astype(int)

        score= evaluatePathway(pathway2, prob2_Adj_3D, prob_assume)

        return score, pathway2

    def stepModifyPathway(doAdd, step, pathway, superPathway, prob2_Adj_3D, prob_assume):

        M = prob_assume.shape[0]

        set1 = copy.deepcopy(pathway[step])
        if doAdd or (len(set1) > 1):
            set2 = copy.deepcopy(superPathway[step])
            set3 = set2[np.isin(set2, set1) != doAdd]

            pathways = []
            scores = []
            for position in set3:
                score, pathway2 = singleModifyPathway(doAdd, step, position, pathway, prob2_Adj_3D, prob_assume)

                pathways.append(copy.deepcopy(pathway2))
                scores.append(score)

            return pathways, scores

        else:

            return [], []


    def iterOptimizePathway(doAddList, stepList, pathway, superPathway, prob2_Adj_3D, prob_assume):

        score = evaluatePathway(pathway, prob2_Adj_3D, prob_assume)
        pathway2 = copy.deepcopy(pathway)

        pathways2, scores2 = [pathway2], [score]
        for doAdd in doAddList:
            for step in stepList:
                pathways, scores = stepModifyPathway(doAdd, step, pathway, superPathway, prob2_Adj_3D, prob_assume)
                pathways2 = pathways2 + pathways
                scores2 = scores2 + scores





        for step in stepList:
            set1 = [-1] + list(range(M))
            for pos_add in set1:

                pathway2 = copy.deepcopy(pathway)

                if pos_add == -1:
                    pathway2[step] = list(np.arange(M))
                else:
                    pathway2[step] = [pos_add]

                score = evaluatePathway(pathway2, prob2_Adj_3D, prob_assume)

                pathways2.append(copy.deepcopy(pathway2))
                scores2.append(copy.copy(score))






        bestOne = np.argmax(np.array(scores2))
        bestPathway = pathways2[bestOne]
        bestScore = np.max(scores2)

        return bestPathway, bestScore




    def iterOptimizePathway2(doAddList, stepList, pathway, superPathway, prob2_Adj_3D, prob_assume):

        M = prob_assume.shape[0]

        score = evaluatePathway(pathway, prob2_Adj_3D, prob_assume)
        pathway2 = copy.deepcopy(pathway)

        pathways2, scores2 = [pathway2], [score]

        for step in stepList:

            set1 = copy.deepcopy(pathway[step])
            set2 = copy.deepcopy(superPathway[step])

            set_add = set2[np.isin(set2, set1) != True]
            set_rem = set2[np.isin(set2, set1) != False]

            set_add = [-1] + list(set_add)
            set_rem = [-1] + list(set_rem)


            '''
            for pos_add in set_add:
                for pos_rem in set_rem:

                    pathway2 = copy.deepcopy(pathway)

                    if pos_add >= 0:
                        score, pathway2 = singleModifyPathway(True, step, pos_add, pathway2, prob2_Adj_3D, prob_assume)

                    if pos_rem >= 0:
                        #print ("T2")
                        #print (pathway2)
                        #print (pos_rem, step)
                        score, pathway2 = singleModifyPathway(False, step, pos_rem, pathway2, prob2_Adj_3D, prob_assume)


                    pathways2.append(copy.deepcopy(pathway2))
                    scores2.append(copy.copy(score))
            '''


            for pos_add in set_add:
                pathway2 = copy.deepcopy(pathway)

                if pos_add >= 0:
                    score, pathway2 = singleModifyPathway(True, step, pos_add, pathway2, prob2_Adj_3D, prob_assume)

                pathways2.append(copy.deepcopy(pathway2))
                scores2.append(copy.copy(score))

            for pos_rem in set_rem:
                pathway2 = copy.deepcopy(pathway)

                if pos_rem >= 0:
                    score, pathway2 = singleModifyPathway(False, step, pos_rem, pathway2, prob2_Adj_3D, prob_assume)

                pathways2.append(copy.deepcopy(pathway2))
                scores2.append(copy.copy(score))

        for step in stepList:
            set1 = [-1] + list(range(M))
            for pos_add in set1:

                pathway2 = copy.deepcopy(pathway)

                if pos_add == -1:
                    pathway2[step] = list(np.arange(M))
                else:
                    pathway2[step] = [pos_add]

                #print (pathway2)

                score = evaluatePathway(pathway2, prob2_Adj_3D, prob_assume)

                pathways2.append(copy.deepcopy(pathway2))
                scores2.append(copy.copy(score))

        #quit()

        bestOne = np.argmax(np.array(scores2))
        bestPathway = pathways2[bestOne]
        bestScore = np.max(scores2)

        return bestPathway, bestScore


    def singleOptimizePathway(doAddList, stepList, pathway, superPathway, prob_Adj_3D, prob_assume):

        pathway2 = copy.deepcopy(pathway)
        bestScoreBefore = -10000
        notDone = True
        while notDone:
            #pathway2, bestScore = iterOptimizePathway(doAddList, stepList, pathway2, superPathway, prob_Adj_3D, prob_assume)
            pathway2, bestScore = iterOptimizePathway2(doAddList, stepList, pathway2, superPathway, prob_Adj_3D, prob_assume)

            #print ("B")
            #print (pathway2)
            #print (bestScore)



            if bestScoreBefore == bestScore:
                notDone = False
            bestScoreBefore = bestScore

        #print (evaluatePathway(pathway2, prob2_Adj_3D, prob_assume, includeProb=True))

        evalScores = evaluatePathway(pathway2, prob2_Adj_3D, prob_assume, includeProb=True)

        #print (evalScores)
        print (np.exp(evalScores[1]), np.exp(evalScores[2]))

        return pathway2



    def removePathFromProb(prob, pathway):

        set1, set2, set3 = np.array(pathway[0]), np.array(pathway[1]), np.array(pathway[2])
        M = prob.shape[0]
        modMask = np.zeros(prob.shape)
        modMask[np.arange(M)[np.isin(np.arange(M), set1) == False]] = 1
        modMask[:, np.arange(M)[np.isin(np.arange(M), set2) == False]] = 1

        if not makeLength2:
            modMask[:, :, np.arange(M)[np.isin(np.arange(M), set3) == False]] = 1

        prob[modMask == 0] = -1000

        return prob


    def doMax(ar1):

        if ar1.size == 0:
            print ("A")
            return 0
        else:
            print ("B")
            return np.max(ar1)




    #pathways = [[  [0], [1, 2], [3]  ], [  [4, 5], [6], [7]  ]]

    #model = torch.load('./Models/savedModel4.pt')
    #model = torch.load('./Models/savedModel14.pt')
    #model = torch.load('./Models/savedModel15.pt')
    #model = torch.load('./Models/savedModel18.pt')
    #model = torch.load('./Models/savedModel19.pt')
    #model = torch.load('./Models/savedModel20.pt')
    #model = torch.load('./Models/savedModel21.pt')
    #model = torch.load('./Models/savedModel22.pt')
    model = torch.load('./Models/savedModel25.pt')
    #model = torch.load('./Models/savedModel26.pt')


    #mutationName = np.load('./data/mutationNamesBreastLarge.npy')[:-2]
    #mutationName = np.load('./data/mutationNamesBreast.npy')[:-2]
    mutationName = np.load('./data/categoryNames.npy')[:-2]
    argsInteresting = np.load('./data/interestingMutations.npy')

    #argsInteresting = np.arange(20)
    #mutationName = np.arange(20)

    #print (len(mutationName))
    #print (np.max(argsInteresting))
    #quit()

    #M = 93
    #M = 116
    #M = 365
    #M = 406
    #M = 20
    M = 24
    #X0 = torch.zeros((M*M, M))
    X0 = torch.zeros((1, M))
    X1 = torch.zeros((M, M))
    X2 = torch.zeros((M*M, M))

    arange0 = np.arange(M)
    arange1 = np.arange(M*M)

    X1[arange0, arange0] = 1

    X2[arange1, arange1 % M] = X2[arange1, arange1 % M] + 1
    X2[arange1, arange1 // M] = X2[arange1, arange1 // M] + 1

    pred0, _ = model(X0)
    pred1, xLatent1 = model(X1)
    pred2, _ = model(X2)


    pred2 = pred2.reshape((M, M, M))
    pred1[arange0, arange0] = -1000
    pred2[arange0, arange0, :] = -1000
    pred2[:, arange0, arange0] = -1000
    pred2[arange0, :, arange0] = -1000
    pred2 = pred2.reshape((M * M, M))



    #print (xLatent1.shape)

    #plt.plot(xLatent1)
    #plt.show()
    #quit()

    if True:
        prob0 = torch.softmax(pred0, dim=1)
        prob1 = torch.softmax(pred1, dim=1)
        prob2 = torch.softmax(pred2, dim=1)

        #plt.imshow(prob1.data.numpy())
        #plt.show()

        prob0 = prob0.data.numpy()
        prob1 = prob1.data.numpy()
        prob2 = prob2.data.numpy()
        prob0_Adj = np.log(np.copy(prob0) + 1e-10)
        outputProb0 = np.log(prob0[0] + 1e-10)
        outputProb0 = outputProb0.repeat(M).reshape((M, M))
        prob1_Adj = outputProb0 + np.log(prob1 + 1e-10)


        #plt.imshow(np.exp(prob1_Adj))
        #plt.show()


        outputProb1 = prob1_Adj.repeat(M).reshape((M*M, M))
        prob2_Adj = outputProb1 + np.log(prob2 + 1e-10)
    else:

        prob0 = pred0.data.numpy() * -1
        prob1 = pred1.data.numpy() * -1
        prob2 = pred2.data.numpy() * -1
        prob0_Adj = np.copy(prob0)
        outputProb0 = prob0[0]
        outputProb0 = outputProb0.repeat(M).reshape((M, M))
        prob1_Adj = addFromLog([outputProb0, prob1])
        outputProb1 = prob1_Adj.repeat(M).reshape((M*M, M))
        prob2_Adj = addFromLog([outputProb1, prob2])
        prob2_Adj = prob2_Adj * -1


    #print (np.sum(np.exp(prob2_Adj)))
    #quit()

    prob2_Adj_3D = prob2_Adj.reshape((M, M, M))


    #plt.imshow(np.sum(prob2_Adj_3D, axis=0))
    #plt.show()
    #quit()

    ###############prob2_Adj_3D = np.exp(prob2_Adj_3D_log)
    #prob2_Adj_3D[arange0, arange0, :] = -1000
    #prob2_Adj_3D[arange0, :, arange0] = -1000
    #prob2_Adj_3D[:, arange0, arange0] = -1000
    #################prob2_Adj_3D = prob2_Adj_3D / np.exp(prob2_Adj_3D)
    prob2_Adj_3D = prob2_Adj_3D - np.log(np.sum(np.exp(prob2_Adj_3D)))



    if True:
        argsBoring = np.arange(M)[np.isin(np.arange(M), argsInteresting) == False]
        for b in range(3):

            sizeBefore = list(prob2_Adj_3D.shape)

            sizeBefore[b] = argsInteresting.shape[0] + 1

            prob2_Adj_3D_new = np.zeros((sizeBefore[0], sizeBefore[1], sizeBefore[2]))

            if b == 0:
                prob2_Adj_3D_new[:-1, :, :] = np.copy(prob2_Adj_3D[argsInteresting, :, :])
                #max1 = np.max(prob2_Adj_3D[argsBoring])
                max1 = doMax(prob2_Adj_3D[argsBoring])
                prob2_Adj_3D_new[-1, :, :] = np.log(np.sum(np.exp(prob2_Adj_3D[argsBoring, :, :] - max1), axis=b)+1e-20) + max1
            if b == 1:
                prob2_Adj_3D_new[:, :-1, :] = np.copy(prob2_Adj_3D[:, argsInteresting, :])
                #max1 = np.max(prob2_Adj_3D[:, argsBoring])
                max1 = doMax(prob2_Adj_3D[:, argsBoring])
                prob2_Adj_3D_new[:, -1, :] = np.log(np.sum(np.exp(prob2_Adj_3D[:, argsBoring, :] - max1), axis=b)+1e-20) + max1
            if b == 2:
                prob2_Adj_3D_new[:, :, :-1] = np.copy(prob2_Adj_3D[:, :, argsInteresting])
                #max1 = np.max(prob2_Adj_3D[:, :, argsBoring])
                max1 = doMax(prob2_Adj_3D[:, :, argsBoring])
                prob2_Adj_3D_new[:, :, -1] = np.log(np.sum(np.exp(prob2_Adj_3D[:, :, argsBoring] - max1), axis=b)+1e-20) + max1

            prob2_Adj_3D = np.copy(prob2_Adj_3D_new)

        mutationName = np.concatenate((mutationName[argsInteresting], np.array(['Generic'])))


    #if argsBoring.size > 0:
    M = argsInteresting.shape[0] + 1
    #else:
    #    M = argsInteresting.shape[0]



    prob2_Adj = prob2_Adj_3D.reshape((M*M, M))

    #print (np.sum(prob2_Adj_3D, axis=2))

    #print (np.sum(np.exp(prob2_Adj_3D), axis=2))

    #plt.imshow(np.sum(np.exp(prob2_Adj_3D), axis=2))
    #plt.imshow(np.exp(prob2_Adj_3D)[18])
    #plt.show()
    #quit()

    #pathways = [[  [4, 5], [6], [7]  ]]



    prob2_sum0 = np.sum(np.sum(np.exp(prob2_Adj_3D), axis=0), axis=0)
    prob2_sum1 = np.sum(np.sum(np.exp(prob2_Adj_3D), axis=1), axis=1)
    if not makeLength2:
        prob2_sum2 = np.sum(np.sum(np.exp(prob2_Adj_3D), axis=0), axis=1)
        prob2_Assume = (prob2_sum0 + prob2_sum1 + prob2_sum2) / 3
    else:
        prob2_Assume = (prob2_sum0 + prob2_sum1) / 2

    #prob2_Assume_3D_i = prob2_Assume.repeat(M*M).reshape((M, M, M))
    #prob2_Assume_3D_0 = prob2_Assume_3D_i
    #prob2_Assume_3D_1 = np.swapaxes(prob2_Assume_3D_i, 0, 1)
    #prob2_Assume_3D_2 = np.swapaxes(prob2_Assume_3D_i, 0, 2)
    #prob2_Assume_3D = np.log(prob2_Assume_3D_0) + np.log(prob2_Assume_3D_1) + np.log(prob2_Assume_3D_2)
    #prob2_Assume_3D = prob2_Assume_3D.reshape((M*M, M))

    #pathway = [(M//2) + np.arange(M // 2), np.arange(M), np.arange(M)]

    #pathway = [np.arange(M), np.arange(M), np.arange(M)]
    #superPathway = [np.arange(M), np.arange(M), np.arange(M)]

    pathway = [np.arange(M-1), np.arange(M-1), np.arange(M-1)]
    #pathway = [[], np.arange(M-1), np.arange(M-1)]
    #pathway = [[], [] , np.arange(M-1)]

    superPathway = [np.arange(M-1), np.arange(M-1), np.arange(M-1)]

    #pathway = [np.arange(M), np.arange(M), np.arange(M)]

    prob2_Adj_3D_mod = np.copy(prob2_Adj_3D)



    for a in range(0, 20):

        #score = evaluatePathway(pathway, prob2_Adj_3D, prob2_Assume)
        doAdd = True
        step = 0
        position = 0
        doAddList = [True, False]
        stepList = [0, 1, 2]
        #score, pathway2 = stepModifyPathway(doAdd, step, pathway, superPathway, prob2_Adj_3D, prob2_Assume)
        #score, pathway2 = stepModifyPathway(doAdd, step, pathway, superPathway, prob2_Adj_3D, prob2_Assume)


        pathway2 = singleOptimizePathway(doAddList, stepList, pathway, superPathway, prob2_Adj_3D_mod, prob2_Assume)



        if True:
            #print ('SUM')
            #print (np.sum(np.exp(prob2_Adj_3D_mod)))
            prob2_Adj_3D_mod = removePathFromProb(prob2_Adj_3D_mod, pathway2)
            #print (np.sum(np.exp(prob2_Adj_3D_mod)))
        else:
            pathway2_full = np.concatenate((pathway2[0], pathway2[1], pathway2[2])).astype(int)
            prob2_Adj_3D_mod[pathway2_full] = -1000
            prob2_Adj_3D_mod[:, pathway2_full] = -1000
            prob2_Adj_3D_mod[:, :, pathway2_full] = -1000
        #print (pathway2_full)
        #quit()
        #prob2_Adj_3D_mod[]



        #print (pathway2)
        for b in range(3):
            properSet = mutationName[pathway2[b]]
            #print (np.intersect1d(properSet, used20).shape)
            #print (np.intersect1d(properSet, used20))
            #if np.intersect1d(properSet, used20).shape[0] == used20.shape[0]:
            #    Extra1 = properSet[np.isin(properSet, used20) == False]
            #    print ("used20 ", Extra1)
            #else:
            #    print (properSet)
        print (mutationName[pathway2[0]])
        print (mutationName[pathway2[1]])
        print (mutationName[pathway2[2]])
        #print (pathway2)
        print (len(pathway2[0]))
        print (len(pathway2[1]))
        print (len(pathway2[2]))
        #quit()

        if a == 4:
            quit()
        #quit()

        #pathways = [[  [0], [1, 2], [3, 4]  ]]



#analyzePathway()
#quit()


import sys

if __name__ == "__main__":

    #print (sys.argv)

    #print (sys.argv[1])



    if sys.argv[1] == 'test':
        True

        if sys.argv[2] == 'causal':
            if sys.argv[3] == 'train':
                trainNewSimulations(4, 20)
            if sys.argv[3] == 'print':
                testOccurSimulations(4, 20)

        if sys.argv[2] == 'proportion':
            doProportionAnalysis()

        if sys.argv[2] == 'pathway':
            if sys.argv[3] == 'train':
                trainNewSimulations(1, 32)
            if sys.argv[3] == 'evaluate':
                savePathwaySimulationPredictions()
            if sys.argv[3] == 'print':
                testPathwaySimulation()

    if sys.argv[1] == 'recap':
        if sys.argv[2] == 'm5':
            name = 'M5_m5'
        if sys.argv[2] == 'm7':
            name = 'M12_m7'
        if sys.argv[2] == 'm12':
            name = 'M12_m12'

        #evaluateSimulations('M5_m5')
        #doRECAPplot('M5_m5', doCluster=False)

        if sys.argv[3] == 'train':
            trainSimulationModels(name)

        if sys.argv[3] == 'evaluate':
            evaluateSimulations('M5_m5')

        if sys.argv[3] == 'plot':
            if sys.argv[4] == 'cluster':
                doRECAPplot(name, doCluster=True)
            if sys.argv[4] == 'accuracy':
                doRECAPplot(name, doCluster=True)

    if sys.argv[1] == 'real':
        if sys.argv[3] == 'train':
            if sys.argv[2] == 'leukemia':
                trainRealData('manual')
            if sys.argv[2] == 'breast':
                trainRealData('breast')

        if sys.argv[3] == 'plot':
            if sys.argv[2] == 'leukemia':
                analyzeModel('manual')
            if sys.argv[2] == 'breast':
                analyzeModel('breast')



    quit()

    #######


    #trainNewSimulations(1, 32)
    #savePathwaySimulationPredictions()
    #testPathwaySimulation()




    #trainRealData('manual')
    #analyzeModel('manual')

    #names1 = ['M5_m5', 'M12_m7', 'M12_m12']
    #trainSimulationModels('M5_m5')
    #evaluateSimulations('M5_m5')
    #doRECAPplot('M5_m5', doCluster=False)
    #trainSimulationModels('M12_m7')
    #evaluateSimulations('M12_m7')
    #doRECAPplot('M12_m7', doCluster=False)
    #trainSimulationModels('M12_m12')
    #evaluateSimulations('M12_m12')
    #doRECAPplot('M12_m12', doCluster=False)
