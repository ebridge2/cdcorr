import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels
import dodiscover as dod
import hyppo
import scipy as sp
import sklearn as sk
from rpy2.robjects.packages import STAP
from rpy2.robjects import numpy2ri
numpy2ri.activate()

def cond_manova(Y, T, X, **kwargs):
    with open('cond_ind.R', 'r') as f:
        string = f.read()
    cond_ind = STAP(string, "cond_ind")
    stat, pval = cond_ind.cmanova(Y, T, X)
    return float(pval), float(stat)


def kcit(Y, T, X, nrep=1000):
    with open('cond_ind.R', 'r') as f:
        string = f.read()
    cond_ind = STAP(string, "cond_ind")
    stat, pval = cond_ind.kcit_wrap(Y, ohe(T), X, nrep=nrep)
    return float(pval), float(stat)


def rcit(Y, T, X, nrep=1000):
    with open('cond_ind.R', 'r') as f:
        string = f.read()
    cond_ind = STAP(string, "cond_ind")
    stat, pval = cond_ind.rcit_wrap(Y, ohe(T), X, nrep=nrep)
    return float(pval), float(stat)


def rcot(Y, T, X, nrep=1000):
    with open('cond_ind.R', 'r') as f:
        string = f.read()
    cond_ind = STAP(string, "cond_ind")
    stat, pval = cond_ind.rcot_wrap(Y, ohe(T), X, nrep=nrep)
    return float(pval), float(stat)


def wgcm(Y, T, X, nrep=1000):
    with open('cond_ind.R', 'r') as f:
        string = f.read()
    cond_ind = STAP(string, "cond_ind")
    stat, pval = cond_ind.wgcm_wrap(Y, ohe(T), X, nrep=nrep)
    return float(pval), float(stat)


def gcm(Y, T, X, nrep=1000):
    with open('cond_ind.R', 'r') as f:
        string = f.read()
    cond_ind = STAP(string, "cond_ind")
    stat, pval = cond_ind.gcm_wrap(Y, ohe(T), X, nrep=nrep)
    return float(pval), float(stat)


def kernelcdtest(Y, T, X, nrep=1000):
    df_dict = {"Covariate" : X, "Group" : T}
    yvars = []
    for i in range(0, Y.shape[1]):
        yvar = "Y{:d}".format(i)
        df_dict[yvar] = Y[:,i]
        yvars.append(yvar)
    df = pd.DataFrame(df_dict)
    xvars = "Covariate"; group_col="Group"
    stat, pval = dod.cd.KernelCDTest(null_reps=nrep).test(df, [xvars], yvars, group_col)
    return pval, stat
    
def cond_dcorr(Y, T, X, nrep=1000):
    DT = sk.metrics.pairwise_distances(ohe(T), metric="l2")
    DY = sk.metrics.pairwise_distances(Y, metric="l2")
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    DX = sk.metrics.pairwise_distances(X, metric="l2")
    stat, pval = hyppo.conditional.CDcorr(compute_distance=None).test(DY, DT, DX, reps=nrep)
    return pval, stat

def dcorr(Y, T, X, nrep=1000):
    DT = sk.metrics.pairwise_distances(ohe(T), metric="l2")
    DY = sk.metrics.pairwise_distances(Y, metric="l2")
    stat, pval = hyppo.independence.Dcorr(compute_distance=None).test(DY, DT, reps=nrep)
    return pval, stat


def causal_prep(Xs, Ts, return_props=False):
    # adopted from Lopez 2017 Matching to estimate the causal effect
    # from multiple treatments
    Xs = sm.add_constant(Xs)
    m = sm.MNLogit(Ts, Xs)

    fit = m.fit()
    pred = fit.predict(Xs)

    Ts_unique = np.unique(Ts)
    K = len(Ts_unique)
    Rtable = []
    # check possible predictions T
    for T in range(0, K):
        Rtab = np.zeros((2, K))
        # for each prediction T, check what elements
        # which are in class Tp, and look at prediction
        # probability for class T
        for Tp in range(0, K):
            Rtab[0,Tp] = pred[Ts == Tp,T].min()
            Rtab[1,Tp] = pred[Ts == Tp,T].max()
        # low and high are the max of the mins and the min of the maxes
        # for class T
        Rtable.append(np.array((Rtab[0,:].max(), Rtab[1,:].min())))

    balance_check = np.zeros((Xs.shape[0], K))
    for T in range(0, K):
        for i in range(0, Xs.shape[0]):
            balance_check[i, T] = pred[i, T] >= Rtable[T][0] and pred[i, T] <= Rtable[T][1]
    balanced_ids = balance_check.all(axis=1)
    if return_props:
        return (balanced_ids, pred)
    else:
        return balanced_ids


def ohe(T):
    K = len(np.unique(T))
    ohe_dat = np.zeros((len(T), K))
    for t in np.unique(T):
        ohe_dat[:,t] = (T == t).astype(int)
    return ohe_dat