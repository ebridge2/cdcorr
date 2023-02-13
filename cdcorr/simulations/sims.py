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

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def simulate_covars(causal_preds, balance=1):
    coefs = []
    #for causal_pred in causal_preds:
    #    if causal_pred == 0:
    #        coefs.append((3, 3*1/balance))
    #    else:
    #        coefs.append((3*1/balance, 3))
    #covars = 2*np.array([float(np.random.beta(alpha, beta, size=1)) for (alpha, beta) in coefs]) - 1
    balance_id = np.random.binomial(1, balance, size=len(causal_preds))
    
    for causal_cl, bal_id in zip(causal_preds, balance_id):
        if bal_id == 1:
            coefs.append((10, 10))
        else:
            if causal_cl == 0:
                coefs.append((2, 8))
            else:
                coefs.append((8, 2))
    covars = 2*np.array([float(np.random.beta(alpha, beta, size=1)) for (alpha, beta) in coefs]) - 1
    return(covars)

def simulate_covars_multiclass(causal_preds, balance=1):
    coefs = []
    balance_id = np.random.binomial(1, balance, size=len(causal_preds))
    
    for causal_cl, bal_id in zip(causal_preds, balance_id):
        if bal_id == 1:
            coefs.append((10, 10))
        else:
            if causal_cl == 0:
                coefs.append((2, 8))
            else:
                coefs.append((8, 2))
    covars = 2*np.array([float(np.random.beta(alpha, beta, size=1)) for (alpha, beta) in coefs]) - 1
    return(covars)
    
def sigmoidal_sim(n, p, balance=1, causal_effect_size=1, covar_effect_size = None, pi=0.5, err_scale = 1):
    Ts = np.random.binomial(1, pi, size=n)
    Xs = simulate_covars(Ts, balance=balance)
    y_base = sigmoid(8*Xs).reshape(n, 1)
    Bs = np.sqrt(np.linspace(1/np.sqrt(p), 1/(p**2), p).reshape(1, p))
    
    if covar_effect_size is None:
        covar_effect_size = 2*causal_effect_size
        Bs_covar = Bs
    else:
        Bs_covar = np.ones((1, p))

    Ys_covar = covar_effect_size*y_base @ Bs
    Ys_causal = -causal_effect_size * Ts.reshape(n, 1) @ Bs
    err = np.random.normal(scale=err_scale, size=(n, p)).reshape(n, p)
    Ys =  Ys_covar + Ys_causal + err
    
    # true signal at a given x
    Ntrue = 200
    true_x = np.linspace(-1, 1, int(Ntrue/2))
    true_x = np.concatenate((true_x, true_x))
    true_y_base = sigmoid(8*true_x).reshape(Ntrue, 1)
    true_y_covar = covar_effect_size*true_y_base @ Bs_covar
    true_t = np.concatenate((np.zeros(int(Ntrue/2)), np.ones(int(Ntrue/2)))).astype(int)
    true_y_causal = -causal_effect_size * true_t.reshape(Ntrue, 1) @ Bs
    true_y = true_y_covar + true_y_causal
    return(Ys, Ts, Xs, true_y, true_t, true_x)

def linear_sim(n, p, balance=1, causal_effect_size=1, covar_effect_size = None, pi=0.5, err_scale = 1):
    Ts = np.random.binomial(1, pi, size=n)
    Xs = simulate_covars(Ts, balance=balance)
    y_base = Xs.reshape(-1, 1)
    Bs = np.sqrt(np.linspace(1/np.sqrt(p), 1/(p**2), p).reshape(1, p))
    
    if covar_effect_size is None:
        covar_effect_size = 2*causal_effect_size
        Bs_covar = Bs
    else:
        Bs_covar = np.ones((1, p))
        
    Ys_covar = covar_effect_size*y_base @ Bs_covar
    Ys_covar = covar_effect_size*y_base @ Bs
    Ys_causal = -causal_effect_size * Ts.reshape(n, 1) @ Bs
    err = np.random.normal(scale=err_scale, size=(n, p)).reshape(n, p)
    Ys =  Ys_covar + Ys_causal + err
    
    # true signal at a given x
    Ntrue = 200
    true_x = np.linspace(-1, 1, int(Ntrue/2))
    true_x = np.concatenate((true_x, true_x))
    true_y_base = true_x.reshape(Ntrue, 1)
    true_y_covar = covar_effect_size*true_y_base @ Bs_covar
    true_t = np.concatenate((np.zeros(int(Ntrue/2)), np.ones(int(Ntrue/2)))).astype(int)
    true_y_causal = -causal_effect_size * true_t.reshape(Ntrue, 1) @ Bs
    true_y = true_y_covar + true_y_causal
    return(Ys, Ts, Xs, true_y, true_t, true_x)

def kclass_sim(n, p, balance=1, causal_effect_size=1, covar_effect_size=None, pi=0.5, err_scale = 1, K=3):
    Ts = np.random.choice(range(0, K), size=n, p=np.concatenate(([pi], (1-pi)*1/(K-1)*np.ones((K-1)))))
    Xs = simulate_covars_multiclass(Ts, balance=balance)
    y_base = sigmoid(8*Xs).reshape(n, 1)
    Bs = np.sqrt(np.linspace(1/np.sqrt(p), 1/(p**2), p).reshape(1, p))
    
    if covar_effect_size is None:
        covar_effect_size = 2*causal_effect_size
        Bs_covar = Bs
    else:
        Bs_covar = np.ones((1, p))
        
    Ys_covar = covar_effect_size*y_base @ Bs_covar
    Ys_causal = -causal_effect_size * (Ts != 0).astype(float).reshape(n, 1) @ Bs
    err = np.random.normal(scale=err_scale, size=(n, p)).reshape(n, p)
    Ys =  Ys_covar + Ys_causal + err
    
    # true signal at a given x
    Ntrue = K*100
    true_x = np.linspace(-1, 1, int(Ntrue/K))
    true_x = np.concatenate([true_x for k in range(0, K)])
    true_y_base = sigmoid(8*true_x).reshape(Ntrue, 1)
    true_y_covar = covar_effect_size*true_y_base @ Bs_covar
    true_t = np.concatenate([np.zeros(int(Ntrue/K)) + k for k in range(0, K)]).astype(int)
    true_y_causal = -causal_effect_size * (true_t != 0).astype(float).reshape(Ntrue, 1) @ Bs
    true_y = true_y_covar + true_y_causal
    return(Ys, Ts, Xs, true_y, true_t, true_x)

def cond_manova(Y, T, X, **kwargs):
    with open('cmanova.R', 'r') as f:
        string = f.read()
    cmanova = STAP(string, "cmanova")
    stat, pval = cmanova.cmanova(Y, T, X)
    return float(pval), float(stat)


def codite(Y, T, X, nrep=1000):
    df_dict = {"Covariate" : X, "Batch" : T}
    yvars = []
    for i in range(0, Y.shape[1]):
        yvar = "Y{:d}".format(i)
        df_dict[yvar] = Y[:,i]
        yvars.append(yvar)
    df = pd.DataFrame(df_dict)
    xvars = "Covariate"; group_col="Batch"
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
        return (balance_ids, pred)
    else:
        return balance_ids

def ohe(T):
    K = len(np.unique(T))
    ohe_dat = np.zeros((len(T), K))
    for t in np.unique(T):
        ohe_dat[:,t] = (T == t).astype(int)
    return ohe_dat

def sigmoidal_sim_cate(n, p, balance=1, causal_effect_size=1, covar_effect_size = None, pi=0.5, err_scale = 1):
    rotation = causal_effect_size * np.pi  # Angle of rotation of the second group
    rot_rescale = np.cos(rotation)  # the rescaling factor for the rotation of the second group
    
    Ts = np.random.binomial(1, pi, size=n)
    Xs = simulate_covars(Ts, balance=balance)
    y_base = sigmoid(8*Xs).reshape(n, 1)
    Bs = 2/(np.power(np.arange(p) + 1, 1.5)).reshape(1, p)
    
    if covar_effect_size is None:
        covar_effect_size = 2*causal_effect_size
        Bs_covar = Bs
    else:
        Bs_covar = np.ones((1, p))
        
    Ys_covar = covar_effect_size*y_base @ Bs
    
    rot_vec = np.ones((n))
    rot_vec[Ts == 0] = rot_rescale
    R = np.eye(n)
    np.fill_diagonal(R, rot_vec)
    
    Ys_covar = R @ (Ys_covar - covar_effect_size/2*Bs)
    Ys_covar = Ys_covar + covar_effect_size/2*Bs
    err = np.random.normal(scale=err_scale, size=(n, p)).reshape(n, p)
    Ys = Ys_covar + err
    
    # true signal at a given x
    Ntrue = 200
    true_x = np.linspace(-1, 1, int(Ntrue/2))
    true_x = np.concatenate((true_x, true_x))
    true_y_base = sigmoid(8*true_x).reshape(Ntrue, 1)
    true_y_covar = covar_effect_size*true_y_base @ Bs
    true_t = np.concatenate((np.zeros(int(Ntrue/2)), np.ones(int(Ntrue/2)))).astype(int)
    rot_vec_true = np.ones((Ntrue))
    rot_vec_true[true_t == 0] = rot_rescale
    R_true = np.eye(Ntrue)
    np.fill_diagonal(R_true, rot_vec_true)
    true_y = R_true @ (true_y_covar - covar_effect_size/2*Bs)
    true_y = true_y + covar_effect_size/2*Bs
    return(Ys, Ts, Xs, true_y, true_t, true_x)

def diff_fn_cate(n, p, balance=1, causal_effect_size=1, covar_effect_size=None, pi=0.5, err_scale=0.5):
    Ts = np.random.binomial(1, pi, size=n)
    Xs = simulate_covars(Ts, balance=balance)
    y_base = np.zeros((n,1))
    y_base[Ts == 0,:] = sigmoid((10*causal_effect_size + 2)*Xs[Ts == 0]).reshape(-1, 1)
    y_base[Ts == 1,:] = sigmoid(2*Xs[Ts == 1]).reshape(-1, 1)
    Bs = 2/(np.power(np.arange(p) + 1, 1.1)).reshape(1, p)
    
    if covar_effect_size is None:
        covar_effect_size = 2*causal_effect_size
        Bs_covar = Bs
    else:
        Bs_covar = np.ones((1, p))
    
    Ys_covar = covar_effect_size*y_base
    err = np.random.normal(scale=err_scale, size=(n, p)).reshape(n, p)
    Ys = Ys_covar @ Bs + err
    
    # true signal at a given x
    Ntrue = 200
    true_x = np.linspace(-1, 1, int(Ntrue/2))
    true_x = np.concatenate((true_x, true_x))
    true_t = np.concatenate((np.zeros(int(Ntrue/2)), np.ones(int(Ntrue/2)))).astype(int)
    true_y_base = np.zeros((Ntrue,1))
    true_y_base[true_t == 0,:] = sigmoid((10*causal_effect_size + 3)*true_x[true_t == 0]).reshape(-1, 1)
    true_y_base[true_t == 1,:] = sigmoid(3*true_x[true_t == 1]).reshape(-1, 1)
    true_y = covar_effect_size*true_y_base @ Bs
    return(Ys, Ts, Xs, true_y, true_t, true_x)
    
def heteroskedastic_cate(n, p, balance=1, causal_effect_size=1, covar_effect_size=None, pi=0.5, err_scale=0.5):
    Ys, Ts, Xs, true_y, true_t, true_x = sigmoidal_sim_cate(n, p, balance=balance, causal_effect_size=0, covar_effect_size=covar_effect_size, pi=pi, err_scale=err_scale)
    Ys[Ts == 0,:] = Ys[Ts == 0,:] + np.random.normal(scale=np.sqrt(2*causal_effect_size), size=(np.sum(Ts == 0), p)).reshape(-1, p)
    return(Ys, Ts, Xs, true_y, true_t, true_x)


def nonmonotonic_sim_cate(n, p, balance=1, causal_effect_size=1, covar_effect_size = None, pi=0.5, err_scale = 1):
    Ts = np.random.binomial(1, pi, size=n)
    Xs = simulate_covars(Ts, balance=balance)
    y_base = Xs.reshape(-1, 1)
    Bs = 2/(np.power(np.arange(p) + 1, 1.5)).reshape(1, p)
    
    if covar_effect_size is None:
        covar_effect_size = 2*causal_effect_size
        Bs_covar = Bs
    else:
        Bs_covar = np.ones((1, p))
    
    Ys_covar = np.zeros((n, p))
    Ys_covar[(Xs >= -.3) & (Xs <= .3),:]  = causal_effect_size*Bs
    Ys_covar[Ts == 0,:] = -Ys_covar[Ts == 0,:]
    err = np.random.normal(scale=err_scale, size=(n, p)).reshape(n, p)
    Ys =  Ys_covar + err
    
    # true signal at a given x
    Ntrue = 200
    true_x = np.linspace(-1, 1, int(Ntrue/2))
    true_x = np.concatenate((true_x, true_x))
    true_y_covar = np.zeros((Ntrue, p))
    true_y_covar[(true_x >= -.3) & (true_x <= .3),:]  = causal_effect_size*Bs
    true_t = np.concatenate((np.zeros(int(Ntrue/2)), np.ones(int(Ntrue/2)))).astype(int)
    true_y_covar[true_t == 0,:] = -true_y_covar[true_t == 0,:]
    
    true_y = true_y_covar
    return(Ys, Ts, Xs, true_y, true_t, true_x)

def kclass_rotation_cate(n, p, balance=1, causal_effect_size=1, covar_effect_size=None, pi=0.5, err_scale = 1, K=3):
    rotation = causal_effect_size * np.pi  # Angle of rotation of the second group
    rot_rescale = np.cos(rotation)  # the rescaling factor for the rotation of the second group
    
    Ts = np.random.choice(range(0, K), size=n, p=np.concatenate(([pi], (1-pi)*1/(K-1)*np.ones((K-1)))))
    Xs = simulate_covars_multiclass(Ts, balance=balance)
    y_base = sigmoid(8*Xs).reshape(n, 1)
    Bs = 2/(np.power(np.arange(p) + 1, 1.1)).reshape(1, p)
    
    if covar_effect_size is None:
        covar_effect_size = 2*causal_effect_size
        Bs_covar = Bs
    else:
        Bs_covar = np.ones((1, p))
        
    Ys_covar = covar_effect_size*y_base @ Bs
        
    rot_vec = np.ones((n))
    rot_vec[Ts == 0] = rot_rescale
    R = np.eye(n)
    np.fill_diagonal(R, rot_vec)
    
    Ys_covar = R @ (Ys_covar - covar_effect_size/2*Bs)
    Ys_covar = Ys_covar + covar_effect_size/2*Bs
    err = np.random.normal(scale=err_scale, size=(n, p)).reshape(n, p)
    Ys = Ys_covar + err    
    
    # true signal at a given x
    Ntrue = K*100
    true_x = np.linspace(-1, 1, int(Ntrue/K))
    true_x = np.concatenate([true_x for k in range(0, K)])
    true_y_base = sigmoid(8*true_x).reshape(Ntrue, 1)
    true_y_covar = covar_effect_size*true_y_base @ Bs
    true_t = np.concatenate([np.zeros(int(Ntrue/K)) + k for k in range(0, K)]).astype(int)
    rot_vec_true = np.ones((Ntrue))
    rot_vec_true[true_t == 0] = rot_rescale
    R_true = np.eye(Ntrue)
    np.fill_diagonal(R_true, rot_vec_true)
    true_y = R_true @ (true_y_covar - covar_effect_size/2*Bs)
    true_y = true_y + covar_effect_size/2*Bs
    return(Ys, Ts, Xs, true_y, true_t, true_x)