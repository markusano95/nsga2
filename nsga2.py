from itertools import groupby
from itertools import combinations
import pandas as pd
import numpy as np
import pickle
# from docplex.mp.model import Model
# from docplex.util.environment import get_environment
import gurobipy

# index set
# source nodes g
# change the path name with ur need
with open('/home/markus/IFL/NSGA-Model/data-excel/sn', 'rb') as f:
    sn = pickle.load(f)

# treatment facilities t
with open('/home/markus/IFL/NSGA-Model/data-excel/tf', 'rb') as f:
    tf = pickle.load(f)

# existing treatment facilities t‘  when etf needed-----
with open('/home/markus/IFL/NSGA-Model/data/etf', 'rb') as f:
    etf = pickle.load(f)

# recycling facilities r
with open('/home/markus/IFL/NSGA-Model/data-excel/rf', 'rb') as f:
    rf = pickle.load(f)

# existing recycling facilities  r'  when erf needed----
with open('/home/markus/IFL/NSGA-Model/data/erf', 'rb') as f:
    erf = pickle.load(f)

# disposal facilities f
with open('/home/markus/IFL/NSGA-Model/data-excel/df', 'rb') as f:
    df = pickle.load(f)

# existing disposal facilities f'  when edf needed-----
with open('/home/markus/IFL/NSGA-Model/data/edf', 'rb') as f:
    edf = pickle.load(f)

# treatment technologies q
with open('/home/markus/IFL/NSGA-Model/data-excel/tt', 'rb') as f:
    tt = pickle.load(f)

# hazardous waste types  w
with open('/home/markus/IFL/NSGA-Model/data-excel/hwt', 'rb') as f:
    hwt = pickle.load(f)

# collection vehicles   k
with open('/home/markus/IFL/NSGA-Model/data-excel/cv', 'rb') as f:
    cv = pickle.load(f)

# time periods   h
with open('/home/markus/IFL/NSGA-Model/data-excel/tp', 'rb') as f:
    tp = pickle.load(f)

node_data = sn['sn_id'].copy()

# create model
model = gurobipy.Model()             # set the gurobipy model


# Parameters
# source node
G = sn[['cd_id', 'sn_id', 'unit_cost',
        'population_ij', 'population_i', 'w1_amount', 'storage_cost']].\
    set_index('sn_id').to_dict()
G_id = list(sn['sn_id'].unique())  # get source node list

C = {}     # define C is list set
POP = {}   # define POP is list set
PA = {}    # define PA is list set
D = {}     # define D is list set
HC = {}    # define HC is list set

for m in range(len(G_id)):
    i = sn['sn_id'][m]    # get the variable index i
    j = sn['sn_id'][m]    # like i index, get the variable index j
    g = sn['sn_id'][m]    # get the variable index g
    w1 = sn['sn_id'][m]   # get the variable index w1
    # print(type(g))

    C[i, j] = sn['unit_cost'][m]        # unit-transportation cost of waste
    POP[i, j] = sn['population_ij'][m]  # number of people are affected between i and j
    PA[i] = sn['population_i'][m]       # number of people are affected at i
    D[g, w1] = sn['w1_amount'][m]       # amount of waste w accumulated at g
    HC[i] = sn['storage_cost'][m]       # storage cost of one unit of waste at i


# treatment facilities t
TF = tf[['etf_id', 'tf_id', 'establish _cost_tf', 'capacity_t', 'fixed_cost_q_t', 'variable_cost_q_t', 'tq_id']]. \
    set_index('tf_id').to_dict()
T_id = list(tf['tf_id'].unique())       # get treatment facilities list
T1_id = list(tf['etf_id'].unique())
Q1_id = list(tf['tq_id'].unique())

TC = {}
FT = {}
FCT = {}
VCT = {}
for i in range(len(T_id)):
    t = tf['tf_id'][i]
    q = tf['tq_id'][i]

    TC[t] = tf['capacity_t'][i]               # capacity of treatment facility t
    FT[q, i] = tf['establish _cost_tf'][i]    # establishment cost of a treatment facility with q at i
    FCT[q, t] = tf['fixed_cost_q_t'][i]       # fixed setup cost of q in t
    VCT[q, t] = tf['variable_cost_q_t'][i]    # variable cost of one unit of waste with q in t


# treatment technologies
Q = tt[['tq_id', 'percentage_residues_w_q', 'percentage_mass_reduction_w_q', 'number_q', 'capacity_w']]. \
    set_index('tq_id').to_dict()
Q_id = list(tt['tq_id'].unique())

Beta = {}
R = {}
NQ = {}
for i in range(len(Q_id)):
    w = tt['tq_id'][i]
    q = tt['tq_id'][i]

    Beta[w, q] = tt['percentage_residues_w_q'][i]      # percentage of recyclable waste residues produced w with q
    R[w, q] = tt['percentage_mass_reduction_w_q'][i]   # percentage of mass reduction waste type w with q
    NQ = tt['number_q'][i]                             # number of treatment technologies q


# recycling facilities
RF = rf[['erf_id', 'rf_id', 'establish _cost_rf', 'percentage_recycled_r', 'capacity_r',
         'variable_cost_r']].set_index('rf_id').to_dict()
R_id = list(rf['rf_id'].unique())
R1_id = list(rf['erf_id'].unique())

FR = {}
W = {}
RC = {}
VCR = {}
for i in range(len(R_id)):
    r = rf['rf_id'][i]                        # get r index
    FR[i] = rf['establish _cost_rf'][i]       # establishment cost of a recycling facility at i
    W[r] = rf['percentage_recycled_r'][i]     # percentage of total waste recycled in r
    RC[r] = rf['capacity_r'][i]               # capacity of recycling facility r
    VCR[r] = rf['variable_cost_r'][i]         # variable cost of unit of waste in r


# disposal facilities
DF = df[['edf_id', 'df_id', 'establish _cost_df', 'capacity_f', 'variable_cost_f']].set_index('df_id').to_dict()
F_id = list(df['df_id'].unique())
F1_id = list(df['edf_id'].unique())

FF = {}
FC = {}
VCF = {}
for i in range(len(F_id)):
    f = df['df_id'][i]

    FF[i] = df['establish _cost_df'][i]   # establishment cost of a disposal facility at i
    FC[f] = df['capacity_f'][i]           # capacity of disposal facility f
    VCF[f] = df['variable_cost_f'][i]     # variable cost of one unit of waste in f


# hazardous waste types  w
W = hwt[['capacity_w', 'hwt_id']].set_index('hwt_id').to_dict()
W_id = list(hwt['hwt_id'].unique())

Delta = {}
for i in range(len(W_id)):
    w = i
    Delta[w] = hwt['capacity_w'][i]     # capacity of a vehicle that is compatible with type w


# collection vehicles   k
K = cv[['cv_id']].set_index('cv_id').to_dict()
K_id = list(cv['cv_id'].unique())

# time periods   h
H = tp[['tp_id']].set_index('tp_id').to_dict()
H_id = list(tp['tp_id'].values)

# collection
GT_id = G_id + T_id              # collection of G and T
GRT_id = G_id + R_id + T_id      # collection of G and T and R
RT_id = R_id + T_id              # collection of R and T
T2_id = T_id - T1_id             # means removing T1 from the T set
R2_id = R_id - R1_id             # means removing R1 from the R set
F2_id = F_id - F1_id             # means removing F1 from the F set
model.update()                   # model update

# decision variable
# binary parameter (1 if w is compatible with q ; 0 otherwise)
COM = model.addVars(W_id, Q_id, vtype=gurobipy.GRB.BINARY, name="COM[w,q]")

# binary parameter (1 if waste w is compatible with k ; 0 otherwise)
V = model.addVars(W_id, K_id, vtype=gurobipy.GRB.BINARY, name="V[w,k]")

# binary parameter (1 if existing treatment t1 is equipped with q ; 0 otherwise)
A = model.addVars(Q_id, T1_id, vtype=gurobipy.GRB.BINARY, name="A[q,t1]")

# binary variable (1 if j is visited just after i by k in h ; 0 otherwise)
X = model.addVars(G_id, G_id, K_id, H_id, vtype=gurobipy.GRB.BINARY, name="X[i,j,k,h]")

# binary variable (1 if q is established at node i; 0 otherwise)
T = model.addVars(Q_id, T2_id, vtype=gurobipy.GRB.BINARY, name="T[q,i]")

# binary variable (1 if a recycling facility is established at r ; 0 otherwise)
R = model.addVars(R_id, vtype=gurobipy.GRB.BINARY, name="R[r]")

# binary variable (1 if a disposal facility is established at f ; 0 otherwise)
F = model.addVars(F_id, vtype=gurobipy.GRB.BINARY, name="F[F]")

# binary variable (1 if accumulated wastes at t are treated with q in h ; 0 otherwise)
PT = model.addVars(Q_id, T_id, H_id, vtype=gurobipy.GRB.BINARY, name="PT[q,t,h]")

# load of vehicle k after visiting node i in period h
L = model.addVars(G_id, K_id, H_id, vtype=gurobipy.GRB.BINARY, name="L[i,k,h]")

# amount of waste type w transferred to treatment facility t in period h
XT = model.addVars(W_id, T_id, H_id, vtype=gurobipy.GRB.BINARY, name="XT[w,t,h]")

# amount of recyclable waste transferred to recycling facility r in period h
XR = model.addVars(R_id, H_id, vtype=gurobipy.GRB.BINARY, name="XR[r,h]")

# amount of waste transferred to disposal facility f in period h
XF = model.addVars(F_id, H_id, vtype=gurobipy.GRB.BINARY, name="XF[f,h]")

# amount of recyclable waste residue transported from t to r in period h
XTR = model.addVars(T_id, R_id, H_id, vtype=gurobipy.GRB.BINARY, name="XTR[t,r,h]")

# amount of waste residue transported from t to f in period h
XTF = model.addVars(T_id, F_id, H_id, vtype=gurobipy.GRB.BINARY, name="XTF[t,f,h]")

# amount of waste residue transported from r to f in period h
XRF = model.addVars(R_id, F_id, H_id, vtype=gurobipy.GRB.BINARY, name="XRF[r,f,h]")

# amount of working stock of w in t in the end of period h
ST = model.addVars(W_id, T_id, H_id, vtype=gurobipy.GRB.BINARY, name="ST[w,t,h]")

# Constraints
# Equation 5
model.addConstrs(gurobipy.quicksum(X[i, j, k, h] for i in G_id for j in G_id if i != j)
                 == 1 for k in K_id for h in H_id)

# Equation 6
model.addConstrs(gurobipy.quicksum(X[i, j, k, h] for i in G_id - X[i + 1, j, k, h] for i + 1 in GRT_id)
                 == 0 for k in K_id for j in G_id for h in H_id)

# Equation 7
model.addConstrs(gurobipy.quicksum(X[i, j, k, h] * V[w, k] for h in H_id for k in K_id for j in GRT_id)
                 == 1 for w in W_id for i in G_id)

# Equation 8
model.addConstrs(gurobipy.quicksum(X[i, j, k, h] for i in G_id - X[i + 1, j, k, h] for i in G_id)
                 == 0 for k in K_id for j in RT_id for h in H_id)

# Equation 9
model.addConstrs(gurobipy.quicksum(V[w, k] * COM[w, q] * T[q, j] for w in W_id for q in Q_id)
                 >= X[i, j, k, h] for k in K_id for h in H_id for i in G_id for j in T_id)

# Equation 10
model.addConstrs((gurobipy.quicksum(V[k, w] * ((abs(NQ) - gurobipy.quicksum(COM[w, q] for q in Q_id)) * R[j]) / abs(NQ))
                  for w in W_id)
                 >= X[i, j, k, h] for k in K_id for h in H_id for i in G_id for j in R_id)

# Equation 11
model.addConstrs((l[i, k, h] - L[j, k, h] + gurobipy.quicksum(V[k, w] * Delta_w * X[i, j, k, h] for w in W_id))
                 <= gurobipy.quicksum(V[k, w] * (Delta[w] - D[j, w]) for w in W_id)
                 for k in K_id for h in H_id for i in G_id for j in G_id)

# Equation 12
model.addConstrs(gurobipy.quicksum(D[i, w] * V[k, w] for w in W_id)
                 <= L[i, k, h] <= gurobipy.quicksum(V[k, w] * Delta[w] for w in W_id)
                 for i in G_id for k in K_id for h in H_id)

# Equation 13
model.addConstrs(X[i, j, k, h] * (gurobipy.quicksum(D[j, w] * V[k, w] for w in W_id)) <= L[j, k, h]
                 for i in G_id for j in G_id for k in K_id for h in H_id)

# Equation 14
model.addConstrs(L[i, j, k, h] <= gurobipy.quicksum(V[w, k] *
                                                    (Delta[w] + (gurobipy.quicksum((D[j, w] - Delta[w]) * X[i, j, k, h])
                                                                 for i in G_id)))
                 for j in G_id for k in K_id for h in H_id)

# Equation 15
model.addConstrs(XT[w, j, h] == gurobipy.quicksum(X[i, j, k, h] * L[i, k, h] * V[k, w] for k in K_id for i in G_id)
                 for j in T_id for w in W_id for h in H_id)

# Equation 16
model.addConstrs(gurobipy.quicksum((PT[q, i, h] * (ST[w, i, h - 1] + XT[w, i, h]) * (1 - R[w, q]) * Beta[w, q])
                                   for w in W_id for q in Q_id)
                 == gurobipy.quicksum(XTR[i, j, h] for j in R_id) for i in T_id for h in H_id)

# Equation 17
model.addConstrs(gurobipy.quicksum(X[i, j, k, h] * L[i, k, h] * V[k, w] for w in W_id for k in K_id for i in G_id)
                 + gurobipy.quicksum(XTR[i + 1, j, h] for i + 1 in K_id)
                 == XR[j, h] for j in R_id for h in H_id)

# Equation 18
model.addConstrs(gurobipy.quicksum((PT[q, i, h] * (ST[w, i, h - 1] + XT[w, i, h]) * (1 - Beta[w, q]))
                                   for w in W_id for q in Q_id)
                 == gurobipy.quicksum(XTF[i, j, h] for j in F_id) for i in T_id for h in H_id)

# Equation 19
model.addConstrs((gurobipy.quicksum(XRF[i, j, h] for j in F_id)
                  == XR[i, h] * (1 - W[i])) for i in R_id for h in H_id)

# Equation 20
model.addConstrs((gurobipy.quicksum(XTF[i, j, h] for j in T_id) + gurobipy.quicksum(XRF[i, j, h] for j in R_id)
                  == XF[i, h]) for i in F_id for h in H_id)

# Equation 21
model.addConstrs(((1 - gurobipy.quicksum(PT[q, i, h] for q in Q_id)) * (ST[w, i, h - 1] + XT[w, i, h]))
                 == ST[w, i, h] for i in T_id for w in W_id for h in H_id)

# Equation 22
model.addConstrs((T[q, i] >= PT[q, i, h]) for i in T_id for q in Q_id for h in H_id)

# Equation 23
model.addConstrs((A[q, i] == T[q, i]) for i in T1_id for q in Q_id)

# Equation 24
model.addConstrs(R[i] == 1 for i in R1_id)

# Equation 25
model.addConstrs(F[i] == 1 for i in F1_id)

# Equation 26
model.addConstrs(gurobipy.quicksum(T[q, i] for q in Q_id) <= 1 for i in T_id)

# Equation 27
model.addConstrs(ST[w, i, 0] + ST[w, i, max(H)] == 0 for i in T_id for w in W_id)

# Equation 28
model.addConstrs(gurobipy.quicksum(ST[w, i, h - 1] + XT[w, i, h] for w in W_id) <=
                 TC[i] * gurobipy.quicksum(T[q, j] for q in Q_id) for i in T_id for h in H_id)

# Equation 29
model.addConstrs(XR[i, h] <= RC[i] * R[i] for i in R_id for h in H_id)

# Equation 30
model.addConstrs(XF[i, h] <= FC[i] * F[i] for i in F_id for h in H_id)

# Equation 31
model.addConstrs(XTF[i, j, h] >= 0 & XF[j, h] >= 0 for i in T_id for j in F_id for h in H_id)

model.addConstrs(XTR[i, j, h] >= 0 & XT[w, i, h] >= 0 & ST[w, i, h] >= 0
                 for i in T_id for j in R_id for w in W_id for h in H_id)

# Equation 32
model.addConstrs(L[i, k, h] >= 0 for i in G_id for h in H_id for k in K_id)
model.addConstrs(X[i, j, k, h] == or_(0, 1) for i in DGRT_id for j in GRT_id for k in K_id for h in H_id)

model.addConstrs(R[i] == or_(0, 1) for i in R_id)
model.addConstrs(F[i] == or_(0, 1) for i in F_id)
model.addConstrs(T[q, i] == or_(0, 1) & PT[q, i, h] == or_(0, 1) for i in T_id for q in Q_id for h in H_id)

model.update()

# Object Function1
# aim at minimizing the total cost  parameter k = 1.5
model.setObjective1(
    gurobipy.quicksum(C[i, j] * X[i, j, k, h] * L[i, k, h]
                      for h in H_id for k in K_id for i in G_id for j in GRT_id) +

    gurobipy.quicksum(C[i, j] * XTF[i, j, h] for h in H_id for i in T_id for j in F_id) +

    gurobipy.quicksum(C[i, j] * XRF[i, j, h] for h in H_id for i in T_id for j in F_id) +

    gurobipy.quicksum(FT[q, i] * T[q, i] for q in Q_id for i in T2_id) +

    gurobipy.quicksum(FR[i] * R[i] for i in R2_id) +

    gurobipy.quicksum(FF[i] * F[i] for i in F2_id) +

    gurobipy.quicksum(PT[q, i, h] * (FCT[q, i] + VCT[q, i] * (ST[w, i, h - 1] + XT[w, i, h]))
                      for h in H_id for w in W_id for q in Q_id for i in T_id) +

    gurobipy.quicksum(VCR[i] * XR[i, h] for h in H_id for i in R_id) +

    gurobipy.quicksum(VCF[i] * XF[i, h] * k for h in H_id for i in F_id) +

    gurobipy.quicksum(HC[i] for h in H_id for i in G_id) * (gurobipy.quicksum(D[i, w] for w in W_id) -
                                                            gurobipy.quicksum(
                                                                X[j, i, k, h + 1] * D[i, w] * V[w, k] for h + 1 in
                                                                [1, h]
                                                                for k in K_id for w in W_id for j in G_id)) +

    gurobipy.quicksum(HC[i] * (1 - PT[q, i, h]) * (ST[w, i, h - 1] + XT[w, i, h])

                      for h in H_id for w in W_id for q in Q_id for i in T_id),

    gurobipy.GRB.MINIMIZE)

# Object Function2
# computes the total site risk and set minimal
model.setObjective2(
    gurobipy.quicksum(PA[i] * (ST[w, i, h - 1] + XT[w, i, h]) for h in H_id for w in W_id for i in T_id) +

    gurobipy.quicksum(PA[j] * XF[i, h] for h in H_id for i in F_id),

    gurobipy.GRB.MINIMIZE)

# Object Function3
# minimize the transportation risk for the people present on and along all possible routes
model.setObjective3(
    gurobipy.quicksum(POP[i, j] * X[i, j, k, h] * L[i, k, h] for h in H_id for k in K_id for i in G_id
                      for j in GT_id) +

    gurobipy.quicksum(POP[i, j] * XTF[i, j, h] for h in H_id for i in T_id for j in F_id),

    gurobipy.GRB.MINIMIZE)

# optimizing model
logfile = open('/home/markus/IFL/NSGA-Model/data-excel/nsga.log', 'w')   # create a log file
model._logfile = logfile
model.Params.TimeLimit = 300  # set a time limit if needed

model.optimize()
logfile.close()
