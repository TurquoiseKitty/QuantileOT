import numpy as np
from mip import Model, xsum, minimize, maximize, OptimizationStatus
import time
from numpy.linalg import inv
from scipy.stats import norm, chi2



def data_generator_X2(SAMPLE_AMOUNT, SIGMA = 1, seedCode = 100):
    B1 = np.array([
        [2,1],
        [3,2],
        [1,0],
        [0,1]
    ])
    
    B2 = np.array([
        [0.04,0],
        [0.04,0.08]
    ])

    H = np.array([
        [1/2, 1/2],
        [0, 1/2]
    ])

    X1 = np.zeros((SAMPLE_AMOUNT, 4))
    X2 = np.zeros((SAMPLE_AMOUNT, 2))
    Y = np.zeros((SAMPLE_AMOUNT, 2))



    # the scheme follows our introduced formulation
    np.random.seed(seedCode)
    X1[:,0] = np.array([1.0]*SAMPLE_AMOUNT)
    X1[:,1] = np.random.randint(2,size=SAMPLE_AMOUNT).astype(float)
    X1[:,2] = np.random.uniform(0,3,SAMPLE_AMOUNT)
    X1[:,3] = np.random.normal(0, 1, SAMPLE_AMOUNT)

    X21_latent = np.random.normal(0, 1, SAMPLE_AMOUNT)
    X22 = np.random.normal(0, 1, SAMPLE_AMOUNT)
    X21 = 1/np.sqrt(2) * X22 + 1/np.sqrt(2) * X21_latent

    X2[:,0] = X21
    X2[:,1] = X22

    Y_fix = np.matmul(X1,B1) + np.matmul(X2,B2)

    
    EPSI1_latent_t = np.random.uniform(0,1,SAMPLE_AMOUNT)
    EPSI2_latent_t = np.random.uniform(0,1,SAMPLE_AMOUNT)

    EPSI1_transed = H[0,0] * EPSI1_latent_t + H[0,1] * EPSI2_latent_t
    EPSI2_transed = H[1,0] * EPSI1_latent_t + H[1,1] * EPSI2_latent_t

    
    # scale effect
    for i in range(SAMPLE_AMOUNT):
        
        EPSI1 = norm.ppf(EPSI1_transed[i], loc=0, scale=SIGMA * np.abs(X2[i,0]))
        EPSI2 = norm.ppf(EPSI2_transed[i], loc=0, scale=SIGMA * np.abs(X2[i,1]))

        Y_error1 = H[0,0] * EPSI1 + H[1,0] * EPSI2
        Y_error2 = H[0,1] * EPSI1 + H[1,1] * EPSI2

    
        Y[i,0] = Y_fix[i,0] + Y_error1
        Y[i,1] = Y_fix[i,1] + Y_error2

    return (X1, X2, Y)


# X1 : n * p
# X2 : n * q
# Y : n * d
# phi : lambda x : phi(x)
# Aphi
# t_level : quantile levels for each dimension
def OT_Tscore_2d(X1, X2, Y, phi = lambda x : 1/2*(x[0]+x[1])-1/2, Aphi=1/24, t_level = 50, echo = True):
    start_time = time.time()

    SAMPLE_AMOUNT = len(X1)
    p = len(X1[0])
    q = len(X2[0])
    d = len(Y[0])
    if not d==2:
        print("Only for 2d Y")
        return

    Q = 1/SAMPLE_AMOUNT * np.matmul(X2.T,X2)

    dual_density = np.zeros((t_level, t_level, SAMPLE_AMOUNT))

    m = Model()
    pi = [[[m.add_var(lb=0,name="pi") for i in range(SAMPLE_AMOUNT)] for t2 in range(t_level)] for t2 in range(t_level)]    
   
    for t1 in range(t_level):
        for t2 in range(t_level):
            for idx in range(p):
                m += xsum(pi[t1][t2][i]*X1[i,idx] for i in range(SAMPLE_AMOUNT)) == xsum(X1[i,idx] for i in range(SAMPLE_AMOUNT))/t_level**2
        
    for i in range(SAMPLE_AMOUNT):
        m += xsum(xsum(pi[t1][t2][i] for t1 in range(t_level)) for t2 in range(t_level)) == 1
        
    m.objective = maximize(
        xsum(
            xsum(
                xsum(
                    (t1/t_level * Y[i,0] + t2/t_level * Y[i,1] * pi[t1-1][t2-1][i]) for i in range(SAMPLE_AMOUNT)
                ) for t2 in range(1,t_level+1)
            )for t1 in range(1,t_level+1)
        )
    )
    m.verbose = 0
    status = m.optimize(max_seconds=600)
    
    if echo:
        print("Time spent for LP calculation : ",time.time()-start_time)

    if not status == OptimizationStatus.OPTIMAL:
        print("Calculation failed!")
        return

    # density fit in
    t1=0
    t2=0
    i=0
    for v in m.vars: 
        dual_density[t1][t2][i] = v.x
        i+=1
        if i == SAMPLE_AMOUNT:
            i=0
            t2+=1
        if t2 == t_level:
            t2=0
            t1+=1

    # calculate T value
    b=0

    for t1 in range(1,t_level+1):
        for t2 in range(1,t_level+1):
            b += phi(np.array([t1/t_level, t2/t_level])) * dual_density[t1-1][t2-1]
        
        
    S = 1/np.sqrt(SAMPLE_AMOUNT) * np.matmul(X2.T,b)

    
    T = np.matmul(np.matmul(S.T,inv(Q)),S)/Aphi
    
    return T

def chi2_p(val, freedom = 2):
    return 1-chi2.cdf(val, df = freedom)


if __name__ == "__main__":
    EXP_TIME = 100
    SAMPLE_AMOUNT = 100
    T_LEVELS = 50

    p_val_collection = np.zeros(SAMPLE_AMOUNT)

    for exp in range(EXP_TIME):
        print("experiment : ",exp)
        # X1, X2, Y = data_generator(SAMPLE_AMOUNT, seedCode = 500+exp, SIGMA = 1)
        X1, X2, Y = data_generator_X2(SAMPLE_AMOUNT, seedCode = 500+exp, SIGMA = 1)
        T_val = OT_Tscore_2d(X1, X2, Y, t_level=T_LEVELS)

        p_val = chi2_p(T_val)
        print("p val : ",p_val)
        p_val_collection[exp] = p_val 

    # np.save("P_collection",p_val_collection)
    np.save("P_collection_withB2",p_val_collection)