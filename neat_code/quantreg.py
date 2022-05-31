import numpy as np
from mip import Model, xsum, minimize, maximize, OptimizationStatus
import time
from numpy.linalg import inv
from scipy.stats import norm, chi2



# generator specified for our paper model
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


def LP_Tscore(X1, X2, Yi, phi = lambda x : x-1/2, Aphi=1/12, t_level = 100, echo = True):
    start_time = time.time()

    SAMPLE_AMOUNT = len(X1)
    p = len(X1[0])
    q = len(X2[0])
    d = len(Y[0])
    if not d==2:
        print("Only for 2d Y")
        return

    Q = 1/SAMPLE_AMOUNT * np.matmul(X2.T,X2)

    dual_data = np.zeros((t_level,SAMPLE_AMOUNT))
    
    for t in range(1,t_level+1):
        quantile = t / t_level
        
        a_s = np.zeros(SAMPLE_AMOUNT)
                
        m = Model()
        m.verbose = 0
    
        a = [m.add_var(lb=0,ub=1,name="a") for i in range(SAMPLE_AMOUNT)]


        for idx in range(p):
            m += xsum(a[i]*X1[i,idx] for i in range(SAMPLE_AMOUNT)) == (1-quantile)*xsum(X1[i,idx] for i in range(SAMPLE_AMOUNT))
          
        m.objective = maximize(xsum(Yi[i] * a[i] for i in range(SAMPLE_AMOUNT)))
    
        status = m.optimize(max_seconds=300)

        if not status == OptimizationStatus.OPTIMAL:
            print("SOMETHING WRONG!")
            quit()

        else:
            count = 0
            for v in m.vars: 
                dual_data[t-1][count] = v.x
                count += 1
    
    if echo:
        print("Time spent for LP calculation : ",time.time()-start_time)
    # calculate target statistics
    b=0
    for t in range(1,t_level):
        b +=  phi(t/t_level) * (dual_data[t] - dual_data[t-1])
        
        
        
    S = 1/np.sqrt(SAMPLE_AMOUNT) * np.matmul(X2.T,b)
    
    T_STAT = np.matmul(np.matmul(S.T,inv(Q)),S)/Aphi
    
    return T_STAT

def chi2_p(val, freedom = 2):
    return 1-chi2.cdf(val, df = freedom)


if __name__ == "__main__":
    EXP_TIME = 100
    SAMPLE_AMOUNT = 100
    T_LEVELS = 100

    p_val1_collection = np.zeros(SAMPLE_AMOUNT)
    p_val2_collection = np.zeros(SAMPLE_AMOUNT)

    for exp in range(EXP_TIME):
        print("experiment : ",exp)
        # X1, X2, Y = data_generator(SAMPLE_AMOUNT, seedCode = 500+exp, SIGMA = 1)
        X1, X2, Y = data_generator_X2(SAMPLE_AMOUNT, seedCode = 500+exp, SIGMA = 1)
        T_val1 = LP_Tscore(X1, X2, Y[:,0], t_level=T_LEVELS)

        p_val1 = chi2_p(T_val1)
        print("p val 1 : ",p_val1)

        T_val2 = LP_Tscore(X1, X2, Y[:,1], t_level=T_LEVELS)

        p_val2 = chi2_p(T_val2)
        print("p val 2 : ",p_val2)

        p_val1_collection[exp] = p_val1 
        p_val2_collection[exp] = p_val2


    np.save("P_collection_withB2_reg1",p_val1_collection)
    np.save("P_collection_withB2_reg2",p_val2_collection)