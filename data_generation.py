#generating the training data
import pandas as pd
#for data handling
import numpy as np
import matplotlib.pyplot as plt
import warnings

def sigmoid(z):
    return (1/(1+np.exp(-z)))

np.random.seed(0)

def generate_random_data():
    r=0.05 #covariance measure, >0 means variables related together
    n=1000
    sigma=1 #how spread out the data is

    beta_x, beta_z, beta_v = -1,2,9
    var_x, var_z, var_v = 1,1,4
    x,z=np.random.multivariate_normal([0,0],[[var_x,r],[r,var_z]],n).T
    v=np.random.normal(0,var_v,n)**3
    #print('x: ',x)
    A=pd.DataFrame({'x':x,'z':z,'v':v})
    Input_data=pd.DataFrame({'x':x,'z':z,'v':v})
    input_data=np.array(Input_data)
    
    A['input']=sigmoid(A[['x','z','v']].dot([beta_x,beta_z,beta_v])+sigma*np.random.normal(0,1,n))
    A['Y']=[np.random.binomial(1,p) for p in A.input]
    O=list(A['Y'])
    output=np.array(O)

    return input_data,output



