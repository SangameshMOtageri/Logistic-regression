#Logistic regression
import numpy as np
from data_generation import generate_random_data

def activation_sigmoid(z):
    return (1/(1+np.exp(-z)))

def model_trainer(i_data, i_lables,n_epochs,lr):
    n_images=i_data.shape[0]
    n_features=i_data.shape[1]
    W=np.array([np.random.rand() for i in range(n_features)])
    transpose_i_data=np.transpose(i_data)
    Y=np.transpose(i_lables)
    b=np.random.rand()
    for i_epoch in range(n_epochs):
        #W.Xt + b
        weighted_output=np.dot(W,transpose_i_data)+b
        activation_output=activation_sigmoid(weighted_output)
        #print("activation_output: ",activation_output)
        activation_output_A=np.array([(lambda: i,lambda: 5.0e-40)[i==0]() for i in activation_output])
        temp=1-activation_output
        activation_output_B=np.array([(lambda: i,lambda: 5.0e-40)[i==0]() for i in temp])
        #print("np.log(activation_output): ",temp)
        #print("np.log(1-activation_output): ",activation_output_B)
        cost_func=-Y*np.log(activation_output_A)+(1-Y)*np.log(activation_output_B)
        #Cost function value for each input element is found.
        #For weight and bias update, take the average cost function
        #print("cost_func: ",cost_func)
        average_cost=(np.sum(cost_func))/n_images
        #dl/dw=(a-y).x for individual input
        #Now implementing the same using matrices
        dw=np.dot((activation_output-Y),i_data)/n_images
        #dl/db=(a-y) for individual input
        db=np.sum(activation_output-Y)/n_images
        W=W-lr*dw
        b=b-lr*db
        print("epoch: ",i_epoch," cost: ",average_cost)

    return W,b

def predict(i_data,W,B):
    transpose_i_data=np.transpose(i_data)
    weighted_output=np.dot(W,transpose_i_data)+b
    return(activation_sigmoid(weighted_output))
    

if __name__ == '__main__':
    print("Logistic regression!")
    data,lables=generate_random_data()
    n_epochs=100
    lr=0.5
    print("data: ",data[:800,:])
    print("Lables: ",lables[:800])
    W,b=model_trainer(data[:800,:],lables[:800],n_epochs,lr)
    #W,b=model_trainer(data[:10,:],lables[:10],n_epochs,lr)
    #print(data[90,:],lables[90])
    #print(predict(data[90,:],W,b))
    #print(data[91,:],lables[91])
    #print(predict(data[91,:],W,b))

    test_data=data[800:1000,:]
    test_result=lables[800:1000]
    score=0
    for i in range(200):
        if predict(test_data[i],W,b)<0.5:
            res=0
        else:
            res=1
        if res==test_result[i]:
            score+=1
    print("Pass Percentage: ",((score/200)*100),"%")




        
    
    
    
            
