"""
Created on Thu Dec 29 16:41:13 2011

@author: Christoph Loschen
"""

import numpy as np



#test function
def function(p):
    #y=pow(p[0]-2,2)+pow(p[1]-2,2)+pow(p[2],2)+pow(p[3]-5.5,2)+pow(p[4]-3,2)
    y=pow(p[0]-2,2)+pow(p[1]-22,2)+pow(p[2],2)+pow(p[3]-5.5,2)
    return y

def function3(p):
    y=pow(p[0]-2,2)+pow(p[1]-2,2)+pow(p[2],2)+pow(p[3]-5.5,2)+pow(p[4]-3,2)+pow(p[5]+6,4)+pow(p[6]+2.2,2)
    return y

def function2(p):
    y = -np.sum(np.sin(p))
    y = np.sin(p[0]) + np.cos(p[1])
    return y

# simplex minimisation in n Dimensions
# p0: start vector 
# fnc: function which takes vector as argument 

def simplex(p0=None, fnc=None, maxiter=5000,delta=0.5,ftol=1e-5,lowerbound=None,upperbound=None,verbose=True):
    if verbose: print "Starting Downhill-Simplex Optimization"
    #setup
    ndim=p0.shape[0]   
    ilow=0;ihigh=0;inhigh=0
    
    #unit vectors
    p=np.identity(ndim)*delta      
    #creating N other points, N+1 in total
    p = np.vstack((p0,p))
    y = np.zeros((ndim+1))
    
    for i,row in enumerate(p):
      y[i] = fnc(row)
  
    #start iteration
    for k in range(maxiter):
        if verbose: print "Iteration: %2d y=f(x): %8.4f " %(k,y[ilow]),   
        #determine highest and lowest point index
        for i in range(np.size(y)):
            if y.min() == y[i]:#dangerous
                ilow=i
            if y.max() == y[i]:
                ihigh=i
            #2nd highest
        for i in range(np.size(y)):    
            temp=y.copy()
            temp[ihigh]=y[ilow]
            if temp.max()==y[i]:
                inhigh=i
        #double tolerance
        if ftol>abs(y[ilow]-y[ihigh]):  
	    print "\nSimplex optimization CONVERGED after %d iterations. [ftol: %8.3e]" %(k,ftol)      
            break
        else:
            if verbose: print " - diff: %8.3e" %(abs(y[ilow]-y[ihigh])),
        
        ytry=amoebamove(p,p0,y,ihigh,-1.0, fnc,lowerbound,upperbound)
        if ytry<y[ilow]:
            if verbose: print " -> EXTRAPOLATION"
            ytry=amoebamove(p,p0,y,ihigh,2.0,  fnc,lowerbound,upperbound)
        elif ytry>y[inhigh]:
            if verbose: print " -> CONTRACTION"
            ysave=ytry
            ytry=amoebamove(p,p0,y,ihigh,0.5,  fnc,lowerbound,upperbound)
        else:
	    print ""
    return p[ilow],ytry    

def amoebamove(p,p0,y,ihigh,factor,fnc,lowerbound,upperbound):
    ndim = p0.shape[0]
    psum=p.sum(axis=0)
    
    factor1=(1.0-factor)/ndim
    factor2=factor1-factor

    ptry=np.zeros(np.size(p[0]))
    for j in range(np.size(ptry)):
        ptry[j]=psum[j]*factor1-p[ihigh][j]*factor2
        #Do not allow values outside bound -> use boundary here
        if lowerbound is not None:
	    if ptry[j]<lowerbound[j]:
		ptry[j]=lowerbound[j]
	
	if upperbound is not None:
	    if ptry[j]>upperbound[j]:
		ptry[j]=upperbound[j]
	
    ytry=fnc(ptry)
    if ytry<y[ihigh]:
        y[ihigh]=ytry
        for j in range(np.size(ptry)):
            psum[j]= psum[j]-p[ihigh][j]+ptry[j]
            p[ihigh][j]=ptry[j]
    return ytry   

   
if __name__=="__main__":    
    g = lambda x: np.power(x-2,2)+5*x
    p0=np.zeros(7)
    p_opt,y_opt=simplex(p0=p0,fnc=function3,lowerbound=p0-10,upperbound=p0+2,verbose=True)
    print p_opt
    print y_opt
    
    
