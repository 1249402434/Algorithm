import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

def real_func(x):
    return np.sin(2*np.pi*x)

def fit_func(p,x):
    f=np.poly1d(p)#创建一个多项式函数，按降幂排列
    return f(x)

def residual_func(p,x,y):
    ret=fit_func(p,x)-y
    return ret

x=np.linspace(0,1,10)
y_=real_func(x)
y=[np.random.normal(0,0.1)+i for i in y_]
x_points=np.linspace(0,1,1000)

def fitting(M=0):
    params=np.random.rand(M+1)
    #在目标函数上只拟合10个加入了噪声的点
    fitted_params=leastsq(residual_func,params,args=(x,y))
    print('fitted_params:',fitted_params[0])

    plt.plot(x_points,real_func(x_points),'r',label='real')
    plt.plot(x_points,fit_func(fitted_params[0],x_points),'b',label='fitted')
    plt.plot(x,y,'bo',label='train_points')
    plt.legend()
    plt.show()
    return fitted_params

result=fitting(1)

result=fitting(3)

result=fitting(8)#出现过拟合，几乎所有的点都在预测曲线上，但是没有很好的拟合真实曲线

#正则化
regularization=0.0001

def residual_func_regularization(p,x,y):
    ret=fit_func(p,x)-y
    ret=np.append(ret,np.sqrt(0.5*regularization*np.square(p)))#惩罚所有系数
    return ret

params_re=np.random.rand(9+1)
params_re_fitted=leastsq(residual_func_regularization,params_re,args=(x,y))

plt.plot(x_points,real_func(x_points),'r',label='real')
plt.plot(x_points,fit_func(result[0],x_points),'b',label='fit')
plt.plot(x_points,fit_func(params_re_fitted[0],x_points),'g',label='regularization')
plt.plot(x,y,'bo',label='noise')
plt.legend()
plt.show()




