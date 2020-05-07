import scipy.optimize
import numpy as np
#def min_fun(x, a, b, c):
 #   return b*x+a*(x**2)+c

def min_fun(a, x_left, x_right, y_left, y_right):
    return (np.sum((a[1]*x_left+a[0]*(x_left**2) + a[2] - y_left)**2) + np.sum((a[1]*x_right+a[0]*(x_right**2) + a[3] - y_right)**2))

    #return (np.sum(((a[0]*x_left + a[1]) - y_left)**2) + np.sum((a[0]*x_right + a[2] - y_right)**2))
def level_that_shit(x_right, y_right, x_left, y_left, guess=None):
    
    
    slope_guess = (y_left[-1]-y_left[0])/(x_left[-1]-x_left[0])
    left_height_guess = np.mean(y_left)
    right_height_guess = np.mean(y_left)
    guess = [0.0, slope_guess, left_height_guess, right_height_guess]
    
    arguess = np.asarray(guess)
    
    p = scipy.optimize.fmin(func = min_fun, x0 =arguess, xtol=1e-9, ftol=1e-9, args=(x_left, x_right, y_left, y_right))
    
    #p, chi2 = scipy.optimize.curve_fit(min_fun, x, y, p0 = arguess)#, args =(x_left, x_right,y_left, y_right), maxiter = 1e12, maxfun = 1e12, xtol = 1e-6)
    return p

def min_fun_cusp(a, x_left, x_right, y_left, y_right):
    return (np.sum((a[1]*x_left+a[0]*(x_left**2) + a[2] - y_left)**2) + np.sum((a[1]*x_right+a[0]*(x_right**2) + a[2] - y_right)**2))

    #return (np.sum(((a[0]*x_left + a[1]) - y_left)**2) + np.sum((a[0]*x_right + a[2] - y_right)**2))
def level_that_shit_cusp(x_right, y_right, x_left, y_left, guess=None):
    
    if guess == None:
        slope_guess = (y_left[-1]-y_left[0])/(x_left[-1]-x_left[0])
        left_height_guess = np.mean(y_left)
        right_height_guess = np.mean(y_left)
        guess = [0.0, slope_guess, left_height_guess]
    
    arguess = np.asarray(guess)
    
    p = scipy.optimize.fmin(func = min_fun_cusp, x0 =arguess, xtol=1e-9, ftol=1e-9, args=(x_left, x_right, y_left, y_right))
    
    #p, chi2 = scipy.optimize.curve_fit(min_fun, x, y, p0 = arguess)#, args =(x_left, x_right,y_left, y_right), maxiter = 1e12, maxfun = 1e12, xtol = 1e-6)
    return p
'''
x = np.linspace(0, 501, 500)
def parab(x):
    return -0.5*x+60

y = parab(x)

x_left = x[0:200]
y_left = y[0:200]

x_right = x[300:500]
y_right = y[300:500]

params = level_that_shit(x_right, y_right, x_left, y_left)
resid = y-params[0]*x**2-params[1]*x- params[3]
otherresid = y-params[0]*x**2-params[1]*x- params[2]
import matplotlib.pyplot as plt
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.plot(x, otherresid)
ax.plot(x, resid)
#ax.plot(x, y)
plt.show()'''
