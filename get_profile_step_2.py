#modules and shit <- you need these
import numpy as np
import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox
import matplotlib
matplotlib.use("TkAgg") #<this one fixes the backend issue with tkinter
import matplotlib.pyplot as plt
import scipy.misc as misc
import scipy.ndimage.interpolation as ndimage
from scipy.optimize import curve_fit
import scipy
import os
#what file do you want to analyze?
root = tk.Tk()

root.withdraw()
root.update()
file_path = filedialog.askopenfilename() #asks which file you want to analyze and records the filepath and name to be used in the code
root.destroy()

import get_AFM_data
#hs, N_lines, scan_size = get_AFM_data.get_AFM_data('PMMAonPSsteps_100517CL_S1_thin_150C_0min_tapping.txt')
hs, N_lines, scan_size = get_AFM_data.get_AFM_data(file_path)

deltax = scan_size/float(hs.shape[1])
xs = deltax*(np.linspace(0, hs.shape[1]+1, hs.shape[1]))

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

directory = os.path.split(file_path)[0]
ensure_dir(directory+'/profiles')

        
##let's see the data
f1 = plt.figure(2)
a1 = f1.add_subplot(111)
m = a1.imshow(hs)
m.set_cmap('hot')
f1.colorbar(m)
plt.show()

root = tk.Tk()

root.withdraw()
root.update()
variable=tkinter.messagebox.askquestion('Wat the deal?', 'Step on the right?')
root.destroy()
if variable == 'no':
    hs = np.fliplr(hs)
    plt.close()
else:
    plt.close()


import line_maker
from matplotlib.lines import Line2D
if __name__ == '__main__':
    fig, ax = plt.subplots()
    dat =ax.imshow(hs)
    dat.set_cmap('hot')
    fig.colorbar(dat)
    ax.legend(title = 'Draw the left line')
    line = Line2D([100,100], [10,100], marker = 'o', markerfacecolor = 'red')
    ax.add_line(line)

    linebuilder = line_maker.LineBuilder(line)
    
    ax.set_title('Draw the left line')
    ax.set_xlim(0,hs.shape[1])
    ax.set_ylim(0,hs.shape[0])
    
    plt.show()
    xleft = line.get_xdata()
    yleft = line.get_ydata()
    plt.close(fig)


if __name__ == '__main__':
    fig, ax = plt.subplots()
    data = ax.imshow(hs)
    data.set_cmap('hot')
    fig.colorbar(data)
    line = Line2D([400,400], [10,100], marker = 'o', markerfacecolor = 'red')
    ax.add_line(line)

    linebuilder = line_maker.LineBuilder(line)
    
    ax.set_title('Draw the right line')
    ax.set_xlim(0,hs.shape[1])
    ax.set_ylim(0,hs.shape[0])
    
    plt.show()
    xright = line.get_xdata()
    yright = line.get_ydata()
    plt.close(fig)


pleft = np.polyfit(yleft, xleft, 1) #writing the equation of the line in terms fo the y variables
pright = np.polyfit(yright, xright, 1) 

x_left = np.polyval(pleft, range(N_lines))
x_right = np.polyval(pright, range(N_lines))



x_left = x_left.astype(int)
x_right = x_right.astype(int)
h2s = np.zeros(N_lines)
fit_hs = np.zeros((N_lines, hs.shape[1]))

import level_that_shit
#Now remove a polynomial from each line
for i in range(N_lines):
    h = hs[i,:]
    
    h_left_fit = h[0:x_left[i]]
    h_right_fit = h[x_right[i]:-1]
    h_fit = np.concatenate([h_left_fit, h_right_fit])
    
    x_left_fit = xs[0:x_left[i]]
    x_right_fit = xs[x_right[i]:-1]
    x_fit = np.concatenate([x_left_fit, x_right_fit])
    params = level_that_shit.level_that_shit(x_right_fit, h_right_fit, x_left_fit, h_left_fit)
    if i==0:
        params = level_that_shit.level_that_shit(x_left_fit,h_left_fit,x_right_fit,h_right_fit)
    else:
        params = level_that_shit.level_that_shit(x_left_fit,h_left_fit,x_right_fit,h_right_fit)
    
    alpha = params[0]
    beta = params[1]
    gamma = params[3]
    guess = params
    print(guess)
    fit_hs[i,:] = h-alpha*(xs**2) - beta*xs - gamma
    #fit_hs[i,:] = h - beta*xs - gamma
    h2s[i] = params[2] - params[3]
fig, ax = plt.subplots()
ax.plot(xs, h)
#ax.plot(xs, gamma+beta*xs)
ax.plot(xs, gamma+beta*xs+alpha*xs**2)
plt.show()

h2 = np.mean(h2s)
hmid = np.zeros(N_lines)
for i in range(N_lines):
  idx = np.where(fit_hs[i,:] > (h2s[i]/2))[0][0]
  hmid[i] = idx;


f4 = plt.figure(4)
a4 = f4.add_subplot(111)
backg = a4.imshow(fit_hs)
backg.set_cmap('hot')
f4.colorbar(backg)





#######
#rotate the image
y_line = np.arange(0,N_lines)
a4.plot(hmid, y_line, '.b')
p = np.polyfit(y_line, hmid, 1)
max_line = np.polyval(p, y_line)
a4.plot(max_line, y_line, 'g')


slope = p[0]
angle = np.arctan(slope)*57.2957795 #to degrees

hs_rot = ndimage.rotate(fit_hs, -angle, reshape=False)

#crop image to rotated input lines

#get distances of left and right lines to a point in the centre

d_left_top = np.array([[x_left[0]-x_left[N_lines-1], 1-N_lines],[hmid[round(N_lines/2)]-x_left[N_lines-1], round(N_lines/2)-N_lines]])
d_left_bottom = np.array([x_left[0]-x_left[N_lines-1], 1-N_lines])
d_left = round(abs(np.linalg.det(d_left_top))/np.linalg.norm(d_left_bottom))

d_right_top = np.array([[x_right[0]-x_right[N_lines-1], 1-N_lines],[hmid[round(N_lines/2)]-x_right[N_lines-1], round(N_lines/2)-N_lines]])
d_right_bottom = np.array([x_right[0]-x_right[N_lines-1], 1-N_lines])
d_right = round(abs(np.linalg.det(d_right_top))/np.linalg.norm(d_right_bottom))


f5, a5 = plt.subplots()
leveled_rot =a5.imshow(hs_rot)
leveled_rot.set_cmap('hot')
f5.colorbar(leveled_rot)


#% position of rotated centre
max_loc = hmid[round(len(hs_rot)/2)]

#% find rows that have zeros in the centre regions and therefore need to be deleted

extra_buffer = 50;
centre_region = hs_rot[:,int(max_loc-d_left-extra_buffer):int(max_loc+d_right+extra_buffer)]


goodrows = []
for row in range(len(hs_rot)):
    if all(x !=0 for x in centre_region[row,:]):
        goodrows.append(row)
first_good_line = goodrows[0]
last_good_line = goodrows[-1]

hs_crop = hs_rot[first_good_line:last_good_line,:]

f6, a6 = plt.subplots()
leveled_rot_crop =a6.imshow(hs_crop)
leveled_rot_crop.set_cmap('hot')
f6.colorbar(leveled_rot_crop)


'''
#________________________________________________
#%% shift lines so that V_left = V_right (as compared to a step function)
newx = np.empty((hs_crop.shape[0],hs_crop.shape[1]))

x = deltax*np.linspace(0, hs_crop.shape[1]+1, hs_crop.shape[1])


heavy=np.zeros((hs_crop.shape[1],2))
heavy[:,0] = x
heavy[:,1] = 0
for i in range(hs_crop.shape[0]):
   
    h2 = h2s[i];
    h = hs_crop[i,:]
    
    #now make the heavyside step function
     # the base line
    intsig = sum(h)
    
    #find the minimum in the difference of the areas, initial to final (to match to the heavyside step function as much as possible
    sumheavo=10**16
    ia3=0
    
    for j in range(len(h)):
        heavys = np.zeros(hs_crop.shape[1])
        heavys[j:] = heavys[j:] + h2
        sumheavn = sum(heavys)
        dsigh = abs(sumheavn-intsig) #the difference in the integrals of the heights
        if dsigh<sumheavo:
            sumheavo = dsigh
            ia3 = j
    
    
    newx[i,:] = x-ia3*deltax
#_______________________________________
'''
# build a matrix of the shifted data

#find mininum x value
#find maximum x value
#make appropriate matrix
#go through all lines, look at first x entry, subtract xmin, divide by spacing and add 1 to get starting coordinate
'''
#find max and min x values
xmin = 1000
xmax = -1000

for i in range(hs_crop.shape[0]):
    xs = newx[i]
    xmin_temp = min(xs)
    if xmin_temp < xmin:
        xmin = xmin_temp
    xmax_temp = max(xs)
    if xmax_temp > xmax:
        xmax = xmax_temp


#make matrix padded with zeros and h2
dim = int((xmax - xmin)/deltax) +1
hs_shift = np.zeros((x.shape[0],dim-1))


#put the data in
for i in range(len(newx)):
    
    x = newx[i]
    x1 = x[0]

    coord = int((x1 - xmin)/deltax) + 1
    hs_shift[i, coord:(coord+len(x)) ]  = hs_crop[i,:]

  
#redefining the x scale in terms of um
xs = np.arange(xmin, xmax, deltax)
'''

#don't want average over x-coordinate that don't have enough data, so first
#remove these rows and columns by seeing if the sum of the step is greater than a threshold value
threshold = np.average(h2s)*(len(hs)/2.0-80)

good_stuff =[]
good_columns = []
for row in range(len(hs_crop)):
    if np.sum(hs_crop[row,int(hs_crop.shape[1]/2):])>=threshold:
        good_stuff.append(row)
    
col_min = good_stuff[0]
col_max = good_stuff[-1]



hs_cut = hs_crop[col_min:col_max, :]

for row in range(len(hs_cut)):
    x = np.where(hs_cut[row, int(3*hs_cut.shape[1]/4):] ==0)[0]
    
    if x.size != 0:
        good_columns.append(int(3*hs_cut.shape[1]/4)+x[0])

lowest = min(good_columns)

hs_cut = hs_cut[:, :lowest]
xs_cut = xs[:lowest]

f5, a5 = plt.subplots()
leveled_rot =a5.imshow(hs_cut)
leveled_rot.set_cmap('hot')
f5.colorbar(leveled_rot)


avg_hs = np.mean(hs_cut[int(len(hs_cut)/2)-40:int(len(hs_cut)/2)+40], axis=0)
#avg_hs = np.mean(hs_cut[int(len(hs_cut)/2)-30:int(len(hs_cut)/2)+30], axis=0) #averaging over columns
#[int(len(hs_crop)/2)-64:int(len(hs_crop)/2)+64]


newx = np.empty(len(avg_hs))

x = deltax*np.linspace(0, len(avg_hs)+1, len(avg_hs))


heavy=np.zeros((len(avg_hs),2))
heavy[:,0] = x
heavy[:,1] = 0

   
h2 = np.mean(h2s)

    
#now make the heavyside step function
# the base line
intsig = sum(avg_hs)
   
    #find the minimum in the difference of the areas, initial to final (to match to the heavyside step function as much as possible
sumheavo=10**16

def f(a, x, y): return np.sum(((a[0]*(np.sign(x-a[1]))+a[2])-y)**2)
arguess = np.array((h2/2, len(avg_hs)*deltax/2, h2/2))
p = scipy.optimize.fmin_powell(func = f, x0 = arguess, args=(xs_cut, avg_hs), xtol = 1e-6)
'''popt,pcov = curve_fit(f, xs_cut, avg_hs, bounds=([h2*0.49,0,h2*0.49],[h2*0.51,len(avg_hs)*deltax,h2*0.51]))
print(popt)'''
newx = x-p[1]

'''
ia3=0
print(h2)    
for j in range(len(avg_hs)):
    heavys = np.zeros(len(avg_hs))
    heavys[j:] = heavys[j:] + h2
    sumheavn = sum(heavys)
    dsigh = abs(sumheavn-intsig) #the difference in the integrals of the heights
    if dsigh < sumheavo:
        sumheavo = dsigh
        ia3 = j
print(ia3)
    
newx = x-ia3*deltax

'''
#heaviside = f(p, newx)
f7, a7 = plt.subplots() #Show off that nice profile
a7.plot(newx, p[0]*(np.sign(xs_cut-p[1]))+p[2])
a7.plot(newx, avg_hs, '.')
plt.show()




#save to a csv with the same initial filename with 'profile' added
profile = np.array((newx,avg_hs))


np.savetxt(directory+'/profiles/'+os.path.split(file_path)[1][0:-4]+'_profile.csv', profile, delimiter = ',')
#_________________________________________________________

'''
#pick a single profile
sing_hs = hs_cut[int(len(hs_cut)/2), :]


sing_profile = np.array((newx, sing_hs))
df8, a8 = plt.subplots() #Show off that nice profile
a8.plot(newx, sing_hs)
plt.show()
np.savetxt(directory+'/profiles/'+os.path.split(file_path)[1][0:-4]+'_single.csv',sing_profile, delimiter = ',')

#Good job!
'''
