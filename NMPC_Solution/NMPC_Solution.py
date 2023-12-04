#!/usr/bin/env python3


from __future__ import division

import sys

############################################# Modify Path ##################################################

"""
WARNING: Update this path with the folder containing the built version of the
optimization problem.
"""

sys.path.insert(1, "/path_to_folder_created_in_step1/optimization-engine/open-codegen/PANOC_DYNAMIC_MOTOR_MODEL/dynamic_my_optimizer/dynamic_racing_target_point")

###########################################################################################################


import dynamic_racing_target_point
solver = dynamic_racing_target_point.solver()


from matplotlib import colors
import matplotlib.pyplot as plt
import time
# import casadi.casadi as cs
import numpy as np
import math
import csv


def find_the_center_line(X_fut,Y_fut,center_x,center_y):    
 
    dist_x = np.zeros(len(center_x))
    dist_y = np.zeros(len(center_x))
    r = np.zeros((N,len(center_x)))
    center_x_proj = np.zeros(N)
    center_y_proj = np.zeros(N)

    for j in range(len(X_fut)):
        dist_x = (X_fut[j] - center_x)**2
        dist_y = (Y_fut[j] - center_y)**2
        r = dist_x+dist_y
        x = np.argmin(r)
        center_x_proj[j] = center_x[x]
        center_y_proj[j] = center_y[x]       	
        
    return center_x_proj, center_y_proj


def perception_target_point(X_odom,Y_odom,center_x,center_y,a):

	center_x = np.concatenate((center_x, center_x))
	center_y = np.concatenate((center_y, center_y))
	dist_x = np.empty(len(center_x))
	dist_y = np.empty(len(center_x))
	r = np.empty(len(center_x))

	dist_x = (X_odom - center_x)**2
	dist_y = (Y_odom - center_y)**2
	r = dist_x+dist_y

	x = np.argmin(r)
	target_point_x = center_x[x+a]
	target_point_y = center_y[x+a]
	# print(target_point_x,target_point_y)

	return target_point_x, target_point_y

############################################################################################################################################
############################################# Choose the Track by modifying the paths ######################################################
############################################################################################################################################

"""
WARNING: Update these paths with the name of the folder containing the description of the track,
i.e., where the car will race.
"""

csv_file = np.genfromtxt('/path_to_folder_created_in_step1/689_Project/Maps/Map_track1/center_x_track1.csv', 
                          delimiter=',', dtype=float)
center_x = csv_file[:].tolist()
csv_file = np.genfromtxt('/path_to_folder_created_in_step1/689_Project/Maps/Map_track1/center_y_track1.csv', 
                          delimiter=',', dtype=float)
center_y = csv_file[:].tolist()
csv_file = np.genfromtxt('/path_to_folder_created_in_step1/689_Project/Maps/Map_track1/bound_x1_track1.csv', 
                          delimiter=',', dtype=float)
bound_x1 = csv_file[:].tolist()
csv_file = np.genfromtxt('//path_to_folder_created_in_step1/689_Project/Maps/Map_track1/bound_y1_track1.csv', 
                          delimiter=',', dtype=float)
bound_y1 = csv_file[:].tolist()
csv_file = np.genfromtxt('/path_to_folder_created_in_step1/689_Project/Maps/Map_track1/bound_x2_track1.csv', 
                          delimiter=',', dtype=float)
bound_x2 = csv_file[:].tolist()
csv_file = np.genfromtxt('/path_to_folder_created_in_step1/689_Project/Maps/Map_track1/bound_y2_track1.csv', 
                          delimiter=',', dtype=float)
bound_y2 = csv_file[:].tolist()


############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

lookahead = 90  # P in research paper
N = 38  # Horizon of Planning
T = 0.033 # Sampling time

# Car's paramters
car_width = 0.25
car_length = 0.4
lr = 0.147
lf = 0.178
m  = 5.6922
Iz  = 0.204
# df= 199.622
# dr= 191.2118
# cf= 0.057849
# cr= 0.11159
# bf= 9.2567
# br= 17.7464
# Cm1= 20
# Cm2= 1.3856e-07
# Cm3= 3.9901
# Cm4= 0.66633


df= 134.585
dr= 159.9198
cf= 0.085915
cr= 0.13364
bf= 9.2421
br= 17.7164
Cm1= 20
Cm2= 6.9281e-07
Cm3= 3.9901
Cm4= 0.66633
n_states = 6
n_controls = 2

mpciter = 0
u_cl1 = 0
u_cl2 = 0
xx1 = np.empty(N+1)
xx2 = np.empty(N+1)
xx3 = np.empty(N+1)
xx4 = np.empty(N+1)
xx5 = np.empty(N+1)
xx6 = np.empty(N+1)

x0 = [0, 0, 0, 1, 0, 0]		# initial conditions

xx_hist = []
xy_hist = []
xtheta_hist = []		# history of the states
vx_hist = []
vy_hist = []
omega_hist = []
ucl = []
guess = [0.0]*(2*N)
theta2unwrap = []

#################################################################################################################################
#################################################################################################################################
#################################################################################################################################

#plot the map
fig = plt.figure()

plt.plot(center_x, center_y, color='black', linestyle='dashed',  dashes=(25, 10), lw = 0.5)
plt.plot( bound_x1, bound_y1, color='black', lw = 0.3)
plt.plot( bound_x2, bound_y2, color='black', lw = 0.3)
plt.title('Results')
plt.xlabel('x (metres)')
plt.ylabel('y (metres)')
# plt.show(block=False)


f = open('race_DATA.csv', 'w')
writer = csv.writer(f)


t = 0 
laps = -1

while laps-2 < 0:  # Run for 2 laps

	if mpciter < 1:
		proj_center = find_the_center_line(np.linspace(0,1,N),np.zeros(N),center_x,center_y)
		proj_center_X = proj_center[0]
		proj_center_Y = proj_center[1]
	else:
		proj_center = find_the_center_line(xx1[1:N+1],xx2[1:N+1],center_x,center_y)
		proj_center_X = proj_center[0]
		proj_center_Y = proj_center[1]

	target_point = perception_target_point(x0[0],x0[1],center_x,center_y,lookahead)
	

	parameter = []
	for i in range(n_states):
		parameter.append(x0[i])
	# preU
	parameter.append(u_cl1)
	parameter.append(u_cl2)
	# target point
	parameter.append(target_point[0])
	parameter.append(target_point[1])
	# center line projection
	for i in range(N):
		parameter.append(proj_center_X[i])
		parameter.append(proj_center_Y[i])

	now = time.time()
	result = solver.run(p=[parameter[i] for i in range(n_states + n_controls + 2 + 2*N)],initial_guess=[guess[i] for i in range (n_controls*N)])
	now1 = time.time()
	elapsed = now1-now
	# print(elapsed)

	u_star = np.full(n_controls*N,result.solution)
	guess = u_star

	u_cl1 = u_star[0]
	u_cl2 = u_star[1]

	# print(x0, target_point, u_cl1,u_cl2)

	xx1[0] = x0[0]
	xx2[0] = x0[1]
	xx3[0] = x0[2]    
	xx4[0] = x0[3]
	xx5[0] = x0[4]
	xx6[0] = x0[5]

	for i in range(N):
		xx1[i+1] = xx1[i] + T* (xx4[i]*math.cos(xx3[i])-xx5[i]*math.sin(xx3[i]))
		xx2[i+1] = xx2[i] + T* (xx4[i]*math.sin(xx3[i])+xx5[i]*math.cos(xx3[i]))
		xx3[i+1] = xx3[i] + T* (xx6[i])
		xx4[i+1] = xx4[i] + T* ((1/m)*( (Cm1-Cm2*xx4[i])*guess[2*i]- Cm4*xx4[i]**2 -Cm3 + ((Cm1-Cm2*xx4[i])*guess[2*i]- Cm4*xx4[i]**2 -Cm3)*math.cos(guess[2*i+1]) + m*xx5[i]*xx6[i] - df*math.sin(cf*math.atan(bf*(- math.atan((xx5[i] + lf*xx6[i])/xx4[i]) + guess[2*i+1])))*math.sin(guess[2*i+1]) ))
		xx5[i+1] = xx5[i] + T* ((1/m)*(((Cm1-Cm2*xx4[i])*guess[2*i]- Cm4*xx4[i]**2 -Cm3)*math.sin(guess[2*i+1])-m*xx4[i]*xx6[i] + (df*math.sin(cf*math.atan(bf*(- math.atan((xx5[i] + lf*xx6[i])/xx4[i]) + guess[2*i+1])))*math.cos(guess[2*i+1]) + dr*math.sin(cr*math.atan(br*(- math.atan((xx5[i] - lr*xx6[i])/xx4[i]))))) ))
		xx6[i+1] = xx6[i] + T* ((1/Iz)*( lf*((Cm1-Cm2*xx4[i])*guess[2*i]- Cm4*xx4[i]**2 -Cm3)*math.cos(guess[2*i+1]) + lf*df*math.sin(cf*math.atan(bf*(- math.atan((xx5[i] + lf*xx6[i])/xx4[i]) + guess[2*i+1])))*math.cos(guess[2*i+1])-lr*dr*math.sin(cr*math.atan(br*(- math.atan((xx5[i] - lr*xx6[i])/xx4[i]))))))


	if x0[0] <= 0 and xx1[1] >=0 : # check if a lap is completed
		
		laps = laps + 1
		# print(laps)



	v_total = np.sqrt(x0[3]**2 + x0[4]**2)
	row = [x0[0], x0[1], x0[2], x0[3], x0[4], x0[5],v_total, elapsed,u_cl1,u_cl2,t]
	writer.writerow(row)

	mpciter = mpciter+1
	t = t + T

	x0[0] = xx1[1]
	x0[1] = xx2[1]
	x0[2] = xx3[1]
	x0[3] = xx4[1]
	x0[4] = xx5[1]
	x0[5] = xx6[1]


print("time_to_complete_2_laps = " + str(t))


# Plotting the solution

data = []
with open('race_DATA.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row)

# Convert data to numpy array
data_array = np.array(data, dtype=float)

# Extract relevant columns
x_positions = data_array[:, 0]
y_positions = data_array[:, 1]
velocities = data_array[:, 6]  

# Plot the scatter plot with color based on velocity
plt.scatter(x_positions, y_positions, s=0.5, c=velocities, cmap='turbo', alpha=1, norm = colors.Normalize(vmin=0, vmax=5))
plt.colorbar(label='Velocity Magnitude')
plt.axis('equal')
fig.savefig("NMPC_Result.png")
