import csv
import numpy as np
import matplotlib.pyplot as plt


#  Steering Input Plot 

data = []
with open('race_DATA.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row)

# Convert data to numpy array
data_array = np.array(data, dtype=float)
fig = plt.figure()
plt.title("Steering Input")
steering_input = np.degrees(data_array[:, 9])
time_data = data_array[:, 10]

plt.scatter(time_data, steering_input, label='delta', color='blue', s = 0.5)


data = []
with open('OBS_race_DATA.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row)

# Convert data to numpy array
data_array = np.array(data, dtype=float)
steering_input = np.degrees(data_array[:, 9])
time_data = data_array[:, 10]


plt.scatter(time_data, steering_input, label='delta_obs', color='red', s = 0.5)
plt.plot(time_data , (-30)*np.ones(len(time_data)) , linestyle='dashed', color = "black")
plt.plot(time_data , (+30)*np.ones(len(time_data)) , linestyle='dashed', color = "black")
plt.legend(loc = "lower right")
plt.xlabel("Time (s)")
plt.ylabel("Steering angle delta (degrees")
fig.savefig("Steering_Input.png", dpi = 400)



# Acceleration Input Plot

data = []
with open('race_DATA.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row)

# Convert data to numpy array
data_array = np.array(data, dtype=float)

fig = plt.figure()
plt.title("Acceleration Input")
acceleration_input = data_array[:, 8]
# steering_input = np.degrees(data_array[:, 9])
time_data = data_array[:, 10]

plt.plot(time_data, acceleration_input, label='d', linestyle = "dotted", color='blue', lw = 1.5)



data = []
with open('OBS_race_DATA.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row)

# Convert data to numpy array
data_array = np.array(data, dtype=float)
acceleration_input = data_array[:, 8]
time_data = data_array[:, 10]


plt.plot(time_data, acceleration_input, label='d_obs', linestyle = "dashed", color='red', lw = 1.5, dashes=(4, 4))
plt.plot(time_data , (0)*np.ones(len(time_data)) , linestyle='dashed', color = "black")
plt.plot(time_data , (1)*np.ones(len(time_data)) , linestyle='dashed', color = "black")
plt.legend(loc = "lower right")
plt.xlabel("Time (s)")
plt.ylabel("PWM Acceleration Signal")
fig.savefig("Accln_Input.png", dpi=400)