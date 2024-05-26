## References

[1]Cataffo, V., Silano, G., Iannelli, L., Puig, V., and Glielmo, L., “A Nonlinear Model Predictive Control Strategy for Autonomous Racing of Scale Vehicles”, <i>arXiv e-prints</i>, 2023. doi:10.48550/arXiv.2302.04722.



# Project Setup Instructions

## Steps to Run the Code

### Step 1
Create a folder and run the following command to create the "optimization-engine" folder inside the created folder.

```bash
$ cd path_to_folder_created_in_step1/
$ git clone https://github.com/alphaville/optimization-engine.git optimization-engine
```

### Step 2
Go into the folder:

```bash
$ cd optimization-engine/open-codegen
```

### Step 3
Create a virtual environment using the correct python identifier (e.g., python3 or python3.6):

```bash
$ virtualenv -p python3 venvopen
```

### Step 4
Source the virtual environment created:

```bash
$ source venvopen/bin/activate
```

### Step 5
Run the setup.py script:

```bash
$ python setup.py install
```

Now the optimization engine (OpEn) is successfully installed and ready to use.

### Step 6
Close the terminal with the virtual environment.

### Step 7

```bash
$ cd path_to_folder_created_in_step1/
$ git clone https://github.com/shivam-vashi/689_Project.git
```

The result will be a folder named "689_Project" containing sub-folders "Maps" and "NMPC_Solution" and the scripts "all_wheel_drive_PANOC_DYNAMIC_motor_model.py" and "OBS_all_wheel_drive_PANOC_DYNAMIC_motor_model.py".

### Step 8
Cut and paste the scripts "all_wheel_drive_PANOC_DYNAMIC_motor_model.py" and "OBS_all_wheel_drive_PANOC_DYNAMIC_motor_model.py" to the following location:

```plaintext
/name_of_folder_created_in_step1/optimization-engine/open-codegen/
```

### Step 9
Run the scripts in terminal which will build the optimization solver:

```bash
$ cd name_of_folder_created_in_step1/optimization-engine/open-codegen
$ python3 all_wheel_drive_PANOC_DYNAMIC_motor_model.py
$ python3 OBS_all_wheel_drive_PANOC_DYNAMIC_motor_model.py
```

Note that the scripts take around 2-3 minutes each to finish execution. Also, they need to be run every time a change is made inside them for the changes to take effect.

### Step 10
Go to the directory "NMPC_Solution" at the location shown below:

```bash
$ cd name_of_folder_created_in_step1/689_Project/NMPC_Solution/
```

Read the warnings in both the scripts "NMPC_Solution.py" and "OBS_NMPC_Solution.py" and modify accordingly. There are 2 warnings within each script asking to change paths of some files. Use the paths given by the "Parent Directory" property of those files.

Save and run the scripts to obtain the solution to the optimization problem:

```bash
$ python3 NMPC_Solution.py
$ python3 OBS_NMPC_Solution.py
```

These will generate "race_DATA.csv" and "OBS_race_DATA.csv" files that contain vehicle states along with the plots saved as "NMPC_Result.png" and "OBS_NMPC_Result.png".

### Step 11
Finally, to plot the Acceleration and Steering Input data, run the script "Input_Plot.py":

```bash
$ cd name_of_folder_created_in_step1/689_Project/NMPC_Solution
$ python3 Input_Plot.py
```

**Note:** Any modification in parameters (like N) needs to be updated in all of these scripts: "all_wheel_drive_PANOC_DYNAMIC_motor_model.py", "OBS_all_wheel_drive_PANOC_DYNAMIC_motor_model.py", "NMPC_Solution.py", and "OBS_NMPC_Solution.py". Repeat steps 9, 10, and 11 for obtaining the results with updated parameters.
