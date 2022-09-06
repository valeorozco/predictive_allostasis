from distutils.log import error
from tqdm import tqdm
from attractor import Attractor
from FF import Cerebellum
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import numpy as np
import random
import math
import csv


#output from the ff controller
out_list = []
temperature = True
thirst = True
window_size = 10000 #Number of episodes gathering error to generate averaged signal/ window size for the averaged signal

plotting = False
Group_simulations=True
episodes = 100000 #100.000
num_simulations= 50

simulation = 0

arena_size = 200
arena_limit = 6

thetaList = []
theta = 90

nav_noise = 5
wheel_w = 15

arenaX = [0,arena_size]
arenaY = [0,arena_size]
grid_size= 1

dV_temperature = 1
dV_thirst = 1
aVtemperature_list=[]
aVthirst_list=[]

Drive_temperature_list=[]
Drive_thirst_list=[]

TFtemperature_list=[]
TFthirst_list=[]

Iatt_temperature_list=[]
Iatt_thirst_list=[]

meanGrad_Temp = []

frequency = 2
#Define a sinusoidal signal
gradient_position_list = np.cos(2*np.pi*np.arange(0,frequency*episodes, frequency)/episodes)
#Scaling the sinusoidal signal from 0 to 130 -> min and max position of the gradient.
gradient_position_list = (gradient_position_list+1)*65


if plotting == True:
    plt.ion()
    plt.style.use('seaborn')
    fig1, ax1 = plt.subplots(1, 2,figsize=(8,4))
    fig2, ax2 = plt.subplots(1, 2,figsize=(16,4))
    fig2.tight_layout(pad=2.0)




class Allostasis():

    def __init__(self):
        self.attractor = Attractor()
        self.robot_x = [random.randint(10,190)]
        self.robot_y = [random.randint(10,190)]

        self.gradient_position_float = 0
        self_temp = 0

        self.create_temp_gradient(0)
        self.create_thirst_gradient()
        self.build_gradients()

        self.aVhomeo_temperature = 1
        self.aVhomeo_thirst = 1
        self.output = 0

 
########################### GRADIENTS FUNCTIONS ###########################
    def kde_quartic(self,d,h):
        dn=d/h
        P=(15/16)*(1-dn**2)**2
        return P


    def create_temp_gradient(self, current_episode):
        global meanGrad_Temp
        #Gradient zone
        gradient_column = np.arange(1.01, 0.05, -0.015)
        temp_gradient = np.tile(gradient_column, (200,1))
        temp_gradient = temp_gradient.transpose()

        gradient_position =arena_size-gradient_column.shape[0]

        self.gradient_position_float = gradient_position_list[current_episode]
        self.gradient_position = int(self.gradient_position_float)


        #Zero zone
        zeros_row = [0]*(200)
        zero_columns =  self.gradient_position
        temp_zeros = np.tile(zeros_row, (zero_columns,1))
        

        #One zone
        ones_row = [1]*(200)
        ones_columns = 200 - self.gradient_position - len(gradient_column)

        temp_ones = np.tile(ones_row, (ones_columns,1))

        temp_arena = np.concatenate((temp_ones, temp_gradient, temp_zeros))

        #CONSTRUCT GRID
        x_grid=np.arange(arenaX[0],arenaX[1],grid_size)
        y_grid=np.arange(arenaY[0],arenaY[1],grid_size)
        self.x_mesh_temperature,self.y_mesh_temperature=np.meshgrid(x_grid,y_grid)
        self.temperature_intensity = temp_arena

        meanGrad_Temp.append(self.temperature_intensity.mean())

    def create_thirst_gradient(self):
        self.thirst_intensity_list=[]

        #POINT DATASET
        x= [15]
        y= [185]
        #DEFINE GRID SIZE AND RADIUS(h)
        grid_size=1
        h=180
        #CONSTRUCT GRID
        x_grid=np.arange(arenaX[0],arenaX[1],grid_size)
        y_grid=np.arange(arenaY[0],arenaY[1],grid_size)
        self.x_mesh_thirst,self.y_mesh_thirst=np.meshgrid(x_grid,y_grid)
        #GRID CENTER POINT
        xc=self.x_mesh_thirst+(grid_size/2)
        yc=self.y_mesh_thirst+(grid_size/2)
        #PROCESSING
        for j in range(len(xc)):
            intensity_row=[]
            for k in range(len(xc[0])):
                kde_value_list=[]
                for i in range(len(x)):
                    #CALCULATE DISTANCE
                    d=math.sqrt((xc[j][k]-x[i])**2+(yc[j][k]-y[i])**2) 
                    if d<=h:
                        p=self.kde_quartic(d,h)
                    else:
                        p=0
                    kde_value_list.append(p)
                #SUM ALL INTENSITY VALUE
                p_total=sum(kde_value_list)
                intensity_row.append(p_total)
            self.thirst_intensity_list.append(intensity_row)


    def build_gradients(self):

        if thirst == True:
            thirst_min = 100
            thirst_max = 0
            for i in range(len(self.thirst_intensity_list)):
                for j in range(len(self.thirst_intensity_list[i])):
                    if self.thirst_intensity_list[i][j] < thirst_min:
                        thirst_min = self.thirst_intensity_list[i][j]
                    if self.thirst_intensity_list[i][j] > thirst_max:
                        thirst_max = self.thirst_intensity_list[i][j]

            thirst_intensity=np.array(self.thirst_intensity_list)
            self.thirst_intensity=thirst_intensity/thirst_max


    def plot_gradient(self):
        #...........  GRADIENTS ...........
        if temperature:
            ax1[0].cla()
            ax1[0].grid(False)
            ax1[0].plot(self.robot_x,self.robot_y,'g', markersize=3)
            ax1[0].set_title("Temperature")
            ax1[0].pcolormesh(self.x_mesh_temperature,self.y_mesh_temperature,self.temperature_intensity, cmap = plt.get_cmap('coolwarm'))


        if thirst:
            ax1[1].cla()
            ax1[1].grid(False)
            ax1[1].set_title("Thirst")
            ax1[1].plot(self.robot_x[-1],self.robot_y[-1],'ro')
            ax1[1].pcolormesh(self.x_mesh_thirst,self.y_mesh_thirst,self.thirst_intensity, cmap = plt.get_cmap('viridis'))


        fig1.canvas.flush_events()

        #...........  ATTRACTOR DYNAMICS ...........
        ax2[0].cla()
        ax2[0].grid(False)
        ax2[0].set_title("Mean Firing Rate")
        ax2[0].set_ylim(-0.1, 2)
        ax2[0].plot(TFtemperature_list, color='orange', label='Temperature')
        ax2[0].plot(TFthirst_list, color='blue', label='Thirst')
        ax2[0].legend(loc="upper left")

        ax2[1].cla()
        ax2[1].grid(False)
        ax2[1].set_ylim(-0.1, 1.1)
        ax2[1].set_title("Attractor Inputs")
        ax2[1].plot(Iatt_temperature_list, color='orange', label='Temperature')
        ax2[1].plot(Iatt_thirst_list, color='blue', label='Thirst')

########################### LOCAL VIEWS ###########################

    def temperature_LV(self): 
        """
        Generates an actual value for temperature based on the location of the agent
        """
        
        self.q0_temperature, self.q1_temperature, self.q2_temperature, self.q3_temperature = 0,0,0,0

        for i in range(4):
            for j in range(3):
                self.q0_temperature += self.temperature_intensity[int(self.robot_y[-1]) + (j + 1), int(self.robot_x[-1]) - (i + 1)]
                self.q1_temperature += self.temperature_intensity[int(self.robot_y[-1]) + (j + 1), int(self.robot_x[-1]) + (i + 1)]
                self.q2_temperature += self.temperature_intensity[int(self.robot_y[-1]) - (j + 1), int(self.robot_x[-1]) - (i + 1)]
                self.q3_temperature += self.temperature_intensity[int(self.robot_y[-1]) - (j + 1), int(self.robot_x[-1]) + (i + 1)]

        self.q0_temperature /= 12
        self.q1_temperature /= 12
        self.q2_temperature /= 12
        self.q3_temperature /= 12
        
        
        self.aV_temperature = (self.q0_temperature + self.q1_temperature + self.q2_temperature + self.q3_temperature) / 4
        self.diff_temperature = abs(dV_temperature - self.aV_temperature)

    def thirst_LV(self):
        """
        Generates an actual value for thirst based on the location of the agent
        """
        self.q0_thirst, self.q1_thirst, self.q2_thirst, self.q3_thirst = 0,0,0,0
        for i in range(4):
            for j in range(3):
                self.q0_thirst += self.thirst_intensity[int(self.robot_y[-1]) + (j + 1), int(self.robot_x[-1]) - (i + 1)]
                self.q1_thirst += self.thirst_intensity[int(self.robot_y[-1]) + (j + 1), int(self.robot_x[-1]) + (i + 1)]
                self.q2_thirst += self.thirst_intensity[int(self.robot_y[-1]) - (j + 1), int(self.robot_x[-1]) - (i + 1)]
                self.q3_thirst += self.thirst_intensity[int(self.robot_y[-1]) - (j + 1), int(self.robot_x[-1]) + (i + 1)]

        self.q0_thirst /= 12
        self.q1_thirst /= 12
        self.q2_thirst /= 12
        self.q3_thirst /= 12

        
        self.aV_thirst = (self.q0_thirst + self.q1_thirst + self.q2_thirst + self.q3_thirst) / 4
        self.diff_thirst = abs(dV_thirst - self.aV_thirst)



########################### ORIENTATION ###########################

    def adsign(self):
        global dV_temperature, dV_thirst
        self.adsign_temperature = np.sign(dV_temperature - self.aV_temperature)
        self.adsign_thirst = np.sign(dV_thirst - self.aV_thirst)

    def hsign(self):
        if theta <= 112 and theta > 77: #UP
            self.hsign_temperature = np.sign(self.q1_temperature - self.q0_temperature)
            self.hsign_thirst = np.sign(self.q1_thirst - self.q0_thirst)
        elif theta <= 157 and theta > 112: #UP-L
            self.hsign_temperature = np.sign(((self.q0_temperature + self.q1_temperature)/2) - ((self.q0_temperature + self.q2_temperature)/2))
            self.hsign_thirst = np.sign(((self.q0_thirst + self.q1_thirst)/2) - ((self.q0_thirst + self.q2_thirst)/2))
        elif theta <= 202 and theta > 157: #L
            self.hsign_temperature = np.sign(self.q0_temperature - self.q2_temperature)
            self.hsign_thirst = np.sign(self.q0_thirst - self.q2_thirst)
        elif theta <= 247 and theta > 202: #DOWN-L
            self.hsign_temperature = np.sign(((self.q2_temperature + self.q0_temperature)/2) - ((self.q2_temperature + self.q3_temperature)/2))
            self.hsign_thirst = np.sign(((self.q2_thirst + self.q0_thirst)/2) - ((self.q2_thirst + self.q3_thirst)/2))
        elif theta <= 292 and theta > 247: #DOWN
            self.hsign_temperature = np.sign(self.q2_temperature - self.q3_temperature)
            self.hsign_thirst = np.sign(self.q2_thirst - self.q3_thirst)
        elif theta <= 337 and theta > 292: #DOWN-R
            self.hsign_temperature = np.sign(((self.q3_temperature + self.q2_temperature)/2) - ((self.q3_temperature + self.q1_temperature)/2))
            self.hsign_thirst = np.sign(((self.q3_thirst + self.q2_thirst)/2) - ((self.q3_thirst + self.q1_thirst)/2))
        elif theta <= 22 and theta > 337: #R
            self.hsign_temperature = np.sign(self.q3_temperature - self.q1_temperature)
            self.hsign_thirst = np.sign(self.q3_thirst - self.q1_thirst)
        elif theta <= 77 and theta > 22: #UP-R
            self.hsign_temperature = np.sign(((self.q1_temperature + self.q3_temperature)/2) - ((self.q1_temperature + self.q0_temperature)/2))
            self.hsign_thirst = np.sign(((self.q1_thirst + self.q3_thirst)/2) - ((self.q1_thirst + self.q0_thirst)/2))



########################### HOMEOSTASIS ###########################

    def homeostasis(self):
        """
        Updates values of homeostatic markers for both thirst and temperature
        """
        discount = 0.001 
        bonus = discount*10

        self.aVhomeo_temperature -= discount
        self.aVhomeo_thirst -= discount

        if self.aVhomeo_temperature < 0: self.aVhomeo_temperature = 0
        if self.aVhomeo_thirst < 0: self.aVhomeo_thirst = 0

        if self.diff_temperature <0.2:
            self.aVhomeo_temperature += bonus
        if self.diff_thirst <0.2:
            self.aVhomeo_thirst += bonus

        if self.aVhomeo_temperature > 1: self.aVhomeo_temperature = 1
        if self.aVhomeo_thirst > 1: self.aVhomeo_thirst = 1
       #The attractor input will be 1 - the Actual value, resulting in 0 when the system is satisfied
        self.Itemp_attractor = 1 - self.aVhomeo_temperature
        self.Ithi_attractor = 1 - self.aVhomeo_thirst

        aVthirst_list.append(self.aVhomeo_thirst)
        aVtemperature_list.append(self.aVhomeo_temperature)
    
        Iatt_temperature_list.append(self.Itemp_attractor)
        Iatt_thirst_list.append(self.Ithi_attractor)

        Drive_thirst_list.append(self.Ithi_attractor)
        Drive_temperature_list.append(self.Itemp_attractor)

########################### ATTRACTOR DYNAMICS ##########################
    def attractor_dynamics(self):
        """
        Calls main functions from the attractor class in which conflict resolution from both needs takes place
        """
        self.Itemp_attractor *= 10
        self.Ithi_attractor *= 10
        input = self.output + self.Itemp_attractor 
        self.total_force_temperature, self.total_force_thirst = self.attractor.advance(input, self.Ithi_attractor)

        self.total_force_temperature /= 40
        self.total_force_thirst /= 40

        TFtemperature_list.append(self.total_force_temperature)
        TFthirst_list.append(self.total_force_thirst)

########################### NAVIGATION ###########################

    def conv(self, ang):
        x = np.cos(np.radians(ang)) 
        y = np.sin(np.radians(ang)) 
        return x , y

    def wheel_turning(self):
        self.wheel = -1 * ((self.hsign_temperature * self.adsign_temperature* self.total_force_temperature) + (self.hsign_thirst * self.adsign_thirst* self.total_force_thirst)) * (1/2)

    def random_navigation(self):
        global theta

        theta_extra = 5

        if theta > 360:
            div = math.trunc(theta/360) #num of rounds
            div *= 360                  #round in degrees
            theta = theta%div            #new theta
        if  theta < 0 and theta >= -360:
            theta +=360


        if(self.robot_x[-1]<arena_limit and self.robot_y[-1]<arena_limit): #Left-bottom
            theta = np.random.randint(20,70)
        elif(self.robot_x[-1]<arena_limit and self.robot_y[-1]>arena_size-arena_limit): #Left-top
            theta = np.random.randint(290,340)
        elif(self.robot_x[-1]>arena_size-arena_limit and self.robot_y[-1]<arena_limit): #Right-bottom
            theta = np.random.randint(110, 160)
        elif(self.robot_x[-1]>arena_size-arena_limit and self.robot_y[-1]>arena_size-arena_limit): #Right-top
            theta = np.random.randint(200,250)
        elif( self.robot_x[-1]<arena_limit ): #Left
            if theta <=180:
                theta -= theta_extra
            else:
                theta += theta_extra
        elif(self.robot_x[-1]>arena_size-arena_limit ): #Right
            if theta <=180 and theta >= 0:
                theta += theta_extra
            elif theta <= 0:
                theta = 0
            else:
                theta -= theta_extra
        elif(self.robot_y[-1]<arena_limit): #Bottom
            if theta >= 90 and theta <=270:
                theta -= theta_extra
            else:
                theta += theta_extra
        elif(self.robot_y[-1]>arena_size-arena_limit): #Top
            if theta >= 90 and theta <=270:
                theta += theta_extra
            else:
                theta -=theta_extra
        else:
            theta = theta + random.gauss(0, nav_noise) + self.wheel*wheel_w

    
        check_x = self.robot_x[-1]+self.conv(theta)[0] + np.random.uniform(-0.5,0.5)
        check_y = self.robot_y[-1]+self.conv(theta)[1] + np.random.uniform(-0.5,0.5)
        if check_x >= 4 and check_x <= arena_size - 4 and check_y >= 4 and check_y <= arena_size - 4:
            self.robot_x.append(check_x)
            self.robot_y.append(check_y)
        else:
            self.robot_x.append(self.robot_x[-1])
            self.robot_y.append(self.robot_y[-1])



########################### SAVING DATA ###########################
        
    def save_data(self, current_simulation):
        global aVtemperature_list, aVthirst_list, Drive_temperature_list, Drive_thirst_list, TFtemperature_list, TFthirst_list, meanGrad_Temp , Iatt_temperature_list, out_list
        csv_namefile = 'THESIS/Results/testFF' + str(current_simulation+1) + '.csv'
        print(csv_namefile)
        
        with open(csv_namefile, mode='w') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=['Xposition', 'Yposition', 'aVtemperature', 'aVthirst', 'Error_to_attractor', 'DriveThirst', 'TFtemperature', 'TFthirst', 'Temperature_error', 'Grad_Temp','Output'])
            csv_writer.writeheader()
            for i in range(window_size, episodes):
                csv_writer.writerow({'Xposition': self.robot_x[i], 'Yposition': self.robot_y[i], 'aVtemperature': aVtemperature_list[i], 'aVthirst': aVthirst_list[i],
                    'Error_to_attractor': Drive_temperature_list[i], 'DriveThirst': Drive_thirst_list[i], 'TFtemperature': TFtemperature_list[i], 'TFthirst': TFthirst_list[i],
                    'Temperature_error':Iatt_temperature_list[i],'Grad_Temp': meanGrad_Temp[i], 'Output': out_list[i-window_size]}) 

    def clean(self):
        """ 
        Cleans variables to initialized values once the simulation number is upgraded
        """
        global aVtemperature_list, aVthirst_list, Drive_temperature_list, Drive_thirst_list, TFtemperature_list, TFthirst_list, Iatt_temperature_list, Iatt_thirst_list, out_list
        self.robot_x = [random.randint(10,190)]
        self.robot_y = [random.randint(10,190)]
        self.gradient_position_float = 0
        self.aVhomeo_temperature = 1
        self.aVhomeo_thirst = 1
        aVtemperature_list=[]
        aVthirst_list=[]
        Drive_temperature_list=[]
        Drive_thirst_list=[]
        TFtemperature_list=[]
        TFthirst_list=[]
        Iatt_temperature_list=[]
        Iatt_thirst_list=[]
        out_list = []
        ff_module.reset()

####################### FEEDFORWARD ###################################
    def feedforward(self):
        """ 
        Calls the main functions of the feedforward controller and generates an output 
        """
        input = np.array([[[self.temperature_intensity.mean()]]])
        error = np.array([self.Itemp_attractor])

        self.output = ff_module.activate(input, error, update=True)
        self.output = self.output[0]
        out_list.append(self.output)

            
########################### RUN SIMULATION ###########################
    def run(self, current_simulation):
    
        for i in tqdm(range(episodes)):
            self.create_temp_gradient(i)
            self.temperature_LV()
            self.thirst_LV()
            self.adsign()
            self.hsign()
            self.homeostasis()
            if i>=window_size:
                self.feedforward()
            self.attractor_dynamics()
            self.wheel_turning()
            self.random_navigation()
            if plotting == True:
                self.plot_gradient()
        self.save_data(current_simulation)
        self.clean()


###########################  ###########################

allo = Allostasis()
ff_module = Cerebellum(dt=1e-3, nInputs=1, nOutputs=1, nPCpop=10, nIndvBasis=100, nSharedBasis=200, beta_MF=1e-3, 
                       beta_PF=1e-7, range_delays=[0.05,0.5], range_TC=[0.05, 2.], range_scaling=[1, 100], range_W=[0., 1.])
if __name__ == '__main__':
    try:
        if Group_simulations == True:
            for a in range(num_simulations):
                allo.run(a)
        else:
            allo.run(0)


    except KeyboardInterrupt:
        print('Simulation interrupted')