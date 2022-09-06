import errno
import numpy as np
import csv


class Cerebellum(object):
   
    '''
    Generalized cerebellar adaptive filter with alpha-like impulse responses as temporal basis functions and with a bank of Purkinje cells sensitive to a range of input-output delays.
    '''
    # Creates a Module composed of Alpha-like filters with different temporal profiles.
    def __init__(self, dt=0.01, nInputs=1, nOutputs=1, nPCpop=10, nIndvBasis=50, nSharedBasis=200, beta_MF=1e-3, beta_PF=1e-8, 
                 range_delays=[0.05, 0.5], range_TC=[0.05, 2.], range_scaling=[1, 100], range_W=[0., 1.]):

        # Basic parameters.
        self.dt = dt
        self.nInputs = nInputs
        self.nIndvBasis = nIndvBasis
        self.nSharedBasis = nSharedBasis
        self.nOutputs = nOutputs
        self.nPCpop = nPCpop
        self.nBasis = (self.nInputs*self.nIndvBasis) + self.nSharedBasis
        self.nPC = self.nPCpop*self.nOutputs
        self.range_delays = range_delays
        delays = np.linspace(self.range_delays[0], self.range_delays[1], self.nPCpop)              
        indxs_delays = ((self.range_delays[1] - delays)/self.dt).astype(int)
        self.indxs_delays = np.tile(indxs_delays, self.nOutputs)
        self.range_TC = range_TC
        self.range_scaling = range_scaling
        self.range_W = range_W
        self.error = [0]
        self.basis_f_list = []


        # Purkinje cells' parameters and variables.
        self.PC_buffer = np.zeros((self.nPC, int(self.range_delays[1]/self.dt)))

        # Learning parameters and variables.
        self.beta_PF = beta_PF
        self.beta_MF = beta_MF

        self.create_synapses()
        self.create_basisFunctions()

    def create_synapses(self):

        # Mossy fiber synapses: specific granule processors for each individual input channel and one (plastic) granule processor shared for all Inputs.
        w_indvMF = np.zeros((self.nInputs, int(self.nInputs*self.nIndvBasis)))
        for i in np.arange(self.nInputs):
            w_indvMF[i, int(i*self.nIndvBasis):int((i+1)*self.nIndvBasis)] = 1

        w_sharedMF = np.random.uniform(self.range_W[0], self.range_W[1], self.nInputs*self.nSharedBasis).reshape((self.nInputs, self.nSharedBasis))
        self.w_MF = np.column_stack((w_indvMF, w_sharedMF))
        self.mask_wMF = np.column_stack((np.zeros((self.nInputs, int(self.nInputs*self.nIndvBasis))), np.ones((self.nInputs, self.nSharedBasis))))

        # Parallel fiber synpases: a population of heterogeneous delay-tuned Purkinje cells for each output channel (deep nucleus).
        self.w_PF = np.zeros(self.nBasis*self.nPC).reshape((self.nBasis, self.nPC)) 

    def create_basisFunctions(self):

        # Temporal basis parameters and variables.
        random_indxs = np.random.choice(np.arange(1000), self.nSharedBasis, replace=False)

        TC_indvReservoir = 1/np.logspace(np.log10(self.range_TC[0]), np.log10(self.range_TC[1]), self.nIndvBasis)
        TC_sharedReservoir = 1/np.logspace(np.log10(self.range_TC[0]), np.log10(self.range_TC[1]), 1000)
        t_constants = np.concatenate([np.tile(TC_indvReservoir, self.nInputs), TC_sharedReservoir[random_indxs]])
        self.gammas = np.exp(-self.dt*t_constants)

        scalingInput_indvReservoir = 1e-2/np.logspace(np.log10(self.range_scaling[0]), np.log10(self.range_scaling[1]), self.nIndvBasis)        
        scalingInput_sharedReservoir = 1e-2/np.logspace(np.log10(self.range_scaling[0]), np.log10(self.range_scaling[1]), 1000)
        self.scaling_input = np.concatenate([np.tile(scalingInput_indvReservoir, self.nInputs), scalingInput_sharedReservoir[random_indxs]])

        self.z = np.zeros(self.nBasis)
        self.p = np.zeros(self.nBasis)                                                          
        self.p_buffer = np.zeros((self.nBasis, int(self.range_delays[1]/self.dt)))

    # Activates the basis functions and computes the corresponding output, according to the actual weights.
    # Updates the weights for the basis, based on the modified Widrow-Hoff learning rule, or Least Mean Square method (LMS).
    def activate(self, input, error, update=True):

        # Compute output based on bases activity given the new input.
        x = np.dot(self.w_MF.T, input)                                       # Pons relay activity.
        self.z = self.z*self.gammas + self.scaling_input*x.flatten()
        self.p = self.p*self.gammas + self.z                                 # Granule cells activity based on alpha function.
        PC = np.dot(self.w_PF.T, self.p)                                     # Purkinje cells 'dendritic' activity.
        self.PC_buffer = np.column_stack((self.PC_buffer, PC))[:, 1:]        # FIFO-buffer for PC delayed output.
        delayed_PC = self.PC_buffer[np.arange(self.nPC), self.indxs_delays]  # Purkinje cell delayed output.
        delayed_PC = delayed_PC.reshape((self.nPCpop, self.nOutputs))
        DCN = np.sum(delayed_PC, axis=0)                                     # Deep Cerebellar Nucleus output as the sums of PC populations' delayed responses.

        if update:
            # Update PF synapses based on granule eligibility traces and climbing fiber signal (error).
            eligibility_traces = self.p_buffer[:, self.indxs_delays]
            CF = np.repeat(error.reshape((self.nOutputs, 1)), self.nBasis, axis=1).repeat(self.nPCpop, axis=0).T   # Climbing fiber teaching signal. (shape(error)[0]==self.nOutputs).
            self.w_PF += self.beta_PF * CF * eligibility_traces
            #MAX VALOR WEIGHT
            self.w_PF[self.w_PF>1] = 1

            # Update MF synapses based on Oja's rule (stable Hebbian). Right now only the shared basis are updated ('mask_wMF').
            self.w_MF += self.beta_MF * (np.dot(input.reshape((input.shape[0],1)), [self.p]) - (self.p**2)*self.w_MF) * self.mask_wMF
            self.w_MF[self.w_MF<0] = 0

        # Update the buffer of basis' activity to be used for eligibility traces (proxy of dendritic activity in PC).
        self.p_buffer = np.column_stack((self.p_buffer, self.p))[:, 1:]   
        self.basis_f_list.append(self.p)                                         

        return DCN


    # Saves the weights of the Mossy Fibers (MF) and the Parallel Fibers (PF).
    def save_model(self, id=0, path=''):

        np.savetxt(path+'w_MF_'+str(id)+'.npy', self.w_MF)
        np.savetxt(path+'w_PF_'+str(id)+'.npy', self.w_PF)
        np.savetxt(path+'gammas_'+str(id)+'.npy', self.gammas)
        np.savetxt(path+'scaling_input_'+str(id)+'.npy', self.scaling_input)


    # Loads model: weights and time-related constants.
    def load_model(self, id=0, path=''):

        self.w_MF = np.loadtxt(path+'w_MF_'+str(id)+'.npy')
        self.w_PF = np.loadtxt(path+'w_PF_'+str(id)+'.npy')
        self.gammas = np.loadtxt(path+'gammas_'+str(id)+'.npy')
        self.scaling_input = np.loadtxt(path+'scaling_input_'+str(id)+'.npy')


    # Resets the whole module with the default values.
    def reset(self):

        self.__init__(dt=self.dt, nInputs=self.nInputs, nOutputs=self.nOutputs, nIndvBasis=self.nIndvBasis, nSharedBasis=self.nSharedBasis, 
                      beta_PF=self.beta_PF, beta_MF=self.beta_MF, range_delays=self.range_delays, range_TC=self.range_TC, 
                      range_scaling=self.range_scaling, range_W=self.range_W, nPCpop=self.nPCpop) 
    
    def create_signals(self, mean_temp, target_cue):
        
        time = 100         # The total time of a single trial. (All times in seconds).
        dt = 1e-3          # Each iteration of the simulation corresponds to 0.01 seconds.
        
        # CUE (first PREDICTIVE SIGNAL OF THE DISTURBANCE).
        cue = mean_temp
        cue_array = np.array(cue)
        cue_array = cue_array[:int(time/dt)]#episodes
        cue = np.array([np.tile(cue_array, (1,1,1))])
        cue= np.reshape(cue,[1,1,-1,1])
       

        # TARGET.
        target = target_cue
        target = np.array(target)
        target = target[:int(time/dt)] #episodes
        target = np.squeeze(target)
    
        return cue, target
    
    def learn(self, cue, target, i):

        output = self.activate(input = cue[:,:,[i]], error=np.array(self.error[i]))
        self.error.extend(target[i] - output)   # this is the error to be minized by the model, that is, the discrepancy between the target signal and the output signal
        return output
    
    def MSE_homeo(self,target,output):
        self.error.append(target - output)


   
    def save_data_basis_functions(self, current_simulation):
        namefile = f'Results/basisfunctions{current_simulation}.npy'
        np.save(namefile, self.basis_f_list)
