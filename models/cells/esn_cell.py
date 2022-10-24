import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
import copy

class ESNCell():
    def __init__(self, reservoir_size, radius, sparsity, sigma_input, W_scaling=1, flip_sign=False, resample=True):
        
        self.reservoir_size = reservoir_size
        self.W_scaling = W_scaling
        self.flip_sign = flip_sign
        self.reservoir_size = reservoir_size
        self.radius = radius
        self.sparsity = sparsity
        self.sigma_input = sigma_input
        
        self.resample()
        self.h = np.zeros((self.reservoir_size, 1))


    def _getHiddenMatrix(self):
        
        success = False
        counter = 5
        
        while not success and counter>=0:
            
            try:
                W = sparse.random(self.reservoir_size, self.reservoir_size, density=self.sparsity)
                W *= self.W_scaling 

                eigenvalues, eigvectors = splinalg.eigs(W)
                eigenvalues = np.abs(eigenvalues)

                W = (W / np.max(eigenvalues)) * self.radius
                success = True
                
                if self.flip_sign:
                    W *= (np.random.binomial(1, 0.5, (self.reservoir_size, self.reservoir_size)) - 0.5)*2

            except:
                sucess = False
                counter -= 1
        
        if counter < 0:
            
            print("-------------------- BIG CONVERGENCE PROBLEMS -----------------")
            
        return W

    def _getInputMatrx(self):

        W_in = np.zeros((self.reservoir_size, 1))
        q = int(self.reservoir_size / 1)
        for i in range(0, 1):
            W_in[i * q:(i + 1) * q, i] = self.sigma_input * (-1 + 2 * np.random.rand(q))
        return W_in
    
    def resample(self):
        self.W_h  = self._getHiddenMatrix()
        self.W_in = self._getInputMatrx()
        
    def fix_weights(self, W_in, W_h):
        self.W_h = W_h
        self.W_in = W_in
        
    def forward(self, input):
        i = np.reshape(input, (-1, 1))
        self.h = np.tanh(self.W_h @ self.h + self.W_in @ i)
        return self.h

    def reset(self):
        self.h = np.zeros((self.reservoir_size, 1))
        
  

    '''    
    def sample_set_hyperparams(self):
        
        self.reservoir_size = random.choice(RESERVOIR_SIZES)
        self.radius = random.choice(RADII)
        self.sparsity = random.choice(SPARSITIES)
        self.dynamic_fit_ratio = random.choice(DYNAMICS_FIT_RATIOS)
        self.W_scaling = random.choice(W_SCALINGS)
        self.flip_sign = random.choice(FLIP_SIGN)
        self.solver = random.choice(SOLVERS)
        
        return {"reservoir_size": self.reservoir_size, "activation_func":self.activation_func, "radius":self.radius, "sparsity":self.sparsity, "solver":self.solver, "dynamic_fit_ratio":self.dynamic_fit_ratio, "W_scaling":self.W_scaling, "flip_sign":self.flip_sign}
    
    def set_hyperparams(self, hyperparams):
        
        self.reservoir_size = hyperparams['reservoir_size']
        self.radius = hyperparams['radius']
        self.sparsity = hyperparams['sparsity']
        self.dynamic_fit_ratio = hyperparams['dynamic_fit_ratio']
        self.W_scaling = hyperparams['W_scaling']
        self.flip_sign = hyperparams['flip_sign']
        self.solver = hyperparams['solver']
   '''     
