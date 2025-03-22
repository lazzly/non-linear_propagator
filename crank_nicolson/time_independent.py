import numpy as np

class Grid:
    def __init__(self):
        pass

    def generate_radial_grid(self, N = 1024, r_min = 0, r_max = 0.2):
        # radia grid parameters
        self.N_perpendicular = N # number of inner gid points in r direction (+2 boundry points)
        self.r_min = r_min         # domain radius
        self.r_max = r_max       # 1e-9 # may be > 0 for stability?
        
        self.Delta_r = (r_max - r_min) / self.N_perpendicular
        self.r = np.linspace(r_min, r_max, self.N_perpendicular + 1, dtype=np.clongdouble)

    def generate_propagation_grid(self, N = 1024*4-1, z_0 = 0, z_max = 0.4):
        # propagation grid parameters
        self.z_0 = z_0
        self.z_max = z_max # mm
        self.N_z = N
        self.Delta_z = (z_max - z_0) / N
        self.z = np.linspace(z_0, z_max, N + 1, dtype=np.clongdouble)


class Propagator:
    def __init__(self, grid):
        self.grid = grid
        pass

    def set_initial_conditions(self, omega_0,lamb, f, n0):
        self.omega_0 = 0.01 # beam waist
        self.lamb = 1030e-6 # wavelength
        self.f = 0.2 # focal length of lense
        self.n0 = 1.5 # refrective index
        self.k0 = n0 * 2 * np.pi / lamb

        self.generate_L_matrices()
        self.apply_boundry_to_L_matrices()
        self.invert_L_minus()
        self.generate_L_matrix()

        # initial Field 
        E_0 = 1
        self.E_init = E_0 * np.exp(- self.grid.r**2 / self.omega_0**2 - 1j * self.k0 * self.grid.r**2 / (2*self.f))

    def check_stability(self):
        # stability?
        print('stable when Delta_z << k0*Delta_r**2',
              '\n          %.2Ef << %.2E'%(self.grid.Delta_z, self.k0*self.grid.Delta_r**2),
              '\nThats a factor of %i'%(1/(self.grid.Delta_z / (self.k0*self.grid.Delta_r**2))))
    
    def generate_Delta_j(self, nu = 1):
        # nu = 1 for radial coordinates, nu = 0 for euclidean
        N = self.grid.N_perpendicular
        matrix = np.zeros([N+1, N+1])
        
        for j in range(1, N): #do not set the first and last row since they be overwritten by boundry conditions    
            u_j = 1 - nu/(2*j)
            v_j = 1 + nu/(2*j)    
            matrix[j, j] = -2
            matrix[j, j-1] = u_j
            matrix[j, j+1] = v_j
        return matrix

    def generate_L_matrices(self):
        # calculate L matrices
        Delta_j = self.generate_Delta_j()
        self.delta = self.grid.Delta_z /(4. * self.k0 * self.grid.Delta_r**2)
        self.L_plus = np.identity(self.grid.N_perpendicular+1) + (1j * self.delta * Delta_j)
        self.L_minus = np.identity(self.grid.N_perpendicular+1) - (1j * self.delta * Delta_j)

    def apply_boundry_to_L_matrices(self, nu=1):
        # boundary conditions
        self.L_plus[0,0] = 1 - (4 * 1j * self.delta)
        self.L_plus[0,1] = (4 * 1j * self.delta)
    
        self.L_minus[0,0] =  1 + (4 * 1j * self.delta)
        self.L_minus[0,1] =  (-4 * 1j * self.delta)
        self.L_minus[-1,-1] = 1

    def invert_L_minus(self):
        self.L_minus_inv = np.linalg.inv(self.L_minus)

    def generate_L_matrix(self):
        self.L = np.matmul(self.L_minus_inv, self.L_plus)

    def propagate(self):
        self.E_final = np.zeros([self.grid.N_perpendicular + 1, self.grid.N_z], dtype=np.clongdouble)
        self.E_final[:,0] = self.E_init
        
        for n in range(1, self.grid.N_z):
            self.E_final[:,n] = np.dot(self.L, self.E_final[:,n-1])

        self.I_final = np.abs(self.E_final)**2





        