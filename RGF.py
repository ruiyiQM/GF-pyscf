from pyscf import gto, scf, dft
import numpy as np
class RGF:
    def __init__(self, mf):
        self.mf = mf
        self.mol = mf.mol

        self.mo_coeff = mf.mo_coeff.copy() if hasattr(mf, 'mo_coeff') else None
        self.mo_energy = mf.mo_energy.copy() if hasattr(mf, 'mo_energy') else None
        self.mo_occ = mf.mo_occ.copy() if hasattr(mf, 'mo_occ') else None


    def get_MO_grid(self, grid):

        GTOval = 'GTOval'        
        ao = self.mol.eval_gto(GTOval, grid)
        orb_on_grid = np.dot(ao, self.mo_coeff)
        return orb_on_grid.flatten()

    def get_MO_grid_grad(self, grid):
        GTOval = 'GTOval_sph_deriv1'        
        ao = self.mol.eval_gto(GTOval, grid)[1:]
        orb_on_grid = np.dot(ao, self.mo_coeff)
        return orb_on_grid.reshape(3,-1)
    
    def get_MO_grid_value_and_grad(self, grid):
        GTOval = 'GTOval_sph_deriv1'        
        ao = self.mol.eval_gto(GTOval, grid)
        orb_on_grid = np.dot(ao, self.mo_coeff)
        return orb_on_grid.reshape(4,-1)
    
    def get_GF_grid(self, grid1, grad2, tau = 0.0, index = None ):

        mo_grid1 = self.get_MO_grid(grid1)
        mo_grid2 = self.get_MO_grid(grad2)

        gf_on_grid = 0.0

        self.ef = self.mf.get_fermi_level() if hasattr(self.mf, 'get_fermi_level') else 0.0

        if index is None:
            index = np.arange(self.mo_occ.shape[0]) 
        for i in index:
            if self.mo_occ[i] > 0:
                gf_on_grid += np.exp(-(self.mo_energy[i]) * tau) * mo_grid1[i] * mo_grid2 [i] 
            if self.mo_occ[i] == 0:
                gf_on_grid += np.exp(-(self.mo_energy[i])* tau) * mo_grid1[i] * mo_grid2[i]
        return gf_on_grid
        
    def get_GF_value_and_grad(self, grid1, grad2, tau = 0.0, index = None ):



        mo_grid1 = self.get_MO_grid_value_and_grad(grid1)
        mo_grid2 = self.get_MO_grid(grad2)    
        if index is None:
            index = np.arange(self.mo_occ.shape[0]) 
        factor = np.exp(-(self.mo_energy) * tau)

        gf = np.einsum('j,ij,j->i',factor[index], mo_grid1[:,index].conj(), mo_grid2[index])
    
        return gf[0], gf[1:]

    def get_GF_value_tau_grad(self, grid1, grad2, tau = 0.0, index = None ):

        mo_grid1 = self.get_MO_grid(grid1)
        mo_grid2 = self.get_MO_grid(grad2)
        if index is None:
            index = np.arange(self.mo_occ.shape[0]) 
        factor = np.exp(-(self.mo_energy) * tau) * (-self.mo_energy)
        gf_grad =np.dot (factor[index] * mo_grid1[index].conj(), mo_grid2[index])
        return gf_grad 
    
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    mol = gto.M(
        atom='''
        H 0.000000 0.000000 0.000000
        H 0.000000 0.000000 1.500000
        ''',
        basis='aug-ccpvdz',
        unit="Bohr",
        charge = 1,
        spin = 1
    )
    mf = scf.RHF(mol)
    
    mf.kernel()
    rgf = RGF(mf)
    grid = np.asarray([[0.0, 0.0, 1.5]])  # Example grid points reshaped to 2D
    grid1 = np.asarray([[0.0, 0.0, 1.6]])  # Example grid points reshaped to 2D
    mo_grid = rgf.get_MO_grid(grid)
    print("Molecular Orbitals on Grid Points:", mo_grid.shape)

    mo_grid_grad = rgf.get_MO_grid_grad(grid)
    print("Molecular Orbitals Gradient on Grid Points:", mo_grid_grad.shape)


    print (rgf.get_GF_grid(grid, grid1, tau = 0.0))
    gf_value, gf_grad = rgf.get_GF_value_and_grad(grid, grid1, tau = 0.0)
    print("GF Value on Grid Points:", gf_value, gf_grad)

    tau_grad = rgf.get_GF_value_tau_grad(grid, grid1, tau = 0.0)
    print ("GF Value and Gradient with Tau:", tau_grad)