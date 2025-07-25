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
        
        

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    mol = gto.M(
        atom='''
        H 0.000000 0.000000 0.000000
        H 0.000000 0.000000 1.500000
        ''',
        basis='6-311g**',
        unit="Bohr",
        charge = 1,
        spin = 1
    )
    mf = scf.RHF(mol)
    
    mf.kernel()
    print (mf.mo_energy)
    rgf = RGF(mf)
    grid = np.asarray([[0.0, 0.0, 1.5]])  # Example grid points reshaped to 2D
    mo_grid = rgf.get_MO_grid(grid)
    print("Molecular Orbitals on Grid Points:", mo_grid)


    grid1 = np.asarray([[0.0, 0.0, 10.0]])  # Grid for first set of orbitals
    print (rgf.get_GF_grid(grid1, grid1, tau=0.1))
    print (rgf.get_GF_grid(grid, grid, tau=0.1))
    print (rgf.get_GF_grid(grid1, grid, tau=0.1))
    
    # Example usage of get_GF_grid
    GF=[]
    
    for t in range(0,100):
        GF.append(rgf.get_GF_grid(grid1, grid, tau=0.1*t))
    plt.plot(np.abs(GF))
    plt.show()
