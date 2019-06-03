import numpy as np
import sys
import schottky_lib.CEF_calculations as cef

class NuclearSchottky:
    
    def __init__(self, ion, isotope=None):
        
        self.ion = ion
        self._importHyperfine(ion, isotope)
        isotopeNums = [i[0] for i in self.Isotopes]
        self._importIsotopicComposition(ion, isotopeNums)
        
        
    def _importHyperfine(self, ion, isotope):
        """Import Hyperfine Coupling constants, as listed in Bleanery (1963)"""
        HCCfile = './schottky_lib/HyperfineConstants.txt'
        with open(HCCfile) as f:
            lines = f.readlines()
        
        self.Isotopes = []
        for lnum, line in enumerate(lines):
            if line.startswith(ion):
                line1 = line.split('\t')
                self.ionJ = self._destringify(line1[1])
                iso1num = int(line1[2])
                iso1I = self._destringify(line1[3])
                iso1aJ = self._destringify(line1[4])
                iso1P = self._destringify(line1[5])
                
                self.Isotopes.append([iso1num, iso1I, iso1aJ, iso1P])
                
                if lines[lnum+1].startswith(('\t',' ')):
                    line2 = lines[lnum+1].split('\t')
                    iso2num = int(line2[2])
                    iso2I = self._destringify(line2[3])
                    iso2aJ = self._destringify(line2[4])
                    iso2P = self._destringify(line2[5])
                    self.Isotopes.append([iso2num, iso2I, iso2aJ, iso2P])
                    
        if isotope != None:
            self.Isotopes = [i for i in self.Isotopes if i[0] == isotope]
        
    def _importIsotopicComposition(self, ion, isotopelist):
        """Import isotopic compositions from NIST website"""
        Isotopefile = './schottky_lib/IsotopicCompositions.txt'
        with open(Isotopefile) as f:
            lines = f.readlines()
        
        self.IsotopeComposition = []
        for lnum, line in enumerate(lines):
            if "Atomic Symbol = "+ion[:-2] in line:
                isotope = int(lines[lnum+1].split('=')[1])
                if isotope in isotopelist:
                    self.IsotopeComposition.append( float((lines[lnum+3].split('=')[1]).split('(')[0]) )

                    
    def HeatCapacity(self, orderedmoment, Tarray, quadrupolemoment = 0):
        '''Compute heat capacity based off hyperfine Hamiltonian'''
        kb = 1.38064852e-23  #J/K, Boltzmann constant
        Na = 6.0221409e+23  #Avagadros number
        self.eigenvalues(orderedmoment, quadrupolemoment)
        
        self.T = Tarray
        self.HC = np.zeros(len(Tarray))
        for i, iso in enumerate(self.Isotopes):
            expvals, temps = np.meshgrid(self.Eigenvalues[i], Tarray)
            ZZ = np.sum(np.exp(-expvals/temps/kb), axis=1)
            self.HC += self.IsotopeComposition[i] * Na/(ZZ*kb*Tarray**2) *\
                    (np.sum(expvals**2 * np.exp(-expvals/temps/kb), axis=1) -\
                    1/ZZ * (np.sum(expvals * np.exp(-expvals/temps/kb), axis=1))**2 )
                
                    
                    
    def eigenvalues(self, orderedmoment, quadrupolemoment):
        Jexp = self._jexp(orderedmoment, self.ion)
        h = 6.626070040e-34  #J*s
        
        self.Eigenvalues = []
        for iso in self.Isotopes:            
            aJexp = iso[2]/self.ionJ*Jexp 
            
            Iz = cef.Operator.Jz(J=iso[1])
            H_sch = aJexp*1e6*h * Iz  +  quadrupolemoment*iso[3]*1e6*h* (Iz**2 -(iso[1]+1.)*iso[1]/3. )
            
            self.Eigenvalues.append( np.linalg.eig(H_sch.O)[0])
            
    def Iz(self, orderedmoment, Temp, Field=0):
        Jexp = self._jexp(orderedmoment, self.ion)
        h = 6.626070040e-34  #J*s
        kb = 1.38064852e-23  #J/K, Boltzmann constant
        mu_n = 5.050783699e-27  #J/T, Nuclear Magneton
        
        for iso in self.Isotopes:            
            aJexp = iso[2]/self.ionJ*Jexp 
            
            Iz = cef.Operator.Jz(J=iso[1])
            H_sch = aJexp*1e6*h * Iz  +  iso[3]*1e6*h* (Iz**2 -(iso[1]+1.)*iso[1]/3. ) -\
                    Field*Iz*mu_n
            
            EigVals, EigVecs = np.linalg.eig(H_sch.O)
            ZZ = np.sum(np.exp(-EigVals/Temp/kb))
            
            netIz = 0
            for jj, ev in enumerate(EigVecs.T):
                iizz = np.dot(ev,np.dot(Iz.O,ev))
                netIz += iizz*np.exp(-EigVals[jj]/Temp/kb)/ZZ
            print(netIz)
    
    def _jexp(self, orderedmoment, ion):
        gj = cef.LandeGFactor(ion)
        return orderedmoment/gj
        #return -0.5 + 0.5*np.sqrt(1+ 4*(orderedmoment/gj)**2)

        
    def _destringify(self, string):
        elements = string.split('/')
        try: return float(elements[0]) / float(elements[1])
        except IndexError: return float(elements[0])


def TwoLevelHeatCapacity(splitting, Tarray):
    '''Compute heat capacity based off a two-level system,
    splitting is in units of meV.'''
    kb = 1.38064852e-23  #J/K, Boltzmann constant
    # kb = 8.6173303e-2 #meV/K
    Na = 6.0221409e+23  #Avagadros number
    JpermeV = 1.6021766208e-22
    eigenvalues = np.array([0, splitting]) * JpermeV
    
    T = Tarray
    HC = np.zeros(len(Tarray))

    expvals, temps = np.meshgrid(eigenvalues, Tarray)
    ZZ = np.sum(np.exp(-expvals/temps/kb), axis=1)
    HC += Na/(ZZ*kb*Tarray**2) *\
            (np.sum(expvals**2 * np.exp(-expvals/temps/kb), axis=1) -\
            1/ZZ * (np.sum(expvals * np.exp(-expvals/temps/kb), axis=1))**2 )

    return HC
