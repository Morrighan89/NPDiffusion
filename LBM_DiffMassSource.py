import matplotlib.pyplot as plt
import numpy as np

def main():
    """ Finite Volume simulation """
    
    # Simulation parameters
    Nx                     = 100    # resolution x-dir
    Ny                     = 300    # resolution y-dir
    rho0                   = 0.1e-12    # average density outside injection site
    rho1                   = 10   # average density initial injection
    eps                    = 1 # porosity
    l                      = 1 # tissue tortuosity
    dt                     = 2 #time step
    dx                     = 1.e-4 # m
    lspeed                  = 1480 #sound speed
    Diff                   = 5.e-11 # Diffusivity
    DiffStar               = Diff/l**2
    tau                    = 3*DiffStar*dt/dx**2 +0.5 # collision timescale
    Source                 = 28.9 #Mass Source mg/dx3
    timeInj                = 1200 # time in seconds
    ntInj                  = int(timeInj//dt)
    #tau                    = 0.590   # collision timescale D 3.3e-11
    Nt                     =  4300  # number of timesteps dt 60 s
    plotRealTime = True # switch on for plotting as the simulation goes along
    
    # Lattice speeds / weights
    NL = 9
    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
    cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
    weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36]) # sums to 1
    
    # Initial Conditions
    F = np.ones((Ny,Nx,NL)) #* rho0 / NL
    X, Y = np.meshgrid(range(Nx), range(Ny))
    rho = np.sum(F,2)
    for i in idxs:
        F[:,:,i] = rho0*weights[i]

    # Initial injection site
    X, Y = np.meshgrid(range(Nx), range(Ny))

    spot = (X - Nx/2)**2 + (Y - Ny/2)**2 < 0.1**2
    #for i in idxs:
    #    F[spot,i] = F[spot,i] + weights[i]*dt*Source/eps*(tau-0.5)/tau #*F[cylinder,i]
    
    # Set Boundaries
    top= Y==Ny-1
    bottom= Y==0
    left= X==0
    right= X==Nx-1
    reflectiveBoundary=np.logical_or(left,np.logical_or(right,np.logical_or(top,bottom)))
    # Prep figure
    fig = plt.figure(figsize=(4,2), dpi=80)
    
    # Simulation Main Loop
    for it in range(Nt):
        print(it)
        
        # Drift
        for i, cx, cy in zip(idxs, cxs, cys):
            F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
            F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)
        
        
        # Set reflective boundaries
        bndryF = F[reflectiveBoundary,:]
        bndryF = bndryF[:,[0,5,6,7,8,1,2,3,4]]
        

        # Calculate fluid variables
        rho = np.sum(F,2)
        ux  = np.sum(F*cxs,2) / rho
        uy  = np.sum(F*cys,2) / rho
        
        
        # Apply Collision
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Feq[:,:,i] = rho * w * ( 1 + 3*(cx*ux+cy*uy)/lspeed**2  + 9*(cx*ux+cy*uy)**2/(2*lspeed**4) - 3*(ux**2+uy**2)/(2*lspeed**4) )
        
        F += -(1.0/tau) * (F - Feq)
        
        # Apply boundary 
        F[reflectiveBoundary,:] = bndryF
        #F[np.logical_and(cylinder,reflective),:]= bndryF
        if it < ntInj:
            for i in idxs:
                F[spot,i] = F[spot,i] + weights[i]*dt*Source#/eps*(tau-0.5)/tau
        
        # plot in real time - color 1/2 particles blue, other half red
        if (plotRealTime and (it % 300) == 0) or (it == Nt-1):
            plt.cla()
            rho = np.sum(F,2)
            rho[reflectiveBoundary] = np.nan
            cmap = plt.cm.bwr
            cmap.set_bad('black')
            contours = plt.contour(X, Y, rho, levels=[1.e-4,1.e-2,1.e-1,1], colors='black')
            plt.clabel(contours, inline=True, fontsize=18)
            plt.imshow(rho, cmap='bwr')
            np.savetxt(f"Conc_tau{tau}_t{it:04d}.txt",rho)
            #plt.clim(0,15)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)   
            ax.set_aspect('equal')  
            plt.pause(0.0005)
            print("plot")
    
    # Save figure
    plt.colorbar()
    plt.savefig('latticeboltzmann.png',dpi=240)
    plt.show()
        
    return 0


if __name__== "__main__":
  main()