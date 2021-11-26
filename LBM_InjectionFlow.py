import matplotlib.pyplot as plt
import numpy as np
import copy
import matplotlib as mpl

def main():
    """ Finite Volume simulation """
    
    # Simulation parameters
    Nx                     = 100    # resolution x-dir
    Ny                     = 300    # resolution y-dir
    rho0                   = 1000    # average density outside injection site
    rho1                   = 1000  # average density initial injection
    tau                    = 0.6  # collision timescale D 5.e-11
    Nt                     = 12000   # number of timesteps dt 60 s
    plotRealTime = True # switch on for plotting as the simulation goes along
    
    # Lattice speeds / weights
    NL = 9
    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
    cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
    weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36]) # sums to 1
    inlet=np.array([4/9,11/180,1/36,1/9,1/36,29/180,1/36,1/9,1/36])
    # Initial Conditions
    F = np.ones((Ny,Nx,NL)) #* rho0 / NL
    X, Y = np.meshgrid(range(Nx), range(Ny))
    rho = np.sum(F,2)
    for i in idxs:
        F[:,:,i] = rho0*weights[i]

    # Initial injection site
    X, Y = np.meshgrid(range(Nx), range(Ny))

 
    
    # Set Boundaries
    top= Y==Ny-1
    slit = np.logical_and(top,np.logical_and(X<Nx/2+2,X>Nx/2-3))
    bottom= Y==0
    left= X==0
    right= X==Nx-1
    reflectiveBoundary=np.logical_or(left,np.logical_or(right,np.logical_or(np.logical_xor(top,slit),bottom)))
    # Prep figure
    fig = plt.figure(figsize=(4,2), dpi=80)
    for i in idxs:
        F[slit,i] = rho1*inlet[i]
    # Simulation Main Loop
    fig, axes=plt.subplots(1,2)
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
            Feq[:,:,i] = rho * w * ( 1 + 3*(cx*ux+cy*uy)  + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2 )
        
        F += -(1.0/tau) * (F - Feq)
        
        # Apply boundary 
        F[reflectiveBoundary,:] = bndryF
        if it < 1200:
            for i in idxs:
                F[slit,i] = rho1*inlet[i]
        #F[np.logical_and(cylinder,reflective),:]= bndryF

        
        # plot in real time - color 1/2 particles blue, other half red
        if (plotRealTime and (it % 100) == 0) or (it == Nt-1):
            axes[0].clear()
            axes[1].clear()
            
            rho = np.sum(F,2)
            rho[reflectiveBoundary] = np.nan
            velocity=np.sqrt(np.square(ux)+np.square(uy))
            cmap = copy.copy(mpl.cm.get_cmap("bwr"))
            cmap.set_bad('black')
            
            cmap = copy.copy(mpl.cm.get_cmap("bwr"))
            cmap.set_bad('black')
            contours = axes[1].contour(X, Y, rho, levels=[1000,1010], colors='black')
            im=axes[1].imshow(rho, cmap=cmap)
            axes[1].clabel(contours, inline=True, fontsize=16)
        
        
            axes[1].set_ylim([300,0])
            axes[1].invert_yaxis()
            axes[1].get_xaxis().set_visible(False)
            axes[1].get_yaxis().set_visible(False)	
            axes[1].set_aspect('equal')	

            qv=axes[0].quiver(X,Y,ux,uy,velocity)
            axes[0].set_ylim([300,0])
            axes[0].invert_yaxis()
            axes[0].get_xaxis().set_visible(False)
            axes[0].get_yaxis().set_visible(False)	
            axes[0].set_aspect('equal')	

            #fig.savefig(f"flow{it}.png",dpi=1000)

            plt.pause(0.0005)
            print("plot")
    
    # Save figure
    fig.colorbar(qv,ax=axes[0])
    fig.savefig('latticeboltzmann.png',dpi=240)
    #plt.show()
        
    return 0


if __name__== "__main__":
  main()