import matplotlib.pyplot as plt
import numpy as np

def main():
	""" Finite Volume simulation """
	
	# Simulation parameters
	Nx                     = 100    # resolution x-dir
	Ny                     = 400    # resolution y-dir
	rho0                   = 0    # average density outside injection site
	rho1				   = 20	  # average density initial injection
	tau                    = 0.509    # collision timescale
	Nt                     = 18000   # number of timesteps
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

	spot = (X - Nx/2)**2 + (Y - Ny/2)**2 < (Nx/40)**2
	for i in idxs:
		F[spot,i] = rho1*weights[i]#*F[cylinder,i]
	
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
			Feq[:,:,i] = rho * w #* ( 1 + 3*(cx*ux+cy*uy)  + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2 )
		
		F += -(1.0/tau) * (F - Feq)
		
		# Apply boundary 
		F[reflectiveBoundary,:] = bndryF
		#F[np.logical_and(cylinder,reflective),:]= bndryF
		
		
		# plot in real time - color 1/2 particles blue, other half red
		if (plotRealTime and (it % 100) == 0) or (it == Nt-1):
			plt.cla()
			rho = np.sum(F,2)
			rho[reflectiveBoundary] = np.nan
			cmap = plt.cm.bwr
			cmap.set_bad('black')
			plt.imshow(rho, cmap='bwr')
			plt.clim(0,15)
			ax = plt.gca()
			ax.invert_yaxis()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)	
			ax.set_aspect('equal')	
			plt.pause(0.0005)
	
	# Save figure
	plt.savefig('latticeboltzmann.png',dpi=240)
	plt.show()
	    
	return 0


if __name__== "__main__":
  main()