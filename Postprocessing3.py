import matplotlib.pyplot as plt
import numpy as np
import glob, os, sys
import copy
import matplotlib as mpl

pgf_with_custom_preamble = {
    "pgf.texsystem": "lualatex",
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False,    # don't setup fonts from rc parameters
    "axes.labelsize": 16,
    "font.size": 18,
    "legend.fontsize": 16,
    "axes.titlesize": 16,           # Title size when one figure
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "figure.titlesize": 18,         # Overall figure title
    "pgf.preamble": 
         r'\usepackage{fontspec}'
         r'\usepackage{units}'         # load additional packages
         r'\usepackage{metalogo}'
         r'\usepackage{unicode-math}'  # unicode math setup
         r'\setmathfont{XITS Math}'
         r'\setmonofont{Libertinus Mono}'
         r'\setmainfont{Libertinus Serif}' # serif font via preamble
         
}
mpl.rcParams.update(pgf_with_custom_preamble)

class Dist:
    """
    This class load the data given a filename
    and gives the possibility to generate a plot with the uploaded data
    """
    def __init__(self, filename, is_avoid_zeros=True):
        # It is better to make general x,y arrays
        
        if not os.path.isfile(filename):
            filename='%s%s' %(filename.split(".txt",1)[0],".TXT")
            if not os.path.isfile(filename):
                print("%s file do not exists" % (filename))
                self.rho=[0]
            else:
                self.rho = np.loadtxt(filename, comments="#")
                
        else:
            self.rho = np.loadtxt(filename, comments="#")

def main():
    tau=1.4
    mainDir="W:\\Git\\NPDiffusion\\"
    timeInstants=[1000,2000,3000]
    #timeInstants=[240,2880,10080]
    datas=[]
    for ti in timeInstants:
        datas.append(Dist(f"{mainDir}Conc_tau{tau}_t{ti:04d}.txt"))
    X, Y = np.meshgrid(range(100), range(300))
    fig, axes=plt.subplots(len(timeInstants),2,figsize=(6, 9.3))
    ylim=(0,10)
    for ti, data, ax in zip(timeInstants,datas,axes):
        cmap = copy.copy(mpl.cm.get_cmap("bwr"))
        cmap.set_bad('black')
        contours = ax[1].contour(X, Y, data.rho, levels=[1.e-4,1], colors='black')
        im=ax[1].imshow(data.rho, cmap=cmap,clim=ylim)
        ax[1].clabel(contours, inline=True, fontsize=16)
    
    
    #plt.clim(0,15)
        ax[1].set_ylim([300,0])
        ax[1].invert_yaxis()
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)	
        ax[1].set_aspect('equal')	
        fig.colorbar(im,ax=ax[1], shrink=0.9,pad=0.05)
        ax[0].plot((Y[:,50])*0.1,data.rho[:,50])
        ax[0].set_ylim(ylim)

        asp = np.diff(ax[0].get_xlim())[0] / np.diff(ax[0].get_ylim())[0]
        asp /= np.abs(np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0])
        ax[0].set_xticks(np.arange(min((Y[:,50])*0.1), max((Y[:,50])*0.1)+1, 5))
        #ax[0].set_aspect(asp)
        ax[0].set_ylabel("C mg/mL")
        ax[0].set_xlabel("r mm")
        #ax[0].set_title(f"time = {int(ti/1000)} week")
        ax[0].set_title(f"time = {int(ti/60)} min")
        if int(ti/60)>48:
            ax[0].set_title(f"time = {int(ti/60/24)} days")
    fig.suptitle(f'$ \mathrm{{D}} = 3.3e^{{-11}} \mathrm{{m}}^2/\mathrm{{s}}$', fontsize=20)
    fig.tight_layout(pad=0.6, w_pad=0.0001, h_pad=0.1)
    
    fig.savefig(f"Conc_tau{tau}flow_pres.png",dpi=1000)
    
    plt.show()
    return 0

if __name__== "__main__":
  main()