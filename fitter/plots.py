import numpy as np 
import matplotlib.pyplot as plt 


def ratio_ploter(average_ratio,N_bins,save_path=None,ellmin=10,ymin=0.95,ymax=1.05,dashed_vert=2*256):
    fig, axes = plt.subplots(N_bins, N_bins, figsize=(14, 14))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    ell = np.arange(average_ratio.shape[2])
    for i in range(N_bins):
        for j in range(N_bins):
            if j > i:
                # Upper triangle: turn off
                axes[i, j].axis('off')
                continue
            ax = axes[i, j]
            ell_plot = ell[ellmin:]
            avg_plot = average_ratio[i, j, ellmin:]
            ax.semilogx(ell_plot, avg_plot, label='Average ratio', lw=0.8, color='blue')
            ax.axhline(1, color='black', linestyle='--', lw=1, alpha=0.5)
            ax.set_xlabel(r'$\ell$', fontsize=11)
            ax.set_ylabel(r'$\langle C_\ell^{\rm mock}/(W_\ell^2C_\ell^{\rm true})\rangle$', fontsize=11)
            # Add bin labels
            if i == j:
                title = f'Bin {i+1}'
            else:
                title = f'Bins ({i+1},{j+1})'
            ax.set_title(title, fontsize=11)
            ax.grid(True, alpha=0.3, which='both')
            ax.set_ylim(ymin,ymax)
            if dashed_vert is not None:
                ax.axvline(dashed_vert,linestyle='dashed',color='black',lw=1, alpha=0.5)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if save_path is None:  
        plt.show()
    plt.close()  

def hist_comparisson_plotter(kappa_sim,kappa_lr_pix,kappa_mock,kappa_mock_pix,N_bins,N,save_path=None):
    fig, axes = plt.subplots(2, N_bins//2, figsize=(10, 10))
    axes = axes.flatten()
    for i in range(N_bins):
        ax = axes[i]
        counts, bins, _ = ax.hist(kappa_sim[i], bins=50, histtype='step',
                                linewidth=2.5, label='Simulation', color='black')
        max_count = np.max(counts[counts > 0])
        ax.hist(kappa_lr_pix[i], bins=bins, histtype='step',
        linewidth=2, label=f'Pix + LP sim', color='red',
        linestyle='solid')
        ax.hist(kappa_mock[i], bins=bins, histtype='step',
            linewidth=2, label=f'G{N} mock', color='blue',
            linestyle='dashed')
        ax.hist(kappa_mock_pix[i], bins=bins, histtype='step',
            linewidth=2, label=f'G{N} mock Pix + LP', color='orange',
            linestyle='dashed')
        ax.set_yscale('log')
        ax.set_ylim(1e-1, None)
        ax.set_xlabel(r'$\kappa$', fontsize=16)
        ax.set_ylabel('Density', fontsize=16)
        ax.set_title(f'Z-bin {i+1}', fontsize=18, pad=15)
        if i == 0:
            ax.legend(fontsize=13, framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=13)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if save_path is None:  
        plt.show()
    plt.close()  

def hist_comparisson_plotter_linear(kappa_sim,kappa_lr_pix,kappa_mock,kappa_mock_pix,N_bins,N,save_path=None):
    fig, axes = plt.subplots(2, N_bins//2, figsize=(10, 10))
    axes = axes.flatten()
    for i in range(N_bins):
        ax = axes[i]
        if i in [2,3]:
            counts, bins, _ = ax.hist(kappa_sim[i], bins=50, histtype='step',
                                    linewidth=2.5, label='Simulation', color='black',range=(kappa_sim[i].min(),0.018))
        else: 
            counts, bins, _ = ax.hist(kappa_sim[i], bins=50, histtype='step',
                            linewidth=2.5, label='Simulation', color='black',range=(kappa_sim[i].min(),0.01))
        ax.hist(kappa_lr_pix[i], bins=bins, histtype='step',
        linewidth=2, label=f'Pix + LP sim', color='red',
        linestyle='solid')
        ax.hist(kappa_mock[i], bins=bins, histtype='step',
            linewidth=2, label=f'G{N} mock', color='blue',
            linestyle='dashed')
        ax.hist(kappa_mock_pix[i], bins=bins, histtype='step',
            linewidth=2, label=f'G{N} mock Pix + LP', color='orange',
            linestyle='dashed')
        ax.set_ylim(1e-1, None)
        ax.set_xlabel(r'$\kappa$', fontsize=16)
        ax.set_ylabel('Density', fontsize=16)
        ax.set_title(f'Z-bin {i+1}', fontsize=18, pad=15)
        if i == 0:
            ax.legend(fontsize=13, framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=13)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if save_path is None:  
        plt.show()
    plt.close()  

def ratio_ploter_comp(average_ratio,average_ratio_filt,N_bins,save_path=None,ellmin=10,ymin=0.95,ymax=1.05,dashed_vert=2*256):
    fig, axes = plt.subplots(N_bins, N_bins, figsize=(14, 14))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    ell = np.arange(average_ratio.shape[2])
    for i in range(N_bins):
        for j in range(N_bins):
            if j > i:
                # Upper triangle: turn off
                axes[i, j].axis('off')
                continue
            ax = axes[i, j]
            ell_plot = ell[ellmin:]
            
            avg_plot = average_ratio[i, j, ellmin:]
            avg_plot_filt = average_ratio_filt[i, j, ellmin:]
            ax.semilogx(ell_plot, avg_plot, label='Average ratio', lw=0.8, color='blue')
            ax.semilogx(ell_plot, avg_plot_filt, label='Average ratio filtered', lw=0.8, color='red')
            ax.axhline(1, color='black', linestyle='--', lw=1, alpha=0.5)
            ax.set_xlabel(r'$\ell$', fontsize=11)
            ax.set_ylabel(r'$\langle C_\ell^{\rm mock}/(W_\ell^2C_\ell^{\rm true})\rangle$', fontsize=11)
            # Add bin labels
            if i == j:
                title = f'Bin {i+1}'
            else:
                title = f'Bins ({i+1},{j+1})'
            ax.set_title(title, fontsize=11)
            ax.grid(True, alpha=0.3, which='both')
            ax.set_ylim(ymin,ymax)
            if dashed_vert is not None:
                ax.axvline(dashed_vert,linestyle='dashed',color='black',lw=1, alpha=0.5)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if save_path is None:  
        plt.show()
    plt.close()  