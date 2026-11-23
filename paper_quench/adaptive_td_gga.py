"""Schematic of adaptive td-gGA (paper_quench Fig. 1). B counts the orbitals per spin of the quasiparticle space,
one standard orbital as in GA and B - 1 ghost orbitals (the bath orbitals of the embedding problem), excluding the physical
impurity orbital. Insertion of a ghost pair when the population n_g of the outermost pair reaches the threshold n_ins.
All occupation curves and event times are schematic, not computed td-gGA data.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11*0.66,'svg.fonttype':'none','mathtext.fontset':'cm','axes.linewidth':.6,'pdf.fonttype':42})
blue='#0072B2'; orange='#E69F00'; teal='#009E73'; ink='#243B53'; grey='#8B98A6'; coral='#D55E00'; bath='#00395f'   # Okabe-Ito, as in plot_figs.py
FS, LW = 0.66, 0.55                               # font and line scale for the one-column size
fig=plt.figure(figsize=(3.4,2.75),facecolor='white')   # one column (3.4 in), printed at its nominal size
a=fig.add_axes([.035,.555,.93,.405]); a.set_xlim(0,11.6); a.set_ylim(-.30,2.85); a.set_aspect('equal'); a.axis('off')
# Compact key; the physical site has no fixed-occupation symbol.
for x,label,color in [(0.55,'Physical',coral),(4.25,'Standard bath',bath),(8.45,'Ghost',blue)]:
    a.add_patch(Circle((x,2.70),.080,fc=color,ec=color))
    a.text(x+.17,2.70,label,va='center',fontsize=FS*10.5,color=ink)

def cluster(cx,count,added=False):
    cy=.40; radius=1.30
    angles=[90] if count==1 else np.linspace(168,12,count)
    for j,theta in enumerate(angles):
        rad=np.deg2rad(theta); x=cx+radius*np.cos(rad); y=cy+radius*np.sin(rad)
        new=added and j in [0,count-1]
        isbath=j==count//2
        color=orange if new else bath if isbath else blue
        a.plot([cx,x],[cy,y],color=orange if new else '#ACBFCE',alpha=.85,lw=LW*1.5 if new else 1.05,zorder=1)
        if new:
            a.add_patch(Circle((x,y),.23,fc=orange,ec='none',alpha=.11,zorder=2))
        a.add_patch(Circle((x,y),.135,fc='white' if new else bath if isbath else '#D9ECF5',ec=color,lw=LW*1.8,ls='--' if new else '-',zorder=3))
    a.add_patch(Circle((cx,cy),.27,fc=coral,ec='none',alpha=.10,zorder=2))
    a.add_patch(Circle((cx,cy),.205,fc=coral,ec='white',lw=LW*1.4,zorder=4))

cluster(1.45,1); cluster(5.35,5); cluster(9.60,7,True)
for x,label in [(1.45,r'$B=1$'),(5.35,r'$B$'),(9.60,r'$B+2$')]:
    a.text(x,-.10,label,ha='center',va='top',fontsize=FS*13,color=ink)
a.text(1.45,1.99,r'$t=0$',ha='center',fontsize=FS*10,color=grey)
for p,q,c in [((2.18,.79),(3.65,.79),grey),((6.95,.79),(7.99,.79),orange)]:
    a.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=FS*15,lw=LW*1.7,color=c))
a.text(2.92,1.05,r'$\cdots$',ha='center',fontsize=FS*19,color=grey)
a.text(7.48,1.26,r'$n_{\mathrm{g}}=n_{\mathrm{ins}}$',ha='center',fontsize=FS*12,color=orange)

ax=fig.add_axes([.12,.255,.82,.275]); bx=fig.add_axes([.12,.095,.82,.105],sharex=ax)
t1,t2=.43,.76; threshold=.60
for lo,hi,color in [(0,t1,blue),(t1,t2,orange),(t2,1,teal)]:
    x=np.linspace(lo,hi,160); end=threshold if hi<1 else .29
    y=.030+(end-.030)*((x-lo)/(hi-lo))**2
    ax.axvspan(lo,hi,color=color,alpha=.035,lw=LW*0)
    ax.fill_between(x,0,y,color=color,alpha=.085,lw=LW*0)
    ax.plot(x,y,color=color,lw=LW*2.5,solid_capstyle='round')
    if hi<1:
        xx=np.linspace(hi,min(1,hi+.16),60)
        ax.plot(xx,threshold+1.0*(xx-hi),color=color,lw=LW*1.25,ls=':',alpha=.60)
        ax.scatter([hi],[threshold],s=LW**2*80,color=color,alpha=.13,edgecolor='none',zorder=5)
        ax.scatter([hi],[threshold],s=LW**2*29,color=color,zorder=6)
ax.axhline(threshold,color=grey,lw=LW*1,ls=(0,(4,3)))
ax.text(1.018,threshold,r'$n_{\mathrm{ins}}$',va='center',color=grey,fontsize=FS*13)
for xpos,color in [(t1,orange),(t2,teal)]:
    for axis in [ax,bx]: axis.axvline(xpos,color=color,ls=(0,(3,3)),lw=LW*1,alpha=.58)
    ax.text(xpos+.025,.84,'+',color=color,fontsize=FS*13,va='center')
    ax.plot([xpos+.075,xpos+.11],[.84,.84],ls='',marker='o',ms=LW*5.7,mfc='white',mec=color,mew=LW*1.4)
ax.set_xlim(-.013,1.013); ax.set_ylim(0,.99)
ax.set_ylabel(r'$n_{\mathrm{g}}(t)$',labelpad=6,fontsize=FS*13,color=ink)
ax.set_yticks([0]); ax.tick_params(axis='x',bottom=False,labelbottom=False)
for lo,hi,height,color in [(0,t1,0,blue),(t1,t2,1,orange),(t2,1,2,teal)]:
    bx.fill_between([lo,hi],-.20,height,color=color,alpha=.08,lw=LW*0)
    bx.plot([lo,hi],[height,height],color=color,lw=LW*2.8,solid_capstyle='butt')
bx.plot([t1,t1],[0,1],color=orange,lw=LW*2.3)
bx.plot([t2,t2],[1,2],color=teal,lw=LW*2.3)
bx.set_ylim(-.22,2.55)
bx.set_yticks([0,1,2]); bx.set_yticklabels([r'$B$',r'$B+2$',r'$B+4$'],fontsize=FS*10)
bx.set_xticks([t1,t2]); bx.set_xticklabels([r'$\tau_1$',r'$\tau_2$'])   # insertion times (t_1 is the first minimum in Fig. 3)
bx.set_xlabel(r'$t$',labelpad=-3,x=1.025,fontsize=FS*13,color=ink)
for axes in [ax,bx]:
    axes.spines[['top','right']].set_visible(False)
    axes.spines[['left','bottom']].set_color(ink)
    axes.tick_params(length=2,width=0.6,color=ink,labelcolor=ink)
fig.text(.012,.955,'(a)',fontsize=FS*12,color=ink,va='top')
fig.text(.012,.535,'(b)',fontsize=FS*12,color=ink,va='top')
for ext in ['svg','pdf','png']:
    fig.savefig(OUT/f'adaptive_td_gga.{ext}',dpi=240,facecolor='white')
