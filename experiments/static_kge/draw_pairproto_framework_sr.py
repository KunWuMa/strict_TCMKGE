"""Editable vector method diagram aligned with the actual two-stage encoder."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Arc

HERE=Path(__file__).resolve().parent
OUT=HERE/"outputs"/"scientific_reports_20261026"/"exports"/"figures"
OUT.mkdir(parents=True,exist_ok=True)
BLUE="#287DAD";PURPLE="#7951A1";ORANGE="#C56D32";GRAY="#687582";DARK="#243645"


def main():
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})
    fig,ax=plt.subplots(figsize=(12,6.6));ax.set_xlim(0,150);ax.set_ylim(0,70);ax.axis("off")
    ax.add_patch(FancyBboxPatch((.7,31),148,38,boxstyle="round,pad=.3,rounding_size=2",fc="#F4F7FA",ec="none"))
    ax.text(3,67,"OFFLINE LEARNING AND MEMORY",fontsize=11,color=GRAY,weight="bold")
    ax.text(3,28.5,"COMPOSITIONAL INFERENCE",fontsize=11,color=GRAY,weight="bold")

    def box(x,y,w,h,label,color,fill="white",size=12):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.3,rounding_size=1.2",fc=fill,ec=color,lw=1.4))
        if label:ax.text(x+w/2,y+h-2,label,ha="center",va="top",fontsize=size,color=DARK,weight="bold")
    def arrow(a,b,color=GRAY,style="-",curve=0):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=13,lw=1.55,color=color,
                                    linestyle=style,connectionstyle=f"arc3,rad={curve}"))
    def vector(x,y,color,label):
        for i,alpha in enumerate((.35,.8,.5,.95,.6)):
            ax.add_patch(FancyBboxPatch((x+i*2,y),1.5,3,boxstyle="round,pad=.05,rounding_size=.25",fc=color,alpha=alpha,ec="none"))
        ax.text(x+4.5,y-1,label,ha="center",va="top",fontsize=13,color=color)

    # A small heterogeneous graph illustrates supervision without implying patient labels.
    nodes=[(7,55,BLUE,"h"),(17,61,ORANGE,"f"),(24,54,PURPLE,"t"),(16,48,BLUE,"h"),(6,45.5,PURPLE,"t")]
    for a,b in ((0,1),(1,2),(1,3),(0,4),(3,4)):
        arrow(nodes[a][:2],nodes[b][:2],"#A0ADB8")
    for x,y,c,label in nodes:
        ax.add_patch(Circle((x,y),2.4,fc=c,ec="white",lw=1.5));ax.text(x,y,label,color="white",ha="center",va="center",fontsize=12)
    ax.text(15,63.6,"Training graph",ha="center",fontsize=10.5,color=DARK)
    arrow((27,55),(35,55),BLUE)
    box(35,48,26,16,"Mean-ComplEx",BLUE)
    ax.text(48,58,"backbone learning",ha="center",fontsize=11,color=DARK)
    vector(43,52,BLUE,r"$\mathbf{e}_h,\;\mathbf{e}_r,\;\mathbf{e}_t$")
    ax.add_patch(Arc((60,65.2),3.1,4,theta1=0,theta2=180,color=BLUE,lw=1.4))
    ax.add_patch(FancyBboxPatch((58.6,62.5),2.8,2.7,boxstyle="round,pad=.1",fc=BLUE,ec=BLUE))
    ax.text(65,65,"Freeze after dev selection",fontsize=11,color=BLUE,va="center")
    ax.plot([61.4,65,65,122],[55,55,63,63],color=BLUE,lw=1.55)
    arrow((122,63),(122,61.4),BLUE)
    box(102,48,40,13,"Training-catalogue features",BLUE)
    ax.text(122,51.5,r"$\widetilde{\Phi}_r$  from the fixed decoder",ha="center",fontsize=12,color=BLUE)

    # Count recipes separately from the learned backbone.
    box(3,32.5,26,10,"Group recipes",ORANGE,size=10)
    ax.text(16,35,"Aliases count once",ha="center",fontsize=11,color=GRAY)
    arrow((16,45.4),(16,43),ORANGE)
    arrow((29,37.5),(35,37.5),ORANGE)
    box(35,32.5,26,12,"Pair profiles",ORANGE,size=10)
    ax.text(48,39.2,r"$\ell_g^{(r)}$",ha="center",fontsize=12,color=ORANGE)
    for row in range(2):
        for col,alpha in enumerate((.2,.6,.9,.3)):
            ax.add_patch(FancyBboxPatch((39+col*4,34+row*2.1),3.2,1.5,boxstyle="round,pad=.03",fc=ORANGE,alpha=alpha if row==0 else 1-alpha/1.2,ec="none"))
    ax.text(48,30.7,r"Support $\geq m$; shrinkage $\tau$",ha="center",fontsize=11,color=ORANGE)
    arrow((61,40),(70,51),ORANGE)
    arrow((102,50),(92,50),BLUE)
    box(70,45,21,16,"Ridge projection",PURPLE,size=11)
    ax.text(80.5,51.2,"Profiles to",ha="center",fontsize=11,color=DARK)
    ax.text(80.5,47.8,"KGE head space",ha="center",fontsize=11,color=DARK)
    arrow((80.5,45),(80.5,41),PURPLE)
    box(69,32.5,24,8,"Pair memory",PURPLE,size=10)
    ax.text(81,34.7,r"$g,r\;\mapsto\;\mathbf{v}_g^{(r)}$",ha="center",fontsize=13,color=PURPLE)
    ax.text(118,40,"Train-derived profiles",fontsize=11,color=GRAY,ha="center")
    ax.text(118,36.3,"No dev/test labels in profiles",fontsize=10.5,color=GRAY,ha="center")

    # New formula: visible ingredient identity and two different compositional paths.
    colors=[BLUE,"#469E8D",ORANGE,PURPLE]
    positions=[(7,18),(17,20),(8,9),(20,11)]
    for i,(x,y) in enumerate(positions):
        ax.add_patch(Circle((x,y),2.2,fc=colors[i],ec="white",lw=1.5));ax.text(x,y,chr(65+i),ha="center",va="center",color="white",fontsize=12)
    ax.text(14,24,"Unseen formula $f$",ha="center",fontsize=12,color=DARK,weight="bold")
    ax.text(14,3.7,"Known herb set $H_f$",ha="center",fontsize=11,color=DARK)
    arrow((24,19),(35,21),PURPLE)
    arrow((23,10),(35,7),BLUE)
    box(35,17,26,10,"Supported pair lookup",PURPLE,size=10)
    ax.text(48,20,"AB, AC, ...",ha="center",fontsize=12,color=PURPLE)
    box(35,2,26,11,"Herb mean",BLUE,size=12)
    vector(43,6.2,BLUE,r"$\mu_f$")
    arrow((61,22),(70,22),PURPLE)
    arrow((81,32.5),(81,27.8),PURPLE)
    box(70,17,23,10,"Pair average",PURPLE,size=11)
    ax.text(81.5,20,r"$\mathbf{p}_f^{(r)}$",ha="center",fontsize=14,color=PURPLE)
    ax.text(106,29.3,"Relation $r$",ha="center",fontsize=11,color=PURPLE)
    arrow((96,29.5),(93.3,33.2),PURPLE,curve=-.1)
    arrow((93,22),(102,19),PURPLE)
    arrow((61,7),(102,14),BLUE,curve=.06)
    box(102,10,17,16,"Compose",DARK,size=12)
    vector(106,16.5,DARK,r"$\mathbf{z}_f^{(r)}$")
    ax.text(110.5,10.8,r"Scale + mix $\gamma$",ha="center",fontsize=9.5,color=GRAY)
    arrow((119,18),(125,18),DARK)
    box(125,10,22,16,"ComplEx",BLUE,size=12)
    ax.text(136,20,"decoder",ha="center",fontsize=12,color=BLUE)
    ax.text(136,15.8,"Rank tails in",ha="center",fontsize=11,color=DARK)
    ax.text(136,12,r"$\mathcal{C}_r$",ha="center",fontsize=13,color=BLUE)
    ax.text(104,5,"At inference: compose once per formula and relation",fontsize=10.5,color=GRAY,ha="center")
    for label in ax.texts:label.set_fontsize(label.get_fontsize()*1.15)
    fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.98)
    for ext in ("pdf","svg","png"):fig.savefig(OUT/("pairproto_framework."+ext),dpi=300,bbox_inches="tight")
    plt.close(fig)
    print("FRAMEWORK SAVED",OUT)


if __name__=="__main__":main()
