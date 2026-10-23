"""exp27: non-adaptive binary-code pools for the 4 validation groups that contain EXACTLY ONE hit (solver result, exp26b):
tiofeno (3 no enviados) 1/3, tiofeno (6 de r94) 1/6, quinazolina B 1/14, pirim/pirid r89 1/50.
Each compound gets an index; submission j contains the compounds whose index has bit j set (+ confirmed negatives up to 50). With exactly one hit the
0/1 results of the ceil(log2 n) submissions spell the hit's index in binary; an index >= n signals an inconsistency (transcription error). Files: submissions/Team_candeleroj_116..130.txt"""
import os, numpy as np, pandas as pd
D=pd.read_csv("data/processed/gp/exp26_unidentified_families.csv")
o=pd.read_csv("data/processed/confirmed_submission_outcomes_v2.csv"); neg=sorted(o[o.confirmed_label==0].CatalogID)
GROUPS=[("tiofeno (3 no enviados)",2),("tiofeno (6 de r94)",3),("quinazolina B",4),("pirim/pirid r89",6)]
nxt=116; plan=[]; rng=np.random.default_rng(116)
for g,nb in GROUPS:
    ids=sorted(D[D.grupo_diseno==g].CatalogID); n=len(ids); assert n<=2**nb and 2**(nb-1)<n or n<=2**nb
    for j in range(nb):
        pool=[c for i,c in enumerate(ids) if (i>>j)&1]; assert 0<len(pool)<=50
        pad=list(rng.choice(neg,50-len(pool),replace=False)); L=pool+pad; assert len(set(L))==50
        p=f"submissions/Team_candeleroj_{nxt}.txt"; assert not os.path.exists(p); open(p,"w").write("\n".join(L)+"\n")
        plan.append(dict(envio=nxt,grupo=g,bit=j,n_grupo=n,n_en_pool=len(pool),miembros=";".join(pool))); nxt+=1
P=pd.DataFrame(plan); P.to_csv("data/processed/gp/exp27_binary_pools_plan.csv",index=False)
D[D.grupo_diseno.isin([g for g,_ in GROUPS])].assign(idx=lambda d:d.groupby("grupo_diseno").CatalogID.rank(method="first").astype(int)-1)[["grupo_diseno","CatalogID","idx","smiles"]].sort_values(["grupo_diseno","idx"]).to_csv("data/processed/gp/exp27_binary_pools_index.csv",index=False)
print(P[["envio","grupo","bit","n_grupo","n_en_pool"]].to_string(index=False)); print("\narchivos:",P.envio.min(),"a",P.envio.max(),f"({len(P)} envios)")
