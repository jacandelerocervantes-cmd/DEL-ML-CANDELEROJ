"""exp22: build the small files for the Kaggle route-2 notebook (kaggle/pgk2_route2/). The big DEL file (data/raw/PGK2_selection.parquet, 193 MB) is uploaded as-is.
al_groups.csv = cluster id per AL compound (Butina, sim>=0.3, same as exp20/exp21) so Kaggle uses exactly the same held-out groups."""
import shutil, numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G
from rdkit.ML.Cluster import Butina
RDLogger.DisableLog('rdApp.*')
al=pd.read_csv("data/raw/active_learning_data.csv"); n=len(al); g4=G.GetMorganGenerator(radius=2,fpSize=2048)
bv=[g4.GetFingerprint(Chem.MolFromSmiles(s)) for s in al.RDKit_SMILES]; dist=[]
for i in range(1,n): dist+=[1-s for s in DataStructs.BulkTanimotoSimilarity(bv[i],bv[:i])]
g=np.zeros(n,int)
for k,c in enumerate(Butina.ClusterData(dist,n,0.7,isDistData=True)): g[list(c)]=k
pd.DataFrame({'CatalogID':al.CatalogID,'group':g}).to_csv("kaggle/pgk2_route2/al_groups.csv",index=False)
shutil.copy("data/raw/active_learning_data.csv","kaggle/pgk2_route2/active_learning_data.csv"); shutil.copy("data/raw/Val-Test-set/PGK2_Test_split.csv","kaggle/pgk2_route2/PGK2_Test_split.csv")
print("grupos:",g.max()+1)
