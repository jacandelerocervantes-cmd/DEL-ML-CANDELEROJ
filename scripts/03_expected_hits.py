"""Step 3: expected number of hits of the two lists of step 2, extrapolated from the hit counts that the earlier workflow (earlier_workflow/) obtained in the organizers' blind test.
Known (reported by the challenge platform, totals only; which compounds are hits is NOT known): earlier list A = 29 hits of 50, earlier list B = 12 of 50 (B excludes A's compounds).
Assumptions, all stated and all uncertain: (1) a new-list compound that also belongs to earlier list A (or B) hits at that list's average rate (exchangeability); (2) a compound in neither earlier list hits at a rate between 0 and the rate of list B (0.24),
because it was ranked below the earlier lists by the AL model; the mid-point is shown as a central value. The result is an estimate for planning, not a measured outcome: it can only be measured by evaluating the lists.
Output: results/expected_hits.json. Run after 02_select_and_ablate.py."""
import json, pandas as pd
EARLIER={"A":(29,50),"B":(12,50)}; rA=EARLIER["A"][0]/EARLIER["A"][1]; rB=EARLIER["B"][0]/EARLIER["B"][1]
ref=pd.read_csv("reference/earlier_workflow_selection_ids.csv"); A=set(ref[ref.submission=="AL1"].CatalogID); B=set(ref[ref.submission=="AL2"].CatalogID); out={}
for n in (1,2):
    l=set(pd.read_csv(f"results/selection_list{n}.csv").CatalogID); a,b=len(l&A),len(l&B); new=len(l)-a-b; base=a*rA+b*rB
    out[f"list{n}"]=dict(in_earlier_A=a,in_earlier_B=b,in_neither=new,expected_hits_low=round(base,1),expected_hits_central=round(base+new*rB/2,1),expected_hits_high=round(base+new*rB,1))
out["earlier_workflow_reference"]=dict(A_hits_of_50=29,B_hits_of_50=12,total=41)
json.dump(out,open("results/expected_hits.json","w"),indent=1); print(json.dumps(out,indent=1))
