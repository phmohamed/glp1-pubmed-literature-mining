"""Literature mining of GLP-1 receptor agonist research on PubMed: growth, drug mentions, topics over time.
Topic model: non-negative matrix factorisation written from scratch in numpy."""
import re, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)
cnt = pd.read_csv("data/papers_per_year.csv")
ab = pd.read_csv("data/abstracts.csv").fillna("")
ab["text"] = (ab.title + " " + ab.abstract).str.lower()
ab = ab[ab.abstract.str.len() > 200].reset_index(drop=True)      # keep papers with a real abstract
print(len(ab), "abstracts with text")
n_year = ab.groupby("year").size()

drugs = {"semaglutide": r"semaglutide|ozempic|wegovy|rybelsus", "liraglutide": r"liraglutide|victoza|saxenda",
         "tirzepatide": r"tirzepatide|mounjaro|zepbound", "dulaglutide": r"dulaglutide|trulicity",
         "exenatide": r"exenatide|byetta|bydureon", "lixisenatide": r"lixisenatide"}
themes = {"Obesity / weight loss": r"obes|weight loss|body weight|overweight", "Cardiovascular": r"cardiovascular|heart failure|myocardial|mace\b|stroke|blood pressure",
          "Kidney": r"kidney|renal|nephropathy|albuminuria|egfr", "Liver (NAFLD / MASH)": r"nafld|nash\b|masld|mash\b|steato|fatty liver|liver fat",
          "Brain / addiction": r"alzheimer|dementia|parkinson|neuroprotect|alcohol|addiction|cognitive|depress", 
          "Safety (pancreatitis, GI, thyroid)": r"pancreatit|gastrointestinal|nausea|vomiting|thyroid|gallbladder|retinopathy|gastroparesis",
          "Type 1 diabetes": r"type 1 diabetes|t1d\b", "Sleep apnea / OA / other": r"sleep apnea|osteoarthritis|psoria|asthma|cancer"}

def share(pats):
    out = {}
    for k, p in pats.items():
        hit = ab.text.str.contains(p, regex=True)
        out[k] = (hit.groupby(ab.year).mean() * 100).round(1)
    return pd.DataFrame(out)
dsh, tsh = share(drugs), share(themes)
dsh.to_csv("data/drug_mention_share_by_year.csv"); tsh.to_csv("data/theme_share_by_year.csv"); cnt.to_csv("data/papers_per_year.csv", index=False)
print(cnt.tail(6).to_string()); print(dsh.loc[[2012, 2016, 2020, 2023, 2025]]); print(tsh.loc[[2012, 2016, 2020, 2023, 2025]].T)

# ---- NMF topics ----
STOP = set("the of and in to a with for was were is are as by on at from that this these be or an which not than also have has had been their it its our we study studies results conclusion conclusions background methods method objective objectives aim aimed patients patient mg group groups compared treatment treated use used using versus vs significant significantly p ci mean after baseline both between among based analysis data trial trials week weeks may been can more most other over within without including associated".split())
tok = lambda t: [w for w in re.findall(r"[a-z][a-z\-]{2,}", t) if w not in STOP]
docs = [tok(t) for t in ab.text]
dfc = {}
for d in docs:
    for w in set(d): dfc[w] = dfc.get(w, 0) + 1
vocab = [w for w, c in sorted(dfc.items(), key=lambda x: -x[1]) if 8 <= c <= 0.4 * len(docs)][:3000]
ix = {w: i for i, w in enumerate(vocab)}; X = np.zeros((len(docs), len(vocab)))
for i, d in enumerate(docs):
    for w in d:
        if w in ix: X[i, ix[w]] += 1
idf = np.log(len(docs) / (1 + (X > 0).sum(0))); X = np.log1p(X) * idf; X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9

def nmf(X, k, iters=300):
    W = rng.random((X.shape[0], k)) + 0.1; H = rng.random((k, X.shape[1])) + 0.1
    for _ in range(iters):
        H *= (W.T @ X) / (W.T @ W @ H + 1e-9); W *= (X @ H.T) / (W @ H @ H.T + 1e-9)
    return W, H, np.linalg.norm(X - W @ H)
K = 8
best = min((nmf(X, K) for _ in range(3)), key=lambda r: r[2]); W, H, err = best
topic_terms = [[vocab[i] for i in np.argsort(-H[k])[:9]] for k in range(K)]
dom = W.argmax(1)
tt = pd.DataFrame({"topic": range(K), "top_terms": [", ".join(t) for t in topic_terms], "n_abstracts": np.bincount(dom, minlength=K)})
tt.to_csv("data/topics.csv", index=False); print(tt.to_string())
tyear = pd.crosstab(ab.year, dom, normalize="index").mul(100).round(1); tyear.to_csv("data/topic_share_by_year.csv"); print(tyear.loc[[2012, 2016, 2020, 2023, 2025]])

# ---- figures ----
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(cnt.year, cnt.papers, color="#4472c4"); ax.set_ylabel("PubMed papers per year"); ax.set_title("GLP-1 receptor agonist papers (search terms in README)")
fig.tight_layout(); fig.savefig("figures/papers_per_year.png", dpi=150)
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
for c in dsh: axs[0].plot(dsh.index, dsh[c], label=c)
axs[0].set_ylabel("% of sampled abstracts mentioning it"); axs[0].set_title("Drug mentions"); axs[0].legend(frameon=False, fontsize=8)
for c in tsh: axs[1].plot(tsh.index, tsh[c], label=c)
axs[1].set_title("Themes"); axs[1].legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig("figures/drugs_and_themes.png", dpi=150)
