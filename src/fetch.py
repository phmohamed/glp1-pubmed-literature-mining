"""PubMed (NCBI E-utilities): yearly paper counts and abstracts for GLP-1 receptor agonist research."""
import requests, time, re, pandas as pd, xml.etree.ElementTree as ET
B = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
Q = '("glucagon-like peptide-1 receptor agonist"[Title/Abstract] OR "GLP-1 receptor agonist"[Title/Abstract] OR semaglutide[Title/Abstract] OR liraglutide[Title/Abstract] OR tirzepatide[Title/Abstract])'
import os, sys
Y0, Y1 = int(os.environ.get('Y0', 2012)), int(os.environ.get('Y1', 2025))
counts = []
for y in (range(2005, 2026) if not os.path.exists('data/papers_per_year.csv') else []):
    r = requests.get(B + "esearch.fcgi", params={"db": "pubmed", "term": Q, "datetype": "pdat", "mindate": y, "maxdate": y, "retmode": "json", "retmax": 0}, timeout=60)
    counts.append({"year": y, "papers": int(r.json()["esearchresult"]["count"])}); time.sleep(0.4)
if counts: pd.DataFrame(counts).to_csv("data/papers_per_year.csv", index=False)
rows = []
for y in range(Y0, Y1 + 1):                      # up to 250 abstracts per year, newest first
    r = requests.get(B + "esearch.fcgi", params={"db": "pubmed", "term": Q, "datetype": "pdat", "mindate": y, "maxdate": y, "retmode": "json", "retmax": 200, "sort": "relevance"}, timeout=60)
    ids = r.json()["esearchresult"]["idlist"]; time.sleep(1)
    for i in range(0, len(ids), 100):
        for attempt in range(5):
            x = requests.post(B + "efetch.fcgi", data={"db": "pubmed", "id": ",".join(ids[i:i + 100]), "retmode": "xml"}, timeout=120)
            try:
                root = ET.fromstring(x.content); break
            except ET.ParseError:
                time.sleep(2 + 2 * attempt)      # NCBI rate limit, wait and retry
        else:
            continue
        for a in root.findall(".//PubmedArticle"):
            t = "".join(a.find(".//ArticleTitle").itertext()) if a.find(".//ArticleTitle") is not None else ""
            ab = " ".join("".join(n.itertext()) for n in a.findall(".//Abstract/AbstractText"))
            rows.append({"pmid": a.findtext(".//PMID"), "year": y, "title": t, "abstract": ab})
        time.sleep(1)
out = pd.DataFrame(rows)
if os.path.exists("data/abstracts.csv") and Y0 > 2012: out = pd.concat([pd.read_csv("data/abstracts.csv"), out])
out.to_csv("data/abstracts.csv", index=False); print(len(out))
