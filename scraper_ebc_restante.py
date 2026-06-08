"""Coleta os 800 artigos restantes da Agência Brasil (EBC) via sitemap."""
import csv, os, re, time, random, logging
import xml.etree.ElementTree as ET
import requests, pandas as pd
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler("scraper_ebc.log", encoding="utf-8")])
log = logging.getLogger(__name__)

DATASET_PATH = "dataset/dataset_labeled_expanded.csv"
OUTPUT_TEMP  = "dataset/ebc_temp.csv"
OUTPUT_FINAL = "dataset/dataset_labeled_expanded.csv"
TARGET       = 800
COLUMNS      = ["url", "title", "date", "content", "source", "lang", "label"]

S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                   "Accept-Language": "pt-BR,pt;q=0.9"})

def get_html(url):
    for i in range(3):
        try:
            r = S.get(url, timeout=20)
            if r.status_code == 200:
                return BeautifulSoup(r.content, "lxml")
            if r.status_code in (404, 410):
                return None
        except Exception as e:
            log.warning(f"Erro {i+1}: {e}")
        time.sleep(10 * (i + 1))
    return None

def clean(t):
    return re.sub(r"\s+", " ", t or "").strip()

def count_saved():
    return len(pd.read_csv(OUTPUT_TEMP)) if os.path.exists(OUTPUT_TEMP) else 0

def save(rec):
    new = not os.path.exists(OUTPUT_TEMP)
    with open(OUTPUT_TEMP, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new: w.writeheader()
        w.writerow(rec)

def load_seen():
    seen = set(pd.read_csv(DATASET_PATH, usecols=["url"])["url"].dropna().astype(str))
    if os.path.exists(OUTPUT_TEMP):
        seen |= set(pd.read_csv(OUTPUT_TEMP, usecols=["url"])["url"].dropna().astype(str))
    return seen

def fetch_sitemap_locs(url):
    try:
        r = S.get(url, timeout=20)
        if r.status_code != 200: return []
        root = ET.fromstring(r.content)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        return [el.text for el in root.findall(".//sm:loc", ns) if el.text]
    except Exception as e:
        log.warning(f"Sitemap erro: {e}")
        return []

def extract_article(url, soup):
    h1 = soup.find("h1")
    title = clean(h1.get_text()) if h1 else ""
    if not title: return None
    t = soup.find("time")
    date = t.get("datetime", clean(t.get_text())) if t else ""
    body = (soup.find("div", class_=re.compile(r"field-items|article-body|content-text", re.I))
            or soup.find("article"))
    if not body: return None
    for tag in body.find_all(["script", "style", "figure", "aside"]): tag.decompose()
    content = clean(body.get_text(" "))
    if len(content) < 150: return None
    return {"url": url, "title": title, "date": date, "content": content,
            "source": "agenciabrasil.ebc.com.br", "lang": "pt", "label": "true"}

def main():
    already = count_saved()
    remaining = TARGET - already
    log.info(f"Já salvos: {already} | Faltam: {remaining}")
    if remaining <= 0:
        pass
    else:
        seen = load_seen()
        log.info(f"URLs já vistas: {len(seen)}")

        # Coleta URLs de saúde das páginas 7-11 do sitemap EBC
        saude_urls = []
        for page in range(7, 12):
            locs = fetch_sitemap_locs(f"http://agenciabrasil.ebc.com.br/sitemap.xml?page={page}")
            new = [u for u in locs if u and "/saude/" in u and u not in seen]
            saude_urls.extend(new)
            for u in new: seen.add(u)
            log.info(f"Página {page}: {len(new)} novas URLs de saúde")
            time.sleep(random.uniform(0.5, 1.0))

        log.info(f"Total URLs candidatas: {len(saude_urls)}")
        random.shuffle(saude_urls)

        collected = 0
        for url in saude_urls:
            if collected >= remaining: break
            soup = get_html(url)
            if not soup: continue
            rec = extract_article(url, soup)
            if rec:
                save(rec)
                collected += 1
                if collected % 100 == 0:
                    log.info(f"Coletados: {collected}/{remaining}")
            time.sleep(random.uniform(0.4, 0.9))
        log.info(f"Coleta concluída: {collected}")

    # Integração
    total_new = count_saved()
    if total_new == 0:
        log.error("Nenhum registro coletado.")
        return

    df_orig = pd.read_csv(DATASET_PATH)
    df_new  = pd.read_csv(OUTPUT_TEMP)
    existing = set(df_orig["url"].dropna().astype(str))
    df_new = df_new[~df_new["url"].astype(str).isin(existing)].drop_duplicates("url")
    df_final = pd.concat([df_orig, df_new], ignore_index=True)
    df_final["label"] = df_final["label"].astype(str).str.strip().str.lower()
    df_final.to_csv(OUTPUT_FINAL, index=False, encoding="utf-8")
    log.info(f"Salvo: {OUTPUT_FINAL} ({len(df_final)} registros)")
    print("\n=== RESULTADO FINAL ===")
    print(df_final["label"].value_counts().to_string())

if __name__ == "__main__":
    main()
