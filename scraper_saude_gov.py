"""
Coleta 3.000 notícias verdadeiras de saúde via sitemaps XML (gov.br + EBC).
Descobre URLs pelos sitemaps e busca o conteúdo real de cada artigo.
"""
import csv, os, re, time, random, logging
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import requests, pandas as pd
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("scraper_saude.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

DATASET_PATH = "dataset/dataset_labeled_expanded.csv"
OUTPUT_TEMP  = "dataset/new_true_records_temp2.csv"
OUTPUT_FINAL = "dataset/dataset_labeled_expanded.csv"
TARGET_NEW   = 800
COLUMNS      = ["url", "title", "date", "content", "source", "lang", "label"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0",
    "Accept-Language": "pt-BR,pt;q=0.9",
}
S = requests.Session()
S.headers.update(HEADERS)


def get_html(url, delay=1.0):
    for i in range(3):
        try:
            r = S.get(url, timeout=20)
            if r.status_code == 200:
                return BeautifulSoup(r.content, "lxml")
            if r.status_code in (404, 410):
                return None
        except Exception as e:
            log.warning(f"Erro {i+1} em {url}: {e}")
        time.sleep(delay * (i + 1))
    return None


def fetch_sitemap(url):
    """Retorna lista de <loc> de um sitemap XML."""
    try:
        r = S.get(url, timeout=30)
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        locs = [el.text.strip() for el in root.findall(".//sm:loc", ns) if el.text]
        return locs
    except Exception as e:
        log.warning(f"Sitemap erro {url}: {e}")
        return []


def clean(t):
    return re.sub(r"\s+", " ", t or "").strip()


def count_saved():
    if not os.path.exists(OUTPUT_TEMP):
        return 0
    return len(pd.read_csv(OUTPUT_TEMP))


def save(rec):
    new_file = not os.path.exists(OUTPUT_TEMP)
    with open(OUTPUT_TEMP, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow(rec)


def load_seen():
    seen = set(pd.read_csv(DATASET_PATH, usecols=["url"])["url"].dropna().astype(str))
    if os.path.exists(OUTPUT_TEMP):
        seen |= set(pd.read_csv(OUTPUT_TEMP, usecols=["url"])["url"].dropna().astype(str))
    return seen


# ── Descoberta de URLs via sitemaps ─────────────────────────────────────────

MONTHS_PT = [
    "janeiro","fevereiro","marco","abril","maio","junho",
    "julho","agosto","setembro","outubro","novembro","dezembro"
]
YEARS = list(range(2019, 2027))


def discover_govbr_urls(seen):
    """Varre sitemaps mensais do gov.br/saude e gov.br/anvisa."""
    urls = []
    prefixes = [
        "https://www.gov.br/saude/pt-br/assuntos/noticias/{year}/{month}/sitemap.xml",
        "https://www.gov.br/anvisa/pt-br/assuntos/noticias/anvisa/{year}/{month}/sitemap.xml",
        "https://www.gov.br/inca/pt-br/comunicacao/noticias/{year}/{month}/sitemap.xml",
    ]
    for tmpl in prefixes:
        for year in YEARS:
            for month in MONTHS_PT:
                sm_url = tmpl.format(year=year, month=month)
                locs = fetch_sitemap(sm_url)
                new = [u for u in locs if u not in seen]
                urls.extend(new)
                for u in new:
                    seen.add(u)
                if locs:
                    log.info(f"Sitemap {sm_url.split('gov.br')[1][:60]}: {len(locs)} total, {len(new)} novos")
                time.sleep(random.uniform(0.3, 0.7))
    log.info(f"[GOV.BR] Total URLs descobertas: {len(urls)}")
    return urls


def discover_ebc_urls(seen):
    """Varre sitemaps mensais da Agência Brasil."""
    urls = []
    base = "https://agenciabrasil.ebc.com.br"

    # Tenta sitemap index
    index_locs = fetch_sitemap(f"{base}/sitemap.xml")
    saude_sitemaps = [l for l in index_locs if "saude" in l.lower() or "sitemap" in l.lower()]

    if saude_sitemaps:
        for sm in saude_sitemaps:
            locs = fetch_sitemap(sm)
            saude_locs = [u for u in locs if "/saude/noticia/" in u and u not in seen]
            urls.extend(saude_locs)
            for u in saude_locs:
                seen.add(u)
            if saude_locs:
                log.info(f"[EBC] Sitemap {sm[-60:]}: {len(saude_locs)} novos")
            time.sleep(0.5)
    else:
        # Fallback: sitemaps por YYYY-MM
        for year in YEARS:
            for month in range(1, 13):
                sm_url = f"{base}/saude/noticia/{year}-{month:02d}/sitemap.xml"
                locs = fetch_sitemap(sm_url)
                new = [u for u in locs if u not in seen]
                urls.extend(new)
                for u in new:
                    seen.add(u)
                if locs:
                    log.info(f"[EBC] {year}-{month:02d}: {len(locs)} total, {len(new)} novos")
                time.sleep(random.uniform(0.3, 0.6))

    # Fallback adicional: /ultimas com filtro /saude/
    if len(urls) < 500:
        log.info("[EBC] Poucos resultados via sitemap, tentando /ultimas...")
        for page in range(0, 200):
            list_url = f"{base}/ultimas?b_start:int={page * 10}"
            soup = get_html(list_url, delay=0.8)
            if not soup:
                break
            found = 0
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "/saude/noticia/" in href:
                    full = href if href.startswith("http") else f"{base}{href}"
                    if full not in seen:
                        urls.append(full)
                        seen.add(full)
                        found += 1
            if found == 0 and page > 10:
                break
            time.sleep(random.uniform(0.5, 1.0))

    log.info(f"[EBC] Total URLs descobertas: {len(urls)}")
    return urls


# ── Extração de conteúdo ────────────────────────────────────────────────────

def extract_govbr(url, soup):
    h1 = soup.find("h1")
    title = clean(h1.get_text()) if h1 else ""
    if not title:
        return None

    d = (soup.find("span", class_=re.compile(r"documentModified|documentPublished|data-publicacao", re.I))
         or soup.find("time"))
    date = clean(d.get_text()) if d else ""

    body = (soup.find("div", id=re.compile(r"content-core|parent-fieldname-text", re.I))
            or soup.find("div", class_=re.compile(r"text-body|news-body|content", re.I))
            or soup.find("article"))
    if not body:
        return None
    for tag in body.find_all(["script", "style", "figure", "aside", "nav", "footer"]):
        tag.decompose()
    content = clean(body.get_text(" "))
    if len(content) < 150:
        return None

    return {"url": url, "title": title, "date": date, "content": content,
            "source": urlparse(url).netloc, "lang": "pt", "label": "true"}


def extract_ebc(url, soup):
    h1 = soup.find("h1")
    title = clean(h1.get_text()) if h1 else ""
    if not title:
        return None

    t = soup.find("time")
    date = t.get("datetime", clean(t.get_text())) if t else ""

    body = (soup.find("div", class_=re.compile(r"field-items|article-body|content-text", re.I))
            or soup.find("article"))
    if not body:
        return None
    for tag in body.find_all(["script", "style", "figure", "aside"]):
        tag.decompose()
    content = clean(body.get_text(" "))
    if len(content) < 150:
        return None

    return {"url": url, "title": title, "date": date, "content": content,
            "source": "agenciabrasil.ebc.com.br", "lang": "pt", "label": "true"}


def scrape_urls(url_list, extractor_fn, tag, quota):
    collected = 0
    for i, url in enumerate(url_list):
        if collected >= quota:
            break
        soup = get_html(url, delay=0.6)
        if not soup:
            continue
        rec = extractor_fn(url, soup)
        if rec:
            save(rec)
            collected += 1
            if collected % 100 == 0:
                log.info(f"[{tag}] {collected}/{quota} coletados")
        time.sleep(random.uniform(0.4, 0.9))
    log.info(f"[{tag}] Total: {collected}")
    return collected


# ── Integração final ────────────────────────────────────────────────────────

def integrate():
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
    print(f"Arquivo: {OUTPUT_FINAL}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    already = count_saved()
    remaining = TARGET_NEW - already
    log.info(f"Já salvos: {already} | Faltam: {remaining}")

    if remaining <= 0:
        integrate()
        return

    seen = load_seen()
    log.info(f"URLs no dataset + temp: {len(seen)}")

    # Só Fiocruz para completar as 800 restantes
    fiocruz_urls = []
    base = "https://portal.fiocruz.br"
    for page in range(0, 120):
        if len(fiocruz_urls) >= remaining * 3:
            break
        list_url = f"{base}/noticias?page={page}" if page else f"{base}/noticias"
        soup = get_html(list_url, delay=1.0)
        if not soup:
            break
        found = 0
        for a in soup.find_all("a", href=re.compile(r"/noticia/")):
            full = urljoin(base, a["href"])
            if full not in seen:
                fiocruz_urls.append(full)
                seen.add(full)
                found += 1
        if found == 0:
            break
        time.sleep(random.uniform(0.8, 1.5))
    log.info(f"[FIOCRUZ] {len(fiocruz_urls)} URLs descobertas")

    collected = 0
    for url in fiocruz_urls:
        if collected >= remaining:
            break
        soup = get_html(url, delay=0.7)
        if not soup:
            continue
        h1 = soup.find("h1")
        title = clean(h1.get_text()) if h1 else ""
        if not title:
            continue
        t = soup.find("span", class_=re.compile(r"date|data", re.I)) or soup.find("time")
        date = clean(t.get_text()) if t else ""
        body = (soup.find("div", class_=re.compile(r"field-body|content-text|node-body", re.I))
                or soup.find("article"))
        if not body:
            continue
        for tag in body.find_all(["script", "style", "figure"]):
            tag.decompose()
        content = clean(body.get_text(" "))
        if len(content) < 150:
            continue
        save({"url": url, "title": title, "date": date, "content": content,
              "source": "portal.fiocruz.br", "lang": "pt", "label": "true"})
        collected += 1
        if collected % 100 == 0:
            log.info(f"[FIOCRUZ] {collected}/{remaining}")
        time.sleep(random.uniform(0.5, 1.0))

    log.info(f"[FIOCRUZ] Total: {collected}")
    total = count_saved()
    log.info(f"Total coletado: {total}")
    if total > 0:
        integrate()


if __name__ == "__main__":
    main()
