import csv
import re
import urllib.request
import xml.etree.ElementTree as ET

months = {
    'janeiro': '01', 'fevereiro': '02', 'marco': '03',
    'abril': '04', 'maio': '05', 'junho': '06',
    'julho': '07', 'agosto': '08', 'setembro': '09',
    'outubro': '10', 'novembro': '11', 'dezembro': '12'
}

def url_to_date(url):
    m = re.search(r'/(\d{4})/(\w+)/', url)
    if m:
        year, month_pt = m.group(1), m.group(2)
        month_num = months.get(month_pt, '01')
        return f"{year}-{month_num}-01T00:00:00-03:00"
    return "2025-01-01T00:00:00-03:00"

def slug_to_title(url):
    slug = url.rstrip('/').split('/')[-1]
    return slug.replace('-', ' ').capitalize()

def fetch_sitemap(sitemap_url):
    try:
        req = urllib.request.Request(sitemap_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read().decode('utf-8')
        root = ET.fromstring(content)
        ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        return [loc.text for loc in root.findall('.//sm:loc', ns)]
    except Exception as e:
        print(f"  ERROR fetching {sitemap_url}: {e}")
        return []

months_to_fetch = [
    ('2026', 'junho'), ('2026', 'maio'), ('2026', 'abril'),
    ('2026', 'marco'), ('2026', 'fevereiro'), ('2026', 'janeiro'),
    ('2025', 'dezembro'), ('2025', 'novembro'), ('2025', 'outubro'),
    ('2025', 'setembro'), ('2025', 'agosto'), ('2025', 'julho'),
    ('2025', 'junho'), ('2025', 'maio'), ('2025', 'abril'),
    ('2025', 'marco'), ('2025', 'fevereiro'), ('2025', 'janeiro'),
    ('2024', 'dezembro'), ('2024', 'novembro'), ('2024', 'outubro'),
    ('2024', 'setembro'), ('2024', 'agosto'), ('2024', 'julho'),
    ('2024', 'junho'), ('2024', 'maio'), ('2024', 'abril'),
    ('2024', 'marco'), ('2024', 'fevereiro'), ('2024', 'janeiro'),
]

with open('c:/Projetos/TCC/treinamento/existing_urls.txt', encoding='utf-8') as f:
    existing_urls = set(line.strip() for line in f)

print(f"Loaded {len(existing_urls)} existing URLs\n")

all_new_urls = []
for year, month in months_to_fetch:
    sitemap_url = f"https://www.gov.br/saude/pt-br/assuntos/noticias/{year}/{month}/sitemap.xml"
    urls = fetch_sitemap(sitemap_url)
    new_urls = [u for u in urls if u and u not in existing_urls]
    all_new_urls.extend(new_urls)
    print(f"{year}/{month}: {len(urls)} total, {len(new_urls)} new")

print(f"\nTotal new URLs: {len(all_new_urls)}")

new_records = []
for url in all_new_urls:
    title = slug_to_title(url)
    date = url_to_date(url)
    source = url.split('/')[2]
    new_records.append([url, title, date, title, source, "pt", "true"])

with open('c:/Projetos/TCC/treinamento/dataset/dataset_labeled.csv', 'a',
          encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(new_records)

print(f"Added {len(new_records)} records to dataset_labeled.csv")
