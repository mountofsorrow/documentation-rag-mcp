import os
from firecrawl import Firecrawl
from firecrawl.types import ScrapeOptions

API_KEY = "your_api_key"
firecrawl = Firecrawl(api_key=API_KEY)

scrape_options = ScrapeOptions(
    formats=["markdown"],
    proxy="auto",
    only_main_content=True,
    fast_mode=False,
)

scrape_options.exclude_tags = ["nav", "footer", ".footer", "header", ".header"]

response = firecrawl.crawl(
    url="https://docs.telethon.dev/en/stable/",
    limit=200,
    scrape_options=scrape_options,
    poll_interval=5,
    sitemap="skip"
)

# Create output folder
os.makedirs("telethon_docs", exist_ok=True)

# Save each page as its own .md file
for i, doc in enumerate(response.data):
    url = doc.metadata.url or doc.metadata.sourceURL or f"page_{i}"
    slug = url.replace("https://docs.telethon.dev/", "").strip("/")
    filename = slug.replace("/", "_") or "index"

    with open(f"telethon/{filename}.md", "w", encoding="utf-8") as f:
        f.write(doc.markdown)

print("✔ Finished ️")
print(f"Saved {len(response.data)} Files.")