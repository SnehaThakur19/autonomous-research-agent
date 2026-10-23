import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"


def academic_search(query, max_results=5):
    """Search arXiv for academic papers matching the query.

    Returns a list of {title, url, content, source} dicts shaped
    like the web search results so they can flow through the same
    verification pipeline.
    """
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    })

    url = f"{ARXIV_API_URL}?{params}"

    with urllib.request.urlopen(url, timeout=30) as response:
        xml_data = response.read().decode("utf-8")

    root = ET.fromstring(xml_data)

    results = []

    for entry in root.findall(f"{ATOM_NS}entry"):
        title = " ".join(
            (entry.findtext(f"{ATOM_NS}title") or "").split()
        )
        summary = " ".join(
            (entry.findtext(f"{ATOM_NS}summary") or "").split()
        )
        link = (entry.findtext(f"{ATOM_NS}id") or "").strip()

        if not title or not link:
            continue

        results.append({
            "title": title,
            "url": link,
            "content": summary,
            "source": "arxiv"
        })

    return results
