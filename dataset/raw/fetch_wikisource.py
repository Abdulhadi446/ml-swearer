import urllib.request
import json
import time
import re

USER_AGENT = {'User-Agent': 'CreativeCursesBot/1.0 (data collection for creative swear generator project)'}

def fetch_json(url):
    req = urllib.request.Request(url, headers=USER_AGENT)
    return json.loads(urllib.request.urlopen(req, timeout=30).read().decode())

PLAYS = [
    "Hamlet",
    "The Tragedy of Hamlet, Prince of Denmark",
    "Romeo and Juliet",
    "The Tragedy of Othello",
    "King Lear",
    "Macbeth",
    "The Tempest",
    "Much Ado About Nothing",
    "A Midsummer Night's Dream",
    "The Merchant of Venice",
]

results = {}
for p in PLAYS:
    try:
        url = f"https://en.wikisource.org/w/api.php?action=query&titles={urllib.request.quote(p)}&prop=revisions&rvprop=content&rvslots=main&format=json"
        data = fetch_json(url)
        pages = data['query']['pages']
        content = ""
        for pid, page in pages.items():
            if 'revisions' in page:
                content = page['revisions'][0]['slots']['main']['*']
                break
        results[p] = content
        print(f"OK: {p} - {len(content)} chars")
    except Exception as e:
        print(f"FAIL: {p} - {e}")
    time.sleep(1)

with open('dataset/raw/shakespeare_wikisource.json', 'w') as f:
    json.dump(results, f, indent=1)
print("SAVED")
