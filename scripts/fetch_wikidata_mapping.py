import os
import json
import time
import requests
import urllib.parse
from tqdm import tqdm

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGES_DIR = os.path.join(CODE_DIR, 'data', 'MKG-W', 'images')
OUT_DICT = os.path.join(CODE_DIR, 'data', 'MKG-W', 'wikidata_to_wiki_dict.json')
BATCH_SIZE = 100

def get_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/sparql-results+json'
    })
    return session

def extract_title_from_url(url):
    # e.g., https://en.wikipedia.org/wiki/Elon_Musk -> Elon_Musk
    if 'wikipedia.org/wiki/' in url:
        return urllib.parse.unquote(url.split('wikipedia.org/wiki/')[-1])
    return url

def main():
    # 1. Get Q-nodes that actually have images
    qnodes = []
    if os.path.exists(IMAGES_DIR):
        for f in os.listdir(IMAGES_DIR):
            if f.endswith('.jpg') and f.startswith('Q'):
                qnodes.append(f.replace('.jpg', ''))
                
    print(f"Found {len(qnodes)} downloaded images in MKG-W.")
    
    # Check existing dict to resume
    mapping = {}
    if os.path.exists(OUT_DICT):
        with open(OUT_DICT, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            
    missing_qnodes = [q for q in qnodes if q not in mapping]
    print(f"Need to fetch mapping for {len(missing_qnodes)} Q-nodes.")
    
    if not missing_qnodes:
        print("All Q-nodes mapped!")
        return

    session = get_session()
    url = 'https://query.wikidata.org/sparql'
    
    batches = [missing_qnodes[i:i+BATCH_SIZE] for i in range(0, len(missing_qnodes), BATCH_SIZE)]
    
    for batch in tqdm(batches, desc="Querying Wikidata"):
        query = """
        SELECT ?item ?article ?label WHERE {
          VALUES ?item { """ + " ".join(f"wd:{q}" for q in batch) + """ }
          OPTIONAL {
            ?article schema:about ?item .
            ?article schema:inLanguage "en" .
            ?article schema:isPartOf <https://en.wikipedia.org/> .
          }
          OPTIONAL {
            ?item rdfs:label ?label .
            FILTER(LANG(?label) = "en")
          }
        }
        """
        
        for attempt in range(3):
            try:
                r = session.get(url, params={'query': query}, timeout=30)
                if r.status_code == 429:
                    time.sleep(15)
                    continue
                if not r.ok:
                    time.sleep(5)
                    continue
                    
                results = r.json()['results']['bindings']
                
                for b in results:
                    item_uri = b['item']['value']
                    q_id = item_uri.split('/')[-1]
                    
                    if 'article' in b:
                        # Use Wikipedia Title
                        title = extract_title_from_url(b['article']['value'])
                        mapping[q_id] = title
                    elif 'label' in b:
                        # Fallback to English label (replace spaces with underscores)
                        label = b['label']['value'].replace(' ', '_')
                        mapping[q_id] = label
                    else:
                        # Absolute fallback
                        mapping[q_id] = q_id
                        
                break # Success
            except Exception as e:
                time.sleep(5)
                
        # Save checkpoints
        with open(OUT_DICT, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=4, ensure_ascii=False)
            
        time.sleep(1) # Be nice to Wikidata API
        
    print("Done fetching! Dictionary saved.")

if __name__ == '__main__':
    main()
