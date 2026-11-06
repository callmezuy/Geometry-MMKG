import os
import time
import requests
from tqdm import tqdm

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MKGW_ENTITY = os.path.join(CODE_DIR, 'baselines', 'SNAG', 'SNAG_MKGC', 'benchmarks', 'MKG-W', 'entity2id.txt')
MKGY_ENTITY = os.path.join(CODE_DIR, 'baselines', 'SNAG', 'SNAG_MKGC', 'benchmarks', 'MKG-Y', 'entity2id.txt')

HEADERS = {'User-Agent': 'MKG-Research/1.0 (python-requests)'}
BATCH_SIZE = 50

def check_mkg_w():
    print("=== Checking MKG-W (Wikidata) ===")
    qids = []
    with open(MKGW_ENTITY, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:] # skip header
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 1:
                uri = parts[0]
                qid = uri.split('/')[-1]
                qids.append(qid)
                
    total = len(qids)
    print(f"Total MKG-W entities: {total}")
    
    url = "https://query.wikidata.org/sparql"
    found = 0
    batches = [qids[i:i + BATCH_SIZE] for i in range(0, len(qids), BATCH_SIZE)]
    
    for batch in tqdm(batches, desc="Querying Wikidata"):
        values = " ".join([f"wd:{q}" for q in batch])
        query = f"""
        SELECT ?entity ?img WHERE {{
          VALUES ?entity {{ {values} }}
          ?entity wdt:P18 ?img .
        }}
        """
        params = {'query': query, 'format': 'json'}
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                data = r.json()
                results = data['results']['bindings']
                
                # count unique entities that have at least 1 image
                unique_entities = set()
                for b in results:
                    unique_entities.add(b['entity']['value'])
                found += len(unique_entities)
            elif r.status_code == 429:
                time.sleep(2)
        except Exception as e:
            pass
        time.sleep(0.1)
        
    print(f"MKG-W Coverage: {found}/{total} ({(found/total)*100:.2f}%)\n")

def check_mkg_y():
    print("=== Checking MKG-Y (YAGO/DBpedia) ===")
    entities = []
    with open(MKGY_ENTITY, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 1:
                entities.append(parts[0])
                
    total = len(entities)
    print(f"Total MKG-Y entities: {total}")
    
    url = "https://dbpedia.org/sparql"
    found = 0
    batches = [entities[i:i + BATCH_SIZE] for i in range(0, len(entities), BATCH_SIZE)]
    
    for batch in tqdm(batches, desc="Querying DBpedia"):
        values = " ".join([f"<http://dbpedia.org/resource/{e}>" for e in batch])
        query = f"""
        SELECT ?entity ?img WHERE {{
          VALUES ?entity {{ {values} }}
          ?entity <http://xmlns.com/foaf/0.1/depiction> ?img .
        }}
        """
        params = {'query': query, 'format': 'application/sparql-results+json'}
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                data = r.json()
                results = data['results']['bindings']
                unique_entities = set()
                for b in results:
                    unique_entities.add(b['entity']['value'])
                found += len(unique_entities)
            elif r.status_code == 429:
                time.sleep(2)
        except Exception as e:
            pass
        time.sleep(0.5) # dbpedia is slower
        
    print(f"MKG-Y Coverage: {found}/{total} ({(found/total)*100:.2f}%)\n")

if __name__ == '__main__':
    check_mkg_w()
    check_mkg_y()
