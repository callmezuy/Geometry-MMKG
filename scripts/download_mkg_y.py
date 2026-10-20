"""
MKG-Y Image Downloader (DBpedia SPARQL)
=======================================
Queries DBpedia SPARQL endpoint for foaf:depiction links in batches of 50,
then downloads the actual Wikimedia Commons images.
"""
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm

INDEX_FILE   = r'd:\University\Thesis\Code\baselines\SNAG\SNAG_MKGC\benchmarks\MKG-Y\entity2id.txt'
OUTPUT_DIR   = r'd:\University\Thesis\Code\data\MKG-Y\images'
FAILED_LOG   = r'd:\University\Thesis\Code\data\MKG-Y\images_failed.log'

SPARQL_URL   = 'https://dbpedia.org/sparql'
BATCH_SIZE   = 50
DL_WORKERS   = 1  # Single thread for manual-like download
MAX_RETRIES  = 1  # No retries
TIMEOUT      = 20

SPARQL_HEADERS = {
    'User-Agent': 'MKG-Research-Bot/1.0 (minh@example.com) python-requests'
}

DL_HEADERS = {
    'User-Agent': 'MKG-ImageDownloader/1.0 (Research Project; contact: mduy2204@gmail.com)'
}

_local = threading.local()

def get_session() -> requests.Session:
    if not hasattr(_local, 'session'):
        s = requests.Session()
        s.headers.update(DL_HEADERS)
        _local.session = s
    return _local.session

def batch_sparql_lookup(entities: list) -> dict:
    values = " ".join([f"<http://dbpedia.org/resource/{e}>" for e in entities])
    query = f"""
    SELECT ?entity ?img WHERE {{
      VALUES ?entity {{ {values} }}
      ?entity <http://xmlns.com/foaf/0.1/depiction> ?img .
    }}
    """
    params = {'query': query, 'format': 'application/sparql-results+json'}
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(SPARQL_URL, params=params, headers=SPARQL_HEADERS, timeout=15)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            if not r.ok:
                time.sleep(5 * (attempt + 1))
                continue
            bindings = r.json()['results']['bindings']
            result = {}
            for b in bindings:
                entity_uri = b['entity']['value']
                img_url = b['img']['value']
                entity_name = entity_uri.split('/')[-1]
                result[entity_name] = img_url
            return result
        except Exception:
            time.sleep(5 * (attempt + 1))
    return {}

def download_image(img_url: str, save_path: str) -> str:
    session = get_session()
    for attempt in range(3):
        try:
            r = session.get(img_url, stream=True, timeout=TIMEOUT)
            if r.status_code == 404:
                return '404'
            if r.status_code == 429:
                retry_after = int(r.headers.get('Retry-After', 15 * (attempt + 1)))
                time.sleep(retry_after)
                continue
            if not r.ok:
                time.sleep(5 * (attempt + 1))
                continue
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            if os.path.getsize(save_path) < 1024:
                os.remove(save_path)
                return 'small_file'
            return 'ok'
        except Exception as e:
            time.sleep(5 * (attempt + 1))
    return '429_or_failed'

def download_worker(args):
    qid, img_url = args
    save_path = os.path.join(OUTPUT_DIR, f"{qid.replace('/', '_').replace(':', '_')}.jpg")
    if os.path.exists(save_path) and os.path.getsize(save_path) >= 1024:
        return qid, True, 'exists'
    
    # Delay to simulate manual download and avoid rate limit
    time.sleep(1.0)
    
    status = download_image(img_url, save_path)
    return qid, status == 'ok', status

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    entities = []
    with open(INDEX_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:] # skip header
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 1:
                entities.append(parts[0])

    print(f'Total MKG-Y entities: {len(entities)}')
    URL_CACHE_FILE = os.path.join(os.path.dirname(OUTPUT_DIR), 'url_cache.json')
    url_cache = {}
    if os.path.exists(URL_CACHE_FILE):
        import json
        with open(URL_CACHE_FILE, 'r', encoding='utf-8') as f:
            url_cache = json.load(f)

    failed_entities = set()
    if os.path.exists(FAILED_LOG):
        with open(FAILED_LOG, 'r', encoding='utf-8') as f:
            for line in f:
                # Only permanently skip 404 (Link Rot) or SPARQL empty
                if '404' in line or 'no image' in line:
                    failed_entities.add(line.split('\t')[0].strip())

    missing_entities = []
    for e in entities:
        if e in failed_entities:
            continue
        safe_e = e.replace('/', '_').replace(':', '_')
        p = os.path.join(OUTPUT_DIR, f'{safe_e}.jpg')
        if not (os.path.exists(p) and os.path.getsize(p) >= 1024):
            missing_entities.append(e)
            
    already_on_disk = len([e for e in entities if os.path.exists(p := os.path.join(OUTPUT_DIR, f"{e.replace('/', '_').replace(':', '_')}.jpg")) and os.path.getsize(p) >= 1024])
    
    print(f'Already on disk : {already_on_disk}')
    print(f'Failed previously: {len(failed_entities)}')
    print(f'Need to process : {len(missing_entities)}')

    if not missing_entities:
        print('Nothing to do!')
        return

    cached_missing = [e for e in missing_entities if e in url_cache and url_cache[e] is not None]
    need_query = [e for e in missing_entities if e not in url_cache]
    
    print(f'Already in cache: {len(cached_missing)}')
    print(f'Need SPARQL     : {len(need_query)}')

    print('\n[Step 1 & 2] Querying SPARQL and downloading images simultaneously...')
    batches = [need_query[i:i+BATCH_SIZE] for i in range(0, len(need_query), BATCH_SIZE)]
    ok_count = already_on_disk
    fail_count = 0
    no_image_entities = []

    os.makedirs(os.path.dirname(FAILED_LOG), exist_ok=True)
    
    with open(FAILED_LOG, 'a', encoding='utf-8') as flog, ThreadPoolExecutor(max_workers=DL_WORKERS) as pool:
        futures = []
        import json
        
        # Submit tasks that already have URLs in cache
        for e in cached_missing:
            futures.append(pool.submit(download_worker, (e, url_cache[e])))

        # Phase 1: Dispatch SPARQL queries and submit download tasks
        for batch in tqdm(batches, desc='Querying SPARQL'):
            results = batch_sparql_lookup(batch)
            for e in batch:
                if e not in results:
                    url_cache[e] = None
                    no_image_entities.append(e)
                else:
                    url_cache[e] = results[e]
                    futures.append(pool.submit(download_worker, (e, results[e])))
            
            with open(URL_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(url_cache, f)
            time.sleep(0.5)

        print(f'\nImage URLs found : {len(futures)}')
        print(f'No image found   : {len(no_image_entities)}')

        # Phase 2: Wait for downloads to finish and tally results
        with tqdm(total=len(futures), desc='Downloading') as bar:
            for future in as_completed(futures):
                entity, success, msg = future.result()
                if success:
                    ok_count += 1
                else:
                    fail_count += 1
                    flog.write(f'{entity}\t{msg}\n')
                bar.update(1)
                bar.set_postfix(ok=ok_count, fail=fail_count)

    with open(FAILED_LOG, 'a', encoding='utf-8') as flog:
        for e in no_image_entities:
            flog.write(f'{e}\tno image on DBpedia\n')

    print(f'\nFinished!  OK: {ok_count}  |  Download fail: {fail_count}  |  No image: {len(no_image_entities)}')

if __name__ == '__main__':
    main()
