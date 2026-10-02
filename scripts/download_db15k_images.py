"""
DB15K Image Downloader - DBpedia SPARQL Edition
================================================
Queries DBpedia SPARQL endpoint for foaf:depiction links in batches of 50,
then downloads the actual Wikimedia Commons images.

Coverage is much higher than Wikipedia thumbnail API because DBpedia's
foaf:depiction directly reflects the images used when the dataset was built.

Usage:
    python download_db15k_images.py
"""
import os
import time
import random
import urllib.parse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

# ─── Config ──────────────────────────────────────────────────────────────────
INDEX_FILE   = 'data/raw_downloads/mmkb/DB15K/DB15K_ImageIndex.txt'
OUTPUT_DIR   = 'data/DB15K/images'
FAILED_LOG   = 'data/DB15K/images_failed.log'

SPARQL_URL   = 'https://dbpedia.org/sparql'
BATCH_SIZE   = 50    # entities per SPARQL query
DL_WORKERS   = 1    # Single thread for manual-like download
MAX_RETRIES  = 1    # No retries
TIMEOUT      = 20   # seconds per HTTP request

SPARQL_HEADERS = {
    'User-Agent': 'DB15K-DepthResearch/1.0 (Thesis project; python-requests)'
}

DL_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
}
# ─────────────────────────────────────────────────────────────────────────────


# ---------- Thread-local requests session ------------------------------------
_local = threading.local()

def get_session() -> requests.Session:
    if not hasattr(_local, 'session'):
        s = requests.Session()
        s.headers.update(DL_HEADERS)
        _local.session = s
    return _local.session


# ---------- DBpedia SPARQL batch lookup --------------------------------------

def batch_sparql_lookup(uri_to_imgid: dict) -> dict:
    """
    Query DBpedia SPARQL for foaf:depiction of a batch of URIs.
    Returns {img_id: depiction_url} for entities that have images.
    """
    uris = list(uri_to_imgid.keys())

    # Build VALUES clause
    values = ' '.join(f'<{u}>' for u in uris)
    query = f"""
    SELECT ?entity ?img WHERE {{
      VALUES ?entity {{ {values} }}
      ?entity <http://xmlns.com/foaf/0.1/depiction> ?img .
    }}
    """

    params = {
        'query': query,
        'format': 'application/sparql-results+json',
        'timeout': '30000',
    }

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(SPARQL_URL, params=params,
                             headers=SPARQL_HEADERS, timeout=TIMEOUT)
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
                img_url    = b['img']['value']
                img_id     = uri_to_imgid.get(entity_uri)
                if img_id:
                    result[img_id] = img_url   # last one wins if multiple
            return result
        except Exception as e:
            time.sleep(5 * (attempt + 1))

    return {}  # failed after retries


# ---------- Image download ---------------------------------------------------

def download_image(img_url: str, save_path: str) -> str:
    session = get_session()
    try:
        r = session.get(img_url, stream=True, timeout=TIMEOUT)
        if r.status_code == 404:
            return '404'
        if r.status_code == 429:
            return '429'
        if not r.ok:
            return str(r.status_code)
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        # Sanity-check: reject tiny files (likely HTML error pages)
        if os.path.getsize(save_path) < 1024:
            os.remove(save_path)
            return 'small_file'
        return 'ok'
    except Exception as e:
        return type(e).__name__


def download_worker(args):
    img_id, img_url = args
    save_path = os.path.join(OUTPUT_DIR, f'{img_id}.jpg')
    if os.path.exists(save_path) and os.path.getsize(save_path) >= 1024:
        return img_id, True, 'exists'
        
    # Delay to simulate manual download and avoid rate limit
    time.sleep(1.0)
    
    status = download_image(img_url, save_path)
    return img_id, status == 'ok', status


# ---------- Main -------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Parse index file: URI → DBIMG_ID
    uri_to_imgid = {}
    with open(INDEX_FILE, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            uri = parts[0].strip('<>').strip()
            img_id = parts[1].strip()
            uri_to_imgid[uri] = img_id

    total = len(uri_to_imgid)
    print(f'Total entities  : {total}')

    # Load failed entities to permanently skip 404s
    failed_entities = set()
    if os.path.exists(FAILED_LOG):
        with open(FAILED_LOG, 'r', encoding='utf-8') as f:
            for line in f:
                if '404' in line or 'no depiction' in line:
                    failed_entities.add(line.split('\t')[0].strip())
                    
    # Filter out already-downloaded entities and permanent failures
    missing_uri_to_imgid = {
        uri: img_id for uri, img_id in uri_to_imgid.items()
        if img_id not in failed_entities and not (
            os.path.exists(p := os.path.join(OUTPUT_DIR, f'{img_id}.jpg'))
            and os.path.getsize(p) >= 1024
        )
    }
    already = len([id for u, id in uri_to_imgid.items() if os.path.exists(p := os.path.join(OUTPUT_DIR, f'{id}.jpg')) and os.path.getsize(p) >= 1024])
    print(f'Already on disk : {already}')
    print(f'Failed previously (404) : {len(failed_entities)}')
    
    # Load URL cache
    URL_CACHE_FILE = os.path.join(os.path.dirname(OUTPUT_DIR), 'url_cache.json')
    url_cache = {}
    if os.path.exists(URL_CACHE_FILE):
        import json
        with open(URL_CACHE_FILE, 'r', encoding='utf-8') as f:
            url_cache = json.load(f)

    # Filter out those that are fully cached (found or explicitly no-image)
    cached_missing = {u: id for u, id in missing_uri_to_imgid.items() if id in url_cache and url_cache[id] is not None}
    no_image_cached = {u: id for u, id in missing_uri_to_imgid.items() if id in url_cache and url_cache[id] is None}
    need_query = {u: id for u, id in missing_uri_to_imgid.items() if id not in url_cache}

    print(f'Missing on disk : {len(missing_uri_to_imgid)}')
    print(f'No image (cache): {len(no_image_cached)}')
    print(f'In cache (found): {len(cached_missing)}')
    print(f'Need SPARQL     : {len(need_query)}')

    if len(cached_missing) == 0 and len(need_query) == 0:
        print('Nothing to do!')
        return

    # Step 1: Batch SPARQL lookup to get depiction URLs
    img_id_to_url = {id: url_cache[id] for u, id in cached_missing.items()}
    no_image_ids  = list(no_image_cached.values())

    if need_query:
        print('\n[Step 1] Querying DBpedia SPARQL for missing URLs...')
        uris = list(need_query.keys())
        batches = [uris[i:i+BATCH_SIZE] for i in range(0, len(uris), BATCH_SIZE)]

        import json
        for batch in tqdm(batches, desc='SPARQL batches'):
            batch_map = {u: need_query[u] for u in batch}
            results   = batch_sparql_lookup(batch_map)
            
            for u, img_id in batch_map.items():
                if img_id in results:
                    url_cache[img_id] = results[img_id]
                    img_id_to_url[img_id] = results[img_id]
                else:
                    url_cache[img_id] = None
                    no_image_ids.append(img_id)
            
            # Save cache incrementally
            with open(URL_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(url_cache, f)
                
            time.sleep(0.5)  # be polite between batches

    print(f'\nTotal Depiction URLs found : {len(img_id_to_url)}')
    print(f'Total No depiction found   : {len(no_image_ids)}')

    # Step 2: Download images in parallel
    print('\n[Step 2] Downloading images...')
    ok_count   = already
    fail_count = 0

    with open(FAILED_LOG, 'a', encoding='utf-8') as flog, \
         ThreadPoolExecutor(max_workers=DL_WORKERS) as pool:

        futures = [pool.submit(download_worker, item) for item in img_id_to_url.items()]

        with tqdm(total=len(futures), desc='Downloading') as bar:
            for future in as_completed(futures):
                img_id, success, msg = future.result()
                if success:
                    ok_count += 1
                else:
                    fail_count += 1
                    flog.write(f'{img_id}\t{msg}\n')
                bar.update(1)
                bar.set_postfix(ok=ok_count, fail=fail_count)

    # Log entities with no depiction at all
    with open(FAILED_LOG, 'a', encoding='utf-8') as flog:
        for img_id in no_image_ids:
            flog.write(f'{img_id}\tno depiction in DBpedia\n')

    print(f'\nFinished!  OK: {ok_count}  |  Download fail: {fail_count}  |  No image: {len(no_image_ids)}')
    print(f'Fail log  : {FAILED_LOG}')


if __name__ == '__main__':
    main()
