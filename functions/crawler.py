import requests
import os
import re

import aiohttp
import asyncio
import aiofiles

# CRAWLER.PY
def fetch_page(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Controlla se ci sono errori HTTP
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Errore durante la richiesta: {e}")
        return None
    
# Download a single URL
async def download_url(session, url, folder_name):
    # HTTP headers to mimic a browser request and avoid being blocked
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36',
        'Referer': 'https://guide.michelin.com/',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    try:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                content = await response.text()
                
                # Extract the restaurant name from the URL so that the HTML could be saved using this name
                match = re.search(r'[^/]+$', url)
                restaurant_name = match.group() if match else "unknown"
                filename = os.path.join(folder_name, f"{restaurant_name}.html")

                # Asynchronously save the content of HTML page to a file
                async with aiofiles.open(filename, 'w') as f:
                    await f.write(content)
                print(f"Downloaded: {url}")
            else:
                # Log the failure if the response status is not 200
                print(f"Failed to download {url}: Status {response.status}")
    except Exception as e:
        # Handle and log any other exceptions during the download process
        print(f"Error downloading {url}: {e}")
  
# Download all URLs, organizing them into folders
async def download_all(urls, output_dir):
    CONCURRENT_REQUESTS = 20  # Limit the number of concurrent requests

    # Set up a connector to limit the number of concurrent connections
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    # Create an HTTP session
    async with aiohttp.ClientSession(connector=connector) as session:
        folder_index = 1

        # Process 20 URLs at a time and save them in different folders
        for i in range(0, len(urls), 20):  
            folder_name = os.path.join(output_dir, f"page_{folder_index}")
            os.makedirs(folder_name, exist_ok=True)
            tasks = [download_url(session, url, folder_name) for url in urls[i:i+20]]
            # Run all tasks concurrently
            await asyncio.gather(*tasks)
            folder_index += 1

# Load URLs from a file
async def load_urls(file_path):
    async with aiofiles.open(file_path, 'r') as f:
        urls = [line.strip() for line in await f.readlines()]
    return urls
