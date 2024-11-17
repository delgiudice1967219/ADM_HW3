import requests
import os
import re

import aiohttp
import asyncio
import aiofiles

from bs4 import BeautifulSoup

def crawl_restaurant_links(base_url, txt_out_pathname):
    print("I'm starting to Scrape!")
    page = 1
    # Link list to return as a file.txt
    all_links = []

    while True:
        # Build URL of current page which is base_url + # of the current page
        url = f"{base_url}{page}"
        print(f"Fetching {url}...")
        response = requests.get(url)

        # Check if the HTTP request was successful, otherwise stop scraping and print error message
        if response.status_code != 200:
            print(f"Errore nel caricamento della pagina {page}")
            break

        # Parsing of the HTML page using Beautiful Soup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Dig in HTML structure to search every restaurnt URL in the current page, first element is the <div> tag that has following class
        for class1_div in soup.select("div.card__menu-content.card__menu-content--flex.js-match-height-content"):
            # Next is the <h3> tag that has the following class, this tag also has an attribute "href" 
            # with the link of the current restaurant, we need to retrive this link 
            h3 = class1_div.select_one("h3.card__menu-content--title.pl-text.pl-big.js-match-height-title a")
            if h3:
                link = h3.get("href")
                # Build the link and add to link list
                full_link = "https://guide.michelin.com" + link if link else None
                if full_link:
                    all_links.append(full_link)


        # In this section of the scraping algorithm, we search for the "next page" button so that
        # we can scrape the other pages links.

        # First step is to find the div that contains pages button which is the div with the followind class
        pagination_lis = soup.select("div.js-restaurant__bottom-pagination ul li")

        # Next, we should find the button which has "active" in the class (which indicate the current page)
        active_index = None
        for i, li in enumerate(pagination_lis):
            if li.select_one("a.active"):
                active_index = i
                break

        # After we have found the current button, we search for immediate next button, 
        # who contains link to the next page in the <a> tag, if next page is present.
        if active_index is not None and active_index + 1 < len(pagination_lis):
            next_page = pagination_lis[active_index + 1].select_one("a")
            if next_page and next_page.get("href"):
                page += 1
            else:
                break
        else:
            break

    # Save every URL in a .txt file
    with open(txt_out_pathname, "w") as file:
        for link in all_links:
            file.write(link + "\n")

    print(f"Crawling completed. {len(all_links)} link saved in michelin_restaurant_urls.txt.")

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
