# Point 1.1 - PARSER.PY
def scrape_restaurant_links(base_url):
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
    with open("michelin_restaurant_urls.txt", "w") as file:
        for link in all_links:
            file.write(link + "\n")

    print(f"Scraping completed. {len(all_links)} link saved in michelin_restaurant_urls.txt.")

# Point 1.2 - CRAWLER.PY

# CRAWLER.PY
# Load URLs from a file
async def load_urls(file_path):
    async with aiofiles.open(file_path, 'r') as f:
        urls = [line.strip() for line in await f.readlines()]
    return urls

# CRAWLER.PY
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

# CRAWLER.PY
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

# Point 1.3

# PARSER.PY
# Function to extract restaurant information from HTML
def extract_restaurant_info(file_path):
    # Open HTML file of current restaurant and parse it using BeautifulSoup
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

        # Create dictionary that will contains current restaurant info
        restaurant_info = {}

        # Everything we need is contained in a <div> tag that has the class below
        restaurantDetailsDiv = soup.find("div", class_="restaurant-details__components")

        # Select each nested <div> with class "row":
            # First div contains "restaurant name"
            # Second div contains address, price range, cuisine type 
            # Third div will be ignored
        mainInfo = restaurantDetailsDiv.select("div.data-sheet > div.row")

        if mainInfo[0]:
            restaurant_info['restaurantName'] = mainInfo[0].find("h1", class_="data-sheet__title").text
        if mainInfo[1]:
            indirizzo_price = mainInfo[1].select("div.data-sheet__block > div.data-sheet__block--text")

            # Split address string by comma
            indirizzoList = indirizzo_price[0].text.strip().split(",")


            # Given that different type of address can be found, we retrive data from the inverted and splitted string, so that:
                # First element is always the city
                # Second element is always the postalCode
                # Third element is always the country
                # The remaining string is the full address, so we store everything in "address" key of the dictionary
            restaurant_info['city'] = indirizzoList[-3]
            restaurant_info['postalCode'] = indirizzoList[-2]
            restaurant_info['country'] = indirizzoList[-1]
            restaurant_info['address'] = " ".join(indirizzoList[:-3]).strip().replace("\n", "")

            # Cuisine Type and Price Range are in the same string and are separated by "·", so we split them apart
            restaurant_info['priceRange'], restaurant_info['cuisineType'] = indirizzo_price[1].text.strip().split("·")

            # Price Range
            restaurant_info['priceRange'] = restaurant_info['priceRange'].strip()
            # List of cuisine types
            restaurant_info['cuisineType'] = restaurant_info['cuisineType'].strip().split(",")

        # Description
        restaurant_info['description'] = soup.find("div", class_="data-sheet__description").text.strip().replace("\n", "")

        # Facilities and Services
        facilities = soup.select("div.restaurant-details__services ul li")
        restaurant_info['facilitiesServices'] = [s.text.strip() for s in facilities]

        # Accepted Credit Cards
        credit_cards = soup.select("div.list--card img")
        restaurant_info['creditCards'] = [re.search(r"(?<=\/)[a-z]*(?=-)", c.get("data-src"))[0] for c in credit_cards]

        # Phone Number
        spansDetails = restaurantDetailsDiv.select("section.section.section-main.section__text-componets.section__text-separator div.collapse__block-title div.d-flex span")
        restaurant_info['phoneNumber'] = spansDetails[0].text.strip()

        # URL
        restaurant_info['website'] = soup.find("meta", property="og:url")["content"]

    return restaurant_info

# UTIL.PY
# Function that iterate over folders to scrape each file HTML
def iterate_folders(output_dir):
    # List to store dictionary data current restaurant
    restaurants_data = []

    # Iterate over each restaurant folder and scrape each HTML page
    for folder in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, folder)
        
        # Check that path is a directory
        if os.path.isdir(folder_path):
            # Iterate over files in the current folder
            for filename in os.listdir(folder_path):
                # Check if current file is .html, if so scrape the page
                if filename.endswith(".html"):  
                    print(f"Processing: {filename}")
                    file_path = os.path.join(folder_path, filename)
                    restaurant_info = extract_restaurant_info(file_path)
                    # Add dictionary data to the list of all restaurant data
                    restaurants_data.append(restaurant_info)

    return restaurants_data

# Point 4.1

# PARSER.PY
# Function to extract restaurant information from HTML
def extract_geo_info(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

        # Extract information using HTML structure of the page
        restaurant_info = {}

        script_tag = soup.find("script", type="application/ld+json")

        if script_tag:
            json_content = json.loads(script_tag.string)

            restaurant_info['restaurantName'] = json_content['name']
            restaurant_info['region'] = json_content['address']['addressRegion']
            restaurant_info['latitude'] = json_content['latitude']
            restaurant_info['longitude'] = json_content['longitude']

    return restaurant_info

# UTIL.PY
def iterate_geo_folders(output_dir):
    # List to store restaurant data
    restaurants_data = []

    # Loop through all files in the directory and extract information
    # Itera attraverso le pagine da 1 a 100
    for page_num in range(1, 101):
        # Costruisci il percorso della cartella per ciascuna pagina
        folder_path = os.path.join(output_dir, f"page_{page_num}")
        
        # Trova tutti i file di testo che iniziano con "html_" nella cartella corrente
        html_files = glob.glob(os.path.join(folder_path, "html_*.txt"))
        
        # Itera attraverso i file trovati e processa ciascuno
        for file_path in html_files:
            print(file_path)
            restaurant_info = extract_geo_info(file_path)
            restaurants_data.append(restaurant_info)

    return restaurants_data

# UTIL.PY
def translateRegions(regionColumn):
    translations = {
        "Aosta Valley": "Valle d'Aosta/Vallée d'Aoste",
        "Piedmont": "Piemonte",
        "Lombardy": "Lombardia",
        "Sicily": "Sicilia",
        "Tuscany": "Toscana",
        "Apulia": "Puglia",
        "Trentino-South Tyrol": "Trentino-Alto Adige/Südtirol",
        "Sardinia": "Sardegna"
    }
    regionColumn = regionColumn.replace(translations) 

    return regionColumn

# CRAWLER.PY
def fetch_page(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Controlla se ci sono errori HTTP
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Errore durante la richiesta: {e}")
        return None