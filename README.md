## Point 1. Data collection

This project require to scrape data from [Italian Michelin guide website](https://guide.michelin.com/en/it/restaurants), this process is divided into various steps:

### Technical Implementation

- **Technologies Used:**
  - *requests*: Library for HTTP request
  - *bs4* (BeautifulSoup): Parse HTML and extract information
  - *aiohttp*: Library for asynchronous HTTP requests
  - *asyncio*: Library for asynchronous programming
  - *aiofiles*: Library for asynchronous file I/O
  - *os*: Library for interacting with the operating system
  - *re*: Library for regular expressions, in particual, used to retrive credit card names.

- **Datasets:**
  - At the end of the process, the **restaurant dataset**, that contains information such as restaurant name, city, address, and cuisine types, will be created and stored in a .tsv file (*restaurants_data.tsv*).

### Install dependencies:

Before starting the script, these we install each dependency needed:

```bash
   pip install requests bs4 aiohttp asyncio aiofiles
```

### Steps of the script:

1. #### **Get the list of Michelin restaurants**
   Our script loops through multiple pages, extracting links to individual restaurant pages from the HTML structure using **requests** and **BeautifulSoup**. 
   In particular, to scroll the pages, we look for the button of the active page (button with "active" in its class) and retrieve the link contained in the immediately following button.
   It stops when there are no more pages to scrape or if an error occurs.

   The collected links are saved in a file called *soupUrls.txt*

2. #### **Crawl Michelin restaurant pages** (IMPROVED VERSION)

   Our script downloads HTML content from a list of URLs stored in a file (*soupUrls.txt*) using **asynchronous** and **concurrent** programming with aiohttp, asyncio, and aiofiles to speed up the process. <br />

   It performs the following steps:

   -  **Load URLs**: Reads URLs asynchronously from *soupUrls.txt* file.
   -  **Download Content**: For each URL, sends an HTTP GET request with headers mimicking a browser, saves the response as an HTML file, and names the files uniquely using a hash of the URL to avoid duplicate files.
   -  **Concurrency Management**: Uses asynchronous tasks and a connection limit to efficiently download multiple pages concurrently without overloading the server.
   - **Output Management**: Creates a directory (downloads) to store the downloaded HTML files.

3. #### Parse downloaded pages

   Our script extracts detailed information about restaurants from HTML files downloaded earlier and saves the data into a structured tabular format (TSV file). 
   
   Here's a brief overview:

   -  **Input**: Reads HTML files from the downloads directory.
   -  **Parsing**: Uses BeautifulSoup to parse HTML and extract information needed, at the end, data will be stored in dictionary.
   -  **Output**: Aggregates the restaurant dictionaries into a Pandas DataFrame and saves the data as a TSV file (*restaurants_data.tsv*).

## Point 4. Interactive Map of Italian Regions with restaurants by price range and Top-k restaurants

This project uses Dash and Plotly to create an interactive map of Italian regions. Users can select a region by clicking on the map and view additional details, such as the restaurants in the selected region.

### Features

1. **Visualization of the Map of Italy:**
   - Displays all Italian regions.
   - Each region can be selected by clicking on the map.

2. **Detailed Map of the Selected Region:**
   - Once a region is selected, a detailed map of the region is displayed.
   - Restaurants in the region appear as markers on the map, each with a different color for better viewing.

3. **Interactive Hover:**
   - Hovering over restaurant markers displays detailed information, including:
     - Restaurant name.
     - City.
     - Address.
     - Cuisine types.

4. **Responsive Layout:**
   - The app dynamically adjusts the layout to fit different screen sizes, with both maps displayed side by side, each taking up half the screen width.

### Technical Implementation

- **Technologies Used:**
  - Dash: For building the web application.
  - Plotly: For creating interactive maps.
  - GeoPandas: For handling geospatial data and processing the Italian region boundaries (GeoJSON).
  - Pandas: For data manipulation of the restaurant dataset.

- **Datasets:**
  - The restaurant dataset contains information such as restaurant name, city, address, and cuisine types, along with their geographical coordinates (latitude and longitude).
  - geodata dataset contains region and coordinates information about each restaurant.

### How to Run the Application

1. **Install Dependencies:**
   ```bash
   pip install dash plotly geopandas pandas

2. **Steps to take**
    1. Run Point 3 to create vocabulary.tsv and top_k_restaurant.tsv
    2. Other stuffs
    3. Run Point 4.1 to 

### Screenshot (or GIF) of maps
