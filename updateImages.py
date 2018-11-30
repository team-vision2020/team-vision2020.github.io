import bs4
from urllib.parse import urlparse
import os
from requests import get  # to make GET request

base_path = "img/"

def download(url, file_name):
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        response = get(url)
        # write to file
        file.write(response.content)

# load the file
with open("index_original.html") as inf:
    txt = inf.read()
    soup = bs4.BeautifulSoup(txt)

# create new link
for img_tag in soup.find_all("img"):
    parsed = urlparse(img_tag['src'])
    path = parsed.path

    targPath = os.path.join(base_path, os.path.basename(path))
    download(img_tag['src'], targPath)

    img_tag['src'] = targPath

# save the file again
with open("index.html", "w") as outf:
    outf.write(str(soup))
