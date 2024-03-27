import base64
from datetime import datetime
import hashlib
from pathlib import Path
from random import randint, uniform
import re
import textwrap
from time import sleep
from typing import Any, List, Literal, Optional, Union
import urllib.parse
import urllib.request
from bs4 import BeautifulSoup
import httpx
from typing import TypedDict, List, Literal
from urllib.parse import urlencode
import numpy as np
import pandas as pd
from PIL import Image as PILImage

from parsel import Selector

from pydantic import BaseModel, ConfigDict, HttpUrl, field_validator, Field, model_validator
import requests
from tenacity import retry, stop_after_attempt, wait_fixed


SORTING_MAP = {
    "best_match": 12,
    "ending_soonest": 1,
    "newly_listed": 10,
    "ended_recently": 13,
}

countryDict = {
    "au": ".com.au",
    "at": ".at",
    "be": ".be",
    "ca": ".ca",
    "ch": ".ch",
    "de": ".de",
    "es": ".es",
    "fr": ".fr",
    "hk": ".com.hk",
    "ie": ".ie",
    "it": ".it",
    "my": ".com.my",
    "nl": ".nl",
    "nz": ".co.nz",
    "ph": ".ph",
    "pl": ".pl",
    "sg": ".com.sg",
    "uk": ".co.uk",
    "us": ".com",
}

conditionDict = {
    "all": "",
    "new": "&LH_ItemCondition=1000",
    "opened": "&LH_ItemCondition=1500",
    "refurbished": "&LH_ItemCondition=2500",
    "used": "&LH_ItemCondition=3000",
}

typeDict = {
    "all": "&LH_All=1",
    "auction": "&LH_Auction=1",
    "bin": "&LH_BIN=1",
    "offers": "&LH_BO=1",
}

def thumbnail(image, scale=3):
    return image.resize(np.array(image.size)//scale)

def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def convert_timestamp_to_datetime(timestamp: str) -> str:
    return datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d %H:%M:%S")

async_session = httpx.AsyncClient(
    # for our HTTP headers we want to use a real browser's default headers to prevent being blocked
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.35",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    },
    # Enable HTTP2 version of the protocol to prevent being blocked
    http2=True,
    # enable automatic follow of redirects
    follow_redirects=True
)


class Item(BaseModel):
    title: str
    link: HttpUrl
    price: float | str
    image_url: HttpUrl
    description: Optional[str] = None
    image_base64: Optional[str] = Field(
        None, description="Base64 encoded image content"
    )
    
    def fetch_and_encode_image(self) -> None:
        """
        Fetches the image from `image_url`, encodes it as a base64 string, and stores it in `image_base64`.
        If `image_base64` already contains a value, the fetching process is skipped.
        """
        if self.image_base64 is not None:
            return

        response = requests.get(self.image_url)
        if response.status_code == 200:
            self.image_base64 = base64.b64encode(response.content).decode("utf-8")
        else:
            print(f"Failed to fetch image from {self.image_url}")


class Delivery(BaseModel):
    shipping: Optional[str] = " "
    location: Optional[str] = None


class Bids(BaseModel):
    count: Optional[int | str] = None
    time_left: Optional[int | str] = None


class eBayProduct(BaseModel):
    model_config = ConfigDict(
        extra='allow'
    )
    query: Optional[str] = None
    timestamp: Optional[datetime | str] = None
    ebay_id: int
    item: Item
    condition: Optional[str] = None
    top_rated: Optional[bool] = None
    reviews: Optional[int | str] = None
    watchers_or_sold: Optional[bool | int | str] = None
    buy_now_extension: Optional[bool | int | str] = None
    delivery: Optional[Delivery] = None
    bids: Optional[Bids] = None
    listing_date: Optional[str] = None
    subtitles: Optional[List[str]] = None
    rating: Optional[str | int | float] = None
    rating_count: Optional[str] = None
    sale_end_date: Optional[str] = None
    hash_id: Optional[str] = None
    
    @model_validator(mode="before")
    def generate_hash(cls, values):
        unique_string = str(values["ebay_id"])+str(values["item"]["description"])
        values['hash_id'] = hash_text(unique_string)
        return values
    
    @model_validator(mode="before")
    def generate_tiemstamp(cls, values):
        timestamp = datetime.now()
        values['timestamp'] = timestamp.isoformat(timespec='minutes')
        return values
    
    @property
    def to_data_dict(self):
        data_dict = {
            "id": self.ebay_id,
            "title": self.item.title,
            "price": self.item.price,
            "shipping": self.delivery.shipping,
            "location": self.delivery.location,
            "condition": self.condition,
            "text": self.item.description,
            "listing_url": str(self.item.link),
            "image_url": str(self.item.image_url),
            "image_base64": self.item.image_base64,
            "image_uri": str(self.full_path),
            "hash_id": self.hash_id,
            "sale_end_date": self.sale_end_date,
        }
        return data_dict

    @property
    def filename(self):
        filename = re.sub(
            r'[\\/*?:"<>|]', "", str(self.item.title)
        )  # Remove invalid file name characters
        filename = re.sub(r"\s+", "_", filename)  # Replace spaces with underscores
        filename += ".jpg"  # Append file extension
        return filename

    @property
    def full_path(self):
        folder_path: str = "./data/multimodal"
        # Ensure the folder exists
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        # Combine folder path and filename
        file_path = Path(folder_path) / self.filename
        return file_path
    
    @property
    def image(self) -> PILImage.Image:
        if not Path(self.full_path).exists():
            self.download_image()
        return PILImage.open(self.full_path)

    def download_image(
        self,
        folder_path: str = "./data/multimodal",
    ) -> None:
        """
        Downloads an image from a given URL and saves it to a specified folder with a filename
        based on the cleaned title attribute.

        Args:
            folder_path (str): The path to the folder where the image will be saved.

        Returns:
            None
        """
        # Ensure the folder exists
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        # Combine folder path and filename
        file_path = Path(folder_path) / self.filename

        # Download and save the image
        response = requests.get(self.item.image_url)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"Failed to download image from {self.item.image_url}")
            
    
    async def async_process_images(
        self,
        folder_path: str = "./data/multimodal",
    ) -> None:
        """
        Fetches an image from the item's image_url, encodes it as a base64 string, stores it in item.image_base64,
        and saves the image to a specified folder.

        Args:
            folder_path (str): The path to the folder where the image will be saved. Defaults to "./data/multimodal".

        Returns:
            None
        """
        # Ensure the folder exists
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        # Combine folder path and filename
        file_path = Path(folder_path) / self.filename

        # Fetch the image
        response = await async_session.get(str(self.item.image_url))
        if response.status_code == 200:
            # Encode and store the image in base64 format
            self.item.image_base64 = base64.b64encode(response.content).decode("utf-8")

            # Save the image to disk
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"Failed to fetch image from {str(self.item.image_url)}")

    def __str__(self) -> str:
        """
        Returns a string representation of the eBay item, including title, condition, price, shipping,
        subtitles, bid status, rating, description, eBay item number, listing URL, and image URL.

        Returns:
            str: The string representation of the eBay item.
        """
        display_str = f"**{self.item.title}**\n\n{self.condition}\n\n"
        if self.sale_end_date:
            display_str += f"{self.sale_end_date}\n\n"
        price_str = f"{self.item.price}\n\n{self.delivery.shipping}\n\n"
        display_str += price_str
        if self.subtitles:
            display_str += " | ".join(self.subtitles)
            display_str += "\n\n"
        if self.bids.count:
            display_str += f"Bid Status: {self.bids.count} - {self.bids.time_left}\n\n"
        if self.rating:
            display_str += f"{self.rating_count}\n\n{self.rating}\n\n"
        
        # Wrapping the description text to limit width to 100 characters without splitting words
        wrapped_description = textwrap.fill(self.item.description[5:-1], width=100)
        description_str = f"\n\nItem description from the seller:\n\n{wrapped_description}\n\n"
        
        id_str = f"- eBay item number: {self.ebay_id}\n"
        url_str = f"- [Listing link]({str(self.item.link)})\n\n"
        image_str = f"- [Image link]({str(self.item.image_url)})\n\n"
        display_str += description_str
        display_str += url_str
        display_str += image_str
        display_str += id_str
        display_str += "-" * 100
        return display_str


class AvgSalePrice(BaseModel):
    item: str = Field(
        ..., description="The description of the item that was searched for."
    )
    count: int = Field(
        ..., description="The count of items used to calculate the average."
    )
    price: float = Field(
        ..., description="The average item price with outliers excluded."
    )
    shipping: float = Field(
        default=0, description="The average shipping cost with outliers excluded."
    )
    total: float = Field(
        ...,
        description="The average cost of the item with shipping, excluding outliers.",
    )
    min_price: float = Field(
        ..., description="The lowest selling price with outliers excluded."
    )
    max_price: float = Field(
        ..., description="The highest selling price with outliers excluded."
    )
    min_shipping: float = Field(
        default=0, description="The lowest shipping cost with outliers excluded."
    )
    max_shipping: float = Field(
        default=0, description="The highest shipping cost with outliers excluded."
    )

    def __str__(self):
        display_string = f"Item Description: {self.item}\n\n"
        pricing_str = f"Average Price (based on {self.count} sold items):\n\n"
        item_str = f"* Item: ${self.price} (${self.min_price} to ${self.max_price})\n\n"
        shipping_str = f"* Shipping: ${self.shipping} (${self.min_shipping} to ${self.max_shipping})\n\n"
        total_str = f"* Total: ${self.total}"
        display_string += pricing_str
        display_string += item_str
        display_string += shipping_str
        display_string += total_str
        return display_string


def Items(query, country="us", condition="all", type="all"):
    if country not in countryDict:
        raise Exception(
            "Country not supported, please use one of the following: "
            + ", ".join(countryDict.keys())
        )

    if condition not in conditionDict:
        raise Exception(
            "Condition not supported, please use one of the following: "
            + ", ".join(conditionDict.keys())
        )

    if type not in typeDict:
        raise Exception(
            "Type not supported, please use one of the following: "
            + ", ".join(typeDict.keys())
        )

    soup = __GetHTML(query, country, condition, type, alreadySold=False)
    data = __ParseItems(soup)

    return data


def AverageSalePrice(query, country="us", condition="all", retries: int = 10):
    for attempt in range(retries):
        if country not in countryDict:
            raise Exception(
                "Country not supported, please use one of the following: "
                + ", ".join(countryDict.keys())
            )

        if condition not in conditionDict:
            raise Exception(
                "Condition not supported, please use one of the following: "
                + ", ".join(conditionDict.keys())
            )
        try:
            soup = __GetHTML(query, country, condition, type="all", alreadySold=True)
        except Exception as e:
            print(f"An error occurred: {e}")

        data = __ParsePrices(soup)

        avgPrice = round(__Average(data["price-list"]), 2)
        avgShipping = round(__Average(data["shipping-list"]), 2)
        count = len(data["price-list"])

        # Calculate min and max values
        minPrice = min(data["price-list"], default=0)
        maxPrice = max(data["price-list"], default=0)
        minShipping = min(data["shipping-list"], default=0)
        maxShipping = max(data["shipping-list"], default=0)

        results_dict = {
            "item": query,
            "price": avgPrice,
            "shipping": avgShipping,
            "total": round(avgPrice + avgShipping, 2),
            "count": count,
            "min_price": minPrice,
            "max_price": maxPrice,
            "min_shipping": minShipping,
            "max_shipping": maxShipping,
        }
        # return AvgSalePrice(**results_dict)
        search_results = AvgSalePrice(**results_dict)
        if avgPrice > 0:
            return search_results
        else:
            sleep(uniform(0.5, 1.5))
    return str(
        "Sorry, but this search tool is currently down. Please try again in a bit."
    )


session = httpx.Client(
    # for our HTTP headers we want to use a real browser's default headers to prevent being blocked
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.35",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    },
    # Enable HTTP2 version of the protocol to prevent being blocked
    http2=True,
    # enable automatic follow of redirects
    follow_redirects=True,
)


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def __GetHTML(
    query,
    country,
    condition="all",
    type="all",
    alreadySold=True,
):
    alreadySoldString = "&LH_Complete=1&LH_Sold=1" if alreadySold else ""

    # Build the URL
    parsedQuery = urllib.parse.quote(query).replace("%20", "+")
    url = (
        f"https://www.ebay{countryDict[country]}/sch/i.html?_from=R40&_nkw="
        + parsedQuery
        + alreadySoldString
        + conditionDict[condition]
        + typeDict[type]
    )

    # Get the web page HTML
    response = session.get(url)
    response.raise_for_status()  # Raises stored HTTPError, if one occurred
    soup = BeautifulSoup(response.content, "html.parser")

    return soup


def __ParseItems(soup):
    rawItems = soup.find_all("div", {"class": "s-item__info clearfix"})
    data = []

    for item in rawItems[1:]:
        # Get item data
        title = item.find(class_="s-item__title").find("span").get_text(strip=True)

        price = __ParseRawPrice(
            item.find("span", {"class": "s-item__price"}).get_text(strip=True)
        )

        try:
            shipping = __ParseRawPrice(
                item.find("span", {"class": "s-item__shipping s-item__logisticsCost"})
                .find("span", {"class": "ITALIC"})
                .get_text(strip=True)
            )
        except:
            shipping = 0

        try:
            timeLeft = item.find(class_="s-item__time-left").get_text(strip=True)
        except:
            timeLeft = ""

        try:
            timeEnd = item.find(class_="s-item__time-end").get_text(strip=True)
        except:
            timeEnd = ""

        try:
            bidCount = int(
                "".join(
                    filter(
                        str.isdigit,
                        item.find(class_="s-item__bids s-item__bidCount").get_text(
                            strip=True
                        ),
                    )
                )
            )
        except:
            bidCount = 0

        try:
            reviewCount = int(
                "".join(
                    filter(
                        str.isdigit,
                        item.find(class_="s-item__reviews-count")
                        .find("span")
                        .get_text(strip=True),
                    )
                )
            )
        except:
            reviewCount = 0

        try:
            condition = item.find(class_=".SECONDARY_INFO").get_text(strip=True)
        except:
            condition = ""

        url = item.find("a")["href"]

        itemData = {
            "title": title,
            "price": price,
            "shipping": shipping,
            "condition": condition,
            "time-left": timeLeft,
            "time-end": timeEnd,
            "bid-count": bidCount,
            "reviews-count": reviewCount,
            "url": url,
        }

        data.append(itemData)

    # Remove item with prices too high or too low
    priceList = [item["price"] for item in data]
    parsedPriceList = __StDevParse(priceList)
    data = [item for item in data if item["price"] in parsedPriceList]

    return sorted(data, key=lambda dic: dic["price"] + dic["shipping"])


def __ParsePrices(soup):
    # Get item prices
    rawPriceList = [
        price.get_text(strip=True) for price in soup.find_all(class_="s-item__price")
    ]
    priceList = [
        price
        for price in map(lambda rawPrice: __ParseRawPrice(rawPrice), rawPriceList)
        if price != None
    ]

    # Get shipping prices
    rawShippingList = [
        item.get_text(strip=True)
        for item in soup.find_all(class_="s-item__shipping s-item__logisticsCost")
    ]
    shippingList = map(lambda rawPrice: __ParseRawPrice(rawPrice), rawShippingList)
    shippingList = [0 if price == None else price for price in shippingList]

    # Remove prices too high or too low
    priceList = __StDevParse(priceList)
    shippingList = __StDevParse(shippingList)

    data = {"price-list": priceList, "shipping-list": shippingList}
    return data


def __ParseRawPrice(string):
    parsedPrice = re.search("(\d+(.\d+)?)", string.replace(",", "."))
    if parsedPrice:
        return float(parsedPrice.group())
    else:
        return None


def __Average(numberList):
    if len(list(numberList)) == 0:
        return 0
    return sum(numberList) / len(list(numberList))


def __StDev(numberList):
    if len(list(numberList)) <= 1:
        return 0

    nominator = sum(
        map(lambda x: (x - sum(numberList) / len(numberList)) ** 2, numberList)
    )
    stdev = (nominator / (len(numberList) - 1)) ** 0.5

    return stdev


def __StDevParse(numberList):
    avg = __Average(numberList)
    stdev = __StDev(numberList)
    # Remove prices too high or too low; Accept Between -1 StDev to +1 StDev
    numberList = [nmbr for nmbr in numberList if (avg + stdev >= nmbr >= avg - stdev)]
    return numberList


class ProductPreviewResult(TypedDict):
    """type hint for search scrape results for product preview data"""

    url: str  # url to full product page
    title: str
    price: str
    shipping: str
    list_date: str
    subtitles: List[str]
    condition: str
    photo: str  # image url
    rating: str
    rating_count: str


def parse_search(response: httpx.Response) -> List[ProductPreviewResult]:
    """parse ebay's search page for listing preview details"""
    previews = []
    # each listing has it's own HTML box where all of the data is contained
    sel = Selector(response.text)
    listing_boxes = sel.css(".srp-results li.s-item")
    for box in listing_boxes:
        # quick helpers to extract first element and all elements
        css = lambda css: box.css(css).get("").strip()
        css_all = lambda css: box.css(css).getall()
        sale_end_date = css(".s-item__caption-section .POSITIVE::text") or css(".s-item__caption-section .NEGATIVE::text")
        previews.append(
            {
                "url": css("a.s-item__link::attr(href)").split("?")[0],
                "title": css(".s-item__title>span::text"),
                "price": css(".s-item__price::text"),
                "shipping": css(".s-item__shipping::text"),
                "list_date": css(".s-item__listingDate span::text"),
                "subtitles": css_all(".s-item__subtitle::text"),
                "condition": css(".s-item__subtitle .SECONDARY_INFO::text"),
                "photo": css(".s-item__image img::attr(src)"),
                "rating": css(".s-item__reviews .clipped::text"),
                "rating_count": css(".s-item__reviews-count span::text"),
                "sale_end_date": sale_end_date,
            }
        )
    return previews


asession = httpx.Client(
    # for our HTTP headers we want to use a real browser's default headers to prevent being blocked
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.35",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    },
    # Enable HTTP2 version of the protocol to prevent being blocked
    http2=True,
    # enable automatic follow of redirects
    follow_redirects=True,
)

cache = {}

def eBayWebSearch(
    query,
    category=0,
    items_per_page=120,  # 60, 120, 240
    sort: Literal["best_match", "ending_soonest", "newly_listed"] = "best_match",
    alreadySold: bool = False,
) -> List["eBayProduct"]:
    """Scrape Ebay's search for product preview data for given"""
    
    # Generate a unique key for the current query parameters
    cache_key = (query, category, items_per_page, sort, alreadySold)

    # Check if the results for these parameters are already in cache
    if cache_key in cache:
        return cache[cache_key]

    def make_request(page):
        return "https://www.ebay.com/sch/i.html?" + urllib.parse.urlencode(
            {
                "_nkw": query,
                "_sacat": category,
                "_ipg": items_per_page,
                "_sop": SORTING_MAP[sort],
                "_pgn": page,
            }
        )
    if alreadySold:
        parsedQuery = urllib.parse.quote(query).replace("%20", "+")
        url = (
            f"https://www.ebay.com/sch/i.html?_from=R40&_nkw="
            + parsedQuery
            + "&LH_Complete=1&LH_Sold=1"
        )
    else:
        url = make_request(page=1)
    
    response = asession.get(url)
    soup = BeautifulSoup(response.text, "lxml")
    parcel_ = parse_search(response=response)

    data = []
    i = 0

    for item in soup.select(".s-item__wrapper.clearfix"):
        title = item.select_one(".s-item__title").text
        link = item.select_one(".s-item__link")["href"]
        item_id = link.split("?")[0].split("/")[-1]
        item_descr_url = f"https://vi.vipr.ebaydesc.com/ws/eBayISAPI.dll?item={item_id}"
        item_descr_response = session.get(item_descr_url)
        item_descr_soup = BeautifulSoup(item_descr_response.content, "html.parser")
        description = item_descr_soup.get_text(strip=True, separator="\n")

        image = item.find("img")
        image_url = image["src"]

        try:
            condition = item.select_one(".SECONDARY_INFO").text
        except:
            condition = None
            
        try:
            sale_end_date = parcel_[i]["sale_end_date"]
        except:
            sale_end_date = None

        try:
            list_date = parcel_[i]["list_date"]
        except:
            list_date = None

        try:
            subtitles = parcel_[i]["subtitles"]
        except:
            subtitles = None

        try:
            rating = (
                item.select_one(".s-item__reviews .clipped").text
                if item.select_one(".s-item__reviews .clipped")
                else None
            )
        except:
            rating = None

        try:
            rating_count = (
                item.select_one(".s-item__reviews-count span").text
                if item.select_one(".s-item__reviews-count span")
                else None
            )
        except:
            rating_count = None

        try:
            shipping = item.select_one(".s-item__logisticsCost").text
        except:
            shipping = None

        try:
            location = item.select_one(".s-item__itemLocation").text
        except:
            location = None

        try:
            watchers_sold = item.select_one(".NEGATIVE").text
        except:
            watchers_sold = None

        if item.select_one(".s-item__etrs-badge-seller") is not None:
            top_rated = True
        else:
            top_rated = False

        try:
            bid_count = item.select_one(".s-item__bidCount").text
        except:
            bid_count = None

        try:
            bid_time_left = item.select_one(".s-item__time-left").text
        except:
            bid_time_left = None

        try:
            reviews = item.select_one(".s-item__reviews-count span").text.split(" ")[0]
        except:
            reviews = None

        try:
            extension_buy_now = item.select_one(
                ".s-item__purchase-options-with-icon"
            ).text
        except:
            extension_buy_now = None

        try:
            price = item.select_one(".s-item__price").text
        except:
            price = None

        data.append(
            {
                "query": query,
                "ebay_id": item_id,
                "item": {
                    "title": title,
                    "link": link,
                    "price": price,
                    "image_url": image_url,
                    "description": description,
                },
                "condition": condition,
                "top_rated": top_rated,
                "reviews": reviews,
                "watchers_or_sold": watchers_sold,
                "buy_now_extension": extension_buy_now,
                "delivery": {"shipping": shipping, "location": location},
                "bids": {"count": bid_count, "time_left": bid_time_left},
                "listing_date": list_date,
                "subtitles": subtitles,
                "rating": rating,
                "rating_count": rating_count,
                "sale_end_date": sale_end_date,
            }
        )
        i += 1
    search_results = [eBayProduct(**r) for r in data]
    cache[cache_key] = search_results[1:-1]
    
    return search_results[1:-1]
