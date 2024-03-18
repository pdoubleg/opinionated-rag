import re
from typing import List, Optional, Union
import urllib.parse
import urllib.request
from bs4 import BeautifulSoup
import requests

from pydantic import BaseModel, HttpUrl, field_validator, Field

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


class Item(BaseModel):
    title: str
    link: HttpUrl
    price: float
    image_url: HttpUrl
    description: Optional[str] = None

    @field_validator("price", mode="before")
    def price_to_float(cls, v: Union[str, float]) -> float:
        return float(v.replace("$", ""))


class Delivery(BaseModel):
    shipping: Optional[str] = None
    location: Optional[str] = None


class Bids(BaseModel):
    count: Optional[int | str] = None
    time_left: Optional[int | str] = None


class eBayProduct(BaseModel):
    ebay_id: int
    item: Item
    condition: Optional[str] = None
    top_rated: Optional[bool] = None
    reviews: Optional[int] = None
    watchers_or_sold: Optional[bool | int | str] = None
    buy_now_extension: Optional[bool | int | str] = None
    delivery: Optional[Delivery] = None
    bids: Optional[Bids] = None

    @classmethod
    def sort_by_price(
        cls, products: List["eBayProduct"], reverse: bool = False
    ) -> List["eBayProduct"]:
        return sorted(products, key=lambda product: product.item.price, reverse=reverse)
    
    def __str__(self):
        display_str = f"{self.item.title}\n{self.condition}\n"
        price_str = f"${self.item.price}\n{self.delivery.shipping}\n\n"
        # Skipping initial characters e.g. "eBay\n"
        description_str = f"Item description from the seller:\n{self.item.description[5:-1]}\n\n"
        id_str = f"eBay item number: {self.ebay_id}\n"
        url_str = f"Listing URL: {str(self.item.link)}\n"
        image_str = f"Image URL: {str(self.item.image_url)}\n"
        display_str += price_str
        display_str += description_str
        display_str += id_str
        display_str += url_str
        display_str += image_str
        display_str += "-"*50
        return display_str


class AvgSalePrice(BaseModel):
    item: str = Field(
        ...,
        description="The description of the item that was searched for."
    )
    count: int = Field(
        ...,
        description="The count of items used to calculate the average."
    )
    price: float = Field(
        ...,
        description="The average item price with outliers excluded."
    )
    shipping: float = Field(
        default=0,
        description="The average shipping cost with outliers excluded."
    )
    total: float = Field(
        ...,
        description="The average cost of the item with shipping, excluding outliers."
    )
    min_price: float = Field(
        ...,
        description="The lowest selling price with outliers excluded."
    )
    max_price: float = Field(
        ...,
        description="The highest selling price with outliers excluded."
    )
    min_shipping: float = Field(
        default=0,
        description="The lowest shipping cost with outliers excluded."
    )
    max_shipping: float = Field(
        default=0,
        description="The highest shipping cost with outliers excluded."
    )
    
    def __str__(self):
        display_string = f"Item Description: {self.item}\n"
        pricing_str = f"Average Price (based on {self.count} sold items):\n"
        item_str = f"* Item: ${self.price} (${self.min_price} to ${self.max_price})\n"
        shipping_str = f"* Shipping: ${self.shipping} (${self.min_shipping} to ${self.max_shipping})\n"
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


def AverageSalePrice(query, country="us", condition="all"):
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

    soup = __GetHTML(query, country, condition, type="all", alreadySold=True)
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
        "max_shipping": maxShipping
    }
    return AvgSalePrice(**results_dict)


def __GetHTML(query, country, condition="", type="all", alreadySold=True):
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
    request = urllib.request.urlopen(url)
    soup = BeautifulSoup(request.read(), "html.parser")

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


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582"
}


def eBayWebSearch(search_item: str) -> List['eBayProduct']:
    html = requests.get(
        f"https://www.ebay.com/sch/i.html?_nkw={search_item}", headers=headers
    ).text
    soup = BeautifulSoup(html, "lxml")

    data = []

    for item in soup.select(".s-item__wrapper.clearfix"):
        title = item.select_one(".s-item__title").text
        link = item.select_one(".s-item__link")["href"]
        
        item_id = link.split('?')[0].split('/')[-1]
        item_descr_url = 'https://vi.vipr.ebaydesc.com/ws/eBayISAPI.dll?item={item_id}'
        item_descr_soup = BeautifulSoup(requests.get(item_descr_url.format(item_id=item_id)).content, 'html.parser')
        description = item_descr_soup.get_text(strip=True, separator='\n')

        image = item.find("img")
        image_url = image["src"]

        try:
            condition = item.select_one(".SECONDARY_INFO").text
        except:
            condition = None

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
            {"ebay_id": item_id,
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
            }
        )
    results = data[1:-1]
    return [eBayProduct(**r) for r in results]


