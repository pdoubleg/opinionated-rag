import asyncio
from collections import defaultdict
import json
import math
import os
from typing import List, Literal, TypedDict
from urllib.parse import urlencode
import pandas as pd
import httpx
from parsel import Selector
import nested_lookup



# establish our HTTP2 client with browser-like headers
session = httpx.Client(
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.35",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    },
    http2=True,
    follow_redirects=True
)


def parse_product(response: httpx.Response) -> dict:
    """Parse Ebay's product listing page for core product data"""
    sel = Selector(response.text)
    # define helper functions that chain the extraction process
    css_join = lambda css: "".join(sel.css(css).getall()).strip()  # join all CSS selected elements
    css = lambda css: sel.css(css).get("").strip()  # take first CSS selected element and strip of leading/trailing spaces

    item = {}
    item["url"] = css('link[rel="canonical"]::attr(href)')
    item["id"] = item["url"].split("/itm/")[1].split("?")[0]  # we can take ID from the URL
    item["price"] = css('.x-price-primary>span::text')
    item["name"] = css_join("h1 span::text")
    item["seller_name"] = css_join("[data-testid=str-title] a ::text")
    item["seller_url"] = css("[data-testid=str-title] a::attr(href)").split("?")[0]
    item["photos"] = sel.css('.ux-image-filmstrip-carousel-item.image img::attr("src")').getall()  # carousel images
    item["photos"].extend(sel.css('.ux-image-carousel-item.image img::attr("src")').getall())  # main image
    # description is an iframe (independant page). We can keep it as an URL or scrape it later.
    item["description_url"] = css("div.d-item-description iframe::attr(src)")
    if not item["description_url"]:
        item["description_url"] = css("div#desc_div iframe::attr(src)")
    # feature details from the description table:
    feature_table = sel.css("div.ux-layout-section--features")
    features = {}
    for ft_label in feature_table.css(".ux-labels-values__labels"):
        # iterate through each label of the table and select first sibling for value:
        label = "".join(ft_label.css(".ux-textspans::text").getall()).strip(":\n ")
        ft_value = ft_label.xpath("following-sibling::div[1]")
        value = "".join(ft_value.css(".ux-textspans::text").getall()).strip()
        features[label] = value
    item["features"] = features
    return item


def find_json_objects(text: str, decoder=json.JSONDecoder()):
    """Find JSON objects in text, and generate decoded JSON data"""
    pos = 0
    while True:
        match = text.find("{", pos)
        if match == -1:
            break
        try:
            result, index = decoder.raw_decode(text[match:])
            yield result
            pos = match + index
        except ValueError:
            pos = match + 1
            
            
def parse_variants(response: httpx.Response) -> dict:
    """
    Parse variant data from Ebay's listing page of a product with variants.
    This data is located in a js variable MSKU hidden in a <script> element.
    """
    selector = Selector(response.text)
    script = selector.xpath('//script[contains(., "MSKU")]/text()').get()
    if not script:
        return {}
    all_data = list(find_json_objects(script))
    data = nested_lookup("MSKU", all_data)[0]
    # First retrieve names for all selection options (e.g. Model, Color)
    selection_names = {}
    for menu in data["selectMenus"]:
        for id_ in menu["menuItemValueIds"]:
            selection_names[id_] = menu["displayLabel"]
    # example selection name entry:
    # {0: 'Model', 1: 'Color', ...}

    # Then, find all selection combinations:
    selections = []
    for v in data["menuItemMap"].values():
        selections.append(
            {
                "name": v["valueName"],
                "variants": v["matchingVariationIds"],
                "label": selection_names[v["valueId"]],
            }
        )
    # example selection entry:
    # {'name': 'Gold', 'variants': [662315637181, 662315637177, 662315637173], 'label': 'Color'}

    # Finally, extract variants and apply selection details to each
    results = []
    variant_data = nested_lookup("variationsMap", data)[0]
    for id_, variant in variant_data.items():
        result = defaultdict(list)
        result["id"] = id_
        for selection in selections:
            if int(id_) in selection["variants"]:
                result[selection["label"]] = selection["name"]
        result["price_original"] = variant["binModel"]["price"]["value"]["convertedFromValue"]
        result["price_original_currency"] = variant["binModel"]["price"]["value"]["convertedFromCurrency"]
        result["price_converted"] = variant["binModel"]["price"]["value"]["value"]
        result["price_converted_currency"] = variant["binModel"]["price"]["value"]["currency"]
        result["out_of_stock"] = variant["quantity"]["outOfStock"]
        results.append(dict(result))
    # example variant entry:
    # {
    #     'id': '662315637173',
    #     'Model': 'Apple iPhone 11 Pro Max',
    #     'Storage Capacity': '64 GB',
    #     'Color': 'Gold',
    #     'price_original': 469,
    #     'price_original_currency': 'CAD',
    #     'price_converted': 341.55,
    #     'price_converted_currency': 'USD',
    #     'out_of_stock': False
    # }
    return results


asession = httpx.AsyncClient(
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
            }
        )
    return previews


SORTING_MAP = {
    "best_match": 12,
    "ending_soonest": 1,
    "newly_listed": 10,
}

async def scrape_search(
    query,
    max_pages=1,
    category=0,
    items_per_page=12,
    sort: Literal["best_match", "ending_soonest", "newly_listed"] = "newly_listed",
) -> List[ProductPreviewResult]:
    """Scrape Ebay's search for product preview data for given"""

    def make_request(page):
        return "https://www.ebay.com/sch/i.html?" + urlencode(
            {
                "_nkw": query, # for search keyword
                "_sacat": category, # category restriction
                "_ipg": items_per_page, # listings per page (default 60)
                "_sop": SORTING_MAP[sort], # sorting type
                "_pgn": page, # page number
            }
        )

    first_page = await asession.get(make_request(page=1))
    results = parse_search(first_page)
    if max_pages == 1:
        return results
    # find total amount of results for concurrent pagination
    total_results = first_page.selector.css(".srp-controls__count-heading>span::text").get()
    total_results = int(total_results.replace(",", ""))
    total_pages = math.ceil(total_results / items_per_page)
    if total_pages > max_pages:
        total_pages = max_pages
    other_pages = [session.get(make_request(page=i)) for i in range(2, total_pages + 1)]
    for response in asyncio.as_completed(other_pages):
        response = await response
        try:
            results.extend(parse_search(response))
        except Exception as e:
            print(f"failed to scrape search page {response.url}")
    return results