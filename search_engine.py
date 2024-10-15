import re
import requests
import pandas as pd

def build_payload(query, start=1, num=10, date_restrict='d1', **params):
    """
    Function to build the payload for the Google Search API request.
    
    :param query: Search term
    :param start: The index of the first result to return
    :param link_site: Specifies that all search results should contain a link to a particular URL
    :param search_type: Type of search (default is undefined, 'IMAGE' for image search)
    :param date_restrict: Restricts results based on recency (default is one month "mi")
    """
    payload={
        'key': API_KEY,
        'q': query,
        'cx': SEARCH_ENGINE_ID,
        'start': start,
        'num': num,
        'dateRestrict':date_restrict
    }

    payload.update(params)
    return payload

def make_request(payload):
    response = requests.get('https://www.googleapis.com/customsearch/v1', params=payload)
    if response.status_code !=200:
        raise Exception('Request Failed')
    return response.json()

def main(query, result_total=10):
    items=[]
    reminder=result_total%10
    if reminder>0:
        pages = (result_total//10) + 1
    else:
        pages = (result_total//10)

    for i in range(pages):
        if pages == i+1 and reminder>0:
            payload = build_payload(query, start=(i+1)*10, num=reminder)
        else:
            payload = build_payload(query, start=(i+1)*10)
        response = make_request(payload)
        items.extend(response['items'])
    # query_string_clean = clean_filename(query)
    df=pd.json_normalize(items)
    print(f"{df['link']} || {df['title']}")
    df.to_excel('Google_Search_result_stocks.xlsx', index=False)


if __name__ == '__main__':
    API_KEY= 'AIzaSyD1pPu85DAKaFKRJi4ASb4gTvUlwdSxQGs'
    SEARCH_ENGINE_ID= 'd1bf03b5898ee4339'
    search_query = 'Top stocks to invest today'
    total_results= 12
    main(search_query, total_results)