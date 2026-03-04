import requests

# ทดสอบ NewsAPI โดยตรง
NEWS_API_KEY = "277729d09fc640549010e57ecb99c09d"
NEWS_API_BASE_URL = "https://newsapi.org/v2/everything"

query = "นายกรัฐมนตรี"
params = {
    "q": query,
    "language": "th",
    "sortBy": "relevancy",
    "pageSize": 3,
    "apiKey": NEWS_API_KEY
}

print(f"Testing NewsAPI with query: {query}")
print(f"URL: {NEWS_API_BASE_URL}")
print(f"Params: {params}")
print()

try:
    response = requests.get(NEWS_API_BASE_URL, params=params, timeout=10)
    print(f"Status Code: {response.status_code}")
    
    data = response.json()
    print(f"Response status: {data.get('status')}")
    print(f"Total results: {data.get('totalResults', 0)}")
    
    if data.get('articles'):
        print(f"\nFound {len(data['articles'])} articles:")
        for i, article in enumerate(data['articles'], 1):
            print(f"\n{i}. {article.get('title', 'N/A')}")
            print(f"   Source: {article.get('source', {}).get('name', 'N/A')}")
            print(f"   URL: {article.get('url', 'N/A')[:60]}...")
    else:
        print("\n❌ No articles found")
        print(f"Response: {data}")
        
except Exception as e:
    print(f"❌ Error: {e}")
