"""ทดสอบ NewsAPI ด้วยคำค้นหาภาษาอังกฤษ"""
import requests

NEWS_API_KEY = "277729d09fc640549010e57ecb99c09d"
NEWS_API_BASE_URL = "https://newsapi.org/v2/everything"

# ทดสอบคำค้นหาภาษาอังกฤษ
test_queries = [
    "Thailand Prime Minister",
    "Bangkok Thailand",
    "Thailand news",
    "Thai government"
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"🔍 Testing query: {query}")
    print('='*60)
    
    # ทดสอบด้วย domains
    params = {
        "q": query,
        "domains": "thairath.co.th,manager.co.th,bangkokpost.com,nationthailand.com",
        "sortBy": "relevancy",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY
    }
    
    response = requests.get(NEWS_API_BASE_URL, params=params)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Results: {data.get('totalResults', 0)}")
        
        if data.get("articles"):
            for i, article in enumerate(data["articles"][:3], 1):
                print(f"\n{i}. {article.get('title', 'No title')}")
                print(f"   Source: {article.get('source', {}).get('name', 'Unknown')}")
                print(f"   URL: {article.get('url', '')[:50]}...")
