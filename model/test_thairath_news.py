"""ทดสอบการค้นหาข่าวจาก Thairath เกี่ยวกับตำรวจบุกทลายเว็บพนัน"""
import requests

NEWS_API_KEY = "277729d09fc640549010e57ecb99c09d"
NEWS_API_BASE_URL = "https://newsapi.org/v2/everything"

# ข่าวที่ผู้ใช้ให้มา
news_text = """ตำรวจบุกทลายรังเว็บพนันเวียดนาม เช่าพื้นที่ในคอนโดฯ หรู ย่านพระราม 9 รวบพนักงานได้เกือบร้อยคน ยึดคอมฯ-มือถืออีกเกือบ 500 เครื่อง แต่ยังไม่มีใครยอมรับเป็นหัวหน้าขบวนการ"""

# ทดสอบคำค้นหาที่เกี่ยวข้อง
test_queries = [
    "police raid Vietnam gambling",
    "Bangkok police illegal gambling",
    "Rama 9 condominium raid",
    "illegal call center Bangkok"
]

print("="*80)
print("ทดสอบการค้นหาข่าวที่เกี่ยวข้องกับ: ตำรวจบุกทลายเว็บพนัน")
print("="*80)

for query in test_queries:
    print(f"\n🔍 Query: {query}")
    
    params = {
        "q": query,
        "domains": "thairath.co.th,manager.co.th,bangkokpost.com,nationthailand.com",
        "sortBy": "publishedAt",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY
    }
    
    response = requests.get(NEWS_API_BASE_URL, params=params)
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Total results: {data.get('totalResults', 0)}")
        
        if data.get("articles"):
            for i, article in enumerate(data["articles"][:3], 1):
                print(f"\n   {i}. {article.get('title', 'No title')[:80]}...")
                print(f"      Source: {article.get('source', {}).get('name', 'Unknown')}")
                print(f"      URL: {article.get('url', '')}")
