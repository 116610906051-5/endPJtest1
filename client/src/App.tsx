import { useState } from 'react'
import './App.css'
import { API_URL } from './config'

async function checkNews(text: string) {
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });

  const data = await res.json();
  return data; // { confidence: number, decision_score: number }
}

interface RelatedNews {
  title: string;
  source: string;
  url: string;
  similarity: number;
  is_trusted: boolean;
  publishedAt?: string;
}

function App() {
  const [currentPage, setCurrentPage] = useState<'home' | 'test'>('home')
  const [inputText, setInputText] = useState('')
  const [result, setResult] = useState<{ 
    isAnalyzing: boolean; 
    verdict: string | null;
    confidence?: number;
    relatedNews?: RelatedNews[];
    verificationNote?: string;
    trustedSourcesFound?: number;
  }>({
    isAnalyzing: false,
    verdict: null
  })

  const handleAnalyze = async () => {
    setResult({ isAnalyzing: true, verdict: null })
    
    try {
      const data = await checkNews(inputText)
      const confidence = data.confidence
      const relatedNews = data.related_news || []
      const verificationNote = data.verification_note
      const trustedSourcesFound = data.trusted_sources_found || 0
      
      let verdict = ''
      if (confidence >= 80) {
        verdict = `ระดับความน่าเชื่อถือสูง (${confidence}%)`
      } else if (confidence >= 60) {
        verdict = `ระดับความน่าเชื่อถือปานกลาง-สูง (${confidence}%)`
      } else if (confidence >= 40) {
        verdict = `ระดับความน่าเชื่อถือปานกลาง (${confidence}%)`
      } else if (confidence >= 20) {
        verdict = `ระดับความน่าเชื่อถือต่ำ-ปานกลาง (${confidence}%)`
      } else {
        verdict = `ระดับความน่าเชื่อถือต่ำ (${confidence}%)`
      }
      
      setResult({ 
        isAnalyzing: false, 
        verdict, 
        confidence,
        relatedNews,
        verificationNote,
        trustedSourcesFound
      })
    } catch (error) {
      console.error('Error:', error)
      setResult({ 
        isAnalyzing: false, 
        verdict: '⚠️ เกิดข้อผิดพลาด - ไม่สามารถเชื่อมต่อกับเซิร์ฟเวอร์ได้' 
      })
    }
  }

  if (currentPage === 'home') {
    return (
      <div className="app-container home-page">
        <div className="home-content">
          <div className="fake-news-box">
            <div className="fake-news-text">
              <h1 className="fake-title">Fake</h1>
              <h1 className="news-title">News</h1>
            </div>
            <div className="black-square"></div>
          </div>
          <button className="start-btn" onClick={() => setCurrentPage('test')}>
            เริ่มทดสอบ
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="app-container test-page">
      <nav className="navbar">
        <div className="nav-left">
          <div className="user-avatar">
            <img src="https://media.discordapp.net/attachments/1324308072058851361/1470787788914823351/image.png?ex=6992801e&is=69912e9e&hm=efe1c7ba43e5b20edc14d37c54283491db0cf94877151f2a0633c6296e37409d&=&format=webp&quality=lossless" alt="User" />
          </div>
          <span className="username">Fake News</span>
        </div>
        <div className="nav-right">
          <button className="nav-btn" onClick={() => setCurrentPage('home')}>Home</button>
          <button className="nav-btn">How</button>
          <button className="nav-btn">Report</button>
          <button className="nav-btn">Contact</button>
        </div>
      </nav>

      <main className="main-content">
        <div className="test-container">
          <h2 className="test-title">ทดสอบข่าวปลอมและความน่าเชื่อถือของข่าวที่ได้</h2>
          
          <div className="search-box">
            <textarea
              className="search-input"
              placeholder="ค้นหา URL,ข้อความ, hashtag(#), keyword"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              rows={1}
              onInput={(e) => {
                const target = e.target as HTMLTextAreaElement;
                target.style.height = 'auto';
                target.style.height = Math.min(target.scrollHeight, 200) + 'px';
              }}
            />
            <button 
              className="check-btn"
              onClick={handleAnalyze}
              disabled={result.isAnalyzing || !inputText.trim()}
            >
              {result.isAnalyzing ? '⏳' : '🔍 CHECK IT'}
            </button>
          </div>

          {result.verdict && (
            <div className="result-section">
              <h3>📊 ผลการวิเคราะห์</h3>
              <div className={`result-card ${
                result.confidence ? (
                  result.confidence >= 60 ? 'success' :
                  result.confidence >= 40 ? 'warning' : 'danger'
                ) : 'warning'
              }`}>
                <p>{result.verdict}</p>
                {result.confidence !== undefined && (
                  <div className="confidence-bar-container">
                    <div className="confidence-bar" style={{ width: `${result.confidence}%` }}>
                      <span className="confidence-text">{result.confidence}%</span>
                    </div>
                  </div>
                )}
              </div>

              {result.confidence !== undefined && (
                <div className="confidence-info">
                  <p style={{ fontSize: '0.9rem', color: '#ccc', marginTop: '10px' }}>
                    💡 ยิ่งค่าความน่าเชื่อถือสูง แสดงว่าข้อมูลมีแนวโน้มเป็นข่าวจริงมากขึ้น
                  </p>
                </div>
              )}

              {result.relatedNews && result.relatedNews.length > 0 && (
                <div className="related-news-section">
                  <h3>📰 ข่าวที่เกี่ยวข้องจากแหล่งที่น่าเชื่อถือ</h3>
                  {result.verificationNote && (
                    <p className="verification-note">
                      ℹ️ {result.verificationNote}
                    </p>
                  )}
                  <div className="related-news-grid">
                    {result.relatedNews.map((news, index) => (
                      <div key={index} className="news-card">
                        <div className="news-header">
                          <span className="news-source">
                            🌐 {news.source}
                            {news.is_trusted && <span className="trusted-badge">✓ เชื่อถือได้</span>}
                          </span>
                          <span className="news-similarity">
                            {news.similarity}% ความคล้าย
                          </span>
                        </div>
                        <h4 className="news-title">{news.title}</h4>
                        <a 
                          href={news.url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="news-link"
                        >
                          📖 อ่านบทความเต็ม →
                        </a>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

export default App