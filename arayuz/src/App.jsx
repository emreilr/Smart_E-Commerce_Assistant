import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [yorum, setYorum] = useState('')
  const [sonuc, setSonuc] = useState(null)
  const [yukleniyor, setYukleniyor] = useState(false)

  const analizEt = async () => {
    if (!yorum) return;

    setYukleniyor(true);
    setSonuc(null);

    try {
      // Python API'ye istek atıyoruz
      const response = await axios.post('http://127.0.0.1:8000/analiz-et', {
        yorum: yorum
      });

      setSonuc(response.data);
    } catch (error) {
      console.error("Hata oluştu:", error);
      alert("Bağlantı hatası!");
    } finally {
      setYukleniyor(false);
    }
  }

  return (
    <div className="container">
      <h1>🛍️ Akıllı E-Ticaret Asistanı</h1>
      <p>Müşteri yorumlarını yapay zeka ile analiz edin.</p>

      <div className="input-area">
        <textarea
          rows="4"
          placeholder="Müşteri yorumunu buraya yapıştırın..."
          value={yorum}
          onChange={(e) => setYorum(e.target.value)}
        />
        <button onClick={analizEt} disabled={yukleniyor}>
          {yukleniyor ? 'Yapay Zeka Düşünüyor...' : 'Analiz Et'}
        </button>
      </div>

      {sonuc && (
        <div className="result-card">
          <h3>Analiz Sonucu:</h3>
          <div className="metrics">
            <div className={`tag ${sonuc.duygu === 'Olumsuz' || sonuc.duygu === 'Negatif' ? 'red' : sonuc.duygu === 'Tarafsız' ? 'yellow' : 'green'}`}>
              Duygu: {sonuc.duygu}
            </div>
            <div className="tag blue">
              Niyet: {sonuc.niyet}
            </div>
          </div>
          <p className="raw-output"> Niyet: {sonuc.niyet}, Duygu: {sonuc.duygu}</p>
        </div>
      )}

      {/* Hocanın İstediği İstatistik Alanı */}
      <div className="stats-footer">
        <h4>Model Performansı (Test Seti)</h4>
        <p>Duygu Başarısı: %89.15 | Niyet Başarısı: %92.74 | Ort. F1: 0.91</p>
      </div>
    </div>
  )
}

export default App
