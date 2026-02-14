from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --- AYARLAR ---
BASE_MODEL = "Trendyol/Trendyol-LLM-7b-base-v1.0"
ADAPTER_PATH = "models/eticaret-uzmani-llm"

app = FastAPI()

# CORS Ayarları (React uygulamasının erişebilmesi için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Güvenlik için production'da spesifik domain girilmeli
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global değişkenler
model = None
tokenizer = None

class YorumGiris(BaseModel):
    yorum: str

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    print("🚀 Model yükleniyor... Lütfen bekleyin.")
    
    try:
        # Tokenizer Yükle
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
        
        # 4-bit Quantization Ayarları (GPU Bellek Tasarrufu için)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Base Modeli Yükle
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Eğitilmiş LoRA Adapter'ı Base Model üzerine ekle
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model.eval()
        
        print("Model başarıyla yüklendi ve analize hazır!")
    except Exception as e:
        print(f" Model yüklenirken hata oluştu: {e}")

@app.post("/analiz-et")
async def analiz_et(veri: YorumGiris):
    if not model:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi.")

    yorum = veri.yorum
    
    # Prompt Hazırlığı (Modelin eğitim formatına uygun olmalı)
    # Basit bir "instruction" formatı deniyoruz.
    prompt = f"### Talimat: Aşağıdaki yorumu e-ticaret bağlamında analiz et. Niyet ve Duygu durumunu JSON formatında döndür.\n### Yorum: {yorum}\n### Cevap: "
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            do_sample=True, 
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Sadece üretilen yeni tokenları al
    input_len = inputs.input_ids.shape[1]
    generated_ids = outputs[0][input_len:]
    analiz_sonucu = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Halüsinasyonları (tekrar eden ### Yorum vs.) temizle
    analiz_sonucu = analiz_sonucu.split("###")[0].strip()
    
    # JSON Çıkarımı
    import json
    import re
    
    duygu = "Nötr"
    niyet = "Genel"
    
    try:
        # JSON benzeri yapıyı bulmaya çalış
        json_match = re.search(r"\{.*?\}", analiz_sonucu)
        if json_match:
            data = json.loads(json_match.group(0))
            raw_duygu = data.get("Duygu", "Nötr")
            niyet = data.get("Niyet", "Genel")
            
            # Frontend uyumluluğu için haritalama
            if raw_duygu == "Olumlu":
                duygu = "Pozitif"
            elif raw_duygu == "Olumsuz":
                duygu = "Negatif"
            else:
                duygu = raw_duygu
        else:
            print(f"UYARI: JSON deseni bulunamadı. Ham çıktı: {analiz_sonucu}")
    except Exception as e:
        print(f"JSON Parse Hatası: {e}. Ham Çıktı: {analiz_sonucu}")

    return {
        "duygu": duygu,
        "niyet": niyet,
        "analiz_sonucu": analiz_sonucu
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
