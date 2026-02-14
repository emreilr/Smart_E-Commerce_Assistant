import torch
import time
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# --- AYARLAR ---
TEST_DOSYASI = "../data/eticaret_test_seti.csv"
BASE_MODEL = "Trendyol/Trendyol-LLM-7b-base-v1.0"
ADAPTER_PATH = "../models/eticaret-uzmani-llm"
CIKIS_KLASORU = "../degerlendirme_sonuclari" # Keeping results at root or change to logs/results

# Matplotlib ayarlari
import os
if not os.path.exists(CIKIS_KLASORU):
    os.makedirs(CIKIS_KLASORU)

def parse_double_output(text):
    """
    Model ciktisindan ve Gercek veriden JSON ayiklar.
    Beklenen format: {"Niyet": "...", "Duygu": "..."}
    """
    duygu = "Bilinmiyor"
    niyet = "Bilinmiyor"
    
    # JSON bulmaya calis
    try:
        # Suslu parantez icini yakala
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            json_str = match.group(0)
            # Bazi durumlarda model tek tirnak kullanabilir, duzelt
            json_str = json_str.replace("'", "\"")
            data = json.loads(json_str)
            
            duygu = data.get("Duygu", "Bilinmiyor")
            niyet = data.get("Niyet", "Bilinmiyor")
    except:
        pass
        
    return duygu, niyet

def main():
    print("1. Test Verisi Yukleniyor...")
    if not os.path.exists(TEST_DOSYASI):
        print(f"HATA: {TEST_DOSYASI} bulunamadi! Once egitim.py calistirilmali.")
        return

    df = pd.read_csv(TEST_DOSYASI)
    print(f"Toplam Test Ornegi: {len(df)}")

    # 2. Model Yukleme
    print("\n2. Model Yukleniyor (4-bit)...")
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model.eval()
        print("Model yuklendi.")
    except Exception as e:
        print(f"Model yukleme hatasi: {e}")
        return

    # 3. Degerlendirme Dongusu
    print("\n3. Tahminler Aliniyor...")
    
    gercek_duygular = []
    tahmin_duygular = []
    
    gercek_niyetler = []
    tahmin_niyetler = []
    
    inference_times = []

    hatali_ornekler = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        full_text = row['Egitim_Metni']
        
        # Egitim metnini 'Prompt' ve 'Target' olarak ayir
        parts = full_text.split("### Cevap:")
        if len(parts) < 2:
            continue
            
        prompt = parts[0] + "### Cevap:"
        ground_truth_str = parts[1].strip()
        
        # Gercek etiketleri al
        g_duygu, g_niyet = parse_double_output(ground_truth_str)
        
        # Inference (Tahmin)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False, # Deterministik sonuc icin
                pad_token_id=tokenizer.eos_token_id
            )
        end_time = time.time()
        inference_times.append(end_time - start_time)
        
        # Ciktiyi isleme
        input_len = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_len:]
        prediction_str = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        p_duygu, p_niyet = parse_double_output(prediction_str)
        
        gercek_duygular.append(g_duygu)
        tahmin_duygular.append(p_duygu)
        
        gercek_niyetler.append(g_niyet)
        tahmin_niyetler.append(p_niyet)

        # Hata Analizi: Eger Duygu veya Niyet yanlis ise kaydet
        if g_duygu != p_duygu or g_niyet != p_niyet:
            # Yorumun sadece soru kismini al (Prompt temizle)
            yorum_ham = prompt.replace("### Talimat: Aşağıdaki yorumu e-ticaret bağlamında analiz et. Niyet ve Duygu durumunu JSON formatında döndür.\n### Yorum: ", "").replace("\n### Cevap:", "").strip()
            
            hata_kaydi = {
                "Yorum": yorum_ham[:200], # Ilk 200 karakter
                "Beklenen Cevap": f"Niyet: {g_niyet}, Duygu: {g_duygu}",
                "Verilen Cevap": f"Niyet: {p_niyet}, Duygu: {p_duygu}"
            }
            hatali_ornekler.append(hata_kaydi)

    # Hatalari Kaydet
    if hatali_ornekler:
        hata_df = pd.DataFrame(hatali_ornekler)
        hata_dosyasi = f"{CIKIS_KLASORU}/hatali_tahminler.csv"
        hata_df.to_csv(hata_dosyasi, index=False, encoding='utf-8-sig')
        print(f"\n[BILGI] Toplam {len(hatali_ornekler)} hatali tahmin bulundu.")
        print(f"Detayli hata raporu kaydedildi: {hata_dosyasi}")
        
        print("\n--- ORNEK HATALAR (ILK 5) ---")
        for i, hata in enumerate(hatali_ornekler[:5]):
            print(f"{i+1}. Yorum: {hata['Yorum']}...")
            print(f"   BEKLENEN -> {hata['Beklenen Cevap']}")
            print(f"   VERILEN  -> {hata['Verilen Cevap']}")
            print("-" * 50)

    # 4. Metrik Hesaplama
    print("\n--- SONUCLAR ---")
    
    avg_inference = np.mean(inference_times)
    print(f"Ortalama Inference Suresi: {avg_inference:.4f} saniye/ornek")
    
    # Duygu Analizi Sonuclari
    acc_duygu = accuracy_score(gercek_duygular, tahmin_duygular)
    f1_duygu = f1_score(gercek_duygular, tahmin_duygular, average='weighted')
    
    print(f"\nDuygu - Accuracy: %{acc_duygu*100:.2f}")
    print(f"Duygu - F1 Score: {f1_duygu:.4f}")
    print("\nDuygu Raporu:")
    print(classification_report(gercek_duygular, tahmin_duygular, zero_division=0))
    
    # Niyet Analizi Sonuclari
    acc_niyet = accuracy_score(gercek_niyetler, tahmin_niyetler)
    f1_niyet = f1_score(gercek_niyetler, tahmin_niyetler, average='weighted')
    
    print(f"\nNiyet - Accuracy: %{acc_niyet*100:.2f}")
    print(f"Niyet - F1 Score: {f1_niyet:.4f}")
    
    # 5. Gorsellestirme
    print("\nGrafikler ciziliyor...")
    
    # Duygu Confusion Matrix
    cm_duygu = confusion_matrix(gercek_duygular, tahmin_duygular)
    labels_duygu = sorted(list(set(gercek_duygular + tahmin_duygular)))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_duygu, annot=True, fmt='d', cmap='Blues', xticklabels=labels_duygu, yticklabels=labels_duygu)
    plt.title('Duygu Analizi - Confusion Matrix')
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    plt.savefig(f"{CIKIS_KLASORU}/confusion_matrix_duygu.png")
    plt.close()

    # Niyet Confusion Matrix
    cm_niyet = confusion_matrix(gercek_niyetler, tahmin_niyetler)
    labels_niyet = sorted(list(set(gercek_niyetler + tahmin_niyetler)))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_niyet, annot=True, fmt='d', cmap='Greens', xticklabels=labels_niyet, yticklabels=labels_niyet)
    plt.title('Niyet Analizi - Confusion Matrix')
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    plt.xticks(rotation=45)
    plt.savefig(f"{CIKIS_KLASORU}/confusion_matrix_niyet.png")
    plt.close()
    
    # Metrikler Bar Chart
    metrics = ['Duygu Accuracy', 'Duygu F1', 'Niyet Accuracy', 'Niyet F1']
    vals = [acc_duygu, f1_duygu, acc_niyet, f1_niyet]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, vals, color=['#4287f5', '#42cbf5', '#42f554', '#87f542'])
    plt.ylim(0, 1.1)
    plt.title('Model Performansi Ozeti')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')
        
    plt.savefig(f"{CIKIS_KLASORU}/metric_chart.png")
    plt.close()
    
    print(f"\nGrafikler '{CIKIS_KLASORU}' klasorune kaydedildi.")

if __name__ == "__main__":
    main()
