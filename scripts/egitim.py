import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# --- AYARLAR ---
DOSYA_ADI = "../data/eticaret_yorum_analiz_egitim_seti.csv"
MODEL_ADI = "Trendyol/Trendyol-LLM-7b-base-v1.0"
CIKIS_KLASORU = "../models/eticaret-uzmani-llm"

def main():
    # 1. GPU Kontrolü
    print("Donanım kontrol ediliyor...")
    if not torch.cuda.is_available():
        print(" HATA: NVIDIA GPU bulunamadı!")
        return
    print(f" GPU Algılandı: {torch.cuda.get_device_name(0)}")

    # 2. Verisetini Yükle
    if not os.path.exists(DOSYA_ADI):
        print(f" HATA: '{DOSYA_ADI}' dosyası bulunamadı!")
        return

    print("Veriseti okunuyor...")
    df = pd.read_csv(DOSYA_ADI)
    full_dataset = Dataset.from_pandas(df[["Egitim_Metni"]])

    # Verisetini bol (%80 Egitim, %10 Val, %10 Test)
    # Once %10 Test ayir (Kalan %90)
    train_val_split = full_dataset.train_test_split(test_size=0.1, seed=42)
    test_dataset = train_val_split["test"]
    remaining_dataset = train_val_split["train"]
    
    # Kalani %80 Egitim, %10 Val olarak ayir (Yaklasik 1/9 oraninda)
    # Kalan %90'in %11.1'i toplam verinin %10'u eder.
    train_val_split_2 = remaining_dataset.train_test_split(test_size=0.111, seed=42)
    dataset = train_val_split_2["train"]
    val_dataset = train_val_split_2["test"]

    # Test setini CSV olarak kaydet (Degerlendirme senaryosu icin)
    test_df = test_dataset.to_pandas()
    test_df.to_csv("eticaret_test_seti.csv", index=False)
    
    print(f"Bolumleme Tamamlandi:")
    print(f"- Egitim Seti:     {len(dataset)} satir")
    print(f"- Dogrulama Seti:  {len(val_dataset)} satir")
    print(f"- Test Seti:       {len(test_dataset)} satir")
    print("Test seti 'eticaret_test_seti.csv' olarak kaydedildi.")

    # 3. Model Hazırlığı (4-bit Sıkıştırma - 4070 Ti Super için)
    print("Model yükleniyor...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ADI, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ADI)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4. LoRA Ayarları
    # 4. LoRA Ayarları
    peft_config = LoraConfig(
        lora_alpha=32,    # ARTTIRILDI: Daha fazla ogrenme kapasitesi
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. Eğitim Parametreleri
    # SFTConfig içine sadece genel eğitim ayarları
    # Hata veren max_seq_length vb. buradan çıkarıldı.
    training_args = SFTConfig(
        output_dir="./gecici_sonuclar",
        num_train_epochs=3,             # ARTTIRILDI: Daha iyi ogrenme icin
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,   
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,                       
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none",
        
        # Validation ve Best Model ayarlari
        eval_strategy="steps",          # Egitim sirasinda dogrulama yap
        eval_steps=50,                  # Her 50 adimda bir
        save_total_limit=2,             # Cok fazla checkpoint saklama
        load_best_model_at_end=True,    # En iyi modeli sec
        metric_for_best_model="eval_loss", # Kayip degerine gore
        greater_is_better=False,        # Dusuk loss daha iyidir

        
        # TRL v0.26+ uyumlulugu icin buraya eklendi
        dataset_text_field="Egitim_Metni",
        max_length=512,  # max_seq_length yerine max_length
        packing=False,
    )

    # 6. Başlat
    print("\n Eğitim Başlıyor...")
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=val_dataset,       # Dogrulama seti eklendi
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    trainer.train()

    # 7. Kaydet
    print(f"\n Model kaydediliyor: {CIKIS_KLASORU}")
    trainer.model.save_pretrained(CIKIS_KLASORU)
    trainer.tokenizer.save_pretrained(CIKIS_KLASORU)
    print(" İşlem Başarıyla Tamamlandı!")

if __name__ == "__main__":
    main()