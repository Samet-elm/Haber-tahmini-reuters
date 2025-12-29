import gradio as gr
import tensorflow as tf
import numpy as np
import re
from keras.datasets import reuters

# 1. Modeli ve Kelime İndeksini Yükle
try:
    model = tf.keras.models.load_model("reuters_model.keras")
    word_index = reuters.get_word_index()
    print("Model ve sözlük başarıyla yüklendi.")
except Exception as e:
    print(f"Hata: Model dosyası bulunamadı. Lütfen önce train.py dosyasını çalıştırın. ({e})")
    exit()

# 2. Reuters Etiket Sözlüğü (0-45 Standart İndeksler)
topic_labels = {
    0: ("cocoa", "Kakao"), 1: ("grain", "Tahıl"), 2: ("veg-oil", "Bitkisel Yağ"),
    3: ("earn", "Kazanç / Bilanço"), 4: ("acq", "Şirket Satın Alma"), 5: ("wheat", "Buğday"),
    6: ("copper", "Bakır"), 7: ("housing", "Konut / Emlak"), 8: ("money-supply", "Para Arzı"),
    9: ("coffee", "Kahve"), 10: ("sugar", "Şeker"), 11: ("trade", "Ticaret"),
    12: ("reserves", "Rezervler"), 13: ("ship", "Gemicilik"), 14: ("cotton", "Pamuk"),
    15: ("carcass", "Karkas Et"), 16: ("crude", "Ham Petrol"), 17: ("nat-gas", "Doğal Gaz"),
    18: ("cpi", "Enflasyon"), 19: ("money-fx", "Döviz Piyasası"), 20: ("interest", "Faiz"),
    21: ("gnp", "GSMH"), 22: ("meal-feed", "Yem"), 23: ("alum", "Alüminyum"),
    24: ("oilseed", "Yağlı Tohum"), 25: ("gold", "Altın"), 26: ("tin", "Kalay"),
    27: ("zinc", "Çinko"), 28: ("orange", "Portakal"), 29: ("pet-chem", "Petrokimya"),
    30: ("lead", "Kurşun"), 31: ("potato", "Patates"), 32: ("strategic-metal", "Stratejik Metal"),
    33: ("livestock", "Hayvancılık"), 34: ("retail", "Perakende"), 35: ("ipi", "Sanayi Üretimi"),
    36: ("iron-steel", "Demir Çelik"), 37: ("rubber", "Kauçuk"), 38: ("heat", "Isınma / Yakıt"),
    39: ("jobs", "İstihdam"), 40: ("lei", "Öncü Göstergeler"), 41: ("bop", "Ödemeler Dengesi"),
    42: ("chick", "Kümes Hayvanı"), 43: ("tea", "Çay"), 44: ("coconut-oil", "Hindistan Cevizi Yağı"),
    45: ("jet", "Jet Yakıtı")
}

# 3. Metin İşleme Fonksiyonu
def transform_text(text):
    # Metni temizle ve kelimelere ayır
    words = re.findall(r'\w+', text.lower())
    
    # Reuters kuralına göre +3 kaydırarak indeksleme yap
    sequence = []
    for w in words:
        idx = word_index.get(w)
        if idx is not None and (idx + 3) < 10000:
            sequence.append(idx + 3)
        else:
            sequence.append(2) # Bilinmeyen kelime (OOV)
            
    # Vektörleştirme (10.000 boyutlu)
    vector = np.zeros((1, 10000))
    for idx in sequence:
        vector[0, idx] = 1.
    return vector

# 4. Tahmin Fonksiyonu (Yüzdeler Kaldırıldı)
def predict_news(text):
    if not text.strip():
        return "Lütfen analiz için bir haber metni girin."
        
    try:
        x = transform_text(text)
        prediction = model.predict(x, verbose=0)[0]
        
        # En olası 3 kategorinin indeksini al
        top_indices = prediction.argsort()[-3:][::-1]
        
        results = []
        for idx in top_indices:
            eng, tr = topic_labels.get(idx, ("Unknown", "Bilinmeyen"))
            # Sadece Kategori İsimlerini Yazdır
            results.append(f"{eng.upper()} ({tr})")
            
        return "\n".join(results)
    except Exception as e:
        return f"Tahmin hatası: {str(e)}"

# 5. Gradio Arayüzü
with gr.Blocks(title="Reuters Haber Sınıflandırma") as demo:
    gr.Markdown("# 📰 Reuters News AI Classifier")
    gr.Markdown("Haber metnini girin; sistem metnin ait olduğu en olası kategorileri belirlesin.")
    
    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(
                lines=5, 
                placeholder="Örnek: The company's quarterly profit rose by 15%...",
                label="Haber Metni (İngilizce)"
            )
            analyze_btn = gr.Button("Analiz Et", variant="primary")
        
        with gr.Column():
            output_box = gr.Textbox(
                label="Tahmin Edilen Kategoriler", 
                interactive=False
            )

    analyze_btn.click(fn=predict_news, inputs=input_box, outputs=output_box)

if __name__ == "__main__":
    # share=True parametresi dış bağlantı linki oluşturur
    demo.launch(share=True)