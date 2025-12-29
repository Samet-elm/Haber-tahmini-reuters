import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import reuters
from keras.utils import to_categorical
from keras import models, layers, callbacks

# 1. VERİ YÜKLEME
print("Veri seti yükleniyor...")
num_words = 10000
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=num_words)

# 2. METİN VEKTÖRLEŞTİRME (Binary Vectorization)
def vectorize_sequences(sequences, dimension=num_words):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

# Validation set (Doğrulama seti)
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

# 3. GELİŞMİŞ MODEL MİMARİSİ
model = models.Sequential()
# Nöron sayısını artırdık (128), Batch Normalization ve Dropout ekledik
model.add(layers.Dense(128, activation='relu', input_shape=(num_words,)))
model.add(layers.BatchNormalization()) # Eğitimi hızlandırır ve kararlı kılar
model.add(layers.Dropout(0.3))         # Ezberlemeyi önler

model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))

model.add(layers.Dense(46, activation='softmax'))

# Optimizasyon: 'adam' genellikle rmsprop'tan daha iyi genelleme yapar
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. CALLBACKS (ERKEN DURDURMA VE EN İYİ MODELİ KAYDETME)
# Val_loss iyileşmeyi durdurursa eğitimi bitirir
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True
)

# Sadece en iyi performansı gösteren ağırlıkları kaydeder
checkpoint = callbacks.ModelCheckpoint(
    'reuters_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# 5. EĞİTİM
print("Model eğitiliyor (Early Stopping aktif)...")
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=30, # Max epoch'u artırdık, erken durdurma yönetecektir
    batch_size=512,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, checkpoint]
)

# 6. TEST SONUÇLARI
print("\n--- Final Test Sonuçları ---")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Doğruluğu (Accuracy): %{test_acc*100:.2f}")

# 7. PERFORMANS GRAFİĞİ
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

# Doğruluk Grafiği
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, 'bo', label='Eğitim Doğruluğu')
plt.plot(epochs_range, val_acc, 'b', label='Doğrulama Doğruluğu')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.legend()

# Kayıp (Loss) Grafiği
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'ro', label='Eğitim Kaybı')
plt.plot(epochs_range, val_loss, 'r', label='Doğrulama Kaybı')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.legend()

plt.savefig("egitim_sonuclari.png")
print("✅ Eğitim grafiği 'egitim_sonuclari.png' olarak kaydedildi.")