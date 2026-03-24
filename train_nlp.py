import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# Import mesin NLP yang barusan sukses jalan
from casual_nlp import CausalNLPTransformer

def generate_causal_text_data(n_samples=5000, vocab_size=100):
    """
    Membuat data teks sintetik yang mengandung Spurious Correlation (Bias).
    """
    # Z (Konteks): 0 atau 1
    Z = torch.randint(0, 2, (n_samples,))
    
    # X_seq: Kalimat panjang 10 token
    X_seq = torch.zeros((n_samples, 10), dtype=torch.long)
    Y = torch.zeros(n_samples, dtype=torch.long)
    
    for i in range(n_samples):
        # ── JEBAKAN KORELASI PALSU ──
        # Jika Z=0, 90% kemungkinan kalimatnya Positif.
        # Jika Z=1, 90% kemungkinan kalimatnya Negatif.
        if Z[i] == 0:
            is_positive = torch.rand(1).item() < 0.9 
        else:
            is_positive = torch.rand(1).item() < 0.1
            
        Y[i] = 1 if is_positive else 0
        
        # Susun token (kata) berdasarkan sentimen
        if is_positive:
            # Token 10-19 adalah kata-kata "Positif"
            X_seq[i] = torch.randint(10, 20, (10,))
        else:
            # Token 20-29 adalah kata-kata "Negatif"
            X_seq[i] = torch.randint(20, 30, (10,))
            
        # Z mempengaruhi kata-kata "filler" (kata sambung, dll) untuk memperkuat bias
        filler_token = 1 if Z[i] == 0 else 2
        X_seq[i, 0:3] = filler_token # 3 kata pertama jadi ciri khas Topik Z
        
    return Z, X_seq, Y

def generate_do_text_data(n_samples=1000, vocab_size=100):
    """
    Data Intervensi do(X): Menyebarkan kata Positif dan Negatif 
    secara merata TANPA mempedulikan konteks Z (Memutus bias).
    """
    X_seq = torch.zeros((n_samples, 10), dtype=torch.long)
    Y = torch.zeros(n_samples, dtype=torch.long)
    
    for i in range(n_samples):
        is_positive = torch.rand(1).item() < 0.5 # Murni 50/50
        Y[i] = 1 if is_positive else 0
        
        if is_positive:
            X_seq[i] = torch.randint(10, 20, (10,))
        else:
            X_seq[i] = torch.randint(20, 30, (10,))
            
        # Kata filler diacak penuh, memutus hubungan dengan Z
        X_seq[i, 0:3] = torch.randint(1, 3, (3,))
        
    return X_seq, Y

# ── Main Training Loop ──
if __name__ == "__main__":
    VOCAB_SIZE = 100
    BATCH_SIZE = 64
    
    print("Mempersiapkan Dataset Jebakan Batman...")
    Z_obs, X_obs, Y_obs = generate_causal_text_data(n_samples=5000, vocab_size=VOCAB_SIZE)
    X_do, Y_do = generate_do_text_data(n_samples=1000, vocab_size=VOCAB_SIZE)
    
    obs_loader = DataLoader(TensorDataset(X_obs, Y_obs), batch_size=BATCH_SIZE, shuffle=True)
    do_loader = DataLoader(TensorDataset(X_do, Y_do), batch_size=BATCH_SIZE, shuffle=True)
    
    print("Inisialisasi Causal NLP Transformer (d_model=14)...")
    model = CausalNLPTransformer(vocab_size=VOCAB_SIZE, num_classes=2, d_model=14)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("Mulai Proses Training NLP...")
    for epoch in range(50):
        total_loss = 0
        for (x_batch, y_batch), (x_do_batch, y_do_batch) in zip(obs_loader, do_loader):
            optimizer.zero_grad()
            
            # 1. Belajar dari Observasi (Biasa)
            y_pred_obs = model(x_batch)
            loss_obs = criterion(y_pred_obs, y_batch)
            
            # 2. Belajar dari Intervensi (Anti-Bias)
            y_pred_do = model(x_batch, x_cf_tokens=x_do_batch)
            loss_do = criterion(y_pred_do, y_do_batch)
            
            # Gabungkan dengan Causal Penalty
            loss = loss_obs + (5.0 * loss_do)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:>2}/50 | Loss: {total_loss/len(obs_loader):.4f} | Epsilon: {model.switch.epsilon:.4f}")
            
    print("\n✅ Training Selesai! Modelmu sekarang sudah kebal terhadap bias kata-kata.")
# ==========================================
    # 3. UJIAN OUT-OF-DISTRIBUTION (OOD)
    # ==========================================
    print("\n" + "="*50)
    print("MAMULAI UJIAN OUT-OF-DISTRIBUTION (OOD)")
    print("="*50)
    
    # Bikin data jebakan: Z dan Y kita balik total dari kebiasaan Training!
    n_test = 1000
    Z_test = torch.randint(0, 2, (n_test,))
    X_test = torch.zeros((n_test, 10), dtype=torch.long)
    Y_test = torch.zeros(n_test, dtype=torch.long)
    
    for i in range(n_test):
        # KEBALIKAN DARI TRAINING:
        # Jika Z=0 (Gadget), sekarang 90% Negatif (dulu Positif)
        # Jika Z=1 (Makanan), sekarang 90% Positif (dulu Negatif)
        if Z_test[i] == 0:
            is_positive = torch.rand(1).item() < 0.1 
        else:
            is_positive = torch.rand(1).item() < 0.9
            
        Y_test[i] = 1 if is_positive else 0
        
        # Susun token kata (tetap jujur: 10-19 Positif, 20-29 Negatif)
        if is_positive:
            X_test[i] = torch.randint(10, 20, (10,))
        else:
            X_test[i] = torch.randint(20, 30, (10,))
            
        # Filler token (Topik Z)
        X_test[i, 0:3] = 1 if Z_test[i] == 0 else 2

    # Matikan gradient karena cuma mau ngetes
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test)
        # Ambil tebakan kelas tertinggi (0 atau 1)
        predictions = torch.argmax(y_pred_test, dim=1)
        
        # Hitung akurasi
        correct = (predictions == Y_test).sum().item()
        accuracy = (correct / n_test) * 100
        
    print(f"Akurasi Model Kausal di Data Jebakan (OOD): {accuracy:.2f}%")
    if accuracy > 80.0:
        print("🏆 GILA! AI-mu beneran kebal bias! Dia paham makna kalimat, bukan sekadar hafal topik.")
    else:
        print("⚠️ Hhmm, masih kena jebakan. Mungkin butuh epoch lebih banyak atau epsilon disesuaikan.")