import torch
import torch.nn as nn
from causal_switch_poc import CausalSwitch 

# ==========================================
# 1. BAGIAN ARSITEKTUR (CLASS)
# ==========================================
class CausalNLPTransformer(nn.Module):
    def __init__(self, vocab_size, num_classes=2, d_model=14, n_heads=2, total_steps=10000, max_seq_len=50):
        super().__init__()
        self.d_model = d_model
        self.switch  = CausalSwitch(total_steps=total_steps)

        # ── Pintu Masuk (Embedding Layer) ──
        self.word_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb  = nn.Embedding(max_seq_len, d_model) 
        
        # Z_token & Y_query
        self.z_token  = nn.Parameter(torch.randn(1, 1, d_model))
        self.y_query  = nn.Parameter(torch.randn(1, 1, d_model))

        # ── Core Transformer ──
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # ── Output Head Klasifikasi ──
        self.head_y = nn.Linear(d_model, num_classes)

    def _forward_seq(self, z_emb, x_seq_emb, intervene: bool):
        B = x_seq_emb.shape[0]
        seq_len = x_seq_emb.shape[1]
        
        # Expand Z dan Y untuk setiap batch
        z_emb_batch = z_emb.expand(B, 1, self.d_model)
        y_emb_batch = self.y_query.expand(B, 1, self.d_model)
        
        # Gabungkan semuanya jadi satu sekuens
        seq = torch.cat([z_emb_batch, x_seq_emb, y_emb_batch], dim=1) 

        # ── NLP Causal Masking ──
        mask = torch.zeros((seq.shape[1], seq.shape[1]), device=seq.device)
        if intervene:
            # Memutus panah Z -> X
            mask[1:-1, 0] = float('-inf') 

        attn_out, _= self.attn(seq, seq, seq, attn_mask=mask)
        seq        = self.norm1(seq + attn_out)
        seq        = self.norm2(seq + self.ff(seq))
        
        # Kembalikan representasi Y (berada di posisi paling akhir)
        return seq[:, -1, :] 

    # 👇 INI DIA JANTUNGNYA YANG TADI HILANG / SALAH SPASI 👇
    def forward(self, x_tokens, x_cf_tokens=None):
        B, seq_len = x_tokens.shape
        device = x_tokens.device
        
        # 1. Ubah token ke Embedding lalu tambah Positional Encoding
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(B, seq_len)
        x_emb = self.word_emb(x_tokens) + self.pos_emb(positions)

        # 2. Injeksi Intervensi do(X) DI DALAM ruang laten (Sudah kebal batch beda ukuran)
        if x_cf_tokens is not None:
            B_cf, seq_len_cf = x_cf_tokens.shape
            positions_cf = torch.arange(0, seq_len_cf, device=device).unsqueeze(0).expand(B_cf, seq_len_cf)
            x_cf_emb = self.word_emb(x_cf_tokens) + self.pos_emb(positions_cf)
            x_emb = self.switch.inject(x_cf_emb)
            self.switch.anneal()

        # 3. Masuk ke Transformer
        y_out_emb = self._forward_seq(self.z_token, x_emb, intervene=(x_cf_tokens is not None))
        
        # 4. Prediksi Kelas (Logits)
        return self.head_y(y_out_emb)


# ==========================================
# 2. BAGIAN TESTING (Di Luar Class)
# ==========================================
if __name__ == "__main__":
    print("Membangun Causal NLP Transformer...")
    
    # Parameter Dummy
    VOCAB_SIZE = 1000  
    NUM_CLASSES = 2    
    D_MODEL = 14       
    
    # Inisialisasi Model
    model = CausalNLPTransformer(
        vocab_size=VOCAB_SIZE, 
        num_classes=NUM_CLASSES, 
        d_model=D_MODEL, 
        n_heads=2
    )
    
    # Simulasi Data Teks: 4 Kalimat, masing-masing 10 kata
    x_text_obs = torch.randint(0, VOCAB_SIZE, (4, 10))
    x_text_cf = torch.randint(0, VOCAB_SIZE, (4, 10))
    
    print("\n[Test 1] Observational Pass (Membaca Teks Biasa)")
    print(f"Input kalimat shape: {x_text_obs.shape} -> (Batch, Seq_Len)")
    y_pred_obs = model(x_text_obs)
    print(f"Output prediksi shape: {y_pred_obs.shape} -> (Batch, Num_Classes)")
    
    print("\n[Test 2] Interventional Pass (Mengaktifkan Causal Switch & Graph Surgery)")
    y_pred_do = model(x_text_obs, x_cf_tokens=x_text_cf)
    print(f"Output prediksi do(X) shape: {y_pred_do.shape}")
    print(f"Status Causal Switch Epsilon: {model.switch.epsilon:.6f}")
    
    print("\n✅ Otsukaresama! Jika ini muncul, Causal NLP Transformer-mu berjalan sempurna!")