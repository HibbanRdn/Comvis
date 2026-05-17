# Dokumentasi Notebook: Klasifikasi Diabetic Retinopathy — Hybrid EfficientNetB3 + Swin-Tiny vs InceptionV3

---

## 1. Gambaran Umum Notebook

Notebook ini merupakan implementasi eksperimen pembelajaran mendalam (_deep learning_) untuk mengklasifikasikan tingkat keparahan **Diabetic Retinopathy (DR)** — penyakit retina yang disebabkan oleh diabetes mellitus dan dapat berujung pada kebutaan permanen. Eksperimen mengacu pada penelitian Xu et al. (2024), berjudul _"A hybrid neural network approach for classifying diabetic retinopathy subtypes"_, yang diterbitkan di jurnal _Frontiers in Medicine_.

Dataset yang digunakan adalah **APTOS 2019 Blindness Detection**, sebuah dataset publik dari kompetisi Kaggle yang berisi gambar fundus mata beserta label tingkat keparahan DR dari 0 hingga 4. Terdapat lima kelas klasifikasi:

| Label | Nama Kelas |
|---|---|
| 0 | No DR (Tanpa DR) |
| 1 | Mild (Ringan) |
| 2 | Moderate (Sedang) |
| 3 | Severe (Berat) |
| 4 | Proliferative DR |

Tujuan utama notebook ini adalah sebagai berikut:

1. Membandingkan dua model utama: **Hybrid EfficientNetB3 + Swin Transformer Tiny** dengan **InceptionV3** sebagai baseline.
2. Melakukan _balanced tuning_ untuk meningkatkan keseimbangan performa antar kelas tanpa mengganti arsitektur utama.
3. Menjaga penggunaan RAM tetap aman melalui _disk cache_, _batching_, dan evaluasi _batch-wise_.
4. Mengevaluasi model dengan metrik yang sesuai untuk data tidak seimbang (_imbalanced_) dan bersifat ordinal, terutama **Macro F1** dan **Quadratic Weighted Kappa (QWK)**.

Notebook ini merupakan versi revisi dari eksperimen sebelumnya (v4). Revisi difokuskan pada pemulihan performa pada kelas No DR, Mild, Moderate, dan Severe tanpa mengabaikan kelas Proliferative DR, berdasarkan analisis kelemahan hasil run sebelumnya.

---

## 2. Struktur dan Alur Notebook

Notebook disusun secara berurutan dengan alur kerja sebagai berikut:

1. **Setup dan Instalasi** — Pemasangan dependensi dan inisialisasi library utama.
2. **Load Dataset** — Memuat metadata APTOS 2019 dan memvalidasi lokasi file gambar.
3. **Preprocessing Citra Fundus** — Pipeline peningkatan citra dengan Gaussian subtraction dan resize.
4. **Konfigurasi Eksperimen** — Penetapan seluruh hyperparameter training dalam kamus `CONFIG`.
5. **Build Disk Cache** — Menyimpan hasil preprocessing deterministik ke disk dalam format `.npy`.
6. **Augmentasi, MixUp, Dataset, dan Split Data** — Pembuatan pipeline augmentasi, pembagian data, dan _DataLoader_.
7. **Focal Loss dengan Class Weight** — Definisi fungsi loss yang memperhatikan ketidakseimbangan kelas.
8. **Definisi Model** — Implementasi arsitektur Hybrid dan InceptionV3.
9. **Training Utilities** — Fungsi pembantu untuk training, validasi, checkpoint, dan early stopping.
10. **Training Hybrid Model** — Proses pelatihan model Hybrid.
11. **Training InceptionV3** — Proses pelatihan model InceptionV3.
12. **Evaluasi Final pada Test Set Internal** — Evaluasi metrik lengkap pada data uji yang tidak pernah digunakan selama training.
13. **Perbandingan Akhir** — Tabel dan visualisasi perbandingan dua model beserta baseline jurnal.
14. **Grad-CAM** — Interpretasi visual area yang diperhatikan model saat mengambil keputusan.
15. **Ringkasan, Interpretasi, dan Limitasi** — Kesimpulan dan panduan membaca hasil eksperimen.

---

## 3. Penjelasan Dataset

### Sumber dan Format Data

Dataset yang digunakan adalah **APTOS 2019 Blindness Detection** yang bersumber dari Kaggle. Dataset terdiri dari file `train.csv` (metadata berlabel) dan `test.csv` (tanpa label publik yang digunakan untuk pengiriman kompetisi). File gambar disimpan dalam direktori `train_images` dalam format PNG atau JPG.

File `train.csv` memiliki dua kolom utama: `id_code` (identifikasi gambar) dan `diagnosis` (label kelas 0–4). Notebook hanya menggunakan `train.csv` berlabel untuk seluruh eksperimen, sedangkan `test.csv` tidak dipakai untuk evaluasi metrik.

### Distribusi Kelas

Dataset APTOS 2019 bersifat sangat **tidak seimbang (_highly imbalanced_)**. Berdasarkan informasi notebook, distribusi kelas menunjukkan dominasi kelas No DR hingga sekitar 49,3% dari total data, sedangkan kelas Severe hanya sekitar 5,3% dan Proliferative DR sekitar 8,1%. Kondisi ini menjadi tantangan utama dalam pelatihan karena model cenderung belajar memprediksi kelas mayoritas secara berlebihan.

### Pembagian Data

Notebook membagi data berlabel dari `train.csv` menjadi tiga bagian menggunakan stratified split:

| Subset | Proporsi |
|---|---|
| Training | 80% dari total data berlabel |
| Validation | 10% dari total data berlabel |
| Test Internal | 10% dari total data berlabel |

Stratified split memastikan distribusi kelas di setiap subset mencerminkan distribusi kelas keseluruhan. Notebook secara eksplisit menyatakan bahwa validation set berukuran kecil — hanya sekitar 19 gambar untuk kelas Severe dan 30 gambar untuk kelas Proliferative DR. Ukuran ini membuat metrik kelas minoritas mudah berfluktuasi antar epoch, sehingga pemilihan checkpoint tidak didasarkan pada satu metrik tunggal yang _noisy_.

### Catatan Penting

Test set internal yang digunakan dalam notebook ini **berbeda** dengan `test.csv` milik kompetisi Kaggle. Evaluasi final hanya berlaku pada test set internal ini, bukan pada test set kompetisi yang tidak memiliki label publik.

---

## 4. Preprocessing Data

### Pipeline Preprocessing Citra Fundus

Preprocessing dilakukan sebelum augmentasi dan disimpan ke disk sebagai cache. Setiap gambar melalui tahapan berikut:

1. **Baca gambar** dengan OpenCV dalam format BGR, kemudian dikonversi ke RGB.
2. **Resize** ke ukuran sementara yang lebih besar dari ukuran input akhir.
3. **Gaussian Subtraction** untuk meningkatkan kontras pembuluh darah dan lesi retina.
4. **Center Crop** untuk mendapatkan ukuran input akhir.

Untuk model Hybrid (input 224 × 224 piksel), gambar di-resize terlebih dahulu ke 256 piksel kemudian di-crop ke 224 piksel. Untuk model InceptionV3 (input 299 × 299 piksel), gambar di-resize ke 324 piksel kemudian di-crop ke 299 piksel.

### Gaussian Blur Subtraction

Teknik ini bekerja dengan cara mengurangi versi _blur_ dari gambar asli terhadap gambar asli itu sendiri:

```
enhanced = 4 × original - 4 × blurred + 128
```

Hasilnya adalah gambar yang menonjolkan tepi dan detail halus seperti pembuluh darah dan mikro-aneurisma yang relevan untuk diagnosis DR. Notebook menegaskan bahwa transformasi ini bersifat visual dan bukan prosedur diagnosis klinis.

### Normalisasi

Setelah preprocessing deterministik, setiap gambar dinormalisasi menggunakan mean dan standar deviasi ImageNet:
- **Mean**: [0.485, 0.456, 0.406]
- **Std**: [0.229, 0.224, 0.225]

Normalisasi ini penting karena backbone model (EfficientNetB3, Swin-Tiny, InceptionV3) menggunakan bobot _pretrained_ pada ImageNet yang mengasumsikan statistik warna tersebut.

### Cache `.npy`

Hasil preprocessing deterministik disimpan dalam format NumPy array (`.npy`) ke direktori cache disk. Penggunaan cache ini bertujuan untuk:

- Menghindari perhitungan preprocessing berulang setiap epoch.
- Menjaga penggunaan RAM tetap rendah karena gambar dibaca dari disk per batch, bukan dimuat seluruhnya ke memori.
- Memisahkan preprocessing deterministik dari augmentasi acak yang dijalankan pada saat training.

Dua cache terpisah dibuat: `cache_224` untuk model Hybrid dan `cache_299` untuk InceptionV3.

---

## 5. Pembuatan Dataset dan DataLoader

### Kelas DRDataset

`DRDataset` adalah kelas PyTorch `Dataset` yang membaca gambar dari cache disk (jika tersedia) atau melakukan preprocessing on-the-fly jika cache belum ada. Pada setiap pemanggilan `__getitem__`, kelas ini:

1. Mencari file `.npy` di direktori cache berdasarkan nama gambar.
2. Jika ada, memuat array NumPy dari disk.
3. Menerapkan transformasi augmentasi (untuk training) atau hanya normalisasi (untuk validasi dan test).
4. Mengembalikan pasangan (gambar-tensor, label-integer).

### Pipeline Augmentasi Training

Augmentasi hanya diterapkan pada data training. Revisi ini mengembalikan sebagian augmentasi dari notebook versi lama dengan probabilitas yang lebih terkontrol, karena run fine-tuned sebelumnya terlalu konservatif. Augmentasi yang diterapkan meliputi:

| Jenis Augmentasi | Parameter |
|---|---|
| Horizontal Flip | Probabilitas 0.5 |
| Vertical Flip | Probabilitas 0.15 |
| Random Rotate 90 | Probabilitas 0.15 |
| ShiftScaleRotate | shift=0.08, scale=0.12, rotate=25°, probabilitas=0.55 |
| CLAHE | clip_limit=2.0, tile_grid_size=8×8, probabilitas=0.35 |
| Color Augmentation (Brightness/Contrast atau HSV) | Probabilitas 0.45 |
| Gaussian Noise | var_limit=(5.0, 25.0), probabilitas=0.20 |
| Coarse Dropout | max_holes=6, probabilitas=0.18 |

**CLAHE** (_Contrast Limited Adaptive Histogram Equalization_) secara khusus berguna untuk citra fundus karena mampu meningkatkan kontras lokal tanpa memperkuat noise berlebihan.

**Coarse Dropout** secara acak menghapus blok piksel dari gambar selama training, memaksa model untuk belajar dari fitur yang tidak lengkap sehingga meningkatkan kemampuan generalisasi.

### MixUp

MixUp adalah teknik augmentasi tingkat tinggi yang menggabungkan dua gambar dari batch secara linear menggunakan koefisien acak dari distribusi Beta. Notebook menggunakan:

- `mixup_alpha = 0.30`: Parameter distribusi Beta untuk menghasilkan koefisien pencampuran. Nilai ini menghasilkan pencampuran yang lebih moderat dibandingkan nilai mendekati 0 (hampir tidak ada pencampuran) atau mendekati 1 (pencampuran kuat).
- `mixup_prob = 0.45`: Probabilitas penerapan MixUp per batch selama training. Artinya, sekitar 45% batch diproses dengan MixUp dan 55% batch tanpa MixUp.

Label yang dihasilkan MixUp bersifat _soft_ (kombinasi linear dari dua one-hot vector), bukan label diskrit. MixUp tidak diterapkan saat fase warmup head agar classifier baru dapat belajar dari label yang bersih terlebih dahulu.

### WeightedRandomSampler

Untuk mengatasi ketidakseimbangan kelas pada data training, notebook menggunakan `WeightedRandomSampler`. Setiap sampel diberi bobot berbanding terbalik dengan frekuensi kelasnya, sehingga kelas minoritas (Severe, Proliferative DR) memiliki peluang lebih tinggi untuk dipilih dalam setiap batch. Ini dilakukan hanya pada data training, bukan pada data validasi atau test.

### DataLoader

Dua set DataLoader dibuat secara terpisah: satu untuk model Hybrid (ukuran 224) dan satu untuk InceptionV3 (ukuran 299). Konfigurasi DataLoader mencakup:

- **Batch size**: 16 sampel per batch. Nilai ini dipilih agar penggunaan RAM dan VRAM GPU tetap aman.
- **num_workers**: 0 (data dimuat pada thread utama). Ini dipilih untuk keamanan memori di lingkungan Kaggle.
- **pin_memory**: Aktif jika CUDA tersedia, untuk mempercepat transfer data dari CPU ke GPU.

---

## 6. Arsitektur Model

### Model 1: Hybrid EfficientNetB3 + Swin Transformer Tiny

Model Hybrid merupakan model _dual-branch_ yang menggabungkan dua backbone berbeda untuk mengekstraksi fitur yang saling komplementer dari gambar yang sama:

- **EfficientNetB3**: Convolutional Neural Network yang efisien secara komputasi dengan scaled-depth, width, dan resolution. Model ini unggul dalam mendeteksi fitur lokal seperti tekstur dan edge.
- **Swin Transformer Tiny**: Vision Transformer berbasis _shifted window_ yang mampu menangkap dependensi jangka panjang antar area gambar. Model ini unggul dalam memahami konteks global dan hubungan spasial antar wilayah retina.

Kedua backbone memproses gambar secara paralel dan menghasilkan vektor fitur masing-masing. Vektor fitur tersebut digabungkan (_concatenate_) menjadi satu vektor gabungan yang kemudian diproses oleh classifier head.

**Classifier Head** Hybrid terdiri dari beberapa layer berurutan:
1. `LayerNorm` pada dimensi gabungan
2. `Dropout(0.35/2)` — setengah dari dropout utama
3. `Linear → GELU → LayerNorm → Dropout(0.35)` — dengan 512 unit
4. `Linear → GELU → LayerNorm → Dropout(0.35/2)` — dengan 256 unit
5. `Linear` — output 5 kelas

Penggunaan `GELU` (_Gaussian Error Linear Unit_) sebagai fungsi aktivasi non-linear lebih cocok untuk transformers dibandingkan ReLU tradisional. `LayerNorm` setelah setiap layer linear membantu stabilitas training pada representasi yang berasal dari dua arsitektur berbeda.

**Dropout = 0.35** digunakan untuk regularisasi, mencegah model terlalu bergantung pada neuron tertentu dan mengurangi risiko overfitting. Dropout sedikit lebih tinggi dari InceptionV3 karena Hybrid memiliki kapasitas parameter yang lebih besar.

Model Hybrid menerima input gambar berukuran **224 × 224 piksel**.

### Model 2: InceptionV3

InceptionV3 adalah model CNN klasik yang dikenal dengan blok inception-nya yang menggunakan konvolusi paralel dengan ukuran kernel berbeda (1×1, 3×3, 5×5) untuk menangkap fitur pada berbagai skala sekaligus. Model ini digunakan sebagai **baseline** dalam eksperimen.

**Classifier Head** InceptionV3 memiliki struktur yang identik dengan Hybrid, tetapi menggunakan `dropout = 0.28` (lebih rendah dari Hybrid). Dropout yang lebih rendah dipilih karena output eksperimen sebelumnya menunjukkan InceptionV3 cenderung underfit pada kelas Mild, Moderate, Severe, dan Proliferative DR — mengindikasikan model terlalu diregularisasi.

Model InceptionV3 menerima input gambar berukuran **299 × 299 piksel** sesuai spesifikasi aslinya.

### Transfer Learning

Kedua model diinisialisasi dengan bobot _pretrained_ dari ImageNet (`pretrained=True`) menggunakan library `timm`. Strategi ini memungkinkan model memanfaatkan representasi visual yang sudah dipelajari dari jutaan gambar ImageNet sebagai titik awal, sehingga proses training lebih efisien dan performanya lebih baik dibandingkan training dari awal (_from scratch_).

### Staged Fine-Tuning (Pembekuan Bertahap Backbone)

Notebook menerapkan strategi _staged fine-tuning_:

1. **Fase Warmup Head (epoch 1–3)**: Backbone dibekukan (_frozen_), hanya classifier head yang dilatih dengan learning rate 4e-4. Ini memungkinkan classifier baru beradaptasi dengan representasi backbone pretrained tanpa merusak bobot backbone.
2. **Fase Fine-Tuning Penuh (epoch 4–100)**: Backbone dibuka (_unfrozen_) dan seluruh model dilatih dengan learning rate berbeda untuk backbone (3e-5) dan classifier (3e-4). Backbone dilatih dengan learning rate jauh lebih rendah untuk menghindari perubahan drastis pada bobot pretrained.

Durasi warmup head dipersingkat dari 5 menjadi 3 epoch pada revisi ini agar backbone lebih cepat beradaptasi dengan karakteristik citra fundus.

---

## 7. Konfigurasi Training

Seluruh hyperparameter training disimpan dalam kamus `CONFIG`. Berikut penjelasan parameter-parameter kunci:

### Parameter Training Umum

| Parameter | Nilai | Penjelasan |
|---|---|---|
| `num_epochs` | 100 | Model dilatih maksimal 100 putaran penuh terhadap data training. |
| `batch_size` | 16 | Data diproses per 16 sampel dalam satu langkah training. |
| `weight_decay` | 3e-4 | Parameter L2 regularisasi pada optimizer AdamW untuk mencegah overfitting. |
| `dropout_hybrid` | 0.35 | Probabilitas penghapusan neuron secara acak pada model Hybrid. |
| `dropout_inception` | 0.28 | Probabilitas penghapusan neuron secara acak pada model InceptionV3. |
| `clip_grad` | 1.0 | Batas maksimal norma gradien untuk mencegah gradient explosion. |
| `label_smoothing` | 0.075 | Faktor pelunakan label untuk mencegah model terlalu percaya diri. |

**Label Smoothing = 0.075** berarti probabilitas target kelas benar tidak lagi 1.0, melainkan 1.0 - 0.075 + (0.075/5) ≈ 0.94. Probabilitas kelas lain masing-masing mendapat nilai kecil 0.075/5 = 0.015. Ini membantu model menghasilkan probabilitas yang lebih terkalibrasi dan mengurangi overconfidence.

### Learning Rate dan Optimizer

| Parameter | Nilai | Penjelasan |
|---|---|---|
| `warmup_lr_classifier` | 4e-4 | Learning rate classifier selama fase warmup head. |
| `lr_backbone` | 3e-5 | Learning rate backbone selama fase fine-tuning penuh. Sangat kecil untuk melindungi bobot pretrained. |
| `lr_classifier` | 3e-4 | Learning rate classifier selama fase fine-tuning penuh. 10× lebih besar dari backbone. |
| `min_lr` | 1e-6 | Batas minimum learning rate setelah penurunan oleh scheduler. |

**Optimizer yang digunakan adalah AdamW**, yang merupakan varian Adam dengan weight decay yang diimplementasikan secara lebih tepat (terpisah dari gradient update). AdamW cocok untuk model besar karena adaptif terhadap gradien per-parameter.

### Scheduler: ReduceLROnPlateau

| Parameter | Nilai | Penjelasan |
|---|---|---|
| `plateau_factor` | 0.5 | Learning rate dikalikan 0.5 (dikurangi setengah) ketika tidak ada peningkatan. |
| `plateau_patience` | 10 | Scheduler menunggu 10 epoch tanpa peningkatan sebelum menurunkan LR. |
| `plateau_cooldown` | 2 | Setelah penurunan LR, scheduler menunggu 2 epoch sebelum mulai memantau kembali. |

Scheduler ini memantau skor seleksi gabungan (`balanced_score`). Jika tidak ada peningkatan selama 10 epoch, learning rate diturunkan separuh. Parameter `plateau_patience = 10` dibuat lebih sabar dari run sebelumnya agar learning rate tidak turun terlalu dini saat validasi sedang berfluktuasi.

### Early Stopping

| Parameter | Nilai | Penjelasan |
|---|---|---|
| `early_stopping_patience` | 24 | Training dihentikan jika tidak ada peningkatan selama 24 epoch berturut-turut. |
| `early_stopping_min_delta` | 1e-4 | Peningkatan minimum yang dianggap signifikan (0.0001). |

Early stopping mencegah pemborosan komputasi dan overfitting dengan menghentikan training secara otomatis ketika model tidak lagi membaik pada data validasi.

### Focal Loss

**Focal Loss** adalah modifikasi dari Cross-Entropy Loss yang dirancang khusus untuk menangani ketidakseimbangan kelas. Komponen utamanya:

- **Gamma (focal_gamma = 2.0)**: Faktor yang memperkuat penalti pada sampel yang mudah diklasifikasikan dengan benar. Sampel yang sudah diprediksi dengan kepercayaan tinggi mendapat bobot loss lebih kecil, sehingga model fokus belajar dari sampel yang sulit.
- **Alpha (class_weight_power = 0.75)**: Bobot per kelas yang dihitung dari inverse frekuensi kelas, kemudian dipangkatkan 0.75. Nilai power ini berada antara nilai sebelumnya (0.5 = akar kuadrat, terlalu lemah) dan nilai notebook lama (terlalu agresif). Bobot kemudian dibatasi antara 0.45 dan 2.20 untuk mencegah bias terlalu ekstrem.
- **Label Smoothing**: Diintegrasikan langsung ke dalam Focal Loss.

Kombinasi WeightedRandomSampler dan alpha Focal Loss memberikan dua mekanisme berbeda untuk mengatasi ketidakseimbangan kelas: sampler memastikan kelas minoritas lebih sering muncul dalam batch, sementara alpha memastikan kelas minoritas mendapat bobot loss lebih besar meskipun sudah di-oversample.

### Skor Seleksi Checkpoint (Balanced Score)

Model checkpoint dipilih bukan berdasarkan satu metrik tunggal, melainkan berdasarkan **skor gabungan (balanced_score)**:

```
balanced_score = 0.55 × Macro F1 + 0.25 × QWK + 0.20 × Balanced Accuracy
```

Penggunaan skor gabungan ini dilakukan karena validation set berukuran kecil membuat metrik individual (terutama untuk kelas minoritas) bersifat _noisy_ dan dapat menyebabkan pemilihan checkpoint yang tidak optimal jika hanya mengandalkan satu metrik.

---

## 8. Proses Training

### Alur Training Per Epoch

Setiap epoch menjalankan urutan berikut:

1. **Train One Epoch**: Model dalam mode training. MixUp diterapkan secara probabilistik. Gradien dihitung dan bobot diperbarui menggunakan `torch.cuda.amp.GradScaler` (mixed precision) untuk efisiensi memori GPU. Gradient clipping diterapkan dengan norma maksimal 1.0.
2. **Validate**: Model dalam mode evaluasi (`torch.inference_mode()`). Loss dan prediksi dihitung pada seluruh validation set tanpa pembaruan gradien.
3. **Hitung Skor Seleksi**: Macro F1, QWK, Balanced Accuracy, Worst Class F1, dan Balanced Score dihitung dari prediksi validation.
4. **Scheduler Step**: ReduceLROnPlateau memperbarui learning rate berdasarkan balanced_score.
5. **Checkpoint**: Jika skor gabungan saat ini lebih baik dari skor terbaik sebelumnya, bobot model disimpan ke disk. Model terbaik dipilih berdasarkan primary metric `balanced_score` dengan tie-breaker `macro_f1`.
6. **Early Stopping Check**: Jika tidak ada peningkatan selama 24 epoch, training dihentikan lebih awal.
7. **RAM Guard**: Setiap 10 batch, penggunaan RAM diperiksa. Jika melebihi 80%, memory cleanup dilakukan. Jika melebihi 90%, training dihentikan untuk mencegah crash.

### Mekanisme Mixed Precision

Notebook menggunakan `torch.cuda.amp.autocast` dan `GradScaler` untuk mixed precision training (FP16/FP32). Ini mempercepat training dan mengurangi penggunaan VRAM GPU sekitar setengahnya tanpa mengorbankan akurasi model secara signifikan.

### Transisi Fase Training

Pada epoch ke-(warmup_head_epochs + 1) = epoch ke-4, notebook secara otomatis:
1. Membuka semua parameter backbone (melepas pembekuan).
2. Membuat optimizer baru dengan dua kelompok parameter (backbone dan classifier) masing-masing dengan learning rate berbeda.
3. Membuat scheduler baru untuk optimizer yang baru.

Transisi ini ditandai dengan pesan "_Unfreeze backbone mulai epoch X; fine-tuning penuh aktif_" pada log training.

### Monitoring Selama Training

Setiap epoch dicetak dalam format tabel dengan kolom: epoch, phase (warmup_head atau finetune), training loss, training raw accuracy, validation loss, validation accuracy, Validation Macro F1, Validation QWK, Validation Balanced Score, learning rate backbone, learning rate classifier, dan persentase penggunaan RAM. Epoch terbaik ditandai dengan asterisk (*).

---

## 9. Evaluasi Model

### Metrik Evaluasi yang Digunakan

#### Accuracy (Akurasi)
Proporsi prediksi benar dari seluruh prediksi. Metrik ini mudah dipahami tetapi **tidak representatif untuk dataset tidak seimbang** seperti APTOS 2019. Model yang selalu memprediksi "No DR" akan mendapat accuracy tinggi karena kelas ini dominan.

#### Balanced Accuracy
Rata-rata recall per kelas. Lebih adil dari accuracy biasa karena memberikan bobot sama pada setiap kelas terlepas dari frekuensinya.

#### Sensitivity (Recall)
Proporsi sampel positif yang berhasil dideteksi benar oleh model untuk setiap kelas. Dalam konteks DR, sensitivity tinggi berarti model tidak banyak melewatkan kasus yang sebenarnya positif. Notebook melaporkan sensitivity dalam bentuk macro average (rata-rata per kelas).

#### Specificity
Proporsi sampel negatif yang berhasil diidentifikasi sebagai negatif. Dihitung secara manual dalam notebook menggunakan confusion matrix karena sklearn tidak menyediakan metrik ini secara langsung untuk multi-kelas.

#### Precision
Proporsi prediksi positif yang benar-benar positif. Precision tinggi berarti model jarang memberikan alarm palsu (false positive).

#### Macro F1
Rata-rata F1-score dari seluruh kelas **tanpa pembobotan berdasarkan frekuensi kelas**. Metrik ini menjadi **metrik utama** dalam notebook karena memberikan bobot yang sama pada kelas minoritas dan mayoritas. Macro F1 rendah mengindikasikan performa buruk pada setidaknya satu kelas, terlepas dari performa kelas majoritasnya.

F1-score adalah rata-rata harmonik dari Precision dan Recall:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

#### Weighted F1
Rata-rata F1-score yang dibobot berdasarkan frekuensi kelas. Metrik ini lebih mencerminkan performa keseluruhan pada distribusi kelas asli, tetapi kurang sensitif terhadap kegagalan pada kelas minoritas.

#### Quadratic Weighted Kappa (QWK)
QWK mengukur tingkat kesepakatan antara prediksi model dan label asli, dengan **penalti lebih besar untuk kesalahan yang jaraknya jauh**. Misalnya, memprediksi kelas "Severe" sebagai "No DR" mendapat penalti lebih besar daripada memprediksinya sebagai "Moderate".

QWK sangat relevan untuk klasifikasi ordinal seperti tingkat keparahan DR karena:
- Label DR memiliki urutan tingkat keparahan yang bermakna (0 < 1 < 2 < 3 < 4).
- Dalam konteks klinis, memprediksi "Severe" sebagai "No DR" jauh lebih berbahaya daripada memprediksinya sebagai "Moderate".
- QWK tinggi menunjukkan bahwa kesalahan prediksi model cenderung pada kelas yang berdekatan, bukan kelas yang jauh.

Nilai QWK berkisar dari -1 hingga 1, di mana 1 berarti kesepakatan sempurna, 0 berarti kesepakatan yang hanya setara dengan tebakan acak, dan negatif berarti lebih buruk dari tebakan acak.

#### AUC (Area Under the ROC Curve)
AUC mengukur kemampuan model membedakan antar kelas berdasarkan skor probabilitas, terlepas dari threshold yang digunakan untuk membuat keputusan kelas akhir. Notebook menghitung macro AUC menggunakan pendekatan One-vs-Rest (OvR). AUC tinggi tidak selalu berarti prediksi kelas akhir (argmax probabilitas) sudah optimal — model mungkin memiliki representasi probabilistik yang baik tetapi threshold yang kurang tepat untuk dataset yang tidak seimbang.

#### Confusion Matrix
Matriks yang menampilkan jumlah prediksi benar dan salah untuk setiap kombinasi kelas asli dan kelas prediksi. Notebook menampilkan confusion matrix dalam dua format: raw count dan normalized (per baris, menunjukkan recall per kelas). Confusion matrix memungkinkan identifikasi kelas mana yang sering dikacaukan dengan kelas lain.

#### Bootstrap Confidence Interval (95%)
Untuk setiap metrik utama (Accuracy, Macro F1, Weighted F1, Balanced Accuracy, QWK, AUC), notebook menghitung interval kepercayaan 95% menggunakan metode bootstrap dengan 1000 iterasi. Ini penting karena test set berukuran kecil, sehingga estimasi metrik tunggal memiliki ketidakpastian yang signifikan.

### Audit TTA (Test Time Augmentation)

Sebelum mengevaluasi test set, notebook melakukan audit TTA pada validation set. TTA yang digunakan adalah **horizontal flip**: setiap gambar dievaluasi dua kali (asli dan versi flip horizontal), kemudian probabilitas dari keduanya dirata-ratakan.

Kebijakan TTA yang digunakan adalah `validation_guarded`: TTA hanya diterapkan pada test set jika evaluasi pada validation set menunjukkan bahwa TTA **tidak menurunkan Macro F1 lebih dari -0.002 dan QWK lebih dari -0.002**. Kebijakan ini lebih konservatif dari asumsi bahwa TTA selalu membantu.

---

## 10. Analisis Hasil dan Output

### Hasil Run Sebelum Balanced Tuning (Baseline Internal)

Notebook mendokumentasikan hasil dari run sebelumnya (sebelum balanced tuning) sebagai referensi perbandingan:

| Model | Best Val Epoch | Test Accuracy | Test Macro F1 | Test QWK | Test AUC | Catatan Utama |
|---|---:|---:|---:|---:|---:|---|
| Hybrid v4 revisi | 83 | 0.8420 | 0.7143 | 0.9087 | 0.8751 | Lebih stabil, unggul pada Macro F1/QWK |
| InceptionV3 v4 revisi | 56 | 0.7847 | 0.6364 | 0.8479 | 0.8813 | AUC cukup baik, prediksi kelas kurang tajam |

**Interpretasi Hasil Baseline**:

- Hybrid mencapai **QWK = 0.9087**, yang termasuk sangat baik. Ini berarti kesalahan prediksi Hybrid cenderung pada kelas yang berdekatan dengan label aslinya, bukan kelas yang jauh.
- Hybrid memiliki **Macro F1 = 0.7143**, menunjukkan bahwa rata-rata kemampuan membedakan antar kelas cukup baik tetapi belum optimal, terutama karena kelas minoritas masih memiliki F1 yang lebih rendah.
- InceptionV3 memiliki **AUC = 0.8813** yang sedikit lebih tinggi dari Hybrid (0.8751), tetapi **Macro F1 = 0.6364** jauh lebih rendah. Ini mengindikasikan InceptionV3 memiliki representasi probabilistik yang cukup baik, tetapi keputusan kelas akhirnya kurang akurat, terutama pada kelas-kelas tengah (Mild, Moderate, Severe).
- Best val epoch InceptionV3 dicapai pada epoch 56, jauh sebelum epoch maksimal 100, mengindikasikan early stopping yang efektif atau plateau yang panjang.

### Target Tuning dan Masalah yang Teridentifikasi

Notebook secara eksplisit mengidentifikasi masalah dari hasil baseline:

1. **Ketidakseimbangan dataset** sangat terasa: No DR 49.3%, Severe hanya 5.3%, Proliferative DR 8.1%.
2. **Validation dan test set** memiliki sangat sedikit sampel untuk kelas Severe (19 gambar) dan Proliferative DR (30 gambar), membuat metrik kelas ini tidak stabil.
3. **InceptionV3** banyak menggeser prediksi kelas Moderate, Severe, dan Proliferative DR ke kelas sekitarnya (Mild/Moderate), menunjukkan model kesulitan membedakan kelas-kelas tingkat keparahan yang lebih tinggi.
4. **Hybrid** masih lemah pada Severe dan Proliferative DR, tetapi performa keseluruhannya lebih baik dibandingkan InceptionV3.
5. **Training 100 epoch** menunjukkan plateau panjang; model tidak mengalami peningkatan signifikan pada banyak epoch terakhir.

### Hasil Balanced Tuning (Belum Tersedia dari Output Notebook)

Notebook yang diupload merupakan notebook yang telah direvisi tetapi **belum dieksekusi ulang sepenuhnya**. Hasil metrik dari balanced tuning (sel-sel evaluasi di bagian akhir notebook) tidak memiliki output yang tersimpan. Oleh karena itu, hasil aktual dari konfigurasi balanced tuning ini tidak dapat dilaporkan secara numerik berdasarkan isi notebook yang ada.

### Perbandingan dengan Jurnal (Xu et al., 2024)

| Model | Accuracy (Jurnal) | Macro F1 (Jurnal) | AUC (Jurnal) |
|---|---|---|---|
| Hybrid | 0.97 | 0.96 | 0.97 |
| InceptionV3 | 0.90 | 0.90 | 0.88 |

Terdapat kesenjangan yang signifikan antara hasil notebook (Hybrid Macro F1 ≈ 0.71, InceptionV3 Macro F1 ≈ 0.64) dengan hasil jurnal (Hybrid Macro F1 ≈ 0.96, InceptionV3 Macro F1 ≈ 0.90). Notebook sendiri mencatat bahwa perbandingan langsung dengan jurnal **tidak sepenuhnya valid** karena protokol eksperimen yang berbeda (pembagian data, preprocessing, augmentasi, hyperparameter, dan kemungkinan subset dataset yang berbeda).

---

## 11. Visualisasi dan Interpretasi Visual

### Grafik Training History

Untuk setiap model, notebook menghasilkan grafik tiga panel:
1. **Loss** — Kurva training loss dan validation loss per epoch, dengan garis vertikal menandai transisi dari warmup ke fine-tuning.
2. **Validation Metrics** — Kurva Validation Accuracy, Macro F1, QWK, Balanced Score, dan Worst Class F1.
3. **Staged Fine-Tuning LR** — Kurva perubahan learning rate backbone dan classifier sepanjang training.

Grafik ini penting untuk mendeteksi:
- **Overfitting**: Training loss terus menurun tetapi validation loss mulai naik.
- **Underfitting**: Keduanya masih tinggi dan belum konvergen.
- **Plateau**: Metrik validasi tidak berubah selama banyak epoch, mengindikasikan model sudah mencapai batas kapasitas belajar dengan konfigurasi saat ini.
- **Efektivitas LR Scheduler**: Penurunan LR yang tepat waktu biasanya disertai peningkatan performa.

### Confusion Matrix

Confusion matrix ditampilkan dalam dua format untuk memudahkan analisis:
- **Raw Count**: Menampilkan jumlah sampel absolut untuk setiap kombinasi prediksi-truth.
- **Normalized**: Menampilkan proporsi per baris (recall per kelas), memudahkan identifikasi kelas mana yang sering dikacaukan dengan kelas lain tanpa terpengaruh perbedaan jumlah sampel per kelas.

### ROC Curve Multi-Kelas

ROC Curve ditampilkan per kelas menggunakan pendekatan One-vs-Rest. Area di bawah kurva (AUC) per kelas membantu mengidentifikasi kelas mana yang paling mudah dibedakan oleh model secara probabilistik.

### F1 per Kelas

Grafik batang F1-score per kelas untuk kedua model memungkinkan perbandingan langsung kekuatan dan kelemahan masing-masing model pada setiap kelas.

### Bootstrap Confidence Interval

Error bar dari bootstrap CI ditampilkan untuk memvisualisasikan ketidakpastian estimasi metrik akibat ukuran test set yang terbatas.

### Grad-CAM (Gradient-weighted Class Activation Mapping)

Grad-CAM menghasilkan _heatmap_ yang menunjukkan area gambar yang paling berkontribusi pada keputusan prediksi model. Notebook menghasilkan visualisasi Grad-CAM untuk:

- **Hybrid**: Menggunakan layer `conv_head` dari cabang EfficientNetB3. Visualisasi ini bersifat **parsial** karena hanya mewakili satu dari dua cabang backbone.
- **InceptionV3**: Menggunakan layer `Mixed_7c` yang merupakan blok inception terakhir sebelum global average pooling.

Setiap visualisasi menampilkan gambar asli (baris atas) dan gambar dengan overlay Grad-CAM (baris bawah) untuk satu contoh per kelas. Prioritas diberikan pada contoh yang diprediksi **benar**; jika tidak ada contoh benar untuk suatu kelas, contoh yang salah ditampilkan sebagai gantinya. Judul setiap gambar mencantumkan label asli, prediksi, tingkat kepercayaan, dan status prediksi (benar/salah).

---

## 12. Analisis Kelebihan dan Kekurangan

### Kelebihan Notebook

1. **Manajemen Memori yang Cermat**: Notebook mengimplementasikan sistem monitoring RAM aktif (`print_ram`, `ram_guard`), cleanup berkala setiap 10 batch, dan pembacaan data per batch dari disk cache. Pendekatan ini memungkinkan eksperimen berjalan pada lingkungan dengan RAM terbatas.

2. **Strategi Evaluasi yang Komprehensif**: Tidak bergantung pada accuracy saja, tetapi menggunakan kombinasi Macro F1, QWK, Balanced Accuracy, Specificity, Sensitivity, per-class F1, AUC, Confusion Matrix, dan Bootstrap CI. Ini memberikan gambaran yang jauh lebih lengkap tentang performa model.

3. **Pemilihan Checkpoint yang Robust**: Menggunakan skor gabungan (_balanced_score_) alih-alih metrik tunggal yang rentan terhadap noise pada validation set kecil.

4. **Audit TTA yang Konservatif**: TTA tidak diasumsikan selalu membantu. Audit pada validation set dilakukan terlebih dahulu, dan TTA hanya digunakan jika terbukti tidak merugikan Macro F1 maupun QWK.

5. **Transparansi Keterbatasan**: Notebook secara eksplisit menyebutkan keterbatasan data (jumlah sampel kelas minoritas yang sangat kecil), keterbatasan perbandingan dengan jurnal, dan keterbatasan Grad-CAM yang bersifat parsial untuk model Hybrid.

6. **Dokumentasi Inline yang Baik**: Setiap tahapan memiliki penjelasan tujuan dan alasan perubahan dari versi sebelumnya, memudahkan penelusuran eksperimen.

### Kekurangan dan Keterbatasan Notebook

1. **Dataset yang Sangat Tidak Seimbang**: Ketidakseimbangan yang ekstrem (No DR ~49% vs Severe ~5%) membuat metrik kelas minoritas pada validation dan test set sangat tidak stabil. Dengan hanya 19 sampel Severe pada validation set, perubahan satu prediksi saja dapat mengubah F1 kelas Severe secara signifikan.

2. **Kesenjangan dengan Jurnal Referensi**: Performa notebook masih jauh di bawah klaim Xu et al. (2024), yang mungkin disebabkan oleh perbedaan protokol eksperimen, dataset yang lebih bersih, augmentasi yang berbeda, atau perbedaan dalam cara pembagian data. Notebook tidak dapat memverifikasi kesetaraan protokol dengan jurnal.

3. **Grad-CAM Hybrid Bersifat Parsial**: Visualisasi Grad-CAM untuk model Hybrid hanya menggunakan cabang EfficientNetB3, sehingga tidak mencerminkan bagaimana Swin Transformer berkontribusi pada keputusan akhir. Interpretasi Grad-CAM Hybrid harus dilakukan dengan hati-hati.

4. **Keterbatasan Skalabilitas**: Penggunaan `num_workers = 0` dan batch size 16 membatasi throughput training. Ini adalah trade-off yang diperlukan untuk keamanan memori tetapi memperpanjang durasi training secara signifikan.

5. **Hasil Balanced Tuning Belum Tersedia**: Notebook yang diupload belum dieksekusi ulang dengan konfigurasi baru, sehingga tidak ada konfirmasi numerik bahwa balanced tuning berhasil meningkatkan atau minimal mempertahankan performa dari baseline internal.

6. **Validasi Statistik Terbatas**: Bootstrap CI membantu, tetapi ukuran test set yang kecil (terutama untuk kelas Severe dan Proliferative DR) tetap membatasi validitas statistik kesimpulan yang ditarik.

---

## 13. Kesimpulan

Notebook ini merupakan eksperimen deep learning yang terstruktur dan dokumentatif untuk klasifikasi tingkat keparahan Diabetic Retinopathy menggunakan dataset APTOS 2019. Pendekatan utama yang diambil adalah membandingkan model Hybrid (EfficientNetB3 + Swin Transformer Tiny) dengan InceptionV3 sebagai baseline, dalam kerangka yang memperhatikan ketidakseimbangan kelas dan efisiensi memori.

Berdasarkan hasil yang tersedia (dari run sebelum balanced tuning), model Hybrid mengungguli InceptionV3 pada hampir semua metrik klasifikasi yang relevan: Accuracy (0.842 vs 0.785), Macro F1 (0.714 vs 0.636), dan QWK (0.909 vs 0.848). InceptionV3 memiliki AUC yang sedikit lebih tinggi (0.881 vs 0.875), mengindikasikan representasi probabilistiknya cukup baik tetapi keputusan kelas akhirnya kurang akurat.

Kerangka kerja yang dibangun dalam notebook ini sudah cukup matang untuk eksperimen komparatif: menggunakan strategi training bertahap, loss function yang memperhatikan ketidakseimbangan kelas, pemilihan checkpoint berbasis skor gabungan, dan evaluasi yang tidak bergantung pada accuracy saja. Revisi balanced tuning yang diimplementasikan dalam notebook ini bertujuan untuk mengatasi kelemahan yang teridentifikasi dari run sebelumnya, khususnya terlalu rendahnya performa pada kelas-kelas tingkat keparahan menengah hingga tinggi.

Namun, notebook ini perlu dieksekusi ulang secara lengkap untuk memperoleh konfirmasi numerik dari efektivitas balanced tuning. Selain itu, kesenjangan yang signifikan antara hasil notebook dengan klaim jurnal referensi memerlukan investigasi lebih lanjut mengenai perbedaan protokol eksperimen.

---

## 14. Catatan Perbaikan

Berikut adalah saran peningkatan yang realistis berdasarkan isi notebook:

1. **Stratified K-Fold Cross Validation**: Menggunakan k-fold (misalnya 5-fold) akan memberikan estimasi performa yang jauh lebih stabil dibandingkan single split, terutama untuk kelas minoritas yang sangat sedikit. Ini akan memanfaatkan seluruh data berlabel secara lebih efisien.

2. **Oversampling Sintetis pada Kelas Minoritas**: Selain WeightedRandomSampler, pertimbangkan teknik augmentasi citra yang lebih agresif khusus untuk kelas Severe dan Proliferative DR (misalnya dengan augmentasi warna yang lebih kuat atau pencarian augmentasi otomatis seperti AutoAugment).

3. **Eksplorasi Threshold Kalibrasi**: Untuk dataset tidak seimbang, menggunakan threshold berbeda per kelas (bukan argmax) dapat meningkatkan Macro F1 tanpa mengubah arsitektur. Threshold optimal dapat ditentukan pada validation set.

4. **Penambahan Regularisasi Label**: Selain label smoothing, teknik seperti StochDepth (Stochastic Depth) atau CutMix dapat dicoba untuk meningkatkan generalisasi model Hybrid yang kapasitasnya besar.

5. **Verifikasi Ekuivalensi Protokol dengan Jurnal**: Jika perbandingan dengan Xu et al. (2024) ingin dilakukan secara adil, perlu dilakukan reproduksi ulang preprocessing dan pembagian data seperti yang digunakan dalam jurnal, atau setidaknya dokumentasi perbedaan protokol yang lebih eksplisit.

6. **Ensemble Kedua Model**: Dengan menggabungkan prediksi probabilitas dari Hybrid dan InceptionV3 (misalnya rata-rata tertimbang), performa ensemble sering kali melampaui model individual, terutama karena kedua model belajar dari input gambar yang berbeda (224 vs 299 piksel) dan arsitektur yang berbeda.

7. **Penambahan Metrik Kalibrasi**: Expected Calibration Error (ECE) dapat ditambahkan untuk mengevaluasi apakah probabilitas yang dihasilkan model benar-benar mencerminkan tingkat kepercayaan yang akurat, bukan hanya mengukur apakah prediksi kelas akhirnya benar.

8. **Eksplorasi Grad-CAM yang Lebih Representatif**: Untuk model Hybrid, pertimbangkan visualisasi yang mencakup kedua cabang backbone, misalnya dengan GradCAM++ pada hasil fusi atau menggunakan Layer Relevance Propagation yang lebih cocok untuk arsitektur hybrid.

---

*Dokumen ini dibuat berdasarkan analisis menyeluruh terhadap isi notebook `notebookfixed.ipynb`, termasuk seluruh kode, komentar markdown, konfigurasi, dan output yang tersedia. Bagian-bagian notebook yang belum memiliki output (evaluasi balanced tuning) dicatat secara eksplisit dalam dokumen ini.*
