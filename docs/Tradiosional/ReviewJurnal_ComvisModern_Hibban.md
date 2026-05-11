# Dokumentasi Akademik Review Jurnal: Computer Vision Modern dalam Analisis Citra Medis

**Disusun oleh:** M. Hibban Ramadhan (NIM: 2315061094)  
**Tanggal Review:** 7 Mei 2026  
**Mata Kuliah:** Computer Vision Modern

---

## Pengantar Dokumen

Dokumen ini merupakan narasi penjelasan akademik yang mendalam atas tiga jurnal ilmiah yang berkaitan dengan penerapan Computer Vision dan Deep Learning dalam dunia medis, khususnya pada bidang analisis citra medis berbasis kecerdasan buatan. Ketiga jurnal tersebut dipublikasikan pada tahun 2024 dan berfokus pada penggunaan arsitektur Vision Transformer, baik secara mandiri maupun dalam bentuk hybrid, untuk menyelesaikan persoalan klasifikasi dan segmentasi citra medis.

Dokumen ini dirancang agar dapat dipahami oleh pembaca yang belum memiliki latar belakang mendalam dalam Computer Vision, Deep Learning, maupun Medical Imaging. Setiap istilah teknis yang muncul akan dijelaskan secara langsung beserta fungsi, mekanisme kerja, dan relevansinya dalam konteks penelitian. Narasi disusun secara runtut dari identitas jurnal hingga refleksi akademik di bagian akhir.

---

# BAGIAN I: JURNAL PERTAMA

## Klasifikasi Tumor Otak Berbasis MRI Menggunakan Fine-Tuned Vision Transformer

---

### 1. Identitas Jurnal

**Judul:** A Fine-Tuned Vision Transformer Based Enhanced Multi-Class Brain Tumor Classification Using MRI Scan Imagery

**Jurnal Penerbit:** Frontiers in Oncology, diterbitkan oleh Frontiers Media SA

**Volume dan Halaman:** Volume 14, Article 1400341, 14 halaman

**Tahun Terbit:** 2024

**Penulis:** C Kishor Kumar Reddy, Pulakurthi Anaghaa Reddy, Himaja Janapati, Basem Assiri, Mohammed Shuaib, Shadab Alam, Abdullah Sheneamer

Jurnal ini tergolong dalam bidang onkologi komputasional, yakni cabang ilmu yang menggabungkan ilmu komputer dengan ilmu medis khususnya kanker, untuk membangun sistem cerdas yang mampu mendeteksi dan mengklasifikasikan tumor secara otomatis. Fokus utama penelitian ini adalah pengembangan model klasifikasi multiclass untuk tumor otak menggunakan citra MRI, dengan pendekatan arsitektur Vision Transformer yang telah dioptimalkan melalui proses fine-tuning.

Gambaran umum permasalahan yang ingin diselesaikan adalah bagaimana meningkatkan akurasi klasifikasi tumor otak yang selama ini dilakukan secara manual oleh dokter radiologi. Proses manual tersebut membutuhkan keahlian tinggi, bersifat subjektif, dan rawan kesalahan terutama pada kasus tumor dengan tampilan visual yang mirip satu sama lain. Penelitian ini bertujuan menyediakan solusi berbasis kecerdasan buatan yang lebih objektif, konsisten, dan akurat.

---

### 2. Latar Belakang Masalah

#### Mengapa Masalah Ini Penting

Tumor otak adalah salah satu penyakit paling berbahaya yang menyerang sistem saraf pusat manusia. Keterlambatan diagnosis dapat berakibat fatal karena tumor dapat berkembang pesat dan merusak jaringan otak secara permanen. Oleh karena itu, diagnosis dini yang akurat adalah kunci utama dalam meningkatkan angka harapan hidup pasien. Dalam konteks ini, sistem otomatis berbasis komputer yang mampu menganalisis citra otak menjadi sangat dibutuhkan, terutama di daerah yang kekurangan dokter spesialis.

#### Mengapa Computer Vision Digunakan

Computer Vision adalah cabang ilmu kecerdasan buatan yang memungkinkan mesin untuk memahami, menginterpretasikan, dan menganalisis informasi dari gambar atau video. Dalam konteks medis, Computer Vision digunakan karena diagnosis tumor otak secara konvensional bergantung sepenuhnya pada interpretasi visual terhadap citra MRI oleh dokter. Proses ini memerlukan waktu, tenaga ahli yang langka, dan tetap mengandung unsur subjektivitas. Dengan Computer Vision, proses analisis dapat diotomatisasi, dipercepat, dan standarisasinya dapat ditingkatkan.

#### Apa Itu MRI dan Mengapa Relevan

MRI, singkatan dari Magnetic Resonance Imaging atau Pencitraan Resonansi Magnetik, adalah modalitas pencitraan medis yang menggunakan medan magnet kuat dan gelombang radio untuk menghasilkan gambar detail dari struktur internal tubuh, khususnya jaringan lunak seperti otak. MRI dipilih sebagai modalitas utama dalam penelitian ini karena kemampuannya menghasilkan gambar otak dengan resolusi dan kontras jaringan yang jauh lebih baik dibandingkan foto Rontgen atau CT Scan, sehingga sangat cocok untuk mendeteksi keberadaan, ukuran, dan lokasi tumor.

#### Tantangan Analisis Citra Medis

Analisis citra medis secara komputasional memiliki beberapa tantangan khusus yang tidak dijumpai pada analisis citra biasa. Pertama, citra medis memiliki variasi penampilan yang tinggi antar pasien karena perbedaan anatomi, usia, dan kondisi medis. Kedua, tumor yang berbeda jenis dapat memiliki tampilan visual yang sangat mirip, sehingga sulit dibedakan bahkan oleh sistem berbasis pembelajaran mesin. Ketiga, jumlah data berlabel untuk citra medis biasanya terbatas karena proses anotasi membutuhkan waktu dan keahlian dokter. Keempat, beberapa jenis tumor, seperti Meningioma, memiliki karakteristik visual yang tidak konsisten sehingga menyulitkan model untuk belajar pola yang representatif.

#### Keterbatasan CNN dan Mengapa Transformer Mulai Digunakan

Convolutional Neural Network, atau CNN, adalah arsitektur jaringan saraf tiruan yang telah lama digunakan dalam analisis citra. CNN bekerja dengan cara mengekstraksi fitur lokal dari gambar melalui operasi konvolusi yang diterapkan secara bertahap dari area kecil ke area yang lebih besar. Meskipun CNN terbukti efektif dalam banyak tugas pengenalan citra, arsitektur ini memiliki keterbatasan fundamental, yaitu keterbatasan pada receptive field lokal. Receptive field adalah area piksel dalam gambar yang dipertimbangkan oleh satu neuron dalam jaringan. CNN memproses gambar secara lokal, artinya hubungan antara bagian gambar yang berjauhan tidak dapat ditangkap secara langsung.

Dalam analisis citra medis, keterbatasan ini menjadi masalah nyata. Tumor otak tidak hanya dikenali dari tekstur lokalnya saja, tetapi juga dari hubungan spasialnya terhadap struktur otak lain yang berada di bagian berbeda dalam gambar. CNN tidak dirancang untuk menangkap hubungan jarak jauh semacam ini secara efisien.

Transformer, yang awalnya dirancang untuk pemrosesan bahasa alami, memiliki mekanisme self-attention yang mampu mempertimbangkan seluruh bagian dari sebuah input secara bersamaan, termasuk hubungan antar bagian yang berjauhan. Vision Transformer mengadaptasi mekanisme ini untuk citra, memungkinkan model untuk memahami hubungan spasial global dalam gambar. Inilah alasan utama mengapa Transformer mulai menggantikan CNN dalam berbagai tugas analisis citra medis.

---

### 3. Dataset

Dataset yang digunakan dalam penelitian ini merupakan gabungan dari tiga sumber data publik yang berbeda, yaitu Figshare, SARTAJ, dan Br35H. Penggabungan ketiganya menghasilkan total 7.023 citra MRI otak yang telah dilabeli. Dataset gabungan ini tersedia secara publik di platform Kaggle dengan nama Brain Tumor MRI Dataset.

#### Kelas Klasifikasi

Dataset ini mengandung empat kelas, yaitu:

- **Glioma:** Tumor yang berasal dari sel glia, yaitu sel pendukung di dalam otak dan sumsum tulang belakang. Glioma adalah jenis tumor otak yang paling umum dan paling agresif. Ia dapat menyerang berbagai bagian otak dan biasanya memiliki batas yang tidak jelas, sehingga sulit untuk dibedakan dari jaringan sehat di sekitarnya dalam citra MRI.
- **Meningioma:** Tumor yang tumbuh dari meninges, yaitu lapisan pelindung yang menyelubungi otak dan sumsum tulang belakang. Meningioma umumnya bersifat jinak, namun karena letaknya yang menekan otak, ia tetap dapat berbahaya. Dalam citra MRI, Meningioma seringkali tampak mirip dengan jenis tumor lain, menjadikannya kelas yang paling sulit diklasifikasikan secara otomatis.
- **Pituitary Tumor:** Tumor yang tumbuh di kelenjar pituitari (hipofisis), sebuah kelenjar kecil di dasar otak yang mengatur produksi hormon. Meskipun sebagian besar bersifat jinak, tumor pituitari dapat mengganggu keseimbangan hormonal tubuh secara signifikan.
- **No Tumor:** Kelas kontrol yang berisi citra MRI otak tanpa tumor, digunakan sebagai kelas negatif dalam klasifikasi.

#### Pembagian Data

Dari total 7.023 citra, sebanyak 5.712 gambar digunakan untuk pelatihan model dan 1.311 gambar digunakan untuk pengujian. Pembagian ini dilakukan secara acak menggunakan fungsi random_split() dari library PyTorch.

#### Mengapa Dataset Ini Dipilih

Dataset ini dipilih karena beberapa alasan. Pertama, ukurannya yang cukup besar untuk pelatihan model deep learning. Kedua, ketersediaan empat kelas yang representatif terhadap kondisi klinis nyata. Ketiga, keterjangkauannya sebagai data publik yang memungkinkan reproduksibilitas penelitian. Keempat, data ini telah digunakan dalam berbagai penelitian sebelumnya, sehingga memberikan baseline perbandingan yang valid.

#### Tantangan Dataset

Tantangan utama dataset ini adalah adanya ketidakseimbangan jumlah citra antar kelas. Kelas Meningioma memiliki jumlah sampel yang lebih sedikit dan variasi visual yang lebih tinggi dibandingkan kelas lain. Hal ini menyebabkan model cenderung kesulitan mempelajari pola yang konsisten untuk kelas tersebut, sebagaimana tercermin dari akurasi per kelas pada model-model CNN yang lebih rendah.

---

### 4. Preprocessing dan Augmentasi Data

Preprocessing adalah serangkaian tahapan pengolahan awal yang dilakukan terhadap data citra sebelum dimasukkan ke dalam model untuk pelatihan. Tujuan utama preprocessing adalah memastikan data dalam kondisi standar, bersih, dan siap diolah oleh jaringan saraf tiruan. Augmentasi data adalah teknik yang digunakan untuk memperluas keberagaman data pelatihan secara artifisial tanpa mengumpulkan data baru.

Seluruh preprocessing dan augmentasi dalam jurnal ini dilakukan menggunakan fungsi transforms.Compose() dari library Torchvision, yang merupakan ekstensi dari framework PyTorch untuk pengolahan citra.

#### Resize ke 224x224 Piksel

Seluruh citra MRI diubah ukurannya menjadi dimensi 224x224 piksel. Standarisasi ukuran ini diperlukan karena jaringan saraf tiruan mengharapkan input dengan dimensi yang tetap dan seragam. Nilai 224x224 bukan dipilih secara sembarangan, melainkan karena ukuran ini merupakan ukuran input standar yang digunakan oleh model-model pretrained yang dilatih pada dataset ImageNet, termasuk Vision Transformer yang digunakan dalam penelitian ini. Dengan menggunakan ukuran yang sama, bobot pretrained dapat dimanfaatkan secara optimal tanpa distorsi informasi spasial yang berlebihan.

Proses resize bekerja dengan menggunakan interpolasi piksel, di mana nilai piksel pada dimensi baru diperkirakan berdasarkan nilai piksel di sekitarnya pada citra asli. Dampak positifnya adalah efisiensi komputasi meningkat karena ukuran gambar lebih seragam, sementara risiko kecilnya adalah hilangnya detail resolusi tinggi jika citra asli memiliki dimensi yang jauh lebih besar.

#### Horizontal Flip dengan Probabilitas 0.5

Augmentasi pertama adalah horizontal flip, yaitu membalik citra secara horizontal, seperti mencerminkannya. Teknik ini diterapkan dengan probabilitas 0.5, artinya setiap citra dalam batch pelatihan memiliki kemungkinan 50% untuk dibalik secara acak. Tujuannya adalah menambah variasi orientasi tanpa mengubah konten semantik gambar.

Dalam konteks citra MRI otak, tumor dapat muncul di sisi kiri maupun kanan otak. Dengan menambahkan variasi horizontal, model tidak hanya belajar mengenali tumor pada posisi aslinya, tetapi juga pada posisi cerminannya. Hal ini memperluas kemampuan generalisasi model terhadap kasus klinis nyata di mana posisi tumor sangat bervariasi antar pasien.

#### Random Rotation 10 Derajat

Rotasi acak hingga 10 derajat diterapkan untuk mensimulasikan variasi posisi pasien saat menjalani pemindaian MRI. Dalam praktik klinis, posisi kepala pasien di dalam mesin MRI tidak selalu sempurna sehingga citra yang dihasilkan dapat sedikit miring. Dengan melatih model pada citra yang sedikit diputar, model menjadi lebih robust terhadap variasi orientasi yang terjadi secara alami dalam kondisi klinis.

Nilai 10 derajat dipilih sebagai batas yang cukup untuk mensimulasikan variasi posisi, namun tidak terlalu besar sehingga citra masih dapat dikenali dengan benar oleh model. Rotasi yang terlalu besar berisiko mengubah citra secara berlebihan sehingga justru mengganggu proses pembelajaran.

#### Color Jitter

Color jitter adalah teknik augmentasi yang mengubah properti warna citra secara acak, meliputi kecerahan (brightness), kontras (contrast), saturasi (saturation), dan rona warna (hue). Tujuannya adalah mensimulasikan perbedaan kondisi pencahayaan dan kalibrasi alat MRI antar fasilitas kesehatan.

Dalam konteks MRI, meskipun modalitas ini tidak menggunakan cahaya tampak, intensitas sinyal yang terekam bergantung pada parameter pengaturan mesin, kondisi kontras agent, dan karakteristik biologis pasien. Color jitter membantu model untuk tidak bergantung pada nilai intensitas piksel yang absolut, melainkan lebih berfokus pada pola tekstural dan struktural yang lebih stabil dan informatif.

#### Konversi ke Tensor PyTorch

Citra yang telah diproses dikonversi dari format PIL Image, yaitu format objek gambar standar Python, menjadi tensor PyTorch. Tensor adalah struktur data multidimensional yang digunakan oleh framework deep learning seperti PyTorch. Konversi ini diperlukan karena seluruh operasi matematika dalam jaringan saraf tiruan dilakukan pada tensor, bukan pada objek gambar biasa. Selama konversi, nilai piksel yang awalnya berada dalam rentang integer 0 hingga 255 secara otomatis diubah menjadi nilai float dalam rentang 0.0 hingga 1.0.

#### Normalisasi Piksel

Normalisasi adalah proses mengubah distribusi nilai piksel agar memiliki rata-rata mendekati nol dan standar deviasi mendekati satu. Hal ini dilakukan dengan menggunakan statistik mean dan standar deviasi yang telah dihitung dari dataset ImageNet. Normalisasi sangat penting untuk mempercepat dan menstabilkan proses konvergensi selama pelatihan, karena gradien yang mengalir melalui jaringan menjadi lebih seimbang dan tidak ada satu dimensi fitur yang mendominasi secara berlebihan.

---

### 5. Arsitektur Model

#### Vision Transformer: Konsep Dasar

Vision Transformer, atau ViT, adalah arsitektur jaringan saraf tiruan yang diadaptasi dari arsitektur Transformer yang awalnya dikembangkan untuk pemrosesan bahasa alami. Ide utama ViT adalah memerlakukan citra sebagai sebuah urutan (sequence) elemen-elemen diskrit, mirip dengan cara kata-kata dalam kalimat diproses oleh Transformer berbasis teks. Alih-alih memproses piksel satu per satu, ViT membagi citra menjadi patch-patch kecil yang tidak tumpang tindih, lalu memproses setiap patch sebagai satu unit.

#### Patch Embedding

Proses pertama dalam ViT adalah membagi citra input menjadi patch-patch kecil berukuran tetap. Pada varian ViT-b16 dan ViT-l16, setiap patch berukuran 16x16 piksel. Pada varian ViT-b32 dan ViT-l32, setiap patch berukuran 32x32 piksel. Untuk citra berukuran 224x224, varian 16x16 menghasilkan 196 patch, sementara varian 32x32 menghasilkan 49 patch.

Setiap patch kemudian diratakan (flatten) menjadi sebuah vektor satu dimensi dan dipetakan ke dalam ruang dimensi yang lebih tinggi melalui lapisan linear. Proses pemetaan ini disebut patch embedding. Hasilnya adalah sebuah deretan vektor yang merepresentasikan setiap patch citra dalam ruang fitur yang lebih kaya. Embeddling di sini dapat dianalogikan seperti menterjemahkan setiap potongan gambar menjadi sebuah representasi numerik yang bermakna.

#### Token dan Class Token

Setiap vektor patch embedding disebut sebagai token. Selain token-token patch, ViT juga menambahkan satu token khusus yang disebut class token, yaitu sebuah vektor yang dapat dipelajari dan ditempatkan di awal urutan. Fungsi class token adalah sebagai representasi agregat dari seluruh informasi dalam citra. Setelah melalui seluruh lapisan Transformer, vektor pada posisi class token inilah yang akan digunakan untuk membuat keputusan klasifikasi akhir.

#### Positional Embedding

Berbeda dengan konvolusi yang secara inheren mempertahankan informasi posisi spasial, mekanisme attention dalam Transformer secara default tidak mengetahui urutan atau posisi dari token-token yang diproses. Oleh karena itu, ditambahkan positional embedding, yaitu sebuah vektor yang dikodekan dengan informasi posisi dan ditambahkan ke setiap token sebelum dimasukkan ke dalam encoder Transformer. Dengan positional embedding, model dapat mengetahui bahwa patch ke-1 berada di pojok kiri atas, patch ke-14 berada di baris pertama paling kanan, dan seterusnya. Tanpa positional embedding, model akan kehilangan informasi tentang di mana setiap bagian citra berada secara spasial.

#### Self-Attention

Self-attention adalah mekanisme inti dari arsitektur Transformer. Mekanisme ini memungkinkan setiap token dalam urutan untuk memperhatikan atau memerhatikan token-token lain, termasuk dirinya sendiri, dan mengukur seberapa relevan setiap token lain dalam menentukan representasi dari token tersebut. Secara matematis, self-attention dihitung menggunakan tiga proyeksi linear dari setiap token, yang disebut Query (Q), Key (K), dan Value (V). Skor perhatian dihitung dengan mengalikan Query dari satu token dengan Key dari seluruh token lain, kemudian dinormalisasi menggunakan fungsi softmax. Skor ini kemudian digunakan untuk menghitung rata-rata tertimbang dari Value seluruh token.

Dalam konteks citra MRI otak, self-attention berarti bahwa ketika model memproses patch yang menampilkan area tertentu dari tumor, ia secara bersamaan dapat mempertimbangkan patch-patch lain yang berada jauh di area berbeda dalam gambar, seperti lokasi ventrikel otak atau struktur anatomi lain yang menjadi referensi. Kemampuan ini sangat penting karena hubungan spasial global semacam ini tidak dapat ditangkap oleh CNN dengan receptive field lokalnya.

#### Multi-Head Self-Attention

Multi-head self-attention adalah perluasan dari mekanisme self-attention standar. Alih-alih menghitung satu mekanisme attention, multi-head self-attention menjalankan beberapa mekanisme attention secara paralel, masing-masing disebut satu head. Setiap head mempelajari pola perhatian yang berbeda. Setelah semua head selesai menghitung perhatiannya masing-masing, hasilnya digabungkan dan diproyeksikan kembali ke dalam dimensi asal. Dengan cara ini, model dapat secara bersamaan mempelajari berbagai jenis hubungan antar patch dari sudut pandang yang berbeda-beda. Satu head mungkin berfokus pada hubungan tekstural, sementara head lain berfokus pada hubungan posisional atau struktural.

#### Encoder Transformer

Encoder Transformer adalah unit pemrosesan utama dalam ViT. Setiap blok encoder terdiri dari dua sub-komponen utama: lapisan Multi-Head Self-Attention dan lapisan Multilayer Perceptron (MLP). Di antara keduanya, terdapat layer normalization dan residual connection yang membantu stabilisasi pelatihan. Dalam ViT varian base (ViT-b), terdapat 12 blok encoder yang disusun secara berurutan. Dalam varian large (ViT-l), terdapat 24 blok encoder. Semakin banyak blok encoder, semakin dalam model tersebut dan semakin kompleks representasi yang dapat dipelajarinya.

#### Multilayer Perceptron (MLP)

MLP dalam konteks ViT adalah sebuah jaringan saraf dengan beberapa lapisan linear yang ditempatkan setelah lapisan multi-head self-attention dalam setiap blok encoder. Fungsinya adalah untuk mentransformasi representasi yang dihasilkan oleh attention layer secara non-linear, menambah kapasitas representasi model. MLP dalam ViT biasanya terdiri dari dua lapisan linear dengan fungsi aktivasi GELU di antaranya.

#### Empat Varian ViT yang Digunakan

Jurnal ini menggunakan empat varian ViT pretrained dari ImageNet21k:

- **ViT-b16:** 12 blok encoder, ukuran patch 16x16 piksel, dimensi embedding 768.
- **ViT-b32:** 12 blok encoder, ukuran patch 32x32 piksel, dimensi embedding 768.
- **ViT-l16:** 24 blok encoder, ukuran patch 16x16 piksel, dimensi embedding 1024.
- **ViT-l32:** 24 blok encoder, ukuran patch 32x32 piksel, dimensi embedding 1024.

Huruf "b" merujuk pada "base" dan "l" merujuk pada "large". Angka 16 dan 32 merujuk pada ukuran patch dalam piksel. Varian large memiliki lebih banyak blok encoder dan dimensi yang lebih besar, sehingga memiliki kapasitas representasi lebih tinggi tetapi juga membutuhkan sumber daya komputasi yang lebih besar.

#### Custom Classifier Head (Fine-Tuned Vision Transformer)

Modifikasi utama yang dilakukan peneliti adalah menggantikan kepala classifier standar yang ada pada ViT pretrained dengan sebuah custom classifier head yang lebih kompleks. Kepala classifier adalah bagian akhir dari jaringan yang mengambil representasi dari class token dan menggunakannya untuk menghasilkan prediksi kelas. Alur custom classifier head yang digunakan adalah sebagai berikut:

Pertama, **Batch Normalization (BN)** diterapkan. Batch Normalization adalah teknik normalisasi yang menormalkan aktivasi dalam setiap batch mini selama pelatihan. Ia menghitung rata-rata dan standar deviasi dari aktivasi dalam satu batch, kemudian menggunakan nilai tersebut untuk menormalisasi setiap aktivasi. Tujuannya adalah mempercepat konvergensi pelatihan, mengurangi sensitivitas terhadap inisialisasi bobot, dan berfungsi sebagai regularisasi ringan.

Kedua, **Linear Layer pertama** mereduksi dimensi fitur dari 768 atau 1024 (tergantung varian) menjadi 512. Lapisan ini adalah lapisan fully connected standar yang melakukan transformasi linear dari ruang fitur berdimensi tinggi ke dimensi yang lebih kecil.

Ketiga, **Fungsi Aktivasi ReLU** diterapkan. ReLU, singkatan dari Rectified Linear Unit, adalah fungsi aktivasi yang bekerja berdasarkan rumus f(x) = max(0, x). Artinya, nilai negatif diubah menjadi nol, sementara nilai positif dibiarkan tidak berubah. Fungsi ini memperkenalkan non-linearitas ke dalam jaringan, yang sangat penting karena tanpanya, seluruh lapisan linear yang disusun secara berurutan akan setara dengan hanya satu lapisan linear saja. Non-linearitas inilah yang memungkinkan jaringan saraf tiruan untuk memodelkan hubungan yang kompleks dan tidak linear.

Keempat, **Dropout dengan rate 0.5** diterapkan. Dropout adalah teknik regularisasi yang bekerja dengan cara menonaktifkan secara acak sejumlah neuron dalam jaringan selama proses pelatihan dengan probabilitas tertentu, dalam hal ini 50%. Neuron yang dinonaktifkan tidak berkontribusi pada forward pass maupun backward pass selama satu iterasi pelatihan. Tujuan Dropout adalah mencegah overfitting, yaitu kondisi di mana model terlalu menghafal data pelatihan dan gagal menggeneralisasi ke data baru. Dengan Dropout, model dipaksa untuk tidak bergantung pada neuron-neuron tertentu saja, melainkan mendistribusikan representasi secara lebih merata.

Kelima, **Linear Layer kedua** mereduksi dimensi dari 512 ke 256, diikuti lagi oleh ReLU dan Dropout sebagai regularisasi tambahan.

Keenam, **Linear Output Layer** sebagai lapisan terakhir yang memetakan 256 dimensi ke 4 output, sesuai dengan empat kelas tumor dalam dataset.

---

### 6. Fine-Tuning dan Training

#### Apa Itu Fine-Tuning dan Pretrained Model

Fine-tuning adalah proses melatih ulang model yang sebelumnya telah dilatih pada dataset besar untuk tugas baru yang lebih spesifik. Model yang telah dilatih sebelumnya disebut pretrained model. Dalam penelitian ini, ViT pretrained yang digunakan dilatih pada ImageNet21k, sebuah dataset besar yang berisi lebih dari 14 juta gambar dari 21.000 kategori.

Alasan penggunaan pretrained model adalah karena melatih model dari nol membutuhkan jumlah data yang sangat besar dan waktu komputasi yang sangat panjang. Dengan fine-tuning, bobot jaringan yang telah mempelajari representasi umum dari jutaan gambar dapat disesuaikan untuk tugas baru, dalam hal ini klasifikasi MRI, dengan menggunakan data yang jauh lebih sedikit dan waktu yang jauh lebih singkat. Representasi visual umum yang dipelajari dari ImageNet, seperti tepian, tekstur, dan pola kompleks, juga relevan untuk analisis citra MRI.

#### Optimizer Adam

Optimizer adalah algoritma yang digunakan untuk memperbarui bobot model berdasarkan gradien yang dihitung melalui backpropagation. Adam, singkatan dari Adaptive Moment Estimation, adalah salah satu optimizer paling populer dalam deep learning. Adam bekerja dengan cara menghitung estimasi rata-rata bergerak dari gradien (momen pertama) dan rata-rata bergerak dari kuadrat gradien (momen kedua) untuk setiap parameter secara adaptif. Ini membuat Adam sangat efektif karena laju pembelajaran efektif untuk setiap parameter disesuaikan secara otomatis berdasarkan riwayat gradiennya.

#### Learning Rate 1e-4

Learning rate atau laju pembelajaran adalah parameter yang menentukan seberapa besar langkah yang diambil optimizer dalam memperbarui bobot model. Nilai 1e-4 atau 0.0001 dipilih karena nilai ini cukup kecil untuk memastikan pembaruan bobot yang halus dan bertahap, menghindari divergensi, tetapi tidak terlalu kecil sehingga konvergensi menjadi terlalu lambat. Dalam proses fine-tuning, learning rate yang lebih kecil dari training from scratch biasanya dipilih untuk menghindari perubahan bobot pretrained yang terlalu drastis.

#### Batch Size 16

Batch size adalah jumlah sampel data yang diproses secara bersamaan dalam satu iterasi pelatihan sebelum bobot diperbarui. Nilai 16 dipilih sebagai keseimbangan antara stabilitas estimasi gradien dan efisiensi penggunaan memori GPU. Batch yang lebih besar memberikan estimasi gradien yang lebih akurat tetapi membutuhkan lebih banyak memori. Batch yang lebih kecil memberikan gradien yang lebih noisy, namun terkadang justru membantu model keluar dari local minima.

#### Epoch 10

Satu epoch berarti model telah melihat seluruh dataset pelatihan satu kali. Pelatihan selama 10 epoch berarti model melewati seluruh data pelatihan sebanyak 10 kali. Dalam jurnal ini, peneliti hanya menggunakan 10 epoch karena keterbatasan sumber daya komputasi. Hal ini menjadi salah satu kelemahan penelitian, karena beberapa varian, terutama ViT-b32, mungkin belum mencapai konvergensi optimal.

#### Cross Entropy Loss

Cross Entropy Loss adalah fungsi kerugian yang digunakan untuk masalah klasifikasi multikelas. Fungsi ini mengukur perbedaan antara distribusi probabilitas yang diprediksi oleh model dan distribusi probabilitas yang sebenarnya (label one-hot). Semakin kecil nilai Cross Entropy Loss, semakin dekat prediksi model dengan label yang benar. Selama pelatihan, optimizer berusaha meminimalkan nilai fungsi kerugian ini melalui backpropagation.

#### Backpropagation

Backpropagation adalah algoritma yang digunakan untuk menghitung gradien dari fungsi kerugian terhadap setiap bobot dalam jaringan. Gradien ini menunjukkan seberapa besar dan ke arah mana setiap bobot perlu diubah untuk mengurangi kesalahan. Algoritma ini bekerja dengan cara menyebarkan sinyal kesalahan dari lapisan output ke lapisan-lapisan sebelumnya secara bertahap menggunakan aturan rantai (chain rule) dalam kalkulus diferensial.

#### GPU T4

Model dilatih menggunakan GPU T4 di platform Kaggle. GPU (Graphics Processing Unit) adalah perangkat keras yang sangat efisien untuk komputasi paralel, sehingga jauh lebih cepat dari CPU dalam menjalankan operasi matriks yang intensif dalam pelatihan deep learning. NVIDIA T4 adalah GPU kelas datacenter dengan 15 GB VRAM yang disediakan secara gratis oleh platform Kaggle untuk eksperimen penelitian. Kebutuhan komputasi yang tinggi ini menjadi salah satu keterbatasan praktis dari penelitian ini.

#### Perbandingan dengan Model CNN

Sebagai pembanding, tiga model CNN juga dilatih pada dataset yang sama menggunakan optimizer SGD (Stochastic Gradient Descent) dengan learning rate 0.001 dan momentum 0.7.

- **ResNet-50:** Jaringan residual dengan 48 lapisan konvolusi dan residual connections yang memungkinkan gradien mengalir langsung dari lapisan awal ke lapisan akhir, mengatasi masalah vanishing gradient pada jaringan yang sangat dalam.
- **EfficientNet-B0:** Menggunakan metode compound scaling yang menyeimbangkan kedalaman, lebar, dan resolusi jaringan secara proporsional, serta memanfaatkan blok Mobile Inverted Bottleneck Convolution (MBConv) yang efisien secara komputasi.
- **MobileNet-V2:** Arsitektur ringan dengan 17 blok bottleneck yang dirancang untuk efisiensi komputasi pada perangkat dengan sumber daya terbatas.

---

### 7. Evaluasi Model

#### Accuracy (Akurasi)

Accuracy mengukur proporsi prediksi yang benar dari seluruh prediksi. Rumusnya adalah (TP + TN) / (TP + TN + FP + FN), di mana TP adalah True Positive, TN adalah True Negative, FP adalah False Positive, dan FN adalah False Negative. Nilai akurasi 98.70% berarti model memprediksi dengan benar sebanyak 98.70% dari seluruh citra yang diuji. Dalam konteks medis, akurasi yang tinggi penting namun tidak cukup jika dievaluasi sendirian karena tidak mempertimbangkan distribusi kesalahan secara rinci.

#### Precision (Presisi)

Precision mengukur proporsi prediksi positif yang benar-benar positif. Rumusnya adalah TP / (TP + FP). Nilai Precision yang tinggi berarti ketika model menyatakan sebuah citra mengandung jenis tumor tertentu, prediksi tersebut hampir selalu benar. Precision penting dalam konteks klinis untuk menghindari alarm palsu yang dapat menyebabkan pasien sehat menjalani prosedur medis yang tidak perlu.

#### Recall (Sensitivitas)

Recall mengukur proporsi kasus positif yang berhasil dideteksi oleh model. Rumusnya adalah TP / (TP + FN). Recall yang tinggi berarti model jarang melewatkan kasus positif yang sebenarnya ada. Dalam konteks tumor otak, Recall yang tinggi sangat kritis karena kegagalan mendeteksi tumor yang ada (False Negative) berpotensi mengancam jiwa pasien.

#### F1-Score

F1-Score adalah rata-rata harmonik dari Precision dan Recall. Rumusnya adalah 2 × (Precision × Recall) / (Precision + Recall). Metrik ini sangat berguna ketika terdapat ketidakseimbangan kelas, karena memberikan gambaran yang seimbang antara kemampuan model menghindari alarm palsu (Precision) dan kemampuannya mendeteksi semua kasus yang ada (Recall).

---

### 8. Hasil Penelitian

Model FTVT secara konsisten mengungguli seluruh model CNN pada semua metrik evaluasi. FTVT-l16 mencapai akurasi tertinggi sebesar 98.70%, diikuti FTVT-l32 sebesar 98.62%, FTVT-b16 sebesar 98.09%, dan FTVT-b32 sebesar 96.87%. Sebagai perbandingan, ResNet-50 hanya mencapai 96.5%, EfficientNet-B0 sebesar 95.1%, dan MobileNet-V2 sebesar 94.9%.

Perbedaan akurasi antara FTVT-l16 (98.70%) dan model CNN terbaik ResNet-50 (96.5%) tampak kecil dalam angka, namun memiliki implikasi klinis yang signifikan. Selisih 2.2% dalam akurasi pada dataset berisi 1.311 citra uji berarti sekitar 29 citra lebih banyak yang diklasifikasikan dengan benar oleh FTVT-l16 dibandingkan ResNet-50. Dalam skenario klinis nyata, setiap citra yang salah diklasifikasikan berpotensi mengganggu keputusan diagnostik.

Analisis per kelas menunjukkan bahwa kelas Meningioma adalah yang paling sulit diklasifikasikan. Pada model ResNet-50, akurasi untuk kelas ini hanya mencapai 86.67%, yang berarti sekitar satu dari tujuh citra Meningioma salah diklasifikasikan. FTVT-l16 berhasil meningkatkan akurasi kelas ini menjadi 98.77%, sebuah peningkatan yang sangat substansial. Ini menunjukkan bahwa mekanisme self-attention ViT mampu menangkap pola visual Meningioma yang tidak dapat ditangkap oleh receptive field lokal CNN.

Perbandingan antara varian patch kecil (b16, l16) dan patch besar (b32, l32) menunjukkan bahwa patch yang lebih kecil menghasilkan akurasi yang lebih tinggi. Hal ini masuk akal karena patch 16x16 menghasilkan lebih banyak token (196 token) dibandingkan patch 32x32 (49 token), sehingga model dapat menangkap detail visual yang lebih halus dan granular dalam citra MRI.

---

### 9. Analisis Kekuatan Penelitian

Kekuatan pertama penelitian ini adalah inovasi pada arsitektur classifier head. Dengan menambahkan kombinasi Batch Normalization, Dropout berlapis, dan lapisan Linear yang lebih dalam, peneliti berhasil mengoptimalkan ViT pretrained untuk tugas spesifik klasifikasi tumor otak, menghasilkan akurasi yang melampaui studi-studi sebelumnya dalam domain yang sama.

Kekuatan kedua adalah analisis komparatif yang komprehensif. Penelitian ini tidak hanya mengusulkan satu model, tetapi membandingkan tujuh model sekaligus, yaitu empat varian FTVT dan tiga model CNN, dengan evaluasi per kelas dan per metrik. Pendekatan ini memberikan gambaran yang lebih lengkap dan lebih dapat dipercaya tentang performa relatif dari setiap arsitektur.

Kekuatan ketiga adalah pembuktian keunggulan self-attention dalam analisis citra medis. Kemampuan multi-head self-attention untuk menangkap dependensi global terbukti secara kuantitatif memberikan keunggulan nyata dibandingkan CNN dalam menangani kelas yang paling menantang, yaitu Meningioma.

Kekuatan keempat adalah keterbukaan dan reproduksibilitas. Penggunaan dataset publik dari Kaggle dan library open-source seperti PyTorch memungkinkan peneliti lain untuk mereproduksi, memverifikasi, dan mengembangkan penelitian ini.

---

### 10. Analisis Kelemahan Penelitian

Kelemahan utama adalah kebutuhan komputasi yang sangat tinggi. Model FTVT varian large membutuhkan GPU T4 dengan 15 GB VRAM dan RAM hingga 29 GB. Persyaratan ini menjadi hambatan serius untuk implementasi di fasilitas kesehatan yang tidak memiliki infrastruktur komputasi canggih, khususnya di negara berkembang.

Kelemahan kedua adalah cakupan dataset yang terbatas. Hanya tiga jenis tumor yang direpresentasikan dalam dataset ini. Dalam praktik klinis, terdapat lebih dari dua puluh jenis tumor otak yang berbeda, termasuk craniopharyngioma, ependymoma, dan metastasis dari tumor primer di organ lain. Model yang tidak pernah melihat jenis tumor tersebut tidak dapat diandalkan untuk mendiagnosisnya.

Kelemahan ketiga adalah tidak adanya validasi klinis. Seluruh evaluasi dilakukan secara in-silico menggunakan dataset yang sudah bersih dan berlabel. Belum ada pengujian dalam kondisi klinis nyata di mana kualitas citra MRI sangat bervariasi, artefak gerakan lebih sering terjadi, dan variasi demografis pasien jauh lebih luas.

Kelemahan keempat adalah jumlah epoch pelatihan yang terbatas, yaitu hanya 10 epoch. Untuk model yang kompleks seperti ViT-large, 10 epoch mungkin tidak cukup untuk mencapai konvergensi yang optimal. Hal ini tercermin dari akurasi ViT-b32 yang lebih rendah dibandingkan varian lain, yang kemungkinan besar disebabkan oleh belum optimalnya proses pembelajaran.

---

### 11. Kesimpulan Jurnal Pertama

Jurnal pertama ini memberikan kontribusi ilmiah yang signifikan dengan membuktikan keunggulan arsitektur Vision Transformer yang telah di-fine-tune secara spesifik (FTVT) atas model-model CNN konvensional dalam klasifikasi tumor otak multiclass berbasis citra MRI. Keberhasilan FTVT-l16 mencapai akurasi 98.70% dengan peningkatan drastis pada kelas Meningioma yang paling sulit membuktikan bahwa mekanisme self-attention global adalah pendekatan yang lebih tepat untuk tugas analisis citra medis kompleks dibandingkan pendekatan konvolusi lokal. Penelitian ini membuka jalan bagi pengembangan sistem diagnosis tumor otak yang lebih akurat dan dapat diandalkan, meskipun masih membutuhkan kerja lebih lanjut untuk mengatasi keterbatasan komputasi, memperluas cakupan jenis tumor, dan melakukan validasi klinis.

---

# BAGIAN II: JURNAL KEDUA

## Segmentasi dan Klasifikasi Kanker Kulit Berbasis Citra Dermoskopi Menggunakan Vision Transformer

---

### 1. Identitas Jurnal

**Judul:** Skin Cancer Segmentation and Classification Using Vision Transformer for Automatic Analysis in Dermatoscopy-Based Noninvasive Digital System

**Jurnal Penerbit:** International Journal of Biomedical Imaging, diterbitkan oleh Hindawi/Wiley

**Volume dan Halaman:** Volume 2024, Article ID 3022192, 17 halaman

**Tahun Terbit:** 2024

**Penulis:** Galib Muhammad Shahriar Himel, Md Masudul Islam, Kh Abdullah Al-Aff, Shams Ibne Karim, Md Kabir Uddin Sikder

Jurnal ini bergerak di bidang dermatologi komputasional, yaitu penerapan kecerdasan buatan untuk analisis penyakit kulit melalui citra digital. Fokus penelitian adalah membangun sistem dua tahap yang sepenuhnya otomatis untuk mendeteksi dan mengklasifikasikan kanker kulit dari citra dermoskopi.

---

### 2. Latar Belakang Masalah

#### Kanker Kulit dan Urgensi Deteksi Dini

Kanker kulit adalah salah satu jenis kanker dengan insidensi tertinggi di seluruh dunia. Melanoma, jenis kanker kulit yang paling ganas, memiliki angka kematian yang tinggi jika tidak terdeteksi secara dini. Namun, jika ditemukan pada stadium awal, tingkat kesembuhan melanoma dapat mencapai lebih dari 90%. Oleh karena itu, sistem skrining yang cepat, murah, dan akurat menjadi sangat penting, terutama di negara-negara dengan rasio dermatologis terhadap penduduk yang rendah.

#### Dermoskopi sebagai Modalitas Pencitraan

Dermoskopi adalah teknik pencitraan noninvasif yang menggunakan alat bernama dermatoskop, yaitu sebuah kaca pembesar yang dilengkapi dengan sistem pencahayaan khusus. Dermatoskop memungkinkan dokter untuk melihat struktur kulit di bawah permukaan yang tidak terlihat dengan mata telanjang, seperti pola pembuluh darah, pigmentasi, dan tekstur lesi. Citra yang dihasilkan oleh dermatoskop disebut citra dermoskopi.

Dibandingkan dengan foto kulit biasa, citra dermoskopi mengandung informasi visual yang jauh lebih kaya dan lebih relevan secara klinis untuk membedakan lesi jinak dari lesi ganas. Namun, interpretasi citra dermoskopi membutuhkan keahlian khusus dan pengalaman bertahun-tahun. Sistem berbasis Computer Vision yang dapat menganalisis citra dermoskopi secara otomatis berpotensi membantu dokter umum yang tidak memiliki spesialisasi dermatologi untuk melakukan skrining awal yang akurat.

#### Pentingnya Segmentasi Sebelum Klasifikasi

Dalam citra dermoskopi, area lesi yang mencurigakan biasanya dikelilingi oleh kulit sehat. Jika model klasifikasi dilatih langsung pada seluruh citra tanpa memisahkan area lesi terlebih dahulu, model berpotensi terdistraksi oleh fitur-fitur yang tidak relevan dari kulit sehat, rambut, atau artefak pencahayaan di sekitar lesi. Segmentasi, yaitu proses mengidentifikasi dan mengisolasi area lesi dari latar belakangnya, dilakukan sebagai langkah pertama untuk memastikan bahwa model klasifikasi hanya berfokus pada informasi yang relevan secara klinis.

---

### 3. Dataset

Dataset yang digunakan adalah **HAM10000 (Human Against Machine with 10000 training images)**, sebuah dataset benchmark yang sangat terkenal dalam komunitas riset Computer Vision medis. Dataset ini terdiri dari 10.015 citra lesi kulit dermoskopi beresolusi tinggi yang telah dianotasi oleh dermatologis ahli. Dataset ini tersedia secara publik di Kaggle dengan nama Skin Cancer MNIST: HAM10000.

#### Kelas dalam HAM10000

Dataset ini aslinya mengandung tujuh kategori lesi kulit yang berbeda, namun dalam penelitian ini, ketujuh kategori tersebut dikonsolidasikan menjadi dua kelas utama untuk tugas klasifikasi biner:

- **Benign (Jinak):** Lesi yang tidak bersifat kanker, sebanyak 6.705 citra. Kelompok ini mencakup lesi-lesi seperti melanocytic nevi (tahi lalat biasa) dan lesi jinak lainnya.
- **Malignant (Ganas):** Lesi yang bersifat kanker, sebanyak 3.310 citra. Kelompok ini mencakup melanoma, karsinoma sel basal, dan jenis kanker kulit lainnya.

#### Ketidakseimbangan Kelas

Terdapat ketidakseimbangan kelas yang signifikan antara benign (6.705 citra) dan malignant (3.310 citra), dengan rasio sekitar 2:1. Ketidakseimbangan ini berpotensi menyebabkan model lebih condong memprediksi kelas mayoritas (benign) karena secara statistis lebih sering muncul. Dalam konteks klinis, bias ini sangat berbahaya karena dapat meningkatkan jumlah kasus malignant yang tidak terdeteksi.

#### Lesi Kulit: Terminologi Medis

- **Melanoma:** Jenis kanker kulit paling mematikan yang berasal dari melanosit, yaitu sel yang memproduksi pigmen kulit. Melanoma dapat menyebar ke organ lain dengan cepat jika tidak segera ditangani.
- **Karsinoma Sel Basal (Basal Cell Carcinoma):** Jenis kanker kulit yang paling umum, berasal dari sel basal di lapisan terbawah epidermis. Meskipun jarang menyebar ke organ lain, ia dapat menyebabkan kerusakan jaringan lokal yang signifikan.
- **Melanocytic Nevi:** Lebih dikenal sebagai tahi lalat. Merupakan pertumbuhan kulit jinak yang terdiri dari melanosit yang mengelompok. Dalam beberapa kasus, tahi lalat dapat berubah menjadi melanoma, sehingga pemantauan regularnya penting.

---

### 4. Preprocessing dan Augmentasi Data

#### Normalisasi Nilai Piksel

Nilai piksel pada citra dermoskopi dinormalisasi untuk menstabilkan proses pelatihan dan mengurangi sensitivitas model terhadap variasi pencahayaan antar citra. Normalisasi memastikan bahwa semua input memiliki skala nilai yang konsisten, sehingga optimizer dapat bergerak lebih efisien selama proses pembelajaran.

#### Augmentasi untuk Mengatasi Ketidakseimbangan Kelas

Augmentasi dalam penelitian ini memiliki dua tujuan sekaligus: meningkatkan keberagaman data latih secara umum, dan secara khusus mengatasi ketidakseimbangan kelas antara benign dan malignant. Teknik yang digunakan meliputi flip horizontal, flip vertikal, rotasi, dan perubahan kecerahan. Augmentasi dilakukan secara online selama proses pelatihan, artinya transformasi diterapkan secara acak pada setiap batch, sehingga setiap epoch model melihat variasi yang berbeda dari citra yang sama.

Flip vertikal juga digunakan, berbeda dari jurnal pertama yang hanya menggunakan flip horizontal. Hal ini relevan untuk citra dermoskopi karena orientasi lesi dapat sangat bervariasi tergantung lokasi anatomis pada tubuh pasien.

---

### 5. Arsitektur Model

#### Segment Anything Model (SAM) untuk Segmentasi

SAM, atau Segment Anything Model, adalah model segmentasi universal yang dikembangkan oleh Meta AI dan dirilis pada tahun 2023. SAM dirancang untuk dapat melakukan segmentasi pada jenis gambar apa pun dengan menggunakan prompt yang fleksibel, seperti titik, kotak pembatas, atau teks. Ini adalah model yang sangat baru dan tergolong state-of-the-art pada saat penelitian dilakukan.

Dalam penelitian ini, SAM digunakan untuk tahap pertama dari pipeline, yaitu segmentasi area lesi. Prosesnya dimulai dengan anotasi manual pada area lesi yang mencurigakan sebagai prompt awal untuk SAM. Setelah menerima prompt ini, SAM menghasilkan segmentasi otomatis yang memisahkan area lesi dari kulit sehat di sekitarnya. Pendekatan ini memanfaatkan kekuatan generalisasi SAM yang luar biasa untuk menghasilkan segmentasi yang sangat akurat bahkan dengan input minimal.

Penggunaan SAM dalam konteks medis ini tergolong inovatif karena model ini awalnya tidak dirancang secara khusus untuk citra medis. Kemampuannya untuk digeneralisasi ke domain baru dengan minimal fine-tuning adalah salah satu keunggulan utamanya.

#### Enam Varian Vision Transformer untuk Klasifikasi

Setelah segmentasi, area lesi yang telah terisolasi dimasukkan ke dalam model klasifikasi berbasis ViT. Enam varian ViT dievaluasi:

- **ViT-Google (patch-32):** Model ViT standar yang dikembangkan oleh Google dengan ukuran patch 32x32 piksel. Ini adalah implementasi ViT yang relatif sederhana namun terbukti efektif dalam berbagai tugas klasifikasi citra.
- **ViT-MAE (Masked Autoencoder):** Varian ViT dengan strategi pretraining Masked Autoencoder yang dikembangkan oleh Facebook. Dalam pretraining MAE, sebagian besar patch citra disembunyikan (masked) dan model dilatih untuk merekonstruksi patch-patch yang disembunyikan tersebut. Pendekatan ini mendorong model untuk mempelajari representasi visual yang lebih dalam dan komprehensif.
- **ViT-ResNet50 (Hybrid Backbone):** Merupakan arsitektur hybrid yang mengintegrasikan ResNet-50 sebagai backbone untuk ekstraksi fitur awal, kemudian dilanjutkan dengan Transformer untuk pemrosesan sekuensial. Pendekatan hybrid ini menggabungkan kekuatan CNN dalam ekstraksi fitur lokal dengan kekuatan Transformer dalam menangkap hubungan global.
- **ViT-VAN (Visual Attention Network):** Menggabungkan mekanisme attention konvolusi, yang berbeda dari attention standar berbasis dot-product dalam ViT murni. VAN menggunakan deformable convolution untuk mencapai attention yang lebih adaptif terhadap bentuk dan ukuran objek.
- **ViT-BEiT (BERT Pretraining of Image Transformers):** Dikembangkan oleh Microsoft, BEiT mengadopsi strategi pretraining yang terinspirasi dari BERT dalam NLP. Dalam pretraining BEiT, patch citra direpresentasikan sebagai visual token dan model dilatih untuk memprediksi token visual yang disembunyikan. Pendekatan ini menghasilkan representasi yang lebih semantik.
- **ViT-DiT (Document Image Transformer):** Awalnya dirancang untuk analisis dokumen, namun diadaptasi dalam penelitian ini untuk citra medis. Ini adalah varian yang paling tidak umum dalam konteks medis, dan hasilnya yang paling rendah di antara keenam model mengonfirmasi bahwa arsitektur yang dirancang untuk dokumen tidak optimal untuk citra dermoskopi.

#### Optimasi Hyperparameter

Penelitian ini melakukan eksplorasi hyperparameter yang cukup menyeluruh. Empat optimizer diuji: Adam, Adamax, RMSProp, dan SGD. Lima nilai learning rate dievaluasi mulai dari 0.1 hingga 0.00001. Dua loss function dibandingkan: Cross Entropy Loss dan Hinge Loss. Kombinasi terbaik yang ditemukan adalah Adam dengan learning rate 0.00002 dan Cross Entropy Loss.

Nilai learning rate 0.00002 yang sangat kecil mencerminkan kebutuhan untuk pembaruan bobot yang sangat halus dalam proses fine-tuning ViT pretrained pada domain citra medis, di mana perubahan yang terlalu besar dapat merusak representasi yang telah dipelajari selama pretraining.

---

### 6. Evaluasi Model

#### Intersection over Union (IoU)

IoU adalah metrik evaluasi yang digunakan untuk mengukur kualitas segmentasi. IoU dihitung sebagai perbandingan antara luas area irisan (intersection) antara segmentasi yang diprediksi dan segmentasi ground truth (yang benar), dibagi dengan luas area gabungan (union) keduanya. Nilai IoU mendekati 1 berarti segmentasi yang diprediksi hampir sempurna tumpang tindih dengan segmentasi yang sebenarnya. Nilai IoU 96.01% yang dicapai SAM dalam penelitian ini menunjukkan kualitas segmentasi yang sangat tinggi.

#### Dice Coefficient

Dice Coefficient adalah metrik evaluasi lain untuk segmentasi yang mengukur kesamaan antara dua set, dalam hal ini mask segmentasi prediksi dan ground truth. Rumusnya adalah 2 × |A ∩ B| / (|A| + |B|), di mana A dan B adalah mask prediksi dan mask ground truth. Dice Coefficient memberikan bobot yang lebih besar pada irisan dibandingkan IoU, sehingga lebih sensitif terhadap overlapping yang akurat. Nilai 98.14% menunjukkan segmentasi yang hampir sempurna.

#### False Negative Ratio

False Negative Ratio adalah proporsi kasus positif (malignant) yang salah diklasifikasikan sebagai negatif (benign) oleh model. Metrik ini sangat kritis dalam konteks kanker karena kasus kanker yang tidak terdeteksi (false negative) jauh lebih berbahaya daripada alarm palsu (false positive). Seorang pasien dengan kanker yang tidak terdeteksi tidak akan mendapatkan penanganan yang dibutuhkan, sementara pasien sehat yang salah didiagnosis hanya akan menjalani pemeriksaan lanjutan yang tidak perlu. Oleh karena itu, penelitian ini secara khusus menekankan pentingnya meminimalkan false negative ratio sebagai prioritas utama evaluasi klinis.

---

### 7. Hasil Penelitian

Tahap segmentasi menggunakan SAM menghasilkan IoU sebesar 96.01% dan Dice Coefficient sebesar 98.14%. Hasil ini menunjukkan bahwa SAM, meskipun awalnya bukan model yang dirancang khusus untuk citra medis, mampu melakukan segmentasi lesi kulit dengan kualitas yang sangat tinggi ketika diberikan prompt manual yang tepat.

Pada tahap klasifikasi, ViT-Google dengan patch-32 tampil sebagai model terbaik dengan akurasi 96.15% dan false negative ratio yang rendah. ViT-BEiT dan ViT-MAE menunjukkan performa kompetitif, sementara ViT-DiT memberikan hasil terendah. Hasil ini memperkuat pandangan bahwa pretraining yang tepat dan arsitektur yang sesuai dengan karakteristik domain sangat menentukan performa akhir model.

Pipeline dua tahap yang menggabungkan segmentasi SAM dan klasifikasi ViT terbukti lebih efektif dibandingkan pendekatan klasifikasi langsung tanpa segmentasi. Ini mengonfirmasi hipotesis awal bahwa isolasi area lesi dari latar belakang yang tidak relevan memang meningkatkan kemampuan model untuk mengekstraksi fitur yang bermakna secara klinis.

---

### 8. Kekuatan Penelitian

Kekuatan utama penelitian ini adalah inovasi pipeline dua tahap yang mengintegrasikan SAM dan ViT secara seamless. Dengan memisahkan tugas segmentasi dan klasifikasi ke dalam dua komponen yang masing-masing dioptimalkan, penelitian ini mencapai performa yang lebih baik daripada sistem end-to-end tunggal.

Penggunaan SAM yang merupakan model yang sangat baru dan mutakhir pada saat penelitian dilakukan menjadikan penelitian ini sebagai salah satu yang pertama mengadopsi SAM untuk citra medis dermoskopi. Ini merupakan kontribusi ilmiah yang penting karena membuka kemungkinan penggunaan model-model segmentasi universal untuk berbagai domain citra medis lainnya.

Orientasi klinis yang kuat, khususnya penekanan pada false negative ratio sebagai metrik utama, mencerminkan pemahaman mendalam tentang kebutuhan klinis yang sesungguhnya. Penelitian yang baik tidak hanya mengejar akurasi agregat, tetapi memahami konsekuensi asimetris dari jenis kesalahan yang berbeda dalam konteks medis.

---

### 9. Kelemahan Penelitian

Kelemahan pertama adalah klasifikasi biner yang terlalu sederhana dari perspektif klinis. Kondensasi tujuh kategori lesi menjadi dua kategori (benign/malignant) kehilangan banyak informasi klinis yang berharga. Dokter membutuhkan diagnosis yang lebih spesifik, misalnya apakah lesi tersebut melanoma, karsinoma sel basal, atau jenis kanker lain, karena penanganannya berbeda.

Kelemahan kedua adalah ketergantungan pada anotasi manual untuk SAM. Meskipun SAM mampu menghasilkan segmentasi otomatis setelah menerima prompt, prompt itu sendiri masih membutuhkan input manual dari operator. Ini mengurangi tingkat otomatisasi sistem secara keseluruhan dan tetap membutuhkan keterlibatan manusia yang terlatih.

Kelemahan ketiga adalah penanganan ketidakseimbangan kelas yang tidak dibahas secara mendalam. Meskipun augmentasi diterapkan, strategi spesifik untuk mengatasi bias akibat rasio 2:1 antara benign dan malignant tidak dijelaskan secara rinci. Ini dapat memengaruhi kepercayaan terhadap validitas metrik evaluasi yang dilaporkan.

---

### 10. Kesimpulan Jurnal Kedua

Jurnal kedua ini memberikan kontribusi signifikan dengan memperkenalkan pipeline dua tahap yang inovatif untuk deteksi kanker kulit, mengintegrasikan SAM yang mutakhir dengan berbagai varian ViT. Penelitian ini membuktikan bahwa segmentasi yang akurat sebelum klasifikasi meningkatkan performa sistem secara keseluruhan, dan bahwa model segmentasi universal seperti SAM dapat diadaptasi dengan efektif untuk domain citra medis. Meskipun masih memiliki keterbatasan dalam granularitas klasifikasi dan otomatisasi segmentasi, penelitian ini membuka arah yang sangat menjanjikan untuk pengembangan sistem skrining kanker kulit berbasis teknologi digital yang dapat diimplementasikan secara luas.

---

# BAGIAN III: JURNAL KETIGA

## Klasifikasi Subtipe Diabetic Retinopathy Menggunakan Hybrid Neural Network EfficientNet dan Swin Transformer

---

### 1. Identitas Jurnal

**Judul:** A Hybrid Neural Network Approach for Classifying Diabetic Retinopathy Subtypes

**Jurnal Penerbit:** Frontiers in Medicine, diterbitkan oleh Frontiers Media SA

**Volume dan Halaman:** Volume 10, Article 1293019, 12 halaman

**Tahun Terbit:** 2024

**Penulis:** Huanqing Xu, Xian Shao, Dandan Fang, Fangliang Huang

Penelitian ini berada di persimpangan antara oftalmologi komputasional dan kecerdasan buatan, berfokus pada deteksi otomatis komplikasi diabetes pada mata menggunakan citra retina. Kontribusi utamanya adalah arsitektur hybrid yang menggabungkan dua pendekatan ekstraksi fitur yang saling melengkapi, serta penambahan mekanisme interpretabilitas berbasis visualisasi untuk mendukung adopsi klinis.

---

### 2. Latar Belakang Masalah

#### Diabetes dan Komplikasi pada Mata

Diabetes mellitus adalah penyakit metabolik kronis yang ditandai dengan kadar gula darah tinggi akibat gangguan produksi atau fungsi insulin. Salah satu komplikasi paling serius dari diabetes adalah **Diabetic Retinopathy (DR)**, yaitu kerusakan pada pembuluh darah di retina, lapisan jaringan sensitif cahaya yang melapisi bagian belakang bola mata. DR adalah penyebab utama kebutaan yang dapat dicegah pada usia produktif di seluruh dunia.

DR berkembang secara bertahap melalui beberapa stadium, mulai dari tidak ada tanda-tanda (Grade 0) hingga kondisi yang sangat parah yang disebut Proliferative DR (Grade 4) di mana pembuluh darah baru yang abnormal tumbuh di retina dan dapat menyebabkan perdarahan serta pelepasan retina. Deteksi dini pada stadium awal memungkinkan intervensi medis yang dapat menghentikan atau memperlambat perkembangan penyakit secara signifikan.

#### Citra Fundus Retina

Citra fundus retina adalah foto dari bagian dalam bola mata yang diambil menggunakan kamera fundus khusus. Kamera ini diarahkan melalui pupil untuk mengambil gambar retina, termasuk pembuluh darah, makula, dan nervus optikus. Dalam citra fundus, tanda-tanda DR yang dapat diamati meliputi microaneurysm (perluasan kecil pada pembuluh darah kapiler), hemorrhage (perdarahan), hard exudate (endapan lemak), cotton wool spots, dan neovascularization (pertumbuhan pembuluh darah abnormal).

Interpretasi citra fundus retina untuk mendiagnosis DR secara manual oleh dokter spesialis mata membutuhkan waktu dan keahlian khusus. Di banyak daerah, skrining DR tidak dapat dilakukan secara rutin karena keterbatasan jumlah dokter spesialis. Sistem otomatis berbasis Computer Vision menawarkan solusi yang dapat menjangkau populasi yang lebih luas dengan biaya yang lebih rendah.

#### Mengapa CNN Murni Tidak Cukup dan Transformer Murni Juga Tidak Ideal

CNN unggul dalam menangkap fitur-fitur lokal seperti microaneurysm yang kecil dan spesifik lokasinya dalam citra fundus. Namun, CNN tidak efisien dalam memahami hubungan spasial global, seperti distribusi area lesi di seluruh retina atau perubahan pola vaskular secara keseluruhan yang juga merupakan indikator penting DR.

Transformer murni, di sisi lain, unggul dalam menangkap hubungan global melalui mekanisme self-attention, tetapi kurang efisien dalam menangkap detail lokal yang halus. Selain itu, Transformer murni membutuhkan data yang sangat besar untuk dapat dilatih secara efektif karena tidak memiliki inductive bias yang dimiliki CNN, seperti translational invariance dan locality.

Pendekatan hybrid yang menggabungkan keduanya, dengan CNN menangani ekstraksi fitur lokal dan Transformer menangani pemahaman konteks global, secara teoritis dapat menghasilkan representasi yang lebih komprehensif dibandingkan menggunakan salah satu saja secara terpisah.

---

### 3. Dataset

Dataset yang digunakan adalah **APTOS 2019 Blindness Detection**, sebuah dataset yang tersedia dari kompetisi Kaggle yang disponsori oleh Asia Pacific Tele-Ophthalmology Society (APTOS). Dataset ini terdiri dari 3.662 citra fundus retina berlabel yang dikumpulkan dari layanan skrining DR di berbagai klinik di India.

#### Lima Grade DR

| Grade | Tingkat Keparahan | Deskripsi Klinis |
|-------|-------------------|------------------|
| Grade 0 | No DR | Tidak ada tanda DR pada retina |
| Grade 1 | Mild DR | Terdapat microaneurysm saja |
| Grade 2 | Moderate DR | Terdapat microaneurysm, hemorrhage, dan hard exudate |
| Grade 3 | Severe DR | Terdapat hemorrhage di empat kuadran retina, venous beading |
| Grade 4 | Proliferative DR | Terdapat neovascularization dan/atau perdarahan vitreus |

#### Ketidakseimbangan Distribusi Kelas

Distribusi kelas dalam dataset APTOS 2019 sangat tidak seimbang. Mayoritas citra berasal dari pasien tanpa DR (Grade 0), sementara Grade 3 dan Grade 4 yang merupakan kondisi paling parah memiliki jumlah sampel paling sedikit. Ketidakseimbangan ini mencerminkan kondisi epidemiologis nyata di mana sebagian besar pasien yang menjalani skrining adalah pasien awal yang belum mengalami komplikasi parah. Namun, dari perspektif pembelajaran mesin, ketidakseimbangan ini berisiko membuat model lebih baik dalam mengenali kelas mayoritas tetapi kurang akurat untuk kelas minoritas yang justru paling kritis secara klinis.

#### Variasi Kualitas Citra

Karena dikumpulkan dari berbagai klinik dengan peralatan dan kondisi yang berbeda, dataset APTOS 2019 mengandung variasi kualitas citra yang signifikan. Beberapa citra memiliki kecerahan yang rendah, fokus yang kurang tajam, atau artefak dari proses pengambilan gambar. Variasi ini membuat proses pembelajaran lebih menantang tetapi juga membuat model yang berhasil dilatih di atas dataset ini lebih relevan untuk kondisi dunia nyata.

---

### 4. Preprocessing dan Augmentasi Data

#### Gaussian Blur

Gaussian Blur adalah filter penghalusan yang diterapkan pada citra dengan menggunakan kernel Gaussian, yaitu matriks bobot yang memiliki distribusi Gaussian. Setiap piksel dalam citra hasil penerapan Gaussian Blur dihitung sebagai rata-rata tertimbang dari piksel-piksel di sekitarnya, dengan bobot yang menurun seiring bertambahnya jarak dari piksel pusat.

Dalam konteks citra fundus retina, Gaussian Blur diterapkan untuk dua tujuan utama. Pertama, mengurangi noise digital yang tidak relevan yang dapat mengganggu proses pembelajaran. Kedua, menghaluskan tekstur halus yang tidak relevan sehingga model lebih berfokus pada pola-pola lesi yang lebih besar dan lebih signifikan secara klinis. Penggunaan Gaussian Blur dalam preprocessing citra fundus adalah langkah yang cukup spesifik dan berbeda dari dua jurnal lainnya, mencerminkan karakteristik khusus citra fundus yang sering memiliki noise tekstural tinggi.

#### Augmentasi

Teknik augmentasi yang digunakan meliputi random flip, rotasi, dan perubahan kontras. Augmentasi ini diterapkan untuk dua tujuan: meningkatkan keberagaman data latih dan mengatasi ketidakseimbangan distribusi antar grade DR. Perubahan kontras secara khusus relevan untuk citra fundus karena perbedaan tingkat kecerahan dan kontras antar klinik yang mengambil gambar.

#### Normalisasi ke Rentang [0,1]

Nilai piksel dinormalisasi ke rentang [0,1] untuk mempercepat konvergensi dan menstabilkan proses pelatihan. Teknik normalisasi yang digunakan dalam jurnal ini berbeda dari jurnal pertama yang menggunakan normalisasi berbasis mean dan standar deviasi ImageNet. Normalisasi sederhana ke [0,1] lebih mudah diimplementasikan tetapi mungkin tidak seoptimal normalisasi berbasis distribusi dataset yang lebih kaya informasi.

---

### 5. Arsitektur Model

#### EfficientNet sebagai Local Feature Extractor

EfficientNet adalah keluarga arsitektur CNN yang dikembangkan oleh Google Brain pada tahun 2019. Inovasi utama EfficientNet adalah metode compound scaling, yaitu cara sistematis untuk menyeimbangkan tiga dimensi jaringan secara proporsional: depth (kedalaman, jumlah lapisan), width (lebar, jumlah filter), dan resolution (resolusi input). Sebelumnya, peneliti biasanya hanya meningkatkan salah satu dimensi secara terpisah. EfficientNet menunjukkan bahwa menyeimbangkan ketiganya secara proporsional menghasilkan peningkatan akurasi yang lebih efisien.

EfficientNet menggunakan blok **Mobile Inverted Bottleneck Convolution (MBConv)** sebagai unit pembangun utamanya. MBConv pertama-tama memperluas channel dengan faktor tertentu menggunakan konvolusi 1x1, kemudian melakukan konvolusi depthwise untuk ekstraksi fitur spasial, dan terakhir mereduksi channel kembali menggunakan konvolusi 1x1. Struktur ini jauh lebih efisien secara komputasi dibandingkan konvolusi standar karena memisahkan pemrosesan spasial dan channel-wise.

Dalam konteks DR, EfficientNet unggul dalam mendeteksi lesi-lesi mikroskopis yang bersifat lokal dalam citra fundus, seperti microaneurysm yang berdiameter hanya beberapa piksel, dan hemorrhage yang memiliki bentuk dan ukuran bervariasi. Kemampuan CNN dalam mendeteksi fitur-fitur berukuran kecil melalui konvolusi dengan kernel kecil adalah keunggulan yang tidak dimiliki Transformer secara alami.

#### Swin Transformer sebagai Global Feature Extractor

Swin Transformer adalah varian dari Vision Transformer yang dikembangkan oleh Microsoft Research dan merupakan salah satu model computer vision paling berpengaruh yang dikembangkan pada tahun 2021. Inovasi utama Swin Transformer adalah penggunaan **shifted window attention mechanism** yang membagi citra menjadi jendela-jendela lokal yang tidak tumpang tindih dan menerapkan self-attention hanya dalam setiap jendela lokal tersebut.

Pendekatan windowed attention ini mengatasi satu kelemahan utama ViT standar, yaitu kompleksitas komputasi yang kuadratik terhadap jumlah token. Dengan membatasi self-attention pada jendela lokal, kompleksitas dapat ditekan menjadi linear terhadap ukuran citra. Namun, window attention yang sepenuhnya lokal tidak dapat menangkap dependensi antar jendela. Untuk mengatasi ini, Swin Transformer menggunakan **shifted window attention** di lapisan yang berselang-seling, di mana jendela-jendela digeser sebesar setengah ukuran jendela sehingga terjadi pertukaran informasi antar jendela dari lapisan sebelumnya. Mekanisme ini menciptakan jalur komunikasi yang efisien antara bagian-bagian citra yang berjauhan.

Dalam konteks DR, Swin Transformer unggul dalam memahami konteks global retina, seperti distribusi dan pola vaskular keseluruhan, hubungan spasial antara berbagai area lesi, dan perubahan struktural pada tingkat makula dan nervus optikus. Informasi konteks global ini melengkapi informasi detail lokal yang diekstraksi oleh EfficientNet.

#### Feature Fusion

Feature fusion adalah proses menggabungkan representasi fitur dari dua sumber yang berbeda, dalam hal ini output dari cabang EfficientNet dan output dari cabang Swin Transformer, menjadi satu representasi terintegrasi. Representasi gabungan ini mencakup baik detail lokal lesi maupun konteks global retina secara bersamaan.

Sayangnya, jurnal ini tidak menjelaskan secara detail mekanisme feature fusion yang digunakan, apakah menggunakan simple concatenation (penggabungan vektor), element-wise addition (penjumlahan elemen), atau attention-based fusion yang lebih canggih. Kurangnya detail teknis ini adalah salah satu kelemahan yang diidentifikasi dalam analisis jurnal.

Secara konseptual, representasi gabungan yang dihasilkan oleh feature fusion adalah lebih informatif dibandingkan representasi dari satu cabang saja, karena ia mengandung informasi komplementer dari dua perspektif yang berbeda. Representasi inilah yang kemudian dimasukkan ke classifier head untuk menghasilkan prediksi grade DR.

#### Class Activation Map (CAM) untuk Interpretabilitas

Class Activation Map adalah teknik visualisasi yang memungkinkan kita untuk melihat area mana dari citra input yang paling berkontribusi pada keputusan klasifikasi yang dibuat oleh model. CAM bekerja dengan cara memvisualisasikan kombinasi tertimbang dari feature map pada lapisan konvolusi terakhir, menggunakan bobot dari lapisan classifier sebagai bobot.

Hasilnya adalah sebuah heatmap yang ditumpangkan di atas citra asli, di mana area dengan warna hangat (merah/kuning) menunjukkan area yang paling diperhatikan oleh model dalam membuat keputusan, sementara area dengan warna dingin (biru) menunjukkan area yang kurang relevan. Dalam konteks DR, CAM yang baik seharusnya menampilkan highlight pada area-area yang memang mengandung lesi DR, seperti microaneurysm dan hemorrhage, bukan pada area kulit di sekitar retina atau artefak pencahayaan.

Pentingnya CAM dalam konteks klinis sangat besar. Dokter tidak akan dengan mudah mempercayai sistem AI yang bekerja seperti kotak hitam tanpa bisa menjelaskan alasan keputusannya. Dengan CAM, dokter dapat memverifikasi bahwa model memang memperhatikan area yang relevan secara medis. Jika CAM menunjukkan bahwa model lebih berfokus pada artefak atau area yang tidak relevan, ini adalah sinyal bahwa model mungkin membuat keputusan yang benar dengan alasan yang salah, yang dapat menjadi masalah serius ketika digunakan dalam kasus-kasus yang berbeda dari data pelatihan.

---

### 6. Training dan Evaluasi

#### Parameter Pelatihan

Meskipun jurnal ini tidak menyebutkan semua detail parameter pelatihan secara eksplisit, disebutkan bahwa model hybrid dilatih pada dataset APTOS 2019 menggunakan teknik optimasi yang telah disetel. Mengacu pada praktik umum untuk arsitektur serupa dan berdasarkan hasil yang dilaporkan, dapat disimpulkan bahwa proses pelatihan menggunakan strategi augmentasi yang cukup intensif untuk mengatasi keterbatasan ukuran dataset.

#### Metrik Evaluasi

Empat metrik utama digunakan dalam evaluasi model ini:

**Accuracy (Akurasi)** mengukur kebenaran prediksi secara keseluruhan, dihitung sebagai rasio prediksi yang benar terhadap total prediksi. Nilai 0.97 atau 97% dalam penelitian ini berarti 97 dari 100 citra fundus diklasifikasikan ke grade DR yang tepat.

**Sensitivity** dalam konteks multikelas mengukur kemampuan model untuk mendeteksi benar setiap grade DR. Dalam konteks diagnostik, Sensitivity yang tinggi berarti model jarang melewatkan kasus positif. Nilai 0.95 berarti model berhasil mengidentifikasi 95% dari seluruh kasus DR dengan tepat.

**Specificity** mengukur kemampuan model untuk mengidentifikasi dengan benar pasien yang tidak memiliki DR (atau tidak memiliki grade tertentu). Nilai 0.98 berarti model sangat efektif menghindari alarm palsu, mengklasifikasikan 98% dari kasus non-DR sebagai non-DR. Specificity yang tinggi sangat penting dalam konteks skrining massal karena rendahnya tingkat alarm palsu mengurangi beban pemeriksaan lanjutan yang tidak perlu.

**AUC (Area Under the ROC Curve)** adalah metrik yang mengukur kemampuan diskriminatif model secara keseluruhan, yaitu seberapa baik model dapat membedakan antara kelas positif dan negatif di berbagai threshold. ROC Curve adalah grafik yang memplot Sensitivity (True Positive Rate) terhadap 1-Specificity (False Positive Rate) pada berbagai threshold keputusan. AUC 0.97 menunjukkan bahwa model memiliki kemampuan diskriminatif yang sangat baik, hampir mendekati model yang sempurna (AUC = 1.0). Secara intuitif, AUC 0.97 berarti jika kita mengambil satu sampel positif (memiliki DR) dan satu sampel negatif secara acak, model akan menetapkan skor yang lebih tinggi untuk sampel positif dengan probabilitas 97%.

---

### 7. Hasil Penelitian

Model hybrid EfficientNet dan Swin Transformer mencapai performa terbaik di antara seluruh model yang dibandingkan pada dataset APTOS 2019, dengan Sensitivity 0.95, Specificity 0.98, Accuracy 0.97, dan AUC 0.97. Nilai-nilai ini secara signifikan melampaui model berbasis CNN tunggal maupun Transformer murni yang diuji dalam eksperimen yang sama.

Keberhasilan model hybrid ini mengonfirmasi hipotesis bahwa representasi komplementer dari CNN (fitur lokal) dan Transformer (fitur global) memang saling melengkapi dan menghasilkan representasi yang lebih kaya dibandingkan masing-masing saja. Specificity 0.98 yang sangat tinggi menunjukkan bahwa model ini sangat efektif untuk skrining massal karena tingkat alarm palsu yang rendah, mengurangi beban kerja dokter spesialis yang hanya perlu menangani kasus yang memang positif.

Visualisasi CAM berhasil menyorot area retina yang relevan secara klinis sebagai basis keputusan model. Ini membuktikan bahwa model tidak hanya mencapai akurasi tinggi, tetapi juga melakukannya dengan cara yang dapat diinterpretasi dan divalidasi oleh klinisi. Validasi CAM ini adalah bukti bahwa model belajar fitur yang bermakna secara medis, bukan hanya menghafal korelasi statistik yang tidak bermakna.

---

### 8. Kekuatan Penelitian

Kekuatan utama penelitian ketiga ini adalah konsep hybrid yang secara arsitektural kuat dan terbukti secara empiris. Penggabungan EfficientNet sebagai ekstraktor fitur lokal dan Swin Transformer sebagai ekstraktor fitur global menghasilkan sistem yang lebih komprehensif daripada pendekatan tunggal mana pun. Ini adalah contoh yang baik dari desain sistem berdasarkan pemahaman mendalam tentang kekuatan dan kelemahan masing-masing komponen.

Interpretabilitas melalui CAM adalah kekuatan yang sangat penting dari perspektif klinis. Dalam dunia medis, kepercayaan klinisi terhadap sistem AI sangat bergantung pada kemampuan sistem tersebut untuk menjelaskan keputusannya. CAM menjadikan model ini bukan sekadar black box, tetapi sebuah alat yang dapat diaudit dan divalidasi secara medis.

Dataset real-world yang dikumpulkan dari berbagai klinik dengan kondisi yang bervariasi memberikan validasi yang lebih realistis dibandingkan dataset yang dikumpulkan dalam kondisi terkontrol. Performa 0.97 akurasi pada dataset semacam ini lebih mencerminkan kemampuan generalisasi model yang sesungguhnya.

---

### 9. Kelemahan Penelitian

Kelemahan utama adalah ukuran dataset yang relatif kecil. Hanya 3.662 citra untuk melatih model hybrid yang kompleks meningkatkan risiko overfitting meskipun sudah diterapkan augmentasi. Untuk perspektif perbandingan, dataset sejenis seperti EyePACS mengandung lebih dari 80.000 citra fundus retina.

Detail teknis fusion yang tidak dideskripsikan secara eksplisit adalah kelemahan dalam hal reproduksibilitas. Tanpa mengetahui metode fusion yang digunakan, peneliti lain tidak dapat mereproduksi hasil secara tepat atau melakukan ablation study untuk memahami kontribusi relatif dari setiap komponen fusion.

Ketidakseimbangan distribusi kelas yang tidak ditangani secara eksplisit juga menjadi perhatian. Meskipun augmentasi diterapkan, tanpa strategi khusus seperti oversampling kelas minoritas, focal loss, atau class weighting, model berpotensi lebih terlatih untuk kelas mayoritas (Grade 0) dan kurang optimal untuk Grade 3 dan Grade 4 yang secara klinis paling kritis.

Terakhir, validasi pada satu dataset saja tidak cukup untuk membuktikan kemampuan generalisasi model pada populasi yang lebih beragam. Performa pada dataset dari India mungkin tidak akan sama dengan performa pada dataset dari populasi Eropa, Asia Timur, atau Afrika yang memiliki karakteristik epidemiologis dan klinis yang berbeda.

---

### 10. Kesimpulan Jurnal Ketiga

Jurnal ketiga ini memberikan kontribusi yang bermakna dengan membuktikan keunggulan arsitektur hybrid CNN-Transformer atas pendekatan tunggal dalam klasifikasi grade DR yang bertingkat lima. Pencapaian AUC 0.97 dan Specificity 0.98 menjadikan model ini kandidat yang serius untuk implementasi dalam sistem skrining DR berbasis AI. Lebih dari sekadar performanya, penambahan interpretabilitas melalui CAM menjadikan penelitian ini relevan secara klinis dan menunjukkan pemahaman yang baik tentang kebutuhan nyata dalam adopsi AI di dunia medis. Tantangan ke depan adalah validasi pada dataset yang lebih besar dan lebih beragam, serta penjelasan teknis yang lebih rinci tentang mekanisme feature fusion.

---

# BAGIAN IV: PERBANDINGAN ANTAR KETIGA JURNAL

---

### 12. Perbandingan Komprehensif Tiga Jurnal

Tabel berikut menyajikan perbandingan ringkas dari aspek-aspek utama ketiga jurnal secara berdampingan.

| Aspek | Jurnal 1 (Brain Tumor) | Jurnal 2 (Skin Cancer) | Jurnal 3 (Diabetic Retinopathy) |
|-------|----------------------|----------------------|-------------------------------|
| Bidang Medis | Onkologi Saraf | Dermatologi | Oftalmologi |
| Jenis Citra | MRI Otak | Dermoskopi Kulit | Fundus Retina |
| Jumlah Data | 7.023 citra | 10.015 citra | 3.662 citra |
| Jumlah Kelas | 4 kelas (multiclass) | 2 kelas (biner) | 5 kelas (multiclass) |
| Arsitektur Utama | Fine-Tuned ViT | SAM + ViT | EfficientNet + Swin Transformer |
| Pendekatan | Single-stage klasifikasi | Two-stage (segmentasi + klasifikasi) | Single-stage klasifikasi hybrid |
| Backbone | ViT pretrained ImageNet21k | Berbagai varian ViT | EfficientNet + Swin Transformer |
| Akurasi Tertinggi | 98.70% (FTVT-l16) | 96.15% (ViT-Google patch-32) | 0.97 (97%) |
| Metrik Kunci Tambahan | F1-Score, Precision, Recall | IoU 96.01%, Dice 98.14%, FN Ratio | AUC 0.97, Sensitivity 0.95, Specificity 0.98 |
| Interpretabilitas | Tidak ada visualisasi | Tidak ada visualisasi | CAM (Class Activation Map) |
| Augmentasi Khusus | Color Jitter, Flip, Rotation | Flip H/V, Rotation, Brightness | Gaussian Blur, Flip, Rotation |
| Optimizer | Adam (lr=1e-4) | Adam (lr=0.00002) | Tidak disebutkan secara eksplisit |
| Komparasi Model | 7 model (4 ViT + 3 CNN) | 6 varian ViT | Beberapa deep learning model |

#### Perbandingan Pendekatan Model

Jurnal pertama menggunakan pendekatan paling sederhana dari ketiga jurnal, yaitu model klasifikasi langsung berbasis ViT yang di-fine-tune dengan custom classifier head. Keunggulan pendekatan ini adalah kesederhanaannya yang memudahkan implementasi dan interpretasi, sementara kualitas akurasinya yang tinggi membuktikan bahwa pendekatan sederhana yang dioptimalkan dengan baik dapat mengungguli pendekatan yang lebih kompleks.

Jurnal kedua menggunakan pendekatan yang paling kompleks dalam hal jumlah komponen, dengan pipeline dua tahap yang melibatkan dua model berbeda, yaitu SAM untuk segmentasi dan ViT untuk klasifikasi. Kompleksitas ini dijustifikasi oleh sifat tugas yang memang memerlukan isolasi area relevan sebelum klasifikasi, karena lesi kulit dalam citra dermoskopi memiliki latar belakang yang kaya dan berpotensi mengganggu.

Jurnal ketiga menawarkan pendekatan yang paling inovatif dari perspektif arsitektur dengan menggabungkan dua model yang saling komplementer dalam sebuah jaringan hybrid. Pendekatan ini didasarkan pada pemahaman teoritis yang kuat tentang kelebihan dan kekurangan masing-masing komponen dan mengintegrasikannya secara cerdas.

#### Perbandingan Dataset

Dataset terbesar digunakan oleh Jurnal 2 (HAM10000 dengan 10.015 citra), diikuti Jurnal 1 (7.023 citra) dan Jurnal 3 (3.662 citra). Namun, ukuran dataset saja bukan satu-satunya faktor yang menentukan kesulitan tugas. Jurnal 3 menghadapi tantangan tambahan berupa klasifikasi ke dalam 5 kelas yang tidak seimbang distribusinya, sementara Jurnal 2 menyederhanakan menjadi hanya 2 kelas.

#### Perbandingan Jenis Penyakit dan Relevansi Klinis

Ketiga jurnal memiliki relevansi klinis yang tinggi namun dalam konteks yang berbeda. Jurnal 1 berkaitan dengan penyakit yang sangat mengancam jiwa dengan tingkat urgensi diagnosis yang tinggi. Jurnal 2 berkaitan dengan kanker dengan insidensi tinggi di seluruh dunia di mana deteksi dini sangat menentukan prognosis. Jurnal 3 berkaitan dengan komplikasi diabetes yang merupakan epidemi global, di mana skrining massal adalah strategi pencegahan kebutaan yang paling efektif secara biaya.

#### Perbandingan Interpretabilitas

Perbedaan paling mencolok antar ketiga jurnal adalah dalam hal interpretabilitas model. Jurnal 1 dan Jurnal 2 tidak menyertakan mekanisme visualisasi atau interpretabilitas apapun, menjadikan keduanya sebagai sistem black box yang sulit diaudit secara klinis. Jurnal 3, sebaliknya, secara eksplisit mengimplementasikan CAM dan menjadikannya salah satu kontribusi utama penelitian. Dari perspektif adopsi klinis jangka panjang, pendekatan Jurnal 3 yang menekankan interpretabilitas adalah yang paling matang secara konseptual.

#### Perbandingan Kelebihan dan Kelemahan

Jurnal 1 unggul dalam akurasi keseluruhan dan dalam analisis komparatif yang komprehensif, namun lemah dalam persyaratan komputasi dan cakupan jenis tumor. Jurnal 2 unggul dalam inovasi pipeline dan orientasi klinis, namun lemah dalam granularitas klasifikasi dan ketergantungan pada prompt manual. Jurnal 3 unggul dalam interpretabilitas dan relevansi untuk skrining massal, namun lemah dalam ukuran dataset dan transparansi teknis mekanisme fusion.

---

# BAGIAN V: REFLEKSI DAN INSIGHT AKADEMIK

---

### 13. Insight dan Pembelajaran

#### Tren Penggunaan Vision Transformer dalam Computer Vision Medis

Ketiga jurnal yang direview secara konsisten menggunakan Vision Transformer sebagai komponen inti sistem mereka. Ini mencerminkan tren yang lebih luas dalam komunitas Computer Vision, di mana arsitektur Transformer telah melampaui CNN sebagai state-of-the-art dalam berbagai benchmark, termasuk ImageNet, COCO, dan berbagai tugas citra medis. Tren ini tidak terjadi secara tiba-tiba; ia adalah hasil dari beberapa tahun penelitian yang secara bertahap memodifikasi arsitektur Transformer agar lebih efisien dan efektif untuk citra, mulai dari ViT standar pada tahun 2020, hingga Swin Transformer yang jauh lebih efisien pada tahun 2021, dan berbagai varian pretrained yang lebih canggih seperti BEiT dan MAE.

#### Pergeseran dari CNN Murni ke Hybrid dan Transformer Murni

Pergeseran paradigma ini tidak berarti CNN menjadi usang. Jurnal pertama masih membandingkan ViT dengan CNN dan hasilnya ViT menang, namun dengan selisih yang tidak terlalu dramatis untuk semua kelas. Jurnal ketiga justru mempertahankan CNN (EfficientNet) sebagai komponen esensial dalam arsitektur hybridnya karena kemampuan CNN dalam mendeteksi fitur lokal tetap tidak tertandingi oleh Transformer.

Pesan yang tepat dari ketiga jurnal ini bukan bahwa CNN sudah tidak berguna, melainkan bahwa CNN dan Transformer memiliki kelebihan yang berbeda dan saling melengkapi. Model hybrid yang menggabungkan keduanya secara cerdas, seperti yang dilakukan Jurnal 3, berpotensi menjadi pendekatan yang paling optimal untuk banyak tugas analisis citra medis.

#### Pentingnya Hybrid Model

Model hybrid menjanjikan karena menggabungkan inductive bias CNN, yaitu asumsi bahwa fitur penting dalam citra bersifat lokal dan translasi-invariant, dengan kemampuan Transformer untuk menangkap dependensi jarak jauh. Untuk domain medis di mana fitur diagnostik dapat bersifat lokal (lesi kecil) sekaligus global (distribusi dan konteks keseluruhan), model hybrid menawarkan fondasi yang paling kuat secara teoritis.

#### Pentingnya Explainable AI dalam Dunia Medis

Ketiga jurnal ini, terutama Jurnal 3, menggaris-bawahi pentingnya Explainable AI (XAI) dalam konteks medis. Dokter adalah profesional yang terlatih untuk membuat keputusan berdasarkan bukti dan pemahaman mekanis. Sebuah sistem AI yang hanya memberikan output prediksi tanpa penjelasan tidak akan mudah diterima dalam praktik klinis, terutama untuk keputusan yang berimplikasi besar bagi kehidupan pasien.

XAI bukan hanya tentang kepercayaan dokter. Ia juga merupakan mekanisme quality assurance yang penting, karena visualisasi seperti CAM dapat mengungkap kasus di mana model membuat keputusan yang benar tetapi dengan alasan yang salah, sebuah fenomena yang disebut shortcut learning. Mendeteksi shortcut learning jauh lebih mudah dengan visualisasi dibandingkan hanya mengandalkan metrik agregat.

#### Tantangan AI dalam Domain Medis

Ketiga jurnal ini juga secara kolektif mengungkap tantangan-tantangan yang masih belum terselesaikan dalam penerapan AI di domain medis. Pertama, keterbatasan data berlabel yang berkualitas, karena anotasi citra medis membutuhkan keahlian dokter yang langka dan mahal. Kedua, ketidakseimbangan kelas yang mencerminkan distribusi epidemiologis alami namun menyulitkan pembelajaran mesin. Ketiga, kebutuhan komputasi yang tinggi yang dapat menghambat akses di negara berkembang. Keempat, kesenjangan antara validasi in-silico dan validasi klinis yang merupakan jurang yang harus dijembatani sebelum sistem AI dapat diandalkan dalam praktik nyata.

#### Peluang Penelitian Lanjutan

Berdasarkan analisis ketiga jurnal ini, beberapa arah penelitian lanjutan yang paling menjanjikan dapat diidentifikasi.

Pertama, pengembangan model yang lebih ringan (lightweight) yang dapat beroperasi pada perangkat keras terbatas tanpa mengorbankan akurasi secara signifikan. Ini penting untuk memungkinkan penerapan di fasilitas kesehatan primer di daerah terpencil.

Kedua, penelitian tentang metode augmentasi dan synthetic data generation yang lebih canggih, seperti penggunaan Generative Adversarial Networks (GAN) atau difusi model untuk menghasilkan citra medis sintetis yang dapat memperkaya dataset yang kecil dan tidak seimbang.

Ketiga, pengembangan model yang dapat dilatih dengan data dari beberapa domain secara bersamaan (multi-task learning) atau dilatih sekali dan diadaptasi ke tugas baru dengan data minimal (few-shot learning), sehingga lebih fleksibel dalam menghadapi beragam kondisi klinis.

Keempat, penelitian yang berfokus pada validasi klinis prospektif, yaitu pengujian model dalam kondisi klinis nyata dengan protokol yang ketat, untuk membuktikan bahwa performa in-silico dapat diterjemahkan ke dalam manfaat klinis yang nyata bagi pasien.

Kelima, integrasi pengetahuan medis (medical knowledge) secara eksplisit ke dalam arsitektur model melalui pendekatan seperti knowledge graph atau physics-informed neural networks, sehingga model tidak hanya belajar dari data tetapi juga dari prinsip-prinsip medis yang sudah mapan.

---

## Penutup

Ketiga jurnal yang dianalisis dalam dokumen ini bersama-sama memberikan gambaran yang komprehensif tentang kondisi terkini penelitian Computer Vision berbasis deep learning dalam domain citra medis. Mereka membuktikan bahwa Vision Transformer, baik secara mandiri maupun dalam bentuk hybrid, telah menjadi pendekatan yang dominan dan efektif untuk berbagai tugas analisis citra medis, dari klasifikasi tumor otak, deteksi kanker kulit, hingga grading Diabetic Retinopathy.

Kemajuan yang dicapai oleh ketiga penelitian ini sangat signifikan, namun perjalanan dari akurasi yang tinggi di lingkungan penelitian menuju implementasi klinis yang andal dan diterima secara luas masih membutuhkan banyak kerja keras, kolaborasi antara ilmuwan komputer dan klinisi, serta komitmen untuk mengatasi tantangan interpretabilitas, kesetaraan akses, dan validasi klinis yang ketat.

---

*Dokumen ini disusun berdasarkan review jurnal yang dilakukan oleh M. Hibban Ramadhan (NIM: 2315061094) pada tanggal 7 Mei 2026.*
