# Submission 1: Email Classification
Nama: Zhulkhoir Rifat
Username dicoding: zhulkhoirrifat

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Spam Email Dataset](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset) |
| Masalah |	Email adalah salah satu alat untuk berkomunikasi yang paling populer saat ini. Namun, banyak email yang masuk dalam kotak masuk (inbox) pengguna berisi spam yang tidak diinginkan, yang bisa menyebabkan gangguan dan membuang waktu. Oleh karena itu, penting untuk memiliki sistem yang dapat secara otomatis mengidentifikasi dan memfilter spam dari email yang sah.|
| Solusi machine learning | Solusi machine learning yang akan dibuat adalah model klasifikasi yang dapat membedakan antara email spam dan email sah berdasarkan fitur-fitur yang terdapat dalam konten email dari kata yang ada.|
| Metode pengolahan | Data akan diproses dengan cara mengubahnya menjadi lower string atau huruf kecil, kemudian mengubah teks menjadi numerik agar bisa dibaca oleh mesin dan membagi data latih dan evaluasi dengan rasio 8:2|
| Arsitektur model | Model ini digunakan untuk klasifikasi teks. Dimulai dengan input berupa teks dalam bentuk string yang kemudian diubah menjadi vektor melalui layer vectorize_layer. Selanjutnya, teks tersebut diterjemahkan menjadi representasi numerik melalui layer Embedding. Kemudian, teks diproses menggunakan dua layer LSTM, pertama dengan arsitektur bidirectional untuk menangkap informasi dari kedua arah (kiri-kanan dan kanan-kiri), dan kedua dengan LSTM standar untuk mengambil informasi lebih mendalam dari sequence. Hasil dari LSTM kemudian diproses melalui dua layer dense dengan fungsi aktivasi ReLU dan regularisasi menggunakan layer Dropout untuk mencegah overfitting. Output akhirnya berupa satu neuron dengan aktivasi sigmoid, menghasilkan prediksi biner (0 atau 1) dari teks input. |
| Metrik evaluasi | Metrik evaluasi yang digunakan yaitu ExampleCount, AUC, FalsePositives, TruePositives, FalseNegatives, TrueNegatives, dan BinaryAccuracy untuk evaluasi performa model dalam menentukan email yang berisi spam atau bukan spam. |
| Performa model | Performa model yang dibuat menunjukkan hasil yang sangat baik dengan mencatatkan loss sebesar 0.0082 dan BinaryAccuracy sebesar 99.89%. Hasil ini menunjukkan bahwa model mampu melakukan klasifikasi dengan tingkat kesalahan yang sangat rendah dan akurasi yang sangat tinggi.|
