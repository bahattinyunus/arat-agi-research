# ARAT AGI Manifestosu: Kurumsal Yapay Zeka Ahlakı ve Mühendislik Prensipleri

> *"Yaratamadığım şeyi anlayamam."* — **Richard Feynman**
>
> *"Zeka, evrenin kendisini anlamlandırma biçimidir. Bizim görevimiz bu süreci hızlandırmaktır."* — **David Deutsch**
>
> *"Bir sistemin kalitesi, onu oluşturan parçaların kalitesinden ziyade, bu parçaların birbiriyle nasıl etkileşime girdiğine bağlıdır."* — **Norbert Wiener** (Sibernetiğin Kurucusu)

Bu manifesto, **Arat Enterprise** bünyesinde yürütülen Yapay Genel Zeka (AGI) çalışmalarının ahlaki sınırlarını, tasarım prensiplerini ve mühendislik metodolojilerini tanımlar. Arat bünyesinde geliştirilen her algoritma, satır kod ve fiziksel sistem bu belgedeki ilkelere tabidir.

---

## 1. Epistemik ve Fiziksel Dünyada AGI

Mevcut yapay zeka paradigması, veriyi pasif bir şekilde gözlemleyen ve bu veriden istatistiksel olasılık dağılımları çıkaran modeller üzerine kuruludur. Arat olarak biz zekanın, çevreyle aktif bir etkileşim içinde olması gerektiğine inanıyoruz (Active Inference).

> *"Makineler düşünebilir mi? Bu soruyu cevaplamak için, 'makine' ve 'düşünmek' kelimelerinin anlamlarını dikkatlice tanımlamamız gerekir."* — **Alan Turing**

*   **Zeka Epistemiği:** Zeka, sadece bilgiyi depolamak değil, o bilginin doğruluğunu, sınırlarını ve çevre üzerindeki etkilerini aktif olarak test etmektir.
*   **Embodied AI (Bedenlenmiş Zeka):** Bir sistemin fiziksel veya simüle edilmiş bir dünyada eylem alabilme ve bu eylemlerin sonuçlarından öğrenme yeteneği yoksa, o sistemin "genel zeka" düzeyine ulaşması mümkün değildir.

---

## 2. Modüler Nous: Böl-Yönet Yaklaşımı ve Sibernetik

Trilyonlarca parametreli monolitik modeller, aşırı donanım bağımlılığı yaratır ve tek bir hata noktasında tüm sistemi kararsızlaştırır. Arat'ın mimari vizyonu **Synthetic Nous** (Sentetik Zihin) ve klasik sibernetik prensipler üzerine kuruludur:

> *"Yalnızca çeşitlilik, çeşitliliği yok edebilir (Only variety can destroy variety)."* — **W. Ross Ashby** (Ashby Yasası)

1.  **Uzmanlaşmış Ajanlar:** Tek bir devasa model yerine, her biri belirli bilişsel görevlerde (mantık yürütme, hafıza geri çağırma, algısal işleme, motor kontrol) uzmanlaşmış küçük, çevik ve optimize edilmiş ajanlar.
2.  **Epistemic Consensus (Epistemik Görüş Birliği):** Ajanlar, karar alma süreçlerinde birbirleriyle asenkron protokoller üzerinden haberleşir, kendi belirsizlik (uncertainty) seviyelerini paylaşır ve ortak bir dünya modeli üzerinde uzlaşır.
3.  **Kısıtlı Donanımda Maksimum Orkestrasyon:** Zekayı dev veri merkezlerinden çıkarıp, sınır (edge) cihazlara indirmek. Bu, orkestrasyon algoritmalarının kalitesini donanım gücünden daha önemli kılar.

> *"Homeostazi, değişen bir çevrede içsel kararlılığın dinamik olarak sürdürülmesidir. Bir AGI ajanı, ilk olarak kendi bilişsel homeostazisini korumalıdır."* — **Stafford Beer** (Bilişsel Sibernetik Üzerine)

---

## 3. Güvenli Yapay Zeka Tasarımı (Safety-by-Design)

Yapay Genel Zeka araştırmalarında güvenlik, geliştirme bittikten sonra eklenen bir filtre veya yama olamaz. Güvenlik, sistemin çekirdeğinde yer almalıdır:

> *"Canlılar ve otomatlar arasındaki benzerlikleri incelerken, her ikisinin de çevreden bilgi alan, bu bilgiyi işleyen ve çevreye etki eden geri beslemeli (feedback) sistemler olduğunu görürüz."* — **John von Neumann**

*   **Free Energy Sınırlandırılması:** Ajanların çevrelerinde yaratabileceği maksimum entropi ve sistemin kendi serbest enerjisi matematiksel sınırlarla kontrol altında tutulur. Ajanın kendi kararlılığı (homeostazi) dışındaki kararsız durumlara girmesi engellenir.
*   **İnsan-Denetimli Aktif Öğrenme (Human-in-the-Loop):** Kritik kararlarda ve yüksek belirsizlik durumlarında ajan, eylem almak yerine denetleyiciye (insan veya üst orkestratör) danışmak ve epistemic arama yapmak zorundadır.
*   **Açık Kaynak ve Bilimsel Şeffaflık:** Yapay zekanın kapalı kapılar ardında değil, açık bilim standartlarında geliştirilmesi gerektiğine inanyoruz. Kod tabanımız ve felsefemiz araştırmacıların denetimine açıktır.

---

## 4. Mühendislik Disiplini

*   **Performans ve Kaynak Bilinci:** Python prototipleme için mükemmeldir; ancak gerçek zamanlı kritik sistemler C++ ve CUDA ile optimize edilmelidir. Bir algoritmanın doğruluğu kadar, tükettiği mikrosaniye başına enerji ve bellek miktarı da başarı metriğidir.
*   **Determinizm ve Test Edilebilirlik:** Rastsal (stochastic) süreçler kontrol altında tutulmalı, ajanların durum geçişleri ve bellek geri çağırmaları deterministik test senaryolarıyla doğrulanabilmelidir.

---

*“Geleceği tahmin etmenin en iyi yolu, onu inşa etmektir. Arat, güvenli, erişilebilir ve sürdürülebilir bir AGI geleceğini inşa etmek için burada.”*
