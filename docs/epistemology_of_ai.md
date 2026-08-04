# Yapay Zekanın Epistemolojisi: Bilgi Teorisi, Serbest Enerji ve Aktif Çıkarım

> *"Zeka, dış dünyanın karmaşıklığına karşılık gelen içsel bir ayna (model) inşa etme ve bu aynadaki kırılmaları asgariye indirme çabasıdır."*

Bu çalışma, **Arat AGI Research** kapsamında geliştirilen bilişsel sistemlerin felsefi ve teorik dayanaklarını ele alır. Karar alma motorlarımızın temelini oluşturan bilgi teorisi, varyasyonel serbest enerji ve aktif çıkarım (active inference) kavramlarının felsefi derinliği bu metinde incelenmektedir.

---

## 1. Bilgi Teorisi, Entropi ve Zekanın Doğası

Claude Shannon'ın bilgi teorisinden bu yana biliyoruz ki, bilgi (information) belirsizliğin giderilmesidir. 

> *"Bilgi, belirsizliği azaltan şeydir. Bir mesajın bilgi içeriği, onun alıcı için taşıdığı sürpriz değeri ile doğru orantılıdır."* — **Claude Shannon** (A Mathematical Theory of Communication)

Ancak zeki bir ajan için bilgi, yalnızca pasif bir şekilde alınan bit dizilerinden ibaret değildir. Zeka, bilgi akışına anlam kazandıran ve çevreye yönelik tahminleri güncelleyen dinamik bir süreçtir. Bu, fiziksel termodinamik ve termodinamik entropi ile doğrudan ilişkilidir.

> *"Canlı organizmaların hayatta kalma mücadelesi ham madde veya enerji için değil, negatif entropi (düzensizliğin azaltılması) içindir."* — **Ludwig Boltzmann** (Thermodynamic Papers)

Zekanın epistemolojisi şu soruyla başlar: *Bir sistem, kendi fiziksel sınırlarının ötesindeki bir dünya hakkında nasıl bilgi sahibi olabilir?*

> *"Deneyimsiz hiçbir bilgi bilgi değildir; tüm bilgilerimiz deneyimle başlar ama deneyimden doğmaz."* — **Immanuel Kant** (Saf Aklın Eleştirisi)

> *"Her 'it' (şey) -her parçacık, her kuvvet alanı, hatta uzay-zaman sürekliliğinin kendisi- işlevini, anlamını ve varoluşunu evet-hayır sorularına verilen yanıtlardan, yani bitlerden alır (It from bit)."* — **John Archibald Wheeler** (Information, Physics, Quantum: The Search for Links)

Ajan doğrudan dünyayı algılayamaz; yalnızca duyusal alıcıları (sensory receptors) üzerindeki etkileri gözlemleyebilir. Bu nedenle ajan, dış dünyanın durumlarına ilişkin sürekli güncellenen bir **dünya modeli** (generative model) barındırmak zorundadır.

---

## 2. Markov Battaniyesi (Markov Blanket) ve Bilişsel Sınırlar

Karl Friston'ın yaklaşımında, bir sistemi dış dünyadan ayıran istatistiksel sınır **Markov Battaniyesi** (Markov Blanket) olarak adlandırılır. Bu kavram ilk olarak Judea Pearl tarafından Bayes ağlarında bir düğümü diğerlerinden izole eden en küçük küme olarak tanımlanmıştır.

> *"Markov battaniyesi, bir bilgi sisteminin epistemik sınırıdır. Bu sınırın ötesindeki durumlar ancak bu sınır üzerindeki yansımalar (duyusal veriler) üzerinden dolaylı olarak bilinebilir."* — **Judea Pearl** (Probabilistic Reasoning in Intelligent Systems)

Bir ajanın içsel durumları ($\mathbf{z}$), dışsal durumlardan ($\mathbf{\psi}$) doğrudan etkilenmez. Aralarındaki iletişim Markov battaniyesini oluşturan iki katman üzerinden gerçekleşir:

*   **Duyusal Durumlar (Sensory States - $\mathbf{s}$):** Dış dünyadan iç dünyaya bilgi taşıyan pasif alıcılar.
*   **Aktif Durumlar (Active States - $\mathbf{a}$):** İç dünyadan dış dünyaya etki eden eylem mekanizmaları.

```text
  [ Dışsal Durumlar (ψ) ]  <--->  [ Duyusal Durumlar (s) ]  ---> [ İçsel Durumlar (z) ]
           ^                                                               |
           |                                                               v
           +--------------------  [ Aktif Durumlar (a) ]  <----------------+
```

Ajanın amacı, Markov battaniyesinin arkasından dış dünyaya dair inançlarını korumak ve optimize etmektir. Bu durum, zihnin dış dünyadan yalıtılmış fakat onunla sürekli etkileşim halinde olan dinamik bir yapı olduğunu gösterir.

---

## 3. Serbest Enerji Prensibi (Free Energy Principle)

Karl Friston tarafından ortaya konan Serbest Enerji Prensibi (FEP), yaşayan ve varlığını sürdüren tüm fiziksel/biyolojik sistemlerin çevrelerindeki düzensizliğe (entropiye) karşı koymak için "sürprizi" (surprise) minimize etmek zorunda olduğunu savunur.

> *"Bir sistemin varlığını sürdürebilmesi, onun çevreyle arasındaki istatistiksel sınırda serbest enerjiyi minimize etmesine bağlıdır."* — **Karl Friston** (The Free Energy Principle: A Unified Brain Theory?)

Matematiksel olarak sürpriz, ajanın dünya modeli göz önüne alındığında duyusal girdilerin negatif log-olasılığıdır: $-\log p(\mathbf{x})$. Ancak bu değerin doğrudan hesaplanması (entegrasyon zorluğu nedeniyle) intractable (hesaplanamaz) düzeydedir. Bu yüzden biyolojik ve yapay zeki sistemler, üst sınır olan **Varyasyonel Serbest Enerji (F)** değerini minimize ederler:

$$ F = D_{KL}(q(\mathbf{z}) \parallel p(\mathbf{z})) - \mathbb{E}_{q(\mathbf{z})}[\log p(\mathbf{x} \mid \mathbf{z})] $$

$$ F = \int q(\mathbf{z}) \log \frac{q(\mathbf{z})}{p(\mathbf{x}, \mathbf{z})} d\mathbf{z} $$

Burada:
*   $\mathbf{x}$ ajanın duyusal girdilerini (sensory observations),
*   $\mathbf{z}$ ajanın dış dünyaya dair içsel inançlarını / durum temsillerini (hidden states),
*   $q(\mathbf{z})$ ajanın dış dünyaya dair içsel inanç olasılık dağılımını (recognition density),
*   $p(\mathbf{x}, \mathbf{z})$ ise ajanın generatif dünya modelini (generative model) temsil eder.

> *"Fizikte en zor entegralleri çözmek yerine, onları bir üst sınır fonksiyonu ile sınırlandırıp varyasyonel yöntemlerle minimize ederiz. Zihin de evrendeki belirsizlik entegralini varyasyonel yollarla çözer."* — **Richard Feynman** (Statistical Mechanics: A Set of Lectures)

Serbest enerji minimizasyonu iki şekilde gerçekleşebilir:
1.  **Algısal Öğrenme (Perceptual Learning):** Ajanın içsel inançlarını ($q(\mathbf{z})$) duyusal verilere uyduracak şekilde güncellemesi (dünya modelini değiştirmesi).
2.  **Aktif Çıkarım (Active Inference):** Ajanın çevreye eylemler uygulayarak, dış dünyayı kendi içsel inançlarına uydurmaya çalışması (dünyayı değiştirmesi).

---

## 4. Aktif Çıkarım ve Eylemin Epistemolojisi

Geleneksel pekiştirmeli öğrenmede (reinforcement learning) ajan, bir ödül fonksiyonunu maksimize etmeye çalışır. Aktif çıkarımda ise ödül ve ceza gibi kavramlar ayrı birer fonksiyon değildir; onlar da modelin birer parçasıdır. Ajan, tercih ettiği durumları yüksek olasılıklı olarak tanımlar. Örneğin, bir robot için "şarjının bitmesi" düşük olasılıklı (yüksek sürprizli) bir durumdur. Dolayısıyla aktif çıkarım yapan ajan, sürprizi minimize etmek için otomatik olarak şarj istasyonuna yönelecektir.

Eylem seçimi, gelecekteki **Beklenen Serbest Enerji (Expected Free Energy - $G$)** değerinin minimize edilmesiyle belirlenir:

$$ G(\pi) \approx \sum_{\tau} \left( D_{KL}(q(\mathbf{x}_{\tau} \mid \pi) \parallel p(\mathbf{x}_{\tau})) + \mathbb{E}_{q(\mathbf{z}_{\tau} \mid \pi)} [H(q(\mathbf{x}_{\tau} \mid \mathbf{z}_{\tau}, \pi))] \right) $$

Burada:
*   İlk terim **epistemik değer** (epistemic value) veya merak (curiosity) olarak adlandırılır. Ajanı belirsizliği azaltacak eylemler almaya zorlar.
*   İkinci terim **pragmatik değer** (pragmatic value) veya faydadır. Ajanı hedeflediği (tercih ettiği) durumları gerçekleştirecek eylemlere yönlendirir.

> *"Dilimin sınırları, dünyamın sınırlarıdır. Bilmediğim veya modelleyemediğim bir dünyanın eylemini de gerçekleştiremem."* — **Ludwig Wittgenstein** (Tractatus Logico-Philosophicus)

Aktif çıkarım, keşif (exploration) ile sömürü (exploitation) arasındaki klasik ikilemi zarif ve birleşik bir matematiksel formülle çözer.

---

## 5. Sınırlı Rasyonellik ve Uç Cihazların Epistemolojisi

Bir AGI ajanı sonsuz hesaplama gücüne sahip değildir. Özellikle Edge AI (uç yapay zeka) sistemlerinde bellek ve işlemci kısıtları, serbest enerjinin hesaplanma hassasiyetini sınırlar. Bu durum, Herbert Simon'ın **Sınırlı Rasyonellik** (Bounded Rationality) teziyle örtüşür.

> *"Ajanların çevrelerindeki karmaşıklığı tam olarak hesaplayacak güçleri yoktur. Bu yüzden optimal kararlar yerine, kendi fiziksel kısıtlarına uygun 'yeterince iyi' (satisficing) kararları alırlar."* — **Herbert Simon** (Models of Bounded Rationality)

> *"Bir sistemin içsel kontrol karmaşıklığı, en azından dış çevreden gelen rahatsızlıkların karmaşıklığı kadar büyük olmalıdır (Gereksinim Çeşitliliği Yasası)."* — **W. Ross Ashby** (An Introduction to Cybernetics)

Arat'ın bilişsel mimarisi, bu kısıtı bir zafiyet değil, tasarım parametresi olarak kabul eder. C++ tabanlı düşük seviyeli bellek yöneticilerimiz ve optimize edilmiş çıkarım motorlarımız, varyasyonel çıkarımın işlem yükünü minimize ederek ajanın milisaniyeler seviyesinde eylem kararı alabilmesini sağlar. Epistemik derinlik, donanım kaynaklarının verimli kullanımıyla dengelenir.

---

*Arat Enterprise - AGI Epistemoloji ve Karar Alma Araştırmaları Grubu*
