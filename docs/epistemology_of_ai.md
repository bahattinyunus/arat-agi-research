# Yapay Zekanın Epistemolojisi: Bilgi Teorisi, Serbest Enerji ve Aktif Çıkarım

> *"Zeka, dış dünyanın karmaşıklığına karşılık gelen içsel bir ayna (model) inşa etme ve bu aynadaki kırılmaları asgariye indirme çabasıdır."*

Bu çalışma, **Arat AGI Research** kapsamında geliştirilen bilişsel sistemlerin felsefi ve teorik dayanaklarını ele alır. Karar alma motorlarımızın temelini oluşturan bilgi teorisi, varyasyonel serbest enerji ve aktif çıkarım (active inference) kavramlarının felsefi derinliği bu metinde incelenmektedir.

---

## 1. Bilgi Teorisi ve Zekanın Doğası

Claude Shannon'ın bilgi teorisinden bu yana biliyoruz ki, bilgi (information) belirsizliğin giderilmesidir. Ancak zeki bir ajan için bilgi, yalnızca pasif bir şekilde alınan bit dizilerinden ibaret değildir. Zeka, bilgi akışına anlam kazandıran ve çevreye yönelik tahminleri güncelleyen dinamik bir süreçtir.

Zekanın epistemolojisi şu soruyla başlar: *Bir sistem, kendi fiziksel sınırlarının ötesindeki bir dünya hakkında nasıl bilgi sahibi olabilir?*

Bu soru bizi Immanuel Kant'ın "kendinde şey" (noumenon) kavramına ve oradan modern sibernetik ve fizik teorilerine götürür. Ajan doğrudan dünyayı algılayamaz; yalnızca duyusal alıcıları (sensory receptors) üzerindeki etkileri gözlemleyebilir. Bu nedenle ajan, dış dünyanın durumlarına ilişkin sürekli güncellenen bir **dünya modeli** (generative model) barındırmak zorundadır.

---

## 2. Serbest Enerji Prensibi (Free Energy Principle)

Karl Friston tarafından ortaya konan Serbest Enerji Prensibi (FEP), yaşayan ve varlığını sürdüren tüm fiziksel/biyolojik sistemlerin çevrelerindeki düzensizliğe (entropiye) karşı koymak için "sürprizi" (surprise) minimize etmek zorunda olduğunu savunur.

Matematiksel olarak sürpriz, ajanın dünya modeli göz önüne alındığında duyusal girdilerin negatif log-olasılığıdır: $-\log p(\mathbf{x})$. Ancak bu değerin doğrudan hesaplanması (entegrasyon zorluğu nedeniyle) intractable (hesaplanamaz) düzeydedir. Bu yüzden biyolojik ve yapay zeki sistemler, üst sınır olan **Varyasyonel Serbest Enerji (F)** değerini minimize ederler:

$$ F = D_{KL}(q(\mathbf{z}) \parallel p(\mathbf{z})) - \mathbb{E}_{q(\mathbf{z})}[\log p(\mathbf{x} \mid \mathbf{z})] $$

Burada:
*   $\mathbf{x}$ ajanın duyusal girdilerini (sensory observations),
*   $\mathbf{z}$ ajanın dış dünyaya dair içsel inançlarını / durum temsillerini (hidden states),
*   $q(\mathbf{z})$ ajanın dış dünyaya dair içsel inanç olasılık dağılımını (recognition density),
*   $p(\mathbf{x}, \mathbf{z})$ ise ajanın generatif dünya modelini (generative model) temsil eder.

Serbest enerji minimizasyonu iki şekilde gerçekleşebilir:
1.  **Algısal Öğrenme (Perceptual Learning):** Ajanın içsel inançlarını ($q(\mathbf{z})$) duyusal verilere uyduracak şekilde güncellemesi (dünya modelini değiştirmesi).
2.  **Aktif Çıkarım (Active Inference):** Ajanın çevreye eylemler uygulayarak, dış dünyayı kendi içsel inançlarına uydurmaya çalışması (dünyayı değiştirmesi).

---

## 3. Aktif Çıkarım ve Eylemin Epistemolojisi

Geleneksel pekiştirmeli öğrenmede (reinforcement learning) ajan, bir ödül fonksiyonunu maksimize etmeye çalışır. Aktif çıkarımda ise ödül ve ceza gibi kavramlar ayrı birer fonksiyon değildir; onlar da modelin birer parçasıdır. Ajan, tercih ettiği durumları yüksek olasılıklı olarak tanımlar. Örneğin, bir robot için "şarjının bitmesi" düşük olasılıklı (yüksek sürprizli) bir durumdur. Dolayısıyla aktif çıkarım yapan ajan, sürprizi minimize etmek için otomatik olarak şarj istasyonuna yönelecektir.

Eylem seçimi, gelecekteki **Beklenen Serbest Enerji (Expected Free Energy - $G$)** değerinin minimize edilmesiyle belirlenir:

$$ G(\pi) \approx \sum_{\tau} \left( D_{KL}(q(\mathbf{x}_{\tau} \mid \pi) \parallel p(\mathbf{x}_{\tau})) + \mathbb{E}_{q(\mathbf{z}_{\tau} \mid \pi)} [H(q(\mathbf{x}_{\tau} \mid \mathbf{z}_{\tau}, \pi))] \right) $$

Burada:
*   İlk terim **epistemik değer** (epistemic value) veya merak (curiosity) olarak adlandırılır. Ajanı belirsizliği azaltacak eylemler almaya zorlar.
*   İkinci terim **pragmatik değer** (pragmatic value) veya faydadır. Ajanı hedeflediği (tercih ettiği) durumları gerçekleştirecek eylemlere yönlendirir.

Aktif çıkarım, keşif (exploration) ile sömürü (exploitation) arasındaki klasik ikilemi zarif ve birleşik bir matematiksel formülle çözer.

---

## 4. Sınırlı Rasyonellik ve Uç Cihazların Epistemolojisi

Bir AGI ajanı sonsuz hesaplama gücüne sahip değildir. Özellikle Edge AI (uç yapay zeka) sistemlerinde bellek ve işlemci kısıtları, serbest enerjinin hesaplanma hassasiyetini sınırlar. Bu durum, Herbert Simon'ın **Sınırlı Rasyonellik** (Bounded Rationality) teziyle örtüşür.

Arat'ın bilişsel mimarisi, bu kısıtı bir zafiyet değil, tasarım parametresi olarak kabul eder. C++ tabanlı düşük seviyeli bellek yöneticilerimiz ve optimize edilmiş çıkarım motorlarımız, varyasyonel çıkarımın işlem yükünü minimize ederek ajanın milisaniyeler seviyesinde eylem kararı alabilmesini sağlar. Epistemik derinlik, donanım kaynaklarının verimli kullanımıyla dengelenir.

---

*Arat Enterprise - AGI Epistemoloji ve Karar Alma Araştırmaları Grubu*
