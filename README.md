# Arat AGI Research (arat-agi-research)

![Arat AGI Banner](assets/arat_agi_banner.jpg)

[![Arat Enterprise](https://img.shields.io/badge/Arat-Research%20Division-000000?style=for-the-badge&logo=spacex)](#)
[![Python Architecture](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](#)
[![C++ Edge AI](https://img.shields.io/badge/C++-17%2F20-00599C?style=for-the-badge&logo=c%2B%2B)](#)
[![CUDA / HPC](https://img.shields.io/badge/CUDA-11.8%2B-76B900?style=for-the-badge&logo=nvidia)](#)
[![Tests Status](https://img.shields.io/badge/Tests-14%2F14%20Passed-success?style=for-the-badge&logo=github)](#)
[![License](https://img.shields.io/badge/License-Apache_2.0-orange?style=for-the-badge)](#)

> *"Yaratamadığım şeyi anlayamam."* - **Richard Feynman**
>
> *"Zeka, evrenin kendisini anlamlandırma biçimidir. Bizim görevimiz bu süreci hızlandırmaktır."* - **David Deutsch** (The Beginning of Infinity)
>
> *"Gözlemlediğimiz şey doğanın kendisi değil, doğanın bizim soru sorma yöntemimize maruz kalmış halidir."* - **Werner Heisenberg** (Physics and Philosophy)

**Arat AGI Research**, Arat ekosisteminin Yapay Genel Zeka (AGI), otonom bilişsel mimariler, uç sistemler (Edge AI) ve nöro-sembolik hesaplama üzerine yürüttüğü resmi, çok disiplinli araştırma manifestosudur. Bu depo, salt bir kod arşivi olmanın ötesinde; zekanın donanım sınırlarına hapsolmadan, hem dijital hem de fiziksel dünyada otonom olarak var olabilmesi için atılmış kavramsal ve mühendislik adımlarını içerir.

---

## Ⅰ. Bölüm I: Vizyon ve Teknoloji Felsefesi

> *"Makineler düşünebilir mi? Bu soruyu cevaplamak için, 'makine' ve 'düşünmek' kelimelerinin anlamlarını dikkatlice tanımlamamız gerekir."* - **Alan Turing** (Computing Machinery and Intelligence)

Arat olarak, mevcut yapay zeka paradigmasının temel bir sınırlaması olduğuna inanıyoruz: Zeka, devasa veri merkezlerinde hapsolmuş istatistiksel bir olasılık dağılımından ibaret olamaz. Gerçek zeka; çevresiyle etkileşime giren, kaos anında hayatta kalabilen ve kısıtlı donanım kaynaklarıyla optimal kararları alabilen **otonom bir yapı** olmalıdır. 

AGI araştırmalarımız, teknoloji felsefesi ile sistem mimarisi prensiplerinin kesişim noktasında durur. Hedefimiz, trilyonlarca parametreli hantal modeller yaratmak değil, sistem orkestrasyonunu maksimize ederek modüler, çevik ve "kendi varlığının farkında olan" otonom sistemler (Synthetic Nous) inşa etmektir. 

> *"Hayatın şu anki en üzücü yönü, bilimin bilgiyi toplumun bilgelik toplamasından daha hızlı toplamasıdır."* - **Isaac Asimov**

---

## Ⅱ. Bölüm II: Teorik ve Matematiksel Çerçeve

Geliştirdiğimiz otonom bilişsel mimarilerin temelinde, ajanların çevresel belirsizliği (sürprizi) minimize etmesini hedefleyen **Serbest Enerji Prensibi (Free Energy Principle - Karl Friston)**, kuantum olasılık teorisi ve evrensel zeka denklemleri yatmaktadır.

### 1. Varyasyonel Serbest Enerji (Variational Free Energy)
Ajanın (agent) çevresinden aldığı gözlemler $\mathbf{x}$ ve içsel durum temsilleri $\mathbf{z}$ olmak üzere, sistemin minimize etmeye çalıştığı varyasyonel serbest enerji şu şekilde ifade edilir:

$$ F = \mathbb{E}_{q(\mathbf{z})}[\log q(\mathbf{z}) - \log p(\mathbf{x}, \mathbf{z})] $$

$$ F = D_{KL}(q(\mathbf{z}) \parallel p(\mathbf{z})) - \mathbb{E}_{q(\mathbf{z})}[\log p(\mathbf{x} \mid \mathbf{z})] $$

### 2. Beklenen Serbest Enerji (Expected Free Energy) ve Hiyerarşik Yapı
Karar alma döngüsünde eylem seçimi, gelecekteki Beklenen Serbest Enerjinin ($G$) minimize edilmesiyle belirlenir:

$$ G(\pi) \approx D_{KL}(q(\mathbf{x}_{\tau} \mid \pi) \parallel p(\mathbf{x}_{\tau})) + \mathbb{E}_{q(\mathbf{z}_{\tau} \mid \pi)} [H(q(\mathbf{x}_{\tau} \mid \mathbf{z}_{\tau}, \pi))] $$

Depomuzdaki **Hiyerarşik Aktif Çıkarım (Hierarchical Active Inference)** mimarisi, bu süreci iki seviyeli bir şema ile yönetir:
*   **Üst Seviye (Goal/Context Layer):** Görev hedeflerini belirler ve soyut gözlemlerle durum çıkarımı yapar. Buradan seçilen eylemler, alt seviye için duyusal inanç önceliklerini ($C$ matrisi) dinamik olarak günceller.
*   **Alt Seviye (Reactive/Motor Layer):** Üst seviyeden gelen hedefler doğrultusunda fiziksel motor/eylem komutlarını seçer.

### 3. Kuantum Biliş (Quantum Cognition) Entegrasyonu
Zihinsel durum geçişleri ve inanç süperpozisyonları, Hilbert uzayındaki durum vektörleri $|\psi\rangle$ ile ifade edilerek klasik Bayesçi olasılıkların ötesine taşınır. Açık sistem bilişsel süreçleri, gürültülü duyusal uyaranlar altında Lindblad Master Denklemi ile simüle edilir:

$$ \frac{d\rho}{dt} = -i[H, \rho] + \sum_{k} \left( L_k \rho L_k^\dagger - \frac{1}{2} \{L_k^\dagger L_k, \rho\} \right) $$

### 4. Non-Linear Durum Tahmini (Extended Kalman Filter)
Fiziksel dünyadaki uzantımız için geliştirilen EKF sensor fusion algoritması, robotun durumunu $\mathbf{x} = [x, y, \theta]^T$ (konum ve yön) şeklinde takip eder. Doğrusal olmayan durum geçiş modeli $f(x, u)$ ve onun Jacobian matrisi $F_j$ şu şekildedir:

$$ x_{t+1} = x_t + v_t \cos(\theta_t) dt, \quad y_{t+1} = y_t + v_t \sin(\theta_t) dt, \quad \theta_{t+1} = \theta_t + \omega_t dt $$

$$ F_j = \begin{bmatrix} 1 & 0 & -v_t \sin(\theta_t) dt \\ 0 & 1 & v_t \cos(\theta_t) dt \\ 0 & 0 & 1 \end{bmatrix} $$

---

## Ⅲ. Bölüm III: Araştırma Odak Alanları

Bu depo altındaki çalışmalar üç ana mühendislik ve araştırma sütunu üzerinde yükselmektedir:

### 1. Bilişsel Mimari ve Çoklu-Ajan Orkestrasyonu (Cognitive Orchestration)
Büyük dil modellerinin düşünme, hafıza yönetimi (vektörel ve epistemik hafıza) ve hareket etme yeteneklerini modüler bir düzlemde birleştiriyoruz. Tek bir monolitik model yerine, belirli görevlerde uzmanlaşmış alt ajanların `Python` tabanlı mikroservisler aracılığıyla senkronize çalıştığı bir "zihin ağı" kurguluyoruz. 

### 2. Uç Yapay Zeka (Edge AI) ve Donanım Optimizasyonu
Zeka buluta bağımlı olmamalıdır. Yüksek frekanslı karar alması gereken otonom mimariler, gecikmeyi (latency) tolere edemez. Geliştirdiğimiz CUDA/C++ altyapısında, özel **GELU** ve **SwiGLU** aktivasyon fonksiyonlarını GPU üzerinde hızlandırıyor; Arena bellek yöneticileri (Arena Memory Allocators) ile gömülü sistemlerde deterministik çalışma zamanları sağlıyoruz.

### 3. Otonom Sistemler ve Robotik Kontrol Mekanizmaları
İleri mühendislik projelerimizden, yarışma geçmişimizden ve sistem tasarımı vizyonumuzdan edindiğimiz tecrübelerle, AGI'nin fiziksel dünyadaki uzantısını yaratıyoruz. Sensör füzyonu, algı algoritmaları ve kinematik kontrol döngüleri, yapay zekanın "bedenlenmiş" (embodied AI) halini temsil eder.

---

## Ⅳ. Bölüm IV: Sistem Mimarisi ve Teknoloji Yönü

Yüksek performanslı hesaplama (HPC) ve çevik araştırma ortamları arasında mükemmel dengeyi kurmak için hibrit bir dil ve mimari yaklaşım benimsiyoruz:

*   **Algoritma Prototipleme & Orkestrasyon:** `Python 3.10+` (PyTorch, LangChain, ROS2 entegrasyonları, Asenkron orkestrasyon)
*   **Düşük Gecikmeli Çıkarım & Sistem Programlama:** `C++ 17/20` (CUDA, bellek güvenliği, donanım seviyesi kontrol düğümleri, gömülü sistem entegrasyonu)

### 📂 Dizin Yapısı (Repository Topology)

```bash
arat-agi-research/
├── docs/
│   ├── ARAT_MANIFESTO.md         # Kurumsal AI ahlakı ve mühendislik prensipleri
│   ├── epistemology_of_ai.md     # Zekanın doğası üzerine felsefi denemeler
│   ├── system_architecture.md    # OMEGA/AGI çekirdek mimari şemaları
│   └── quantum_cognition_and_constructor_theory.md # Kuantum biliş ve kurucu teorisi felsefesi [NEW]
├── core_cognition/
│   ├── reasoning_engine/         # LLM tabanlı aktif çıkarım algoritmaları (Python)
│   │   ├── active_inference.py   # Klasik ve Hiyerarşik Aktif Çıkarım Ajanları [MODIFIED]
│   │   └── llm_reasoner.py       # LLM entegrasyon arayüzü
│   ├── episodic_memory/          # Vektör veri tabanı ve bellek yönetimi
│   └── agent_orchestration/      # Çoklu-ajan (Multi-agent) iletişim protokolleri
├── edge_inference/
│   ├── cpp_tensor_runtimes/      # Donanıma özel optimize edilmiş çıkarım motorları (C++)
│   ├── memory_allocators/        # Kısıtlı RAM ortamları için özel bellek yöneticileri
│   └── cuda_kernels/             # Özel donanım hızlandırma çekirdekleri (SwiGLU, GELU) [MODIFIED]
├── autonomous_control/
│   ├── sensor_fusion/            # Gerçek zamanlı veri akış işleme (Kalman, EKF)
│   │   ├── fusion_node.py        # 1D Kalman Filtresi düğümü
│   │   └── ekf_fusion_node.py    # 2D Extended Kalman Filtresi düğümü [NEW]
│   └── kinematic_loops/          # Fiziksel eylem ve kontrol mekanizmaları
├── tests/
│   ├── test_cognition.py         # Bilişsel modüller testleri (Hiyerarşik test dahil) [MODIFIED]
│   ├── test_orchestration.py     # Asenkron iletişim testleri
│   ├── test_control.py           # Kontrol ve füzyon testleri
│   └── test_ekf.py               # EKF non-linear tahmin testleri [NEW]
├── assets/
│   └── arat_agi_banner.jpg       # Depo kapak görseli
└── README.md
```

> *"Bir sistemin kalitesi, onu oluşturan parçaların kalitesinden ziyade, bu parçaların birbiriyle nasıl etkileşime girdiğine bağlıdır."* - **Norbert Wiener** (Sibernetiğin Kurucusu)

---

## Ⅴ. Bölüm V: Yol Haritası ve Kritik Metrikler (Roadmap)

* [x] **Aşama 1: Bilişsel Temeller (Q3 2026)**
  * LLM ajanları arasında asenkron iletişim protokollerinin inşası.
  * Epistemik hafıza modülünün vektör veri tabanlarına entegrasyonu.
  * Aktif çıkarım motorunun online Dirichlet öğrenimi ile genişletilmesi.

* [x] **Aşama 2: Edge Optimizasyonu ve EKF Entegrasyonu (Q4 2026)**
  * Karar alma motorunun uç çıkarımları ve Arena Allocator modülünün entegrasyonu.
  * CUDA GELU/SwiGLU aktivasyon çekirdekleri ve TensorRT çalışma zamanı entegrasyonları.
  * Extended Kalman Filter (EKF) ile 2D kinematik durum tahmini entegrasyonu.
  * Hiyerarşik Aktif Çıkarım (Hierarchical Active Inference) sisteminin inşası.

* [ ] **Aşama 3: Bedenlenmiş Zeka (Embodied AI) ve Kuantum Karar Alma (2027 ve Ötesi)**
  * Arat donanım altyapısı ve robotik şasiler ile entegrasyon testleri.
  * Simülasyondan gerçeğe (Sim2Real) pekiştirmeli öğrenme transferlerinin tamamlanması.
  * Lindblad Master Denklemi tabanlı Kuantum Biliş simülatörünün kod tabanına entegre edilmesi.

---

## Ⅵ. Bölüm VI: İletişim, Katkı ve Kurumsal Kimlik

Bu araştırma deposu, **Arat Enterprise** inisiyatiflerinin çekirdek projelerinden biridir. Bilimin ve mühendisliğin kapalı kapılar ardında değil, şeffaf ve işbirliğine açık ortamlarda ilerlediğine inanıyoruz. Teknoloji felsefesi, sistem mimarisi veya otonom yapılar üzerine tartışmak, katkı sağlamak veya araştırma ağımıza dahil olmak için kapımız açıktır.

**Baş Sistem Mimarı ve Geliştirici:** [@arch-yunus](https://github.com/arch-yunus)

**Araştırma Kuruluşu:** Arat Enterprise / Startup Division

**İletişim & Katkı:** (Issue ve Pull Request'ler üzerinden sürece dahil olabilirsiniz)

---

*© 2026 Arat. All rights reserved. Building the mind of tomorrow, today.*
