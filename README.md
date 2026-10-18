# Arat AGI Research (arat-agi-research)

![Arat AGI Banner](assets/arat_agi_banner.jpg)

[![Arat Enterprise](https://img.shields.io/badge/Arat-Research%20Division-000000?style=for-the-badge&logo=spacex)](#)
[![Python Architecture](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](#)
[![C++ Edge AI](https://img.shields.io/badge/C++-17%2F20-00599C?style=for-the-badge&logo=c%2B%2B)](#)
[![CUDA / HPC](https://img.shields.io/badge/CUDA-11.8%2B-76B900?style=for-the-badge&logo=nvidia)](#)
[![Tests Status](https://img.shields.io/badge/Tests-32%2F32%20Passed-success?style=for-the-badge&logo=github)](#)
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

Geliştirdiğimiz otonom bilişsel mimarilerin temelinde, ajanların çevresel belirsizliği (sürprizi) minimize etmesini hedefleyen **Serbest Enerji Prensibi (Free Energy Principle - Karl Friston)**, kuantum olasılık teorisi, David Deutsch'un **Kurucu Kuramı (Constructor Theory)** ve non-linear durum kestirimi yatmaktadır.

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

### 3. Kuantum Biliş (Quantum Cognition) & Lindblad Dinamiği
Klasik Kolmogorov aksiyomlarının (sıra etkileri ve bağlamsallık) yetersiz kaldığı bilişsel belirsizlik anlarında zihinsel durumlar, yoğunluk matrisi $\rho$ ($Tr(\rho)=1$, $\rho=\rho^\dagger$) ile temsil edilir. Açık sistem kuantum bilişi, çevre gürültüsü altında Runge-Kutta 4 (RK4) çözücülü **Lindblad Master Denklemi** ile evrilir:

$$ \frac{d\rho}{dt} = -i[H, \rho] + \sum_{k} \gamma_k \left( L_k \rho L_k^\dagger - \frac{1}{2} \{L_k^\dagger L_k, \rho\} \right) $$

Sistemin belirsizliği **von Neumann Entropisi** ile ölçülür: $S(\rho) = -Tr(\rho \ln \rho)$. Born kuralı ile yapılan izdüşümsel ölçümler ($P(m) = Tr(M_m \rho M_m^\dagger)$), süperpozisyonu kesin bir eyleme çökertir.

### 4. Kurucu Kuramı (Constructor Theory) ile Fiziksel Yapılabilirlik
Görevler girdi-çıktı durum dönüşümleri olarak modellenir: $T = \{ S_{in} \rightarrow S_{out} \}$. Bir kurucu $C$, dönüşümü gerçekleştiren ve döngü sonunda değişmeden kalan (veya restore edilen) bir aktördür ($C \otimes S_{in} \rightarrow C \otimes S_{out}$). Termodinamik kabul edilebilirlik denetimi ($\Delta E \le W$, $\Delta S_{tot} \ge 0$), robotun imkansız eylemlere girişmesini engeller.

### 5. Non-Linear Sensör Füzyonu (EKF & Unscented Kalman Filter - UKF)
Robotun durum tahmini $\mathbf{x} = [x, y, \theta]^T$, analitik Jacobian gerektirmeyen **Scaled Unscented Transform** tabanlı UKF ($2n+1$ sigma noktaları) ve EKF ile çift yönlü çapraz doğrulamayla yürütülür:

$$ \chi_0 = \bar{x}, \quad \chi_i = \bar{x} + \left(\sqrt{(n + \lambda) P}\right)_i, \quad \chi_{i+n} = \bar{x} - \left(\sqrt{(n + \lambda) P}\right)_i $$

### 6. Dinamik Pencere Yaklaşımı (DWA) & Quintic Jerk-Minimizing Yörünge
Engel engelleme ve kinematik kontrol, hız uzayında $V_d$ dinamik penceresini tarayan DWA algoritması ve minimum sarsıntı (jerk) sağlayan 5. derece (quintic) polinom yörünge üreteci ile optimize edilir:

$$ J(v, \omega) = \alpha \cdot \text{heading}(v, \omega) + \beta \cdot \text{dist}(v, \omega) + \gamma \cdot \text{velocity}(v, \omega) $$

---

## Ⅲ. Bölüm III: Araştırma Odak Alanları

1.  **Bilişsel Mimari, Kuantum Karar Alma ve Çoklu-Ajan Orkestrasyonu:** Kuantum süperpozisyon çökmesi, doğrusal ve log-doğrusal (Product of Experts) inanç havuzu birleştirme, Dirichlet öğrenimi ve epizodik-semantik bellek konsolidasyonu.
2.  **Uç Yapay Zeka (Edge AI) ve Donanım Optimizasyonu:** Özel CUDA çekirdekleri (**GELU, SwiGLU, RMSNorm, RoPE**), simetrik INT8 tensör kuantizasyonu, bloklu (tiled) GEMM matris çarpımı ve SIMD hizalı filigran (watermark rollback) destekli **Arena Allocator**.
3.  **Bedenlenmiş Zeka (Embodied AI) ve Otonom Robotik:** EKF/UKF sensör füzyonu, diferansiyel sürüş ileri/ters kinematiği, PID denetleyicileri ve dinamik engelleri aşan DWA yerel yol planlayıcıları.

---

## Ⅳ. Bölüm IV: Sistem Mimarisi ve Teknoloji Yönü

*   **Algoritma Prototipleme & Orkestrasyon:** `Python 3.10+` (PyTorch, NumPy, Asenkron asyncio orkestrasyon)
*   **Düşük Gecikmeli Çıkarım & Sistem Programlama:** `C++ 17/20` (CUDA, bellek güvenliği, donanım seviyesi kontrol düğümleri, gömülü sistem entegrasyonu)

### 📂 Dizin Yapısı (Repository Topology)

```bash
arat-agi-research/
├── docs/
│   ├── ARAT_MANIFESTO.md         # Kurumsal AI ahlakı ve mühendislik prensipleri
│   ├── epistemology_of_ai.md     # Zekanın doğası üzerine epistemolojik denemeler
│   ├── system_architecture.md    # OMEGA/AGI çekirdek mimari ve veri akış şemaları
│   └── quantum_cognition_and_constructor_theory.md # Kuantum biliş ve kurucu teorisi temelleri
├── core_cognition/
│   ├── reasoning_engine/         # FEP tabanlı aktif çıkarım algoritmaları
│   │   ├── active_inference.py   # Klasik & Hiyerarşik Aktif Çıkarım Ajanları
│   │   └── llm_reasoner.py       # LLM entegrasyon arayüzü ve VFE sürpriz hesabı
│   ├── episodic_memory/          # Epizodik, çalışma belleği ve semantik vektör konsolidasyonu
│   ├── agent_orchestration/      # Çoklu-ajan Pub/Sub ve doğrusal / log-doğrusal inanç füzyonu
│   ├── quantum_cognition/        # Kuantum Biliş & Lindblad açık sistem dinamikleri [NEW]
│   │   ├── quantum_state.py      # Hilbert uzayı, yoğunluk matrisi (rho), von Neumann entropisi
│   │   └── quantum_decision_agent.py # Süperpozisyon, karar çökmesi ve sıra etkileri (order effects)
│   └── constructor_theory/       # David Deutsch Kurucu Kuramı görev ağı [NEW]
│       └── task_network.py       # Fiziksel durumlar, termodinamik kabul edilebilirlik, görev planlayıcı
├── edge_inference/
│   ├── cpp_tensor_runtimes/      # INT8 kuantizasyon, Tiled GEMM ve çalışma zamanı (C++)
│   ├── memory_allocators/        # 16-bayt SIMD Arena Allocator ve filigran geri alma (rollback)
│   ├── cuda_kernels/             # CUDA & CPU Fallback çekirdekleri (SwiGLU, GELU, RMSNorm, RoPE)
│   └── edge_demo.cpp             # Bağımsız C++ donanım ve çıkarım doğrulama demosu [NEW]
├── autonomous_control/
│   ├── sensor_fusion/            # Gerçek zamanlı sensör füzyonu
│   │   ├── kalman_filter.py      # Standart ve Genişletilmiş (EKF) Kalman Filtreleri
│   │   ├── ekf_fusion_node.py    # 2D EKF robot durum kestirim düğümü
│   │   └── unscented_kalman_filter.py # Scaled Unscented Transform UKF düğümü [NEW]
│   ├── kinematic_loops/          # Diferansiyel kinematiği ve PID hız/açı denetleyicileri
│   └── trajectory_planning/      # Otonom rota ve hareket planlama [NEW]
│       ├── polynomial_planner.py # 5. derece (Quintic) jerk-minimizing yörünge üreteci
│       └── dwa_planner.py        # Dinamik Pencere Yaklaşımı (DWA) engel sakınma planlayıcısı
├── tests/
│   ├── test_cognition.py         # Bilişsel ve aktif çıkarım testleri
│   ├── test_orchestration.py     # Asenkron iletişim ve inanç füzyon testleri
│   ├── test_control.py           # Kinematik ve 1D Kalman testleri
│   ├── test_ekf.py               # 2D non-linear EKF testleri
│   ├── test_quantum_cognition.py # Kuantum durum, entropi ve Lindblad testleri [NEW]
│   ├── test_constructor_theory.py # Kurucu teorisi görev yapılabilirlik testleri [NEW]
│   ├── test_ukf.py               # UKF sigma noktaları ve non-linear takip testleri [NEW]
│   └── test_trajectory_planning.py # Polinom yörünge ve DWA engel sakınma testleri [NEW]
├── main.py                       # OMEGA/AGI v1.2 Master simülasyon yürütme döngüsü [UPGRADED]
├── CMakeLists.txt                # C++ / CUDA derleme betiği
└── README.md
```

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

* [x] **Aşama 3: Kuantum Karar Alma, Kurucu Teorisi ve UKF/DWA Robotik (Tamamlandı)**
  * Lindblad Master Denklemi tabanlı Kuantum Biliş simülatörü (`QuantumDecisionAgent`) ve Born kuralı izdüşümsel ölçüm entegrasyonu.
  * David Deutsch Kurucu Kuramı (`ConstructorTaskPlanner`) ile fiziksel yapılabilirlik ve termodinamik kısıtların entegrasyonu.
  * Unscented Kalman Filter (`UKFSensorFusionNode`) ile türevsiz hassas non-linear durum takibi.
  * Dynamic Window Approach (`DWA`) ve Quintic Jerk-Minimizing yörünge planlayıcılarının inşası.
  * RMSNorm ve RoPE CUDA/CPU çekirdekleri, INT8 kuantizasyon ve Tiled GEMM eklemeleri.

* [ ] **Aşama 4: Bedenlenmiş Donanım Entegrasyonu & Sim2Real (2027)**
  * Arat fiziksel robotik şasisi üzerinde donanım seviyesi ROS2/DDS köprüsü testleri.
  * Sim2Real pekiştirmeli transfer doğrulama döngüleri.

---

## Ⅵ. Bölüm VI: İletişim, Katkı ve Kurumsal Kimlik

Bu araştırma deposu, **Arat Enterprise** inisiyatiflerinin çekirdek projelerinden biridir. Bilimin ve mühendisliğin kapalı kapılar ardında değil, şeffaf ve işbirliğine açık ortamlarda ilerlediğine inanıyoruz. Teknoloji felsefesi, sistem mimarisi veya otonom yapılar üzerine tartışmak, katkı sağlamak veya araştırma ağımıza dahil olmak için kapımız açıktır.

**Baş Sistem Mimarı ve Geliştirici:** [@arch-yunus](https://github.com/arch-yunus)

**Araştırma Kuruluşu:** Arat Enterprise / Startup Division

**İletişim & Katkı:** (Issue ve Pull Request'ler üzerinden sürece dahil olabilirsiniz)

---

*© 2026 Arat. All rights reserved. Building the mind of tomorrow, today.*
