# AEVUM

**Continuous-Time Event-Driven Neural Speech Codec**

Версия спецификации: 0.1

---

## 1. Основная идея

AEVUM — потоковый нейронный речевой кодек, в котором аудио не представляется последовательностью токенов с фиксированной частотой.

Кодек непрерывно наблюдает входной аудиосигнал и поддерживает внутреннее динамическое состояние. Новый дискретный пакет создаётся только тогда, когда текущее состояние речи достаточно отличается от того, что декодер уже способен предсказать самостоятельно.

Главный принцип:

> Передавать не состояние речи, а новую информацию, которую невозможно достаточно хорошо предсказать из предыдущего состояния.

Таким образом, время и bitrate становятся адаптивными.

Во время стабильного участка речи кодек может долго не передавать новые токены. Во время согласных, атак, переходов между фонемами, резких изменений pitch или других информационно плотных участков события могут возникать значительно чаще.

---

# 2. Цели

AEVUM должен обеспечивать:

* полностью causal streaming;
* отсутствие future lookahead;
* динамическую частоту событий;
* динамическое количество бит на событие;
* continuous-time hidden state;
* низкую алгоритмическую задержку;
* отдельное представление content, prosody и acoustic innovation;
* возможность использования токенов последующей speech language model;
* устойчивость токенов при повторном encode → decode → encode;
* адаптацию temporal resolution к сложности текущего участка речи.

Первоначальный целевой режим:

```text
sample rate:             24 kHz
channels:                mono
observation interval:    10 ms
internal update rate:    100 Hz

average events:          8–20 events/sec
average bitrate:         0.5–1.5 kbps
future lookahead:        0 ms
```

Частота внутренних обновлений 100 Hz не означает частоту токенизации 100 Hz.

Encoder может обновляться 100 раз в секунду, но передать, например, только 10 событий.

---

# 3. Представление времени

Базовая единица наблюдения:

$$
\Delta t = 10ms
$$

При sample rate 24 kHz:

$$
24000 \cdot 0.01 = 240
$$

Каждый streaming step получает:

$$
x_t \in \mathbb{R}^{B \times 1 \times 240}
$$

То есть 240 новых PCM samples.

Важно различать:

$$
observation\ rate \neq token\ rate
$$

Observation rate фиксирован.

Event rate — динамический.

---

# 4. Общая архитектура

```text
PCM 24 kHz
   │
   ▼
Causal Acoustic Frontend
   │
   │ 100 Hz
   ▼
Multi-Timescale Continuous Encoder
   │
   ├── Fast State
   ├── Mid State
   └── Slow State
   │
   ▼
Fused latent z(t)
   │
   ├───────────────┐
   │               │
   ▼               ▼
Predictor      Factorization
   │               │
   ▼               ├── Content
predicted z         ├── Prosody
   │               └── Acoustic residual
   ▼
Innovation
   │
   ▼
Surprise
   │
   ▼
Event Gate
   │
   ├── no event ──────────────┐
   │                          │
   ▼                          │
Quantization                  │
   │                          │
   ▼                          │
Event Packet                  │
   │                          │
   ▼                          │
Continuous Decoder ◄──────────┘
   │
   ▼
100 Hz acoustic representation
   │
   ▼
Causal waveform generator
   │
   ▼
PCM 24 kHz
```

Ключевая особенность: decoder существует и эволюционирует даже тогда, когда новых токенов нет.

---

# 5. Causal Acoustic Frontend

Frontend превращает PCM в компактный acoustic feature vector.

Целевой output rate:

$$
100Hz
$$

Общий stride:

$$
240
$$

Рекомендуемая стартовая конфигурация:

```text
Conv1D
channels: 1 → 64
stride: 5

Conv1D
64 → 128
stride: 4

Conv1D
128 → 192
stride: 3

Conv1D
192 → 256
stride: 2

Conv1D
256 → 384
stride: 2
```

Общий stride:

$$
5\cdot4\cdot3\cdot2\cdot2=240
$$

Output:

$$
f_t\in\mathbb{R}^{B\times384}
$$

Все convolutions должны быть causal.

Padding выполняется только слева.

Рекомендуемые activation:

$$
SiLU
$$

или gated activation:

$$
GLU
$$

Normalization желательно делать независимой от будущего времени:

* RMSNorm;
* LayerNorm по channels;
* WeightNorm.

BatchNorm использовать нежелательно.

---

# 6. Multi-Timescale Continuous Encoder

Вместо одного recurrent state используются три динамических состояния:

$$
h_t^F
$$

$$
h_t^M
$$

$$
h_t^S
$$

где:

**F — fast dynamics**

примерный диапазон:

$$
10-80ms
$$

Отвечает за:

* атаки;
* plosives;
* fricatives;
* быстрые formant transitions;
* transient information.

**M — medium dynamics**

$$
50-500ms
$$

Отвечает за:

* phoneme;
* syllable;
* локальный pitch contour;
* energy envelope.

**S — slow dynamics**

$$
300ms-5000ms
$$

Отвечает за:

* voice identity;
* speaking style;
* долгую prosody;
* room/acoustic context;
* медленные изменения речи.

Рекомендуемый размер:

```text
fast:   192
medium: 192
slow:   192
```

---

# 7. Continuous-Time Cell

Каждая ветвь использует adaptive time constant.

Для состояния:

$$
h_{t-1}
$$

строится candidate:

$$
u_t =
\tanh(
W_xx_t+
W_hh_{t-1}+b
)
$$

Time constant вычисляется динамически:

$$
\tau_t =
\tau_{min}
+
(\tau_{max}-\tau_{min})
\sigma(
W_\tau[x_t;h_{t-1}]+b_\tau
)
$$

Decay:

$$
\alpha_t =
e^{-\frac{\Delta t}{\tau_t}}
$$

Обновление:

$$
h_t =
\alpha_t h_{t-1}
+
(1-\alpha_t)u_t
$$

Это основная continuous-time операция AEVUM.

Она имеет важное свойство.

Если:

$$
\tau_t \gg \Delta t
$$

то:

$$
h_t \approx h_{t-1}
$$

Состояние меняется медленно.

Если:

$$
\tau_t \approx \Delta t
$$

состояние быстро реагирует на новый сигнал.

---

# 8. Adaptive temporal resolution

Time constant должна зависеть не только от входа, но и от неожиданности происходящего.

Можно добавить modulation:

$$
\tau'_t=
\frac{\tau_t}
{1+\beta s_t}
$$

где:

$$
s_t
$$

— surprise.

При высокой неожиданности:

$$
s_t\uparrow
$$

следовательно:

$$
\tau'_t\downarrow
$$

и сеть временно становится быстрее.

При стабильной речи:

$$
s_t\downarrow
$$

$$
\tau'_t\uparrow
$$

состояние становится инерционным.

На v0 это необязательно включать сразу.

Рекомендую сначала обучить обычные adaptive \(\tau\), а surprise modulation добавить отдельным ablation.

---

# 9. Cross-timescale communication

Fast, medium и slow ветви не должны быть полностью независимыми.

Для fast:

$$
u_t^F=
\phi(
W_Ff_t+
U_Fh_{t-1}^F+
C_{MF}h_{t-1}^M
)
$$

Для medium:

$$
u_t^M=
\phi(
W_Mf_t+
U_Mh_{t-1}^M+
C_{FM}h_{t-1}^F+
C_{SM}h_{t-1}^S
)
$$

Для slow:

$$
u_t^S=
\phi(
W_Sf_t+
U_Sh_{t-1}^S+
C_{MS}h_{t-1}^M
)
$$

Cross connections желательно делать небольшими projection layers, а не full dense matrices.

---

# 10. Fused latent

Состояния объединяются:

$$
h_t =
[h_t^F;h_t^M;h_t^S]
$$

Размер:

$$
576
$$

Далее:

$$
z_t=
W_z RMSNorm(h_t)
$$

Рекомендуем:

$$
z_t\in\mathbb{R}^{512}
$$

Это основное непрерывное representation encoder.

---

# 11. Decoder-side prediction

Receiver должен уметь предсказывать дальнейшее развитие сигнала без новых токенов.

Поэтому encoder во время работы поддерживает локальную копию decoder state.

Пусть:

$$
d_t
$$

— состояние decoder непосредственно перед обработкой нового event.

Predictor вычисляет:

$$
\hat z_t=P(d_t)
$$

где:

$$
P:\mathbb{R}^{D_d}\rightarrow\mathbb{R}^{512}
$$

Prediction error:

$$
e_t=z_t-\hat z_t
$$

Именно:

$$
e_t
$$

является главным кандидатом на передачу.

Необходимо избегать ситуации, когда encoder просто начинает искусственно делать \(z_t\) легко предсказуемым.

Поэтому для surprise branch рекомендуется:

$$
e_t^{gate}=
stopgrad(z_t)-\hat z_t
$$

Decoder predictor учится догонять encoder representation, но event gate не заставляет encoder уничтожать информацию ради уменьшения количества событий.

---

# 12. Prediction uncertainty

Predictor должен оценивать не только ожидаемый latent, но и uncertainty.

Он выдаёт:

$$
(\hat z_t,\sigma_t)
$$

где:

$$
\sigma_t=
softplus(W_\sigma d_t)+\epsilon
$$

Тогда normalized surprise:

$$
s_t=
\frac{1}{D}
\sum_i
\left(
\frac{z_{t,i}-\hat z_{t,i}}
{\sigma_{t,i}+\epsilon}
\right)^2
$$

Это лучше простого:

$$
||z-\hat z||^2
$$

потому что модель способна учитывать естественную неопределённость разных latent dimensions.

---

# 13. Event Gate

Event gate отвечает на вопрос:

> Нужно ли decoder сейчас получить новую информацию?

Вход:

$$
q_t =
[
e_t;
s_t;
h_t^F;
h_t^M;
h_t^S;
age_t
]
$$

где:

$$
age_t
$$

— время с предыдущего события.

Logit:

$$
a_t=MLP(q_t)
$$

Вероятность:

$$
p_t=\sigma(a_t)
$$

Во время inference:

$$
event_t =
p_t > \theta
$$

Начальное:

$$
\theta=0.5
$$

---

# 14. Ограничение максимального gap

Decoder не должен бесконечно жить без синхронизации.

Устанавливаем:

$$
T_{max}=250ms
$$

Если:

$$
age_t\ge T_{max}
$$

event принудительно создаётся независимо от gate.

Это является safety mechanism, а не основным механизмом токенизации.

В дальнейшем \(T_{max}\) можно увеличить до 500–1000 ms.

---

# 15. Differentiable Event Gate

Во время обучения обычный boolean event нарушает backpropagation.

Используем Straight-Through Gumbel-Sigmoid.

$$
g\sim Gumbel(0,1)
$$

$$
p_t^{soft}
=
\sigma
\left(
\frac{a_t+g}{T_g}
\right)
$$

Hard decision:

$$
p_t^{hard}
=
\mathbb{1}[p_t^{soft}>0.5]
$$

Straight-through value:

$$
p_t^{ST}
=
p_t^{soft}
+
stopgrad(
p_t^{hard}-p_t^{soft}
)
$$

Forward использует hard decision.

Backward использует gradient soft decision.

Начальная temperature:

$$
T_g=1.0
$$

К концу training:

$$
T_g\rightarrow0.1
$$

---

# 16. Factorized Representation

Каждое событие раскладывается на несколько логически разных компонентов.

AEVUM использует:

```text
ΔT — temporal distance
C  — linguistic/content information
P  — prosody
R  — unpredictable acoustic residual
```

Дополнительно можно использовать редкий:

```text
G — global acoustic context
```

---

# 17. Content representation

Content должен преимущественно кодировать:

* phonetic identity;
* speech units;
* linguistic information;
* локальные переходы между phonetic states.

Content head:

$$
c_t=W_cz_t
$$

Например:

$$
c_t\in\mathbb{R}^{128}
$$

Vector Quantizer:

```text
codebook size: 4096
embedding dim: 128
```

Получаем:

$$
C_t\in\{0,\ldots,4095\}
$$

Стоимость без entropy coding:

$$
12bits
$$

Content representation должно быть максимально устойчивым к:

* amplitude;
* microphone;
* небольшому noise;
* room response;
* speaker identity.

---

# 18. Content invariance

Берём один speech segment и создаём две версии:

$$
x
$$

и:

$$
x'
=
Augment(x)
$$

Допустимые augmentations:

* gain;
* небольшой EQ;
* mild noise;
* room impulse response;
* codec artifacts;
* small resampling perturbation.

Encoder получает:

$$
c
$$

и:

$$
c'
$$

Loss:

$$
L_{content-consistency}
=
1-
cos(c,c')
$$

Можно добавить внешний speech representation teacher, однако архитектура не должна принципиально зависеть от него.

---

# 19. Prosody representation

Prosody head:

$$
p_t=W_pz_t
$$

Например:

$$
p_t\in\mathbb{R}^{64}
$$

Codebook:

```text
512 entries
64 dimensions
```

Token:

$$
P_t\in[0,511]
$$

9 bits.

Prosody должна хранить преимущественно:

* F0;
* voicing;
* pitch movement;
* loudness;
* stress;
* rhythm;
* spectral tilt;
* voice dynamics.

Можно использовать auxiliary targets:

$$
\log F0
$$

$$
V/UV
$$

$$
RMS
$$

$$
spectral\ tilt
$$

---

# 20. Acoustic Innovation

После удаления предсказуемого состояния остаётся:

$$
e_t=z_t-\hat z_t
$$

Создаём projection:

$$
r_t=W_re_t
$$

Например:

$$
r_t\in\mathbb{R}^{256}
$$

Далее применяется residual quantization.

Но глубина quantization является переменной.

---

# 21. Variable Residual Depth

Используем максимум:

$$
K_{max}=4
$$

residual codebooks.

Каждый:

```text
1024 entries
256 dimensions
```

Первый quantizer:

$$
q_1=Q_1(r_t)
$$

Residual:

$$
r_t^{(1)}=r_t-q_1
$$

Второй:

$$
q_2=Q_2(r_t^{(1)})
$$

и так далее.

Но encoder дополнительно предсказывает:

$$
K_t\in\{0,1,2,3,4\}
$$

То есть на простом событии можно передать:

```text
C
P
```

а residual вообще отсутствует.

На сложном:

```text
C
P
R1
R2
R3
R4
```

---

# 22. Depth Gate

Depth predictor:

$$
l_t=W_K[e_t;s_t;h_t]
$$

где:

$$
l_t\in\mathbb{R}^5
$$

и:

$$
P(K_t=k)=softmax(l_t)
$$

Во время training использовать Straight-Through Gumbel-Softmax.

Rate loss должен учитывать expected depth:

$$
E[K_t]
=
\sum_{k=0}^{4}
kP(K_t=k)
$$

Таким образом модель сама учится решать, сколько информации стоит потратить на текущий event.

---

# 23. Temporal token

Каждое событие хранит время после предыдущего события.

$$
\Delta T_t
$$

Для v0 достаточно quantization:

$$
10ms
$$

Диапазон:

```text
1 ... 25
```

соответствует:

```text
10 ... 250 ms
```

5 bits.

Важно:

online decoder знает реальное wall-clock время и не обязан использовать этот token непосредственно.

\(\Delta T\) нужен прежде всего:

* для хранения токенов;
* offline reconstruction;
* language modeling;
* replay;
* пакетной передачи.

---

# 24. Global context token

Опциональный token:

$$
G_t
$$

используется для очень медленно меняющейся информации:

* timbre;
* speaker;
* microphone;
* room;
* overall voice style.

Codebook:

```text
1024 entries
128 dimensions
```

Token стоит:

$$
10bits
$$

Он должен обновляться редко.

Например:

```text
initial event
+
при большом изменении slow-state
+
не чаще 1 раза в 500 ms
```

Это позволяет не заставлять каждый acoustic residual повторно кодировать speaker identity.

---

# 25. Event Packet

Базовое событие:

```text
Event {
    dt
    content
    prosody
    depth
    residual[depth]
}
```

Расширенное:

```text
Event {
    dt

    optional global_context

    content
    prosody

    residual_depth

    residual_1
    residual_2
    residual_3
    residual_4
}
```

Пример битовой стоимости:

```text
dt:        5 bits
content:  12 bits
prosody:   9 bits
depth:     3 bits

base:     29 bits

each residual:
          10 bits
```

При среднем:

```text
10 events/sec
1.5 residual/event
```

получаем:

$$
(29+15)\cdot10
=
440bps
$$

без entropy coding и transport overhead.

Это не обязательный конечный bitrate, а удобный ориентир.

---

# 26. Continuous Decoder

Decoder не должен быть системой вида:

```text
token → frame
token → frame
token → frame
```

Он обязан существовать между токенами.

Decoder имеет свои состояния:

$$
d_t^F,d_t^M,d_t^S
$$

аналогичные encoder states.

На каждом 10 ms step:

$$
d_t=F(d_{t-1},u_t,\Delta t)
$$

Если события нет:

$$
u_t=0
$$

но:

$$
d_t\neq d_{t-1}
$$

Состояние продолжает естественно эволюционировать.

Если event есть:

$$
u_t=
E_C(C_t)
+
E_P(P_t)
+
E_G(G_t)
+
\sum_iE_{R_i}(R_{t,i})
$$

---

# 27. Decoder update

Для каждой temporal branch:

$$
v_t=
\phi(
W_ud_t^{event}
+
W_hd_{t-1}
)
$$

Adaptive time constant:

$$
\tau_t^D=
\tau_{min}
+
(\tau_{max}-\tau_{min})
\sigma(W_\tau[d_{t-1};u_t])
$$

Decay:

$$
\alpha_t^D=
e^{-\Delta t/\tau_t^D}
$$

State:

$$
d_t=
\alpha_t^Dd_{t-1}
+
(1-\alpha_t^D)v_t
$$

При отсутствии event модель продолжает trajectory на основе предыдущего состояния.

---

# 28. Decoder latent

Из decoder state:

$$
y_t=
W_y[d_t^F;d_t^M;d_t^S]
$$

Размер:

$$
y_t\in\mathbb{R}^{384}
$$

Это 100 Hz acoustic representation, которое передаётся waveform generator.

---

# 29. Waveform Generator

Для v0 использовать causal neural generator.

Input:

$$
[B,384,T_{100Hz}]
$$

Output:

$$
[B,1,T_{24kHz}]
$$

Upsampling:

```text
×2
×2
×3
×4
×5
```

Итого:

$$
240
$$

Можно использовать:

* causal transposed convolutions;
* nearest-neighbor upsampling + causal Conv1D;
* residual convolutional blocks.

Для первой реализации я рекомендую:

```text
nearest-neighbor / repeat upsampling
+
causal residual conv
```

Это проще корректно стримить, чем ConvTranspose.

---

# 30. Closed-loop training

Критически важно:

encoder во время обучения не должен вычислять event относительно идеального ground truth decoder state.

Нужно запускать настоящий decoder в loop.

То есть:

```text
encoder observes audio
      │
      ▼
event decision
      │
      ▼
quantization
      │
      ▼
decoder receives exactly transmitted information
      │
      ▼
decoder evolves
      │
      ▼
decoder state returns to predictor
```

Именно этот state используется для:

$$
\hat z_t
$$

Иначе training и inference будут фундаментально различаться.

---

# 31. Reconstruction losses

Основной reconstruction loss:

$$
L_{recon}
=
\lambda_{wav}L_{wav}
+
\lambda_{mel}L_{mel}
+
\lambda_{stft}L_{stft}
$$

### Waveform

$$
L_{wav}
=
|x-\hat x|_1
$$

### Mel

$$
L_{mel}
=
|Mel(x)-Mel(\hat x)|_1
$$

### Multi-resolution STFT

Для нескольких FFT resolutions:

$$
L_{STFT}
=
\sum_j
\left(
L_{spectral-convergence}^{(j)}
+
L_{log-mag}^{(j)}
\right)
$$

Например:

```text
FFT:
256
512
1024
2048
```

---

# 32. Predictor loss

$$
L_{pred}
=
|
stopgrad(z_t)-\hat z_t
|_1
$$

Можно дополнительно использовать Gaussian NLL:

$$
L_{predNLL}
=
\sum_i
\left[
\frac{
(z_i-\hat z_i)^2
}{
2\sigma_i^2
}
+
\log\sigma_i
\right]
$$

Этот вариант предпочтительнее, если uncertainty используется для surprise.

---

# 33. VQ Loss

Для каждого codebook:

$$
L_{VQ}
=
||
stopgrad(z)-q
||^2
+
\beta
||
z-stopgrad(q)
||^2
$$

Рекомендуем:

$$
\beta=0.25
$$

Codebooks можно обновлять:

* gradient descent;
* EMA.

Для начала рекомендую EMA.

---

# 34. Codebook anti-collapse loss

Следить за распределением использования кодов.

Пусть:

$$
p_i
$$

— empirical probability code \(i\).

Entropy:

$$
H=-\sum_ip_i\log p_i
$$

Можно оптимизировать:

$$
L_{usage}
=
-H
$$

или penalize divergence от uniform:

$$
L_{usage}
=
D_{KL}(p||U)
$$

Не нужно заставлять распределение быть идеально uniform, но сильный collapse должен штрафоваться.

---

# 35. Rate loss

Наша основная optimization target:

$$
Rate
=
\frac{
\sum_t B_t
}{
duration
}
$$

где:

$$
B_t
$$

— expected количество переданных bits.

Для каждого observation step:

$$
E[B_t]
=
p(event_t)
\left(
B_{base}
+
10E[K_t]
\right)
$$

Целевой bitrate:

$$
R^*
$$

Например:

$$
R^*=1000bps
$$

Вместо фиксированного коэффициента лучше использовать Lagrangian constraint:

$$
L=
L_{quality}
+
\lambda_R(R-R^*)
$$

А \(\lambda_R\) обновлять отдельно:

$$
\lambda_R
\leftarrow
max(
0,
\lambda_R+\eta(R-R^*)
)
$$

Если bitrate слишком высокий, стоимость event автоматически растёт.

Если bitrate ниже target, pressure ослабляется.

---

# 36. Token Cycle Consistency

Одна из главных целей AEVUM — стабильные discrete representations.

Берём:

$$
x
$$

Encoder:

$$
T=E(x)
$$

Decoder:

$$
\hat x=D(T)
$$

Повторный encoder:

$$
\hat T=E(\hat x)
$$

Content consistency:

$$
L_{cycle-C}
=
CE(C,\hat C)
$$

Prosody:

$$
L_{cycle-P}
=
CE(P,\hat P)
$$

Event timing:

$$
L_{cycle-time}
=
distance(
\Delta T,\widehat{\Delta T}
)
$$

Не обязательно заставлять residual tokens быть абсолютно идентичными.

Высокая стабильность важнее прежде всего для:

* content;
* prosody;
* event boundaries.

---

# 37. Temporal jitter consistency

Создать:

$$
x'
$$

как тот же сигнал, смещённый, например, на:

```text
±5 ms
±10 ms
±20 ms
```

Content events должны оставаться семантически близкими.

Event timestamps могут немного сдвинуться, но последовательность content units не должна полностью меняться.

Это особенно важно для будущего language model.

---

# 38. Prosody auxiliary loss

Из decoder prosody representation прогнозировать:

$$
F0
$$

$$
VUV
$$

$$
Energy
$$

Например:

$$
L_{prosody}
=
\lambda_fL_{F0}
+
\lambda_v BCE(VUV)
+
\lambda_eL_{energy}
$$

Prosody head не обязан идеально воспроизводить эти параметры.

Это inductive bias.

---

# 39. Adversarial training

GAN не стоит подключать в начале.

Сначала codec должен научиться:

* реконструкции;
* event selection;
* prediction;
* quantization.

После стабилизации добавить:

* multi-period discriminator;
* multi-scale waveform discriminator.

И:

$$
L_G
$$

$$
L_D
$$

Плюс feature matching:

$$
L_{FM}
$$

GAN используется для texture fidelity, а не как основной источник learning signal.

---

# 40. Полный loss

Финальная форма примерно:

$$
L=
L_{recon}
+
\lambda_{pred}L_{pred}
+
\lambda_{vq}L_{VQ}
+
\lambda_{content}L_{content}
+
\lambda_{prosody}L_{prosody}
+
\lambda_{cycle}L_{cycle}
+
\lambda_{usage}L_{usage}
+
\lambda_RL_{rate}
+
\lambda_{GAN}L_G
$$

Начальные коэффициенты:

```text
reconstruction:       1.0

prediction:           0.1
VQ:                   0.25

content consistency:  0.1
prosody:              0.1
cycle:                0.1

usage:                0.01

GAN:
0 initially
```

Rate pressure вводить постепенно.

---

# 41. Training curriculum

## Stage 1 — Continuous Autoencoder

Event gate отключён.

Event происходит каждые 10 ms.

Residual depth:

$$
K=4
$$

Цель:

> получить стабильный causal speech autoencoder.

На этом этапе нет необходимости экономить bitrate.

---

## Stage 2 — Quantized Autoencoder

Активировать:

* content VQ;
* prosody VQ;
* residual VQ.

Но events всё ещё dense.

Цель:

> добиться хорошей реконструкции через discrete bottleneck.

---

## Stage 3 — Predictor

Включить decoder-side predictor.

Обучить:

$$
\hat z_t\approx z_t
$$

Events пока можно оставить dense.

Цель:

> decoder должен научиться понимать temporal trajectory.

---

## Stage 4 — Event Sparsification

Включить Event Gate.

Начальный bitrate target сделать лёгким:

```text
3–4 kbps
```

После стабилизации постепенно уменьшать:

```text
3 kbps
2 kbps
1.5 kbps
1 kbps
```

Не нужно сразу загонять модель в 500 bps.

---

## Stage 5 — Variable Residual Depth

Включить:

$$
K_t\in0..4
$$

Rate constraint теперь действует одновременно на:

* число events;
* количество residual tokens.

---

## Stage 6 — Long Temporal Context

Начальные segments могут быть:

```text
2–4 sec
```

Потом:

```text
10 sec
30 sec
```

Slow dynamics бессмысленно нормально обучать только на коротких fragments.

---

## Stage 7 — Perceptual Refinement

Подключить GAN.

Основную архитектуру на этом этапе желательно уже не менять.

---

## Stage 8 — Stability Training

Добавить:

* token cycle consistency;
* temporal jitter;
* noise consistency;
* encode/decode/re-encode.

---

# 42. Streaming forward pass

На каждом 10 ms step:

```text
1. Получить 240 PCM samples.

2. Frontend:
   f_t = frontend(chunk)

3. Обновить:
   h_fast
   h_mid
   h_slow

4. Получить:
   z_t

5. Эволюционировать локальный decoder state.

6. Predictor:
   z_hat_t, sigma_t

7. Innovation:
   e_t = z_t - z_hat_t

8. Surprise:
   s_t

9. Event Gate:
   emit / no emit

10. Если event:

    content = quantize(content_head(z_t))
    prosody = quantize(prosody_head(z_t))

    K = depth_gate(...)
    residuals = quantize(e_t, K)

    packet = {
        dt,
        content,
        prosody,
        K,
        residuals
    }

11. Передать packet локальному decoder.

12. Decoder обновляет state.

13. Decoder выдаёт 100 Hz latent.

14. Waveform generator генерирует следующие 240 PCM samples.
```

---

# 43. Decoder при отсутствии события

Это принципиальный случай.

Допустим:

```text
t0: EVENT
t1: -
t2: -
t3: -
t4: -
t5: EVENT
```

Decoder не должен делать:

```text
repeat last frame
```

Он должен вычислять:

$$
d_{t+1}=F(d_t,0,\Delta t)
$$

пять раз.

То есть сигнал между событиями является следствием внутренней dynamics decoder, а не interpolation токенов.

---

# 44. Initial state

При начале нового stream:

$$
h_0=0
$$

$$
d_0=0
$$

Но первый event должен быть принудительным.

Для первых примерно:

```text
30–50 ms
```

можно временно разрешить dense events.

Это bootstrap window.

Например:

```text
0 ms   event
10 ms  event
20 ms  event
30 ms  normal event gate begins
```

Это поможет decoder быстро получить:

* voice;
* pitch;
* acoustic context.

---

# 45. Module structure

Рекомендуемая структура проекта:

```text
aevum/

    config.py

    frontend.py

    dynamics/
        cell.py
        multiscale.py

    encoder.py

    predictor.py
    surprise.py
    event_gate.py

    quantization/
        vq.py
        content.py
        prosody.py
        residual.py
        depth_gate.py

    decoder.py
    generator.py

    packet.py

    streaming/
        encoder_stream.py
        decoder_stream.py
        state.py

    losses/
        reconstruction.py
        rate.py
        consistency.py
        prosody.py
        adversarial.py

    metrics/
        bitrate.py
        token_stability.py
        speech_quality.py

    train.py
    inference.py
```

---

# 46. Основные tensor shapes

При batch training:

```text
waveform

[B, 1, samples]
```

После frontend:

```text
features

[B, T, 384]
```

Temporal states:

```text
fast

[B, T, 192]

mid

[B, T, 192]

slow

[B, T, 192]
```

Fused:

```text
[B, T, 576]
```

Latent:

```text
z

[B, T, 512]
```

Content:

```text
[B, T, 128]
```

Prosody:

```text
[B, T, 64]
```

Innovation:

```text
[B, T, 512]
```

Residual projection:

```text
[B, T, 256]
```

Gate:

```text
[B, T, 1]
```

Depth logits:

```text
[B, T, 5]
```

---

# 47. State API

Streaming state желательно сделать explicit.

Пример логической структуры:

```text
EncoderState {
    frontend_buffers

    h_fast
    h_mid
    h_slow

    decoder_shadow_state

    time_since_event
}
```

Decoder:

```text
DecoderState {
    h_fast
    h_mid
    h_slow

    generator_buffers

    absolute_time
}
```

Не хранить streaming state внутри PyTorch modules как скрытые mutable variables.

Передавать state явно.

Это сильно упростит:

* batching;
* tests;
* server inference;
* resets;
* exporting;
* parallel streams.

---

# 48. Codec profiles

Можно сразу архитектурно предусмотреть несколько профилей.

### AEVUM-L

Low bitrate.

```text
target:
0.4–0.8 kbps
```

### AEVUM-M

Balanced.

```text
0.8–1.5 kbps
```

### AEVUM-H

High fidelity.

```text
1.5–3 kbps
```

Желательно использовать одну модель и менять rate target/control variable.

Можно передавать encoder:

$$
r_{target}
$$

как conditioning.

---

# 49. Rate conditioning

Добавить bitrate embedding:

$$
e_R=Embedding(rate\ profile)
$$

и передавать его:

* Event Gate;
* Depth Gate;
* возможно decoder.

Тогда одна модель сможет работать в разных режимах.

Например:

```text
profile 0:
aggressive compression

profile 1:
balanced

profile 2:
quality
```

В будущем можно сделать continuous bitrate conditioning.

---

# 50. Метрики

Оценивать нужно не только качество waveform.

## Audio quality

* SI-SDR;
* STOI;
* PESQ;
* ViSQOL;
* multi-resolution spectral distance.

## Speech preservation

Прогнать ASR на:

```text
original
```

и:

```text
decoded
```

Сравнить WER.

## Speaker preservation

Speaker embedding similarity:

$$
cos(E_{spk}(x),E_{spk}(\hat x))
$$

## Prosody

* F0 RMSE;
* F0 correlation;
* voiced/unvoiced accuracy;
* energy correlation.

## Compression

* average bits/sec;
* events/sec;
* residuals/event;
* p50/p95 event gap.

## Token stability

Encode:

$$
T_1=E(x)
$$

Decode:

$$
\hat x=D(T_1)
$$

Re-encode:

$$
T_2=E(\hat x)
$$

Измерять:

* content token match;
* prosody token match;
* event timing deviation.

---

# 51. Очень важная diagnostic metric

Измерять связь между event emission и реальной acoustic information density.

Например сравнивать events с:

* spectral flux;
* phoneme boundaries;
* pitch changes;
* voice onset;
* plosives;
* consonant transitions.

Хотим увидеть примерно:

```text
stable vowel:

────────────●─────────────●────


rapid phonetic transition:

────●─●─●─●────
```

Если events остаются почти периодическими, значит модель фактически не научилась event-driven representation.

---

# 52. Ablation experiments

Обязательные сравнения:

```text
fixed τ
vs
adaptive τ
```

```text
single timescale
vs
fast/mid/slow
```

```text
fixed events
vs
learned events
```

```text
fixed residual depth
vs
variable depth
```

```text
raw latent transmission
vs
prediction residual
```

```text
without content/prosody separation
vs
factorized representation
```

```text
without cycle loss
vs
with cycle loss
```

Это позволит понять, где находится реальное улучшение.

---

# 53. Failure modes

### Gate collapse: always event

Причина:

reconstruction выигрывает от дополнительных событий.

Решение:

усиливать bitrate constraint постепенно.

---

### Gate collapse: never event

Причина:

rate pressure введён слишком быстро.

Решение:

curriculum.

---

### Codebook collapse

Решение:

EMA VQ, entropy monitoring, dead-code reset.

---

### Content содержит speaker identity

Решение:

augmentation consistency.

В дальнейшем можно добавить gradient reversal speaker classifier.

---

### Prosody захватывает lexical content

Решение:

уменьшить capacity prosody codebook и добавить соответствующие auxiliary objectives.

---

### Decoder дрейфует между events

Решение:

prediction training, max event gap, более сильный continuous decoder.

---

### Все event intervals становятся почти одинаковыми

Это означает, что архитектура превратилась в скрытый fixed-rate codec.

Проверять histogram:

$$
P(\Delta T)
$$

Он должен иметь значимую variance.

---

# 54. Что не делать в первой версии

Не нужно сразу добавлять:

* giant Transformer;
* diffusion decoder;
* autoregressive waveform generation;
* сложный entropy model;
* speaker cloning;
* explicit emotion model;
* text supervision;
* online plasticity;
* adversarial disentanglement;
* огромный набор auxiliary losses.

Главная гипотеза первой версии всего одна:

> Может ли decoder поддерживать качественную речевую trajectory между нерегулярными событиями, получая только prediction innovations?

Пока этот вопрос не доказан, всё остальное вторично.

---

# 55. Минимальный эксперимент AEVUM-v0

Если нужно максимально быстро проверить саму идею, оставить только:

```text
24 kHz PCM

↓

causal frontend
100 Hz

↓

fast/mid/slow continuous encoder

↓

512 latent

↓

predictor

↓

innovation

↓

event gate

↓

single VQ:
4096 codes

↓

continuous decoder

↓

causal waveform generator
```

Пока без:

* content/prosody separation;
* variable residual depth;
* global token.

То есть event содержит всего:

```text
ΔT
Q(innovation)
```

Сначала ответить на вопрос:

> Может ли нерегулярный event stream вообще эффективно управлять continuous decoder?

Если да — переходить к полной factorized архитектуре.

---

# 56. Порядок реализации

Рациональный порядок:

```text
1. Streaming causal frontend.

2. Continuous-time cell.

3. Multi-timescale encoder.

4. Continuous decoder.

5. Waveform generator.

6. Dense autoencoder.

7. VQ.

8. Predictor.

9. Innovation coding.

10. Event gate.

11. Rate loss.

12. Variable event frequency.

13. Content/prosody factorization.

14. Variable residual depth.

15. Global slow context.

16. Cycle consistency.

17. Adversarial refinement.
```

Не стоит писать всю систему сразу.

---

# 57. Главная математическая модель

В наиболее компактном виде AEVUM можно описать следующим образом.

Encoder dynamics:

$$
\frac{dh(t)}{dt}
=
F(h(t),x(t),\tau(t))
$$

Latent:

$$
z(t)=G(h(t))
$$

Decoder prediction:

$$
\hat z(t)=P(d(t))
$$

Innovation:

$$
e(t)=z(t)-\hat z(t)
$$

Surprise:

$$
s(t)=
\mathbb{E}
\left[
\frac{e(t)^2}{\sigma(t)^2}
\right]
$$

Event decision:

$$
g(t)
\sim
Bernoulli(
\pi(s(t),h(t),age(t))
)
$$

Передаваемая информация:

$$
q(t)=Q(e(t))
$$

только если:

$$
g(t)=1
$$

Decoder dynamics:

$$
\frac{dd(t)}{dt}
=
D(d(t),q(t))
$$

Если нового события нет:

$$
q(t)=0
$$

но:

$$
\frac{dd(t)}{dt}\neq0
$$

Waveform:

$$
\hat x(t)=V(d(t))
$$

Оптимизация:

$$
\min
D(x,\hat x)
+
\lambda R
$$

где:

$$
D
$$

— perceptual distortion,

а:

$$
R
$$

— количество информации, переданной за секунду.

---

# 58. Исследовательская гипотеза AEVUM

Классический вопрос neural codec:

> Как эффективно дискретизировать каждый временной frame?

AEVUM ставит другой вопрос:

> Когда вообще появилась новая информация, которую необходимо дискретизировать?

То есть compression происходит не только в пространстве значений.

Она происходит одновременно:

$$
value
$$

$$
time
$$

$$
depth
$$

AEVUM адаптивно выбирает:

**когда передавать информацию,**

**какую информацию передавать,**

**сколько информации передавать.**

Именно эти три свойства являются центральной архитектурной идеей проекта.
