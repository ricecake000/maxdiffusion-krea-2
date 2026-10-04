# Krea 2 배치-1 추론 커널 설계 (TPU v5e / v6e)

작성일: 2026-09-26. 대상: `krea/Krea-2-Turbo` (8 스텝, CFG 없음) 배치 1 텍스트→이미지.
근거 데이터: `artifacts/krea2_2k_xprof` (v5e-4, TP=4, 2048², 커밋 `1a05a72` 기준 코드).

## 1. 현재 구현 요약

| 항목 | 현재 |
|---|---|
| 모델 | 12.9B 단일 스트림 MMDiT, 28 블록, hidden 6144 = 48 heads × 128, GQA 12 kv heads, SwiGLU 16384, adaRMSNorm(공유 temb_mod + 블록별 테이블), 어텐션 출력 sigmoid 게이트 |
| 시퀀스 | `[text 512 | image (H/16)(W/16)]`. 텍스트는 항상 512로 고정, 중간 패딩을 마스크로 처리. 1024² → 4608 토큰, 2048² → 16896 토큰 |
| 병렬화 | 배치 1이면 `generate_krea2.py`가 슬라이스 내 모든 칩에 TP 자동 설정 (`heads`,`mlp` → tensor 축). 가중치 bf16 26 GB → v5e-4 기준 칩당 6.5 GB |
| 어텐션 | `attention: flash` → JAX 내장 splash (`splash_mha_fwd_segmented`), bq=bkv=bkv_compute=1024, FullMask + segment id(텍스트 패딩 마스크), GQA는 커널 진입 전 `jnp.repeat`로 48 heads로 복제, RoPE는 f32 interleaved-pair 레이아웃, 시퀀스는 1024 배수로 패딩(16896 → 17408) |
| 실행 구조 | 스텝마다 prelude(text_fusion, txt_in, img_in, temb) → 블록 28회(staged면 블록별 jit, 아니면 단일 jit) → final. AOT 캐시로 재컴파일 없음 |
| 실측 | v5e-4 2048²: 디노이즈 10.40 s = 1.30 s/스텝 = 46.06 ms/블록. HBM 피크 15.4 / 15.75 GiB |

이 저장소에는 이미 Wan용으로 튠된 커스텀 Pallas 커널(`kernels/custom_splash_attention.py`)이 있다. exp2 + 스케일 폴딩, bf16 P·V, 전치 레이아웃(kv를 sublane, q를 lane에 두어 m/l 통계를 (1, bq) 벡터로 유지), VPU 레지스터 타일링, GQA index_map, fixed-m, heads_per_tile을 지원한다. 그러나 (a) `ulysses_custom`/`tokamax_ring_custom` 경로에서만 쓰이고 TP head 샤딩에서는 도달할 수 없으며, (b) `attention_mask`를 받으면 `NotImplementedError`를 낸다. Krea 2는 항상 텍스트 마스크를 넘기므로 지금은 쓸 수 없다.

## 2. 프로파일 분해 (v5e-4, 2048², 블록 1개, 칩 1개 기준 46.06 ms)

| 구분 | 시간 | 비율 | 비고 |
|---|---|---|---|
| 어텐션 (splash) | 13.42 ms | 29% | 1.86 TFLOP/칩 → bf16 피크의 70% |
| 매트멀 8개 | 20.0 ms | 43% | gate/up 4.6 ms(183~187 TF/s), down 4.3, q/gate/out 1.7~1.9, k/v 0.45. **MXU 피크의 90~95%** |
| all-reduce ×2 | 7.2 ms | 16% | 207 MB bf16 × 2, 86 GB/s 실효. 컴퓨트와 **전혀 겹치지 않음** |
| RoPE 글루 | 2.4 ms | 5% | `reshape(...,64,2)`가 T(2,128) 타일 레이아웃을 만들어 f32 relayout copy 발생 |
| norm/modulation/residual | 2.1 ms | 5% | 207 MB 텐서를 700 GB/s로 여러 번 왕복. 4칩 모두 **중복 실행** |
| 전치/패딩 카피 | 0.9 ms | 2% | (B,L,H,D)↔(B,H,L,D), 17408 패딩, GQA repeat |

디스패치 갭은 스텝당 0.14 ms, prelude는 스텝당 약 8 ms(prompt만 의존하는데 매 스텝 재계산)로 둘 다 작다.

결론: 매트멀은 이미 한계다. 남은 57%(어텐션 30 + 통신 16 + 글루 12)가 설계 대상이다.

## 3. 설계

### 3.1 병렬화: 커널보다 먼저 정해야 하는 것

| 토폴로지 | 권장 | 이유 |
|---|---|---|
| v6e-4 / v6e-8 | **Ulysses 시퀀스 병렬** (`ici_context_parallelism=N`, 가중치 복제) | 레이어당 칩당 통신 98 MB(a2a q,k,v,o) vs TP 622 MB. norm/residual 글루가 1/N. bf16 26 GB + 텍스트 인코더 1.9 GB(별도 FSDP 메시로 4칩 샤딩; 복제하면 7.5 GB라 안 들어감)가 32 GB에 들어감 |
| v5e-4 bf16 | **TP + 시퀀스 병렬 residual** (reduce-scatter/all-gather, windowed einsum) | 가중치 복제 불가(16 GB). 바이트는 같지만 매트멀과 겹치고, 글루는 1/4 |
| v5e-4 int8 | Ulysses 가능 | W8 12.2 GB + TE 샤드 1.9 GB + VAE 0.24 GB ≈ 14.5 GB. 빡빡함 |
| v6e-1 | 통신 없음. TE 오프로드(현행) 또는 int8로 상주 | |

TP+SP 구현: residual stream에 `with_sharding_constraint(x, P(None, 'tensor', None))`을 걸어 GSPMD가 all-reduce 대신 RS+AG를 넣게 하고, XLA 플래그 `--xla_tpu_enable_windowed_einsum_for_reduce_scatter`, `--xla_tpu_enable_windowed_einsum_for_all_gather`, `--xla_tpu_enable_async_collective_fusion*`, `--xla_tpu_overlap_compute_collective_tc=true`로 down_proj/qkv 매트멀과 겹친다. 코드 변경은 샤딩 제약 + 플래그만이다.

Ulysses는 이미 `_ulysses_attention`(stock 커널, 마스크 지원)과 `use_custom_kernel` 경로가 있으므로 Krea 2는 `heads=48, kv=12`가 4로 나눠지는 4칩 구성에서 바로 쓸 수 있다. 8칩은 kv 12가 나눠지지 않아 kv를 24로 복제해야 한다.

### 3.2 어텐션 커널 `krea2_splash` (Pallas TPU, 기존 커스텀 커널 확장)

목표: v5e에서 bf16 피크의 85%(13.4 → ~10.5 ms), v6e에서 55% (head_dim 128은 256×256 MXU의 QKᵀ 축적 깊이를 절반만 채우므로 67%가 상한).

1. **레이아웃**: 기존 커스텀 커널의 전치 공식 유지. `qk = k·qᵀ` → (bkv, bq). m, l은 (1, bq) lane 벡터, o 스크래치는 (128, bq). stock 커널의 (bq,128) 통계 타일과 `jnp.tile(m_next, ...)` 브로드캐스트를 없앤다.
2. **스케일 폴딩**: softmax 스케일 1/√128 과 log2(e)를 `norm_q.weight`에 흡수한다. RMSNorm이 `(1+w)`를 곱하므로 로드 시 `w' = (1+w)·c − 1`로 바꾸면 커널은 `exp2(qk − m)`만 계산한다. RoPE는 회전이라 스칼라 스케일과 교환된다. 정확한 변환.
3. **GQA 네이티브**: k/v는 칩당 3 heads 그대로 넘기고 `k_index_map = h // 4`. `jnp.repeat` 제거.
4. **P·V bf16, 통계 f32**: `s_curr.astype(bf16)`로 MXU에 넣고 m/l/o 누적은 f32. stock 커널은 `v.astype(f32)` 매트멀을 쓴다.
5. **마스크 제거 (3.4의 시퀀스 재배열과 결합)**: 시퀀스를 `[image | text]`로 바꾸면 패딩은 항상 꼬리에 온다. 커널은 `orig_kv_seq_len`으로 마지막 kv 블록의 ragged 꼬리만 처리하면 되고 segment id 입력이 사라진다. 모든 kv 블록이 마스크 비교/select 없이 돈다.
6. **블록 크기 (초기값, 자동 탐색으로 확정)**: v5e `bq=1024, bkv=2048, bkv_compute=512, bkv_compute_in=256`, v6e `bq=2048, bkv=2048, bkv_compute=512`(256 배수 필수), `vmem_limit_bytes` 64~96 MiB. bq를 키우면 K/V HBM 재읽기(2K에서 레이어당 1.8 GB)가 절반으로 준다.
7. **에필로그 융합**: `o · sigmoid(gate)` 를 `end()`에서 처리. gate 블록 (bq, 128)을 추가 입력으로 받는다. 52 MB 왕복 1회 절약(선택).
8. **fixed-m**: 이미 구현된 Cauchy-Schwarz 고정 최대값 경로는 Krea 2에서 화질 A/B를 거친 뒤에만 켠다.
9. **레지스트리 등록**: `flash_custom` 이름으로 `KERNEL_REGISTRY`에 추가해 TP head 샤딩(`shard_map` in_specs가 head 축)에서도 도달 가능하게 한다.

### 3.3 RoPE: 커널이 아니라 가중치 순서로 해결

현재 RoPE는 `(2i, 2i+1)` 쌍을 회전한다. 로드 시 `to_q`/`to_k` kernel의 출력 열과 `norm_q.weight`/`norm_k.weight`를 `new[i] = old[2i], new[64+i] = old[2i+1]`로 치환하면 rotate-half 레이아웃이 된다. q와 k에 같은 치환을 적용하므로 q·k는 불변이고 v/out은 건드리지 않는다. RoPE는 `x·cos + concat(−x₂, x₁)·sin`이 되어 64-lane 절반 단위의 lane-dense 연산이 되고 XLA가 projection 에필로그에 융합한다. cos/sin 테이블은 `concat(freqs, freqs)`로 한 번만 만든다. 2.4 ms → 0.3 ms 예상. 첫 32차원(축 0, 항상 위치 0)은 회전이 항등이므로 생략 가능.

### 3.4 파이프라인 구조

1. **텍스트 컴팩션**: Qwen3 인코더는 현행 템플릿(중간 패딩, cumsum 위치)으로 그대로 돌리고, 출력에서 유효 토큰만 gather해 128 배수 버킷(128/256/384/512)으로 패딩한다. DiT는 텍스트 위치 id가 전부 0이고 text_fusion 정제 블록도 위치 정보가 없어 순열-동변이므로 결과가 정확히 같다. 일반 프롬프트(<128 토큰)에서 1024²는 L 4608 → 4224 (매트멀 −8%, 어텐션 −16%). 버킷 4개 = 실행 파일 4벌(AOT 캐시).
2. **`[image | text]` 순서**: 패딩을 꼬리로 보내 3.2-5를 성립시킨다. `finalize_output`은 `[:, :L_img]`로, rotary concat 순서도 같이 바꾼다.
3. **프롬프트 전용 계산을 스텝 밖으로**: text_fusion + txt_in 출력, rotary cos/sin은 생성당 1회. 스텝당 ~8 ms.
4. **QKV/gate 매트멀 병합**: `[6144, 6144+1536+1536+6144]` 하나, `gate/up`을 `[6144, 32768]` 하나로. 입력 207 MB 재읽기 4회 → 1회. 명시적 LoRA 경로는 병합 커널에 down/up을 더하는 식으로 유지.
5. **int8 W8A8 (선택, 화질 검증 필요)**: 매트멀이 MXU 한계이므로 유일한 매트멀 가속 수단. v5e 394 TOPS, v6e 1836 TOPS. `use_qwix_quantization` 플래그가 이미 있다. (2026-09-28: 트랜스포머는 qwix 대신 저장소 내 자체 양자화로 구현, `krea2_transformer_quantization=w8a8`. 같은 날 v6e-1에서 화질을 확인해 v6e-1 프리셋에서만 기본 켬, 범용 설정은 기본 꺼짐. 6.1 참조.) 가중치 전용 int8은 속도가 아니라 HBM 절약(v5e-4 Ulysses 성립)용이다.

### 3.5 하지 않는 것

- RoPE를 커널 프롤로그에 넣기: K 블록은 q 블록 수만큼 재읽히므로 회전을 L/bq 번 반복하게 된다.
- 블록 28개를 `lax.scan`으로 묶기: 디스패치 갭이 스텝당 0.14 ms라 이득이 없다.
- bf16 softmax(v6e): 화질 리스크 대비 이득이 불확실. f32 통계 유지.

## 4. 예상 효과 (roofline 모델, v5e-4 2K 실측 46.06 ms로 보정; v6e는 외삽)

| 구성 | 블록 ms | 8스텝 디노이즈 |
|---|---|---|
| v5e-4 2048² 현행 | 46.4 (실측 46.1) | 10.4 s |
| v5e-4 2048² TP+SP + 커널 + 컴팩션 | 33.4 | 7.5 s |
| v5e-4 2048² Ulysses int8 + 커널 + 컴팩션 | 21.8 | 4.9 s |
| v5e-4 1024² 현행 | 10.2 | 2.3 s |
| v5e-4 1024² TP+SP + 커널 + 컴팩션 | 6.7 | 1.5 s |
| v5e-4 1024² Ulysses int8 + 커널 + 컴팩션 | 3.7 | 0.84 s |
| v6e-4 1024² 현행 (추정) | 3.3 | 0.74 s |
| v6e-4 1024² Ulysses bf16 + 커널 + 컴팩션 | 1.5 | 0.34 s |
| v6e-1 1024² 현행 TE 오프로드 (추정) | 7.0 | 1.6 s |
| v6e-1 1024² 커널 + 컴팩션 | 5.7 | 1.3 s |
| v6e-1 1024² + int8 W8A8 | 3.5 | 0.79 s |

모델 가정: 매트멀 MXU 92%, 어텐션 v5e 85%/v6e 55%, TP+SP 통신 70% 은닉, Ulysses a2a 40% 은닉, 글루는 실측 5.4 ms를 L과 HBM 대역폭으로 스케일.

## 5. 구현 순서와 검증

| 단계 | 변경 파일 | 검증 |
|---|---|---|
| 1. RoPE 가중치 치환 + cos/sin 절반 테이블 | `models/krea2/util.py`(로드 시 치환), `transformer_krea2_flax.py`(rotate-half apply), `embeddings_flax.py` 또는 로컬 pos_embed | 기존 `krea2_transformer_test.py`의 reference 대비 허용오차 유지; 프로파일에서 `reshape.188/copy.20` 소멸 |
| 2. `[image|text]` 순서 + 텍스트 컴팩션 버킷 | `pipelines/krea2/krea2_pipeline.py`, `transformer_krea2_flax.py`(finalize, concat) | 같은 seed/prompt로 픽셀 diff ≈ 0 (bf16 노이즈 수준) |
| 3. 커널 확장: 마스크 없는 ragged 꼬리, gate 에필로그, `flash_custom` 레지스트리, TP head 샤딩 in_specs | `kernels/custom_splash_attention.py`, `models/attention_flax.py` | `interpret=True`로 CPU 정확도 테스트 → TPU에서 블록 크기 자동 탐색 |
| 4. 병렬화: TP+SP 샤딩 제약 + XLA 플래그 / Ulysses 설정 프리셋, 텍스트 인코더용 별도 FSDP 메시 | `krea2_pipeline.py`, `configs/base_krea2*.yml`, `generate_krea2.py`(배치 1 자동 설정 분기, 인코더 메시) | xprof에서 all-reduce 소멸 또는 겹침 확인 |
| 5. (선택) int8 W8A8 (2026-09-28 구현, qwix 대신 자체 양자화; v6e-1 프리셋은 기본 켬, `base_krea2.yml`/`base_krea2_turbo.yml`은 기본 꺼짐) | `models/krea2/transformer_quant.py`, `transformer_krea2_flax.py`, `generate_krea2.py`, `configs/base_krea2*.yml` | 화질 A/B (`artifacts/krea2_v10_ab` 방식; 2026-09-28 v6e-1에서 프롬프트·시드 1개 확인) |

TPU 없이는 1~3단계의 정확성(CPU/interpret)과 HBM 추정(`compile_krea2.py`)까지 확인할 수 있고, 성능 수치는 v5e-4 또는 v6e 인스턴스에서 xprof로 확정해야 한다.

## 6. 타깃 확정: v6e-1 또는 v5e-4

두 구성은 총 bf16 연산량이 비슷하다(918 vs 4×197 = 788 TF/s). 병목 순서는 정반대다.

### 6.1 v6e-1: 병목은 통신이 아니라 HBM 상주

- 통신 0. TP/Ulysses/SP 작업 전부 불필요. 커널은 단일 칩 `shard_map`이 자명해진다.
- 현행 README 구성은 텍스트 인코더(7.5 GiB)와 트랜스포머(23.9 GiB)를 **매 생성마다 호스트에서 HBM으로 다시 올린다**. 31 GiB를 PCIe로 옮기면 실효 10~20 GB/s에서 1.5~3 s가 걸려, 1024² 디노이즈 예상치(1.3~1.6 s)보다 크다. 상주 문제를 먼저 풀지 않으면 커널 최적화 효과가 보이지 않는다.

| 상주 방안 | HBM 사용 (compile_krea2 v6e-1 실측) | 비고 |
|---|---|---|
| 트랜스포머 bf16 상주 + TE만 오프로드 | 인코딩 단계 32.05 GiB → **OOM** (0.8 GiB 초과) | 인코딩 중에는 스왑인된 TE(7.7)와 상주 트랜스포머(24.1), VAE가 동시에 HBM에 있다. 초안의 24.2 GiB 계산은 이 동시 상주를 빠뜨렸다 |
| 양쪽 오프로드 (현재 프리셋) | 피크 24.57 GiB (1024²), 26.34 GiB (2048²) FITS | 생성마다 31.4 GiB 호스트→HBM 스왑. 상주 문제는 미해결 |
| 트랜스포머 int8 가중치 전용 + TE bf16 | ≈ 12.2 + 7.7 + 0.24 + 활성화 ≈ 20.6 GiB | 전부 상주, 스왑 0. dequant는 dot 프롤로그에 융합. 화질 영향 가장 작은 양자화 |
| **트랜스포머 bf16 + TE int8 가중치 전용 + 임베딩 호스트 룩업 (구현됨, 현재 프리셋)** | 1024² 피크 28.84 GiB FITS(여유 2.4). 2048²는 디코드 단계 32.6 GiB OOM → `krea2_offload_components=["text_encoder"]`로 29.2 GiB | 전부 상주, 스왑 0(2048²는 int8 인코더 3.4 GiB 스왑). qwix PTQ, group 128, 스케일 bf16. 실제 체크포인트에서 탭 hidden state 상대 L2 오차 1.4%, 코사인 0.99993 |
| 양자화 없이: TE를 레이어 단위로 스트리밍 | ≈ 24.1 + 0.24 + TE 1레이어 0.2 + 임베딩 룩업을 호스트에서 처리 | 스왑 7.7 GiB/생성(0.4~0.8 s 추정), 트랜스포머는 상주. qwen3_flax에 레이어별 실행 경로 필요 |

- 1024²에서는 매트멀이 블록 시간의 약 76%라 int8 W8A8(1836 TOPS)이 유일한 큰 레버다. 어텐션 커널과 RoPE 치환은 합쳐 15~20%.
- 2048²에서는 어텐션이 매트멀과 비슷해진다(추정 18 vs 17 ms/블록). head_dim 128 때문에 QKᵀ가 256×256 MXU를 절반만 채우므로 커널이 도달할 상한은 67%다. bq=2048, bkv_compute=512(256 배수), exp2, bf16 P·V, f32 통계.

우선순위: (1) 상주(양자화 허용 여부에 따라 int8 가중치 전용 또는 TE 스트리밍) → (2) 텍스트 컴팩션 + 프롬프트 전용 계산 호이스팅 → (3) int8 W8A8 화질 A/B → (4) 커널 + RoPE 치환. 2026-09-26 기준 (1) int8 인코더 상주, (2), (4), `flash_custom` 커널 모두 구현됨. 2026-09-28 트랜스포머 int8 W8A8 매트멀 구현(저장소 내 자체 양자화: 가중치는 로드 시 호스트에서 출력 열별 스케일, 활성값은 스텝 안에서 토큰별 동적 스케일, to_k/to_v는 bf16 유지, `krea2_transformer_quantization=w8a8`). HF int8 ConvRot 체크포인트는 쓰지 않기로 했다(활성값 회전 비용이 W8A8 절감분의 약 40%이고 키 레이아웃도 맞지 않음). v6e-1 매트멀 벤치 기준 예상: 1024² 1.89 s → 약 1.45 s, 트랜스포머 HBM 23.88 → 약 12.6 GiB. 같은 날 v6e-1 실측(Turbo 8스텝, 워밍업 후 생성 시간): 1024² 1.88 s → 1.45 s, 2048² 11.29 s(bf16, TE 오프로드) → 9.03 s(전부 상주), 1024² 런타임 HBM 피크 28.01 → 17.19 GiB, 트랜스포머 가중치 23.88 → 13.06 GiB. 동일 시드 이미지는 세부가 다르지만(1024² PSNR 21.3 dB) 구도가 같고 아티팩트가 없어 bf16과 화질이 동등하다고 판단했다(프롬프트·시드 1개). 그래서 W8A8을 v6e-1 프리셋 `base_krea2_turbo_v6e1.yml`의 기본값으로 켰고(bf16은 `krea2_transformer_quantization=''`), 다른 하드웨어에도 쓰는 범용 `base_krea2.yml`/`base_krea2_turbo.yml`은 기본 꺼짐을 유지한다. 남은 것: staged 트랜스포머와 LoRA를 W8A8로 켠 TPU 실행 확인, 더 많은 프롬프트·시드로 화질 확인. 2048²용 VAE 타일 디코드는 W8A8 프리셋에서는 필요 없어졌고(compile_krea2 추정 피크 21.80 GiB), bf16 트랜스포머로 2048²를 전부 상주시키려 할 때만 의미가 있다.

W8A8 후속 최적화(2026-09-28, v6e-1 실측, 프롬프트·시드 1개): (A) `to_q`/`to_k`/`to_v`의 역양자화를 헤드 레이아웃에서 수행해 q/k RMSNorm이 matmul에 다시 융합되게 함, 1024² 1.45 → 1.36 s. (B) 어텐션 커널 블록 크기 `flash_block_sizes: {block_kv: 2048, block_kv_compute: 1024, block_kv_compute_in: 256}`(block_q는 자동 선택 유지), 커널 호출당 1.512 → 1.348 ms, 2048² 디노이즈 8.34 → 8.05 s, 이미지는 픽셀 단위로 동일. (D) `to_k`/`to_v`도 int8(`to_q`/`to_gate`와 활성값 양자화 공유), 호출당 0.129 → 0.054 ms, 트랜스포머 가중치 12.56 GiB, 같은 시드 이미지는 달라지지만(1024² PSNR 14.8 dB, 구도 변화) 눈에 띄는 화질 저하는 없다고 판단. B와 D를 v6e-1 프리셋 기본값으로 넣은 결과 1024² 1.29 s, 2048² 8.22 s(전부 상주), compile_krea2 추정 피크 17.52 / 21.30 GiB. (C) 같은 입력을 읽는 int8 matmul을 하나로 병합하는 안은 마이크로벤치에서 분리본보다 1.05~1.49배 느려 기각했다(병합 matmul 약 1000~1100 TOP/s, 분리 약 1300 TOP/s, 결과 분할 비용 포함).

자동 block_q 확장(2026-09-29, v6e-1 실측, 프롬프트·시드 1개): 커널은 q 블록마다 모든 k/v 블록을 HBM에서 다시 읽으므로 q 블록이 적을수록 빠르다. 자동 block_q의 상한을 고정값 2048에서 칩별 VMEM 예산(`block_q * (4 * block_kv_compute + 2304) + 1664 * block_kv` 추정치, v6e 36e6 / v5e 18.5e6, 절대 상한 8192)으로 바꿨다. 프리셋의 kv 블록 크기에서 v6e의 선택은 1024² 1408 → 4224(q 블록 1개), 2048² 1664 → 3328(10개 → 5개). 커널 호출당 1.353 → 1.192 ms, 1024² 1.29 → 1.25 s, 2048² 8.24 → 7.96 s, 512 토큰 텍스트(시퀀스 4608)에서는 커널 1.914 → 1.749 ms. 대가는 컴파일 시간뿐이다(콜드 디노이즈 1024² +8 s, 2048² +0.7 s). f32 참조 대비 커널 오차는 block_q와 무관하게 같지만(max_abs_err 동일, rel L2 6자리 일치) 마지막 비트는 달라져서 같은 시드 이미지가 세부에서 달라진다(1024² PSNR 22.0 dB, 2048² 24.2 dB, 구도 동일, 화질 저하 없음으로 판단). 2048²에서 block_q 5504 고정은 커널 단독으로 6% 빠르지만 전체 생성은 8.01 s로 자동 선택보다 느렸고, `vmem_limit_bytes`를 올린 8320 / 16512는 호출당 31~38 ms(3328은 19.4 ms)로 훨씬 느려 둘 다 채택하지 않았다.

자동 block_q 확장 철회(2026-09-30/10-01, v6e-1 실측): 위 확장은 정사각 해상도에서만 쟀다. 종횡비 프리셋의 시퀀스 길이(이미지 토큰 + 텍스트 128)에서는 프리셋 kv 블록 크기(block_kv 2048)로 규칙이 block_q 4096을 고른다: 시퀀스 4016(1k 4:3), 16256(2k 16:9 / 21:9 / 5:4 / 4:5 / 9:16 / 9:21), 16352(2k 3:2 / 2:3). 그런데 이 값은 커널의 느린 구간에 들어간다. 시퀀스 16256에서 커널 호출당 block_q 2048 19.4 ms, 3328 19.1 ms, 3840~4480 55~69 ms, 4608 22.1 ms(구간은 유한하다)이고, block_kv 1024로 바꾸면 4096도 18.6 ms다. 시퀀스 4016에서는 2048 1.58 ms, 4096 / 4224 4.3~4.4 ms, block_kv 1024 1.52 ms다. block_kv_compute나 `vmem_limit_bytes`를 바꿔도 달라지지 않는다. 전체 생성은 2k 16:9 17.98 → 7.41 s, 1k 4:3 1.81 → 1.14 s(block_q 2048로 고정했을 때, 이미지 sha256 동일)였다. 원인은 Mosaic 내부에 있고 밝히지 못했다. 그래서(사용자 결정 2026-10-01) 자동 block_q의 상한을 다시 2048로 되돌렸고(`KREA2_BLOCK_SELECTION_REVISION` 3), 정사각 프리셋은 확장 전 수치인 1024² 1.29 s, 2048² 8.24 s로 돌아간다. 예산 코드(`_VMEM_BUDGET_BYTES`, `max_auto_block_q(..., budget_extension=True)`)는 `AUTO_BLOCK_Q_BUDGET_EXTENSION`(기본 꺼짐) 뒤에 남겨 두었다. block_kv 1024 변형이 커널 단독으로는 잰 모든 곳에서 가장 빨랐으므로 그 변형과 함께 다시 켤 수 있지만, block_kv 1024의 전체 생성 시간은 미측정이다. 상한을 2048로 되돌리자 패딩 낭비만 보는 규칙이 시퀀스 15680(2k 4:3 / 3:4)에서 block_q 512(q 블록 31개, 낭비 192)를 골랐다. 규칙이 q 블록마다 드는 비용을 무시하기 때문이다. v6e-1 실측에서 q 블록을 줄여 아낀 시간을 같은 시퀀스의 쿼리 행으로 환산하면 블록당 약 180~295행이다(시퀀스 4224 블록 1408 3개 → 4224 1개 1.353 → 1.192 ms, 블록당 약 250행; 16512 1664 10개 → 3328 5개 20.53 → 19.41 ms, 약 180행; 4608 1536 3개 → 2304 2개 1.914 → 1.792 ms, 약 295행). 그래서 자동 선택은 `패딩된 길이 + 200 * q 블록 수`를 최소화한다(`_BLOCK_Q_OVERHEAD_ROWS` 200은 보수적인 쪽 끝, 같은 revision 3). 프리셋 시퀀스 길이 중 선택이 바뀌는 것은 15680뿐이고 512 → 1792(블록 9개, 낭비 448)이며, 2026-10-02 v6e-1 실측에서 2k 4:3 디노이즈 7.08 s(전체 7.38 s)로 2026-09-30의 block_q 3968(7.50 s / 7.80 s)보다 빨랐다. 같은 세션에서 자동 선택이 1k 4:3 1.15 s, 2k 16:9 7.41 s로 09-30의 block_q 2048 고정 실행과 같은 시간을 냈고 이미지도 같았으며(8 초과 차이 나는 픽셀 0개), 정사각은 1024² 1.29 s, 2048² 8.20 s였다.

종횡비 프리셋과 사전 컴파일(2026-09-29 구현, 2026-09-30 v6e-1 실행): 해상도를 API처럼 `krea2_aspect_ratio` + `krea2_image_size`로 고른다(`models/krea2/resolution_presets.py`, 둘 중 하나라도 지정하면 `height`/`width`를 무시하고 빠진 쪽은 1:1 / 1k). 가로 기준 표(가로x세로, 세로형 4:5 3:4 2:3 9:16 9:21은 가로·세로를 바꾼 것, 2k는 1k의 정확히 2배):

| 비율 | 1k | 토큰 | 2k | 토큰 |
|---|---|---|---|---|
| 1:1 | 1024x1024 | 4096 | 2048x2048 | 16384 |
| 5:4 | 1152x896 | 4032 | 2304x1792 | 16128 |
| 4:3 | 1152x864 | 3888 | 2304x1728 | 15552 |
| 3:2 | 1248x832 | 4056 | 2496x1664 | 16224 |
| 16:9 | 1344x768 | 4032 | 2688x1536 | 16128 |
| 21:9 | 1536x672 | 4032 | 3072x1344 | 16128 |

트랜스포머 스텝 실행 파일은 패킹된 잠재 `(B, 토큰, 64)`와 격자 좌표를 값으로 받으므로 토큰 수(`(H/16)*(W/16)`)가 같은 해상도끼리 하나를 공유하고, VAE 디코드만 `(H, W)`마다 따로다. 그래서 5:4 / 16:9 / 21:9는 명목 비율에 가까우면서(실제 1.286 / 1.750 / 2.286, 오차 4% 이내) 4032(2k 16128) 토큰에 맞춘 근사값이고, 크기마다 토큰 수는 4개뿐이다. 22개 해상도 x 텍스트 버킷 1개(128)면 계산상 트랜스포머 스텝 8개 + VAE 디코드 22개에 텍스트 쪽 실행 파일 몇 개다. 사전 컴파일은 모델을 한 번 올린 뒤 항목마다 제로 실행 워밍업만 돌리고 항목이 끝날 때마다 AOT 캐시에 저장한다(선점되어도 그때까지 컴파일한 것은 남는다). 텍스트 버킷은 `min_text_tokens`로 강제하므로 짧은 프롬프트 하나로 256/384/512 버킷도 만들 수 있다(추가 위치는 마스크된 패딩이라 결과는 그대로). 강제값은 하한일 뿐이라 긴 프롬프트는 더 큰 버킷을 컴파일하므로, 사전 컴파일은 설정의 프롬프트 대신 고정된 짧은 프롬프트와 빈 네거티브 프롬프트를 쓰고, 파이프라인이 trace에 보고한 실제 버킷이 요청과 다르면 저장 후 중단한다. 사전 컴파일한 실행 파일을 새 프로세스의 지연 로드가 그대로 쓰는지는 CPU 소형 모델 테스트로 확인했다(재컴파일 없음, 결과 일치).

```
python src/maxdiffusion/generate_krea2.py src/maxdiffusion/configs/base_krea2_turbo_v6e1.yml \
  aot_cache_dir=<dir> krea2_precompile=all "krea2_precompile_text_tokens=[128]"
python src/maxdiffusion/generate_krea2.py src/maxdiffusion/configs/base_krea2_turbo_v6e1.yml \
  aot_cache_dir=<dir> krea2_aspect_ratio=16:9 krea2_image_size=2k prompt="..."
```

캐시에 형상이 30개쯤 쌓이면 시작할 때 전부 역직렬화하는 기존 방식은 한 번 쓰고 끝나는 실행에 낭비라서, v6e-1 프리셋은 `aot_cache_lazy_load: True`로 해당 형상이 처음 호출될 때 그 파일만 읽는다(형상당 `os.path.exists` 한 번, 설치당 시도 한 번). 범용 설정은 기존처럼 시작 시 전부 읽는다. 2026-09-30 v6e-1에서 사전 컴파일이 22개 해상도를 14분에 마쳤고(실행 파일 32개, 2.76 GiB), 그중 1k 4:3 / 2:3과 2k 16:9 / 21:9 / 3:4를 생성했다. 당시 자동 block_q로 1k 2:3 1.21 s, 2k 3:4 7.50 s였고, block_q 4096을 고른 1k 4:3(1.81 s)과 2k 16:9 / 21:9(17.98 s)는 느렸으며 block_q 2048로 1.14 s / 7.41 s가 됐다(위 "자동 block_q 확장 철회"). 2026-10-02에는 사전 컴파일이 22개 해상도를 10분에 마쳤고(실행 파일 32개; 22개 항목 중 세션 앞의 생성이 이미 만든 5개는 캐시 적중), 2k 4:3은 revision 3의 block_q 1792로 7.08 s였다(위 문단). 나머지 해상도의 생성 시간과 화질은 미측정이고, 2k의 HBM은 compile_krea2.py 추정뿐이다(v6e-1 3072x1344 피크 21.36 GiB, 2048x2048은 21.30 GiB).

양자화 가중치 캐시(2026-09-29, v6e-1 검증): 매 시작마다 bf16 체크포인트(트랜스포머 ~24 GiB, 텍스트 인코더 ~7.7 GiB)를 다시 읽고 다시 양자화하는 대신, `krea2_weight_cache_dir`를 지정하면 최종 호스트 트리를 한 번 저장해 두고 이후에는 그것을 바로 읽는다(`models/krea2/weight_cache.py`, 기본값 빈 문자열 = 꺼짐). 캐시하는 것은 트랜스포머의 체크포인트 변환 → rotate-half 순열 → W8A8 양자화가 끝난 트리(실모델 12.56 GiB)와 텍스트 인코더의 qwix int8 양자화가 끝난 트리(3.44 GiB), 그리고 `krea2_text_embed_on_host`일 때 호스트 임베딩 테이블(0.72 GiB)이다. 양자화한 구성 요소만 캐시한다. 구성 요소마다 `<dir>/<component>-<fp>/` 디렉터리 하나에 `meta.json`(형식 번호, 핑거프린트 입력, 체크포인트 `*.safetensors`의 이름·크기 목록, 리프별 경로·dtype·형상·오프셋 색인)과 `weights.bin`(모든 리프의 리틀엔디언 원시 바이트를 64바이트 정렬로 이어 붙인 것)을 둔다. 임시 디렉터리에 `weights.bin`, `meta.json` 순으로 쓰고 fsync한 뒤 rename하므로 `meta.json`이 있으면 완성된 디렉터리다. `<fp>`는 핑거프린트 입력의 sha256 앞 12자리이고, 입력은 모델 이름, 스냅숏(Hugging Face 커밋 해시), `weights_dtype`, 양자화 모드와 대상·헤드 기하(`num_attention_heads`, `num_key_value_heads`, `attention_head_dim`; rotate-half 순열이 여기에 의존하고 기하가 달라도 평탄화한 투영 형상은 같을 수 있다)(트랜스포머) 또는 모드·타일 크기·`embed_on_host`·스케일 dtype·qwix 버전(텍스트 인코더), 그리고 코드 리비전 상수(양자화 `KREA2_TRANSFORMER_WEIGHT_QUANT_REVISION`·`KREA2_TEXT_ENCODER_WEIGHT_QUANT_REVISION`, 순열 `KREA2_ROPE_PERMUTATION_REVISION`와 `rope_layout`, 그 밖에 저장값을 만드는 체크포인트 변환·bf16 정규화·임베딩 테이블 읽기의 `KREA2_TRANSFORMER_WEIGHT_BUILD_REVISION`·`KREA2_TEXT_ENCODER_WEIGHT_BUILD_REVISION`)다. 읽을 때는 저장된 입력 전체, 체크포인트 파일 목록(체크포인트가 있을 때만), 파일 크기, 색인이 쓸 때와 같은 정규 배치인지(중복 경로 없이 경로·dtype·형상에서 다시 계산한 오프셋·바이트 수와 전체 크기가 같아야 하므로 겹침·빈틈·순서 뒤바뀜은 불일치), 런타임 모델의 추상 트리와 경로·형상·dtype(트랜스포머는 정확히, 텍스트 인코더는 부동소수 리프면 폭과 무관하게 부동소수, int8 qvalue 같은 나머지는 정확히)을 대조하고, 하나라도 다르거나 `meta.json`이 형식에 어긋나면 예외 없이 이유 한 줄을 남기고 기존 경로로 만든 뒤 다시 저장한다. 적중하면 체크포인트의 safetensors 파일을 열지 않으므로 작은 설정·토크나이저·VAE 파일과 캐시만 있는 머신도 가능하다. 그런 "슬림" 머신은 `transformer/`·`text_encoder/`의 safetensors 없이 설정 파일·토크나이저·VAE(0.47 GiB)만 두고 캐시를 GCS에서 받아 쓰며, 캐시 미스는 거기서 RuntimeError로 바로 끝난다. 2026-10-02 새 v6e-1 VM 실측: 슬림 모델 0.51 GiB 다운로드로 설치 45 s(전체 체크포인트는 ~3분 40초), 캐시 16.7 GiB GCS 수신 104 s, 실행 파일 수신 31 s, 그 뒤 로드 4.5 s(수신 직후라 페이지 캐시가 따뜻해서 구성 요소당 읽기 0.8-1.1 s; 09-30에 본 62 s의 차가운 읽기는 이 흐름에서는 생기지 않는다), 프로세스 시작부터 이미지까지 40 s, 1k@1:1·1k@4:3·2k@16:9 이미지가 전체 체크포인트 실행과 비트 단위로 같았다(디노이즈 1.17 / 1.14 / 7.42 s). 같은 세션에서 프롬프트·시드를 바꾼 1k@1:1 4장(인물, 야경, 제품, 수채 일러스트)도 육안으로 이상 없었다. LoRA 어댑터가 하나라도 있으면 어댑터 가중치가 트리에 합쳐지는데 핑거프린트에는 없으므로 트랜스포머 캐시는 읽지도 쓰지도 않는다(텍스트 인코더 캐시는 그대로). 내용 체크섬은 없다(16 GiB를 해시하면 캐시로 아끼는 시간만큼 든다). 적중 시 결과는 캐시 없이 만든 트리와 바이트 단위로 같아야 하며, CPU 소형 모델 테스트에서 리프 비트 일치와 순전파 출력 일치를 확인했다. 2026-09-29 v6e-1 실측(1024x1024, 프롬프트·시드 하나): 로드는 캐시 없이 72.0 s, 캐시를 만드는 실행에서 171.2 s(트랜스포머 12.56 GiB 쓰기 84.9 s, 텍스트 인코더 4.16 GiB 쓰기 17.1 s), 페이지 캐시가 따뜻한 적중에서 4.48 s(구성 요소마다 읽기 1.0 s, VAE는 다른 로더와 경쟁하지 않게 되어 56 s 대신 3.7 s)이고, 세 실행의 이미지는 비트 단위로 같았다. 페이지 캐시가 차가운 적중(재부팅 직후나 새 VM)은 미측정이다.

빌드 전용 모드(`krea2_weight_cache_build_only`, 기본값 `False`): 캐시를 만드는 데는 TPU가 필요 없다. 체크포인트 읽기, rotate-half 순열, 두 양자화가 모두 호스트 CPU에서 돌기 때문이다. 이 모드는 `krea2_weight_cache_dir`가 있어야 하고 `krea2_precompile`과 함께 쓸 수 없다. 설정·메시·모델 구성·`jax.eval_shape`까지는 평소와 같고, 이미 완성된 구성 요소는 배열을 읽지 않고 검사만 하며(`weight_cache.component_is_valid`: `load_component`의 검사 전부와 `weights.bin` 열기), 캐시 대상이 아닌 구성 요소와 VAE는 로드하지 않는다. 빠진 구성 요소는 평소 미스와 같은 함수를 같은 순서로 불러 만들고 저장하므로 TPU 호스트가 쓰는 파일과 바이트 단위로 같아야 한다. 저장 실패는 이 모드에서만 오류(`RuntimeError`)다. 성공하면 구성 요소별 요약을 남기고 장치 배치 전에 끝난다(토크나이저·파이프라인·생성 없음). 그래서 가속기 없는 CPU VM(`JAX_PLATFORMS=cpu`, 목표는 RAM 64 GiB)에서 ~16.7 GiB 캐시를 만들 수 있다. TPU가 없는 머신에는 JAX 분산 코디네이터가 없으므로(`pyconfig`가 `jax.distributed.initialize()`에서 `coordinator_address should be defined`로 실패한다) `skip_jax_distributed_system=True`를 함께 준다: `JAX_PLATFORMS=cpu python src/maxdiffusion/generate_krea2.py <config> skip_jax_distributed_system=True krea2_weight_cache_dir=<dir> krea2_weight_cache_build_only=True`. 캐시할 구성 요소가 하나도 없으면(두 양자화가 모두 꺼졌거나, 트랜스포머만 양자화했는데 LoRA 어댑터가 설정된 경우) 스냅숏 다운로드·메시·모델 구성 전에 설정만 보고 `ValueError`로 끝난다. **아직 CPU 테스트 밖에서는 한 번도 돌리지 않았다.**

| v6e-1 | 8스텝 디노이즈 (추정) |
|---|---|
| 1024² 현행 (양쪽 오프로드) | 1.6 s + 스왑 1.5~3 s |
| 1024² 상주 + 커널 + 컴팩션 | 1.3 s |
| 1024² + int8 W8A8 | 0.8 s |
| 2048² 상주 + 커널 + 컴팩션 | 7.2 s |
| 2048² + int8 W8A8 | 5.3 s |

### 6.2 v5e-4: 병목은 통신 노출과 어텐션

- 가중치는 이미 상주(칩당 6.5 GiB). 단, 2048² HBM 피크가 15.4/15.75 GiB라 활성화 여유가 거의 없다. bq를 키우거나 QKV 병합으로 임시 버퍼가 늘면 OOM 위험이 있으니 `compile_krea2.py compile_topology=v5e-4`로 매 변경을 확인한다.
- all-reduce가 1024²에서 약 19%, 2048²에서 16% 노출. TP+SP(residual을 tensor 축으로 시퀀스 샤딩 + windowed einsum 플래그)가 첫 작업이다. 가중치 복제가 안 되므로 Ulysses는 int8 가중치 전용을 전제로만 가능하고, 그 경우에도 14.5 GiB로 2048² 활성화가 안 들어간다. **v5e-4는 TP+SP로 고정**한다.
- v5e VPU는 bf16 연산이 없으므로 softmax는 f32 유지, exp2로 스케일 곱을 제거하는 것이 VPU 부담을 줄이는 유일한 수단. bq=1024, bkv=2048, bkv_compute=512, bkv_compute_in=256.
- int8 W8A8은 TP 그대로 적용 가능(394 TOPS).

우선순위: (1) TP+SP 샤딩 제약 + XLA 플래그 → (2) 커널 + RoPE 치환 + 컴팩션 → (3) int8 W8A8.

| v5e-4 | 8스텝 디노이즈 (추정, 2048² 현행은 실측) |
|---|---|
| 1024² 현행 | 2.3 s |
| 1024² TP+SP + 커널 + 컴팩션 | 1.5 s |
| 1024² + int8 W8A8 | 0.93 s |
| 2048² 현행 | 10.4 s (실측) |
| 2048² TP+SP + 커널 + 컴팩션 | 7.5 s |
| 2048² + int8 W8A8 | 5.3 s |

### 6.3 선택 기준

- 같은 최적화 단계에서 두 구성의 디노이즈 시간은 거의 같다(1024² 1.3 vs 1.5 s, int8 0.8 vs 0.9 s). 차이는 v6e-1이 통신 코드와 4칩 메시 없이 단일 실행 파일로 끝나고, 컴파일·AOT 캐시·프로파일이 단순하다는 점이다. 칩 1개 시간당 비용도 일반적으로 v5e 4개보다 낮다.
- v5e-4가 유리한 경우는 2048² 위주에 bf16을 고수할 때다. HBM 총량 63 GiB로 양자화 없이 전부 상주하고, 어텐션이 4칩에 나뉜다.
- 공통 작업(RoPE 치환, `[image|text]` 재배열, 텍스트 컴팩션, 커널 ragged 꼬리, 프롬프트 계산 호이스팅)은 두 구성에서 그대로 재사용되므로 먼저 만들고, 상주(v6e-1) 또는 TP+SP(v5e-4)를 그 위에 얹는다.

### 6.4 텍스트 인코더 int8 검증 (2026-09-26, CPU, krea/Krea-2-Turbo 실제 가중치)

인코더는 stock Qwen/Qwen3-VL-4B-Instruct 언어 모델과 바이트 단위로 동일함을 확인했다(임베딩, 0/17/35층, 최종 norm SHA-256 일치). 프롬프트 8개(534 유효 토큰)에서 12개 탭 레이어의 bf16 대비 int8(group 128, RTN) 출력:

| 탭 레이어 | 상대 L2 | 평균 코사인 |
|---|---|---|
| h[2] | 7.2e-3 | 0.999978 |
| h[11] | 1.13e-2 | 0.999936 |
| h[23] | 1.13e-2 | 0.999933 |
| h[35] | 1.47e-2 | 0.999878 |
| 전체 | 1.38e-2 (bf16 스케일 반올림 포함 1.49e-2) | 0.999933 |

측정 한계: 검증은 f32 활성화로 돌렸으므로 실제 bf16 곱셈 반올림은 포함되지 않았다. VAE 디코드 활성화(1024² 1.3 GiB, 2048² 5.1 GiB)가 전부 상주 레이아웃의 2048² 병목이며, 공간 타일 디코드(mid block 전역 어텐션 때문에 근사)가 후속 항목이다.

### 6.5 어텐션 커널 hybrid 변형과 칩별 선택 (2026-10-03/04)

`kernels/krea2_attention.py`에는 이제 두 변형이 있다(`Krea2BlockSizes.variant`). 래퍼, 그리드 `(B, Hq, q 블록, kv 블록)`, BlockSpec, 입력(q, k, v, `valid_kv_len`)과 출력 `(B, Hq, 128, L)`, 네 개의 `pl.when` 영역(본문 / 마스크 본문 / 마지막 / 마지막 마스크)은 같고, 한 kv 블록 안의 계산 구조만 다르다.

- **flash** (기존): kv 청크(block_kv_compute 행)마다 QKᵀ를 계산하고, block_kv_compute_in 행마다 q 블록 전체(bq 레인)에 대해 m / l / o를 갱신한다.
- **hybrid** (신규): (1) kv 청크마다 `(block_kv_compute, bq)` f32 QKᵀ를 한 번에 계산해 명시적 VMEM 스크래치에 쓴다. (2) 온라인 softmax와 P·V를 block_q_strip(256) 레인 스트립 단위로 돌리고, 스트립의 m과 o는 청크 내내 레지스터에 둔다(마지막 스트립은 더 좁을 수 있다: bq 1408 = 256×5 + 128). (3) l(softmax 분모)을 MXU에서 구한다: kv 블록의 vᵀ를 `(136, block_kv)` 스크래치에 쓰되 128행은 1, 129~135행은 0으로 두면(j == 0에서 한 번 기록, 이후 스텝은 0~127행만 갱신) `o_ext = vᵀ_ext · P`의 128행이 l이 된다. l 스크래치와 VALU 열 합, alpha·l 갱신이 사라지고 finalize가 128행으로 나눈다. (4) 러닝 최대값은 block_kv_compute_in(1024) 행마다 한 번, exp2와 P·V는 block_kv_pv(256) 행(v6e MXU 축적 타일 1개)마다 돈다. exp 결과의 수명이 짧아져 스필이 준다. 기본 크기: block_kv 2048, block_kv_compute 2048, block_kv_compute_in 1024, block_kv_pv 256, block_q_strip 256.

왜: int8 어텐션 실험(F, 2026-10-03)의 LLO 분석에서 v6e flash 커널은 MXU가 아니라 VLIW 이슈 슬롯(VALU 4 / 로드 3 / 스토어 2 per bundle)에 묶여 있고, 본문 로드·스토어의 약 90%가 VMEM 스필/필이었다. 그래서 같은 수학을 슬롯과 레지스터 압력 기준으로 재배치했다. 방법은 정적 번들 계수다: 노트북에서 `v6e:1x1` 토폴로지로 교차 컴파일하고(`--xla_jf_dump_llo_text`), 최종 VLIW 번들 목록에서 본문 영역(region 1, 마스크 없는 전체 kv 블록)의 번들 수와 연산 종류(스코어 vreg당 VALU, EUP, MXU push/pop, 스필 로드/스토어)를 센 뒤, flash의 실측 시간에 번들당 0.54 ns(int8/bf16 차이로 맞춘 값)를 더하고 빼서 예측했다. 변형 후보(레인 스트립 폭 128~512, QKᵀ 청크 1024/2048, 최대값 갱신 256~2048행, P·V 조각 128~1024행, l-sum VPU/MXU, bf16 exp, 래퍼에서 만든 vᵀ)를 이렇게 걸렀고, 승자는 본문 번들 6321 → 5141(seq 4224), 7531 → 6180(seq 16512), 스코어 vreg당 VALU 4.80 → 3.36, 스필 필 3.06 → 0.89다. bf16 exp는 번들이 늘고(+6.6~10.8%) 정확도도 9배 나빠 기각했다. 정적 계수는 DMA 대기·MXU 지연을 보지 못하므로 상위 후보 사이의 1~3% 차이는 잡음 수준이다.

v6e-1 실측(2026-10-03, 같은 래퍼·같은 block_q, 커널 호출당): seq 4224(1024², bq 1408) flash 1.674 → hybrid 1.444 ms, seq 16512(2048², bq 1664) 20.558 → 16.869 ms(−14 / −18%). 정확도는 같다: f64 참조 대비 출력 오차가 flash와 같고(bf16 출력 max 8.5e-3 vs 8.4e-3 at 4224), 두 변형의 차이는 bf16 1 ulp 이내(relL2 7~9e-4). 레포 커밋 e76983a로 다시 잰 v6e-1 세션(2026-10-04, 같은 세션에서 flash 강제와 비교): 커널 호출당 seq 4224 1.657 → 1.400 ms, seq 16512 20.33 → 16.84 ms(probe 수치 재현), 온칩 커널 검사 max 오차 flash 3.15e-3 / hybrid 3.11e-3. 전체 생성 디노이즈(시간 측정 패스)는 1024² 1.169 → 1.119 s(−4.3%), 2048² 7.902 → 7.109 s(−10.0%)이고, hybrid로 1k 4:3 1.09 s, 2k 16:9 6.52 s였다. 이미지는 비트 단위로 같지 않다(PSNR 1024² 20.9 dB, 2048² 24.6 dB): 장면과 구도는 같고 모자 모양 같은 세부만 달라지며 화질 저하는 보이지 않았다(블록 크기만 바꿨을 때의 22~24 dB와 같은 성격). 같은 세션의 사전 컴파일(revision 4)은 9분, 실행 파일 38개(2.96 GB)를 버킷 `aot/`에 올렸다. 노트북 교차 컴파일(v6e:1x1, 2026-10-04): 22개 종횡비 프리셋의 시퀀스 길이 8개(4016 / 4160 / 4184 / 4224 / 15680 / 16256 / 16352 / 16512) 모두 래퍼를 거쳐 hybrid로 컴파일되고 block_q는 flash와 같다(2048 / 1408 / 1408 / 1408 / 1792 / 2048 / 2048 / 1664). `compile_krea2.py` v6e-1 프리셋 HBM 피크는 hybrid와 flash 모두 1024² 17.52 GiB, 2048² 21.30 GiB로 이전과 같다(커널 VMEM은 HBM 추정에 들어가지 않는다).

VMEM: hybrid의 QKᵀ 스크래치는 `4 · block_kv_compute · bq` 바이트(2048 × 1664면 13.6 MB)라 flash보다 VMEM을 훨씬 많이 쓴다. 자동 block_q는 `_estimated_hybrid_vmem_bytes`(명시적 스크래치 + 이중 버퍼 q/out/k/v 블록 + q 레인당 2048 바이트의 컴파일러 임시값)를 칩 기본 scoped VMEM 한도(v6e 32 MiB, v5e 16 MiB; 사용자 `vmem_limit_bytes`가 있으면 그 값) × 0.86 예산과 비교해 상한을 정한다. 교차 컴파일 보정점(block_kv 2048, 성공 최대 / 실패 최소 block_q): v6e block_kv_compute 2048: 2176 / 2304, 1024: 3328 / 3456; v5e 2048: 1024 / 1152, 1024: 1536 / 1664; v5e에 `vmem_limit_bytes` 32 MiB를 주면 v6e와 같은 2176 / 2304. 그래서 v6e 기본 크기에서는 상한(2048)이 걸리지 않아 모든 프리셋 시퀀스에서 flash와 같은 block_q(1408 / 1664 등)를 고르고, v5e에서 hybrid를 강제하면 block_kv_compute 2048로 896, 1024로 1408까지로 제한된다(보정상 한 단계 보수적). 512도 들어가지 않으면 줄일 크기를 알려 주는 `ValueError`다. 사용자가 block_q를 직접 주면 제한하지 않는다(컴파일이 판단).

선택 규칙(`KREA2_BLOCK_SELECTION_REVISION` 4): `krea2_attention_kernel`(기본 `auto`; `flash` / `hybrid` 강제, 빈 문자열은 `auto`, 그 밖의 값은 모델 로드 전 시작 시 오류)이 `CustomFlashBlockSizes.kernel`로 커널에 전달된다. `auto`는 장치 종류가 `TPU v6 lite`일 때만 hybrid, 그 밖(v5e, 알 수 없는 장치, CPU)은 flash다. kv 블록 크기 기본값은 (변형, 칩) 표 `_DEFAULT_KV_BLOCKS`에서 오고 필드마다 `flash_block_sizes`로 덮어쓸 수 있다: flash 기본 1024/512/256, flash on v6e 2048/1024/256(이전 v6e-1 프리셋 값), hybrid 2048/2048/1024/256/256. 그래서 v6e-1 프리셋은 `flash_block_sizes: {}`이고, `krea2_attention_kernel=flash`로 강제해도 v6e 튜닝 크기를 그대로 쓴다. flash 변형에 `block_kv_pv` / `block_q_strip`을 주면 오류다. flash 경로의 코드와 출력은 바뀌지 않았다(CPU interpret 모드에서 비트 단위 동일 확인). AOT 캐시 메타에는 `krea2_block_selection: r4`, `krea2_attention_kernel: <선택>`, 래퍼가 보는 칩 종류(`krea2_attention_device_kind`, 트랜스포머 메시에서 같은 헬퍼로 구함; AbstractMesh는 `abstract_device`로)와 그 칩에서 결정된 변형(`krea2_attention_kernel_variant`)이 들어간다. 그래서 revision 3 실행 파일은 적중하지 않고, 같은 설정·메시 모양이라도 v6e(auto → hybrid)와 v5e(auto → flash)의 핑거프린트가 다르다.

### 6.6 어텐션 글루 융합: q/k norm + RoPE + 패딩을 Pallas 한 패스로 (2026-10-04)

근거: hybrid 커널(6.5) 이후의 v6e-1 xprof 프로파일(2026-10-04, W8A8 + hybrid, Turbo 8 스텝, 이미지당 장치 self time)은 1024² 1176.8 ms = matmul 52.6 %, 어텐션 커널 21.5 %(호출당 1.129 ms), 활성화 양자화 7.2 %, RoPE/norm 글루 6.8 %, 어텐션 글루 4.3 %, VAE 5.1 %였고, 2048² 7385 ms = 어텐션 커널 47.8 %(호출당 15.77 ms), matmul 31.9 %, 활성화 양자화 5.3 %, RoPE/norm 7.5 %, 어텐션 글루 3.6 %, VAE 3.4 %였다. 글루(활성화 양자화 + RoPE/norm + 어텐션 글루)는 16~18 %이고 모두 HBM 바운드다. 가장 큰 덩어리는 `flash_custom` + `rotate_half` 경로의 q/k 후처리로, XLA가 텐서마다 세 융합으로 나눴다: (1) norm_q가 f32 정규화 중간값을 `(B, H, L, 128)` 전치 형태로 HBM에 쓰고, (2) RoPE 융합이 64-wide bf16 두 반쪽을 따로 내보내 128-lane 타일의 절반이 빈 채로 읽고 쓰며, (3) concat + 스케일 + 블록 크기 패딩이 전체를 다시 읽고 쓴다. 블록·스텝당 q 경로는 약 0.35 ms(1024²) / 2.5 ms(2048²)로, 이상적 read+write 약 0.07 / 0.25 ms의 5~10배였다. 별도로 어텐션 입력의 W8A8 양자화가 같은 내용의 융합 두 개로 두 번 계산됐다(to_q/to_gate용 하나, to_k/to_v용 7-D bitcast 형태 하나; 블록·스텝당 0.07 / 0.28 ms).

jnp 재작성은 도움이 되지 않았다: 전폭 cos/sin 표로 RoPE를 한 식으로 쓰고 파트너를 concat 또는 roll로 만든 두 버전 모두, 교차 컴파일 HLO에서 XLA가 여전히 64-lane 반쪽 스왑을 융합하지 못하고 정규화 값과 두 반쪽을 HBM에 materialize했다. 그래서 Pallas 커널로 갔다.

커널 `kernels/krea2_qk_prep.py`(`krea2_qk_prep`, 텐서마다 한 번 읽고 한 번 쓴다):

- 입력은 head-major `(B, H, L, D)` 투영이다. 호출 측의 `(B, L, H, D)` → `(B, H, L, D)` 전치는 비용이 없다: XLA가 W8A8 matmul 결과 `(L, H, D)`를 레이아웃 {2,0,1}로 쓰고, 이는 (8, 128) 타일링에서 row-major `(H, L, D)`와 같은 바이트다. `(B, L, H*D)`로 읽으면 relayout 복사가 두 번 생긴다.
- 그리드 `(B, 행 블록, 헤드 블록)`, 블록 `(heads_per_block ≤ 4, block_rows ≤ 1024, D)`. 헤드가 가장 안쪽이라 `(block_rows, D)` cos/sin 블록이 행 블록 동안 VMEM에 남는다. 출력은 `(B, H, padded_len, D)`이고 L 이후 행은 iota 마스크로 0을 쓴다. 전치와 어텐션 블록 크기로의 패딩이 블록 맵으로 해결되어 추가 패스가 없다.
- 수학은 헤드마다 f32: `normed = x · rsqrt(mean(x²) + eps) · (1 + w)`, `out = normed · cos2 + pltpu.roll(normed, D/2) · sin2`(전폭 표 `[cos, cos]`, `[−sin, sin]`). q는 softmax 스케일을 f32에서 곱하고(스케일 폴딩), 마지막에 bf16 캐스트 한 번이다. 기존 경로는 norm 뒤와 RoPE 뒤에 bf16으로 캐스트하고 스케일을 bf16 결과에 곱했으므로 반올림이 다르다.
- 어텐션 래퍼는 raw head-major 투영과 norm 가중치·표를 `AttentionOp.apply_attention`의 새 `extra_context`로 받아 래퍼 안에서 prep 커널을 부른다.

중복 양자화: `transformer_krea2_flax.py`에서 공유 `quantize_activation(hidden_states)` 결과를 `jax.lax.optimization_barrier`로 감싸 to_q/k/v/gate가 materialize된 `(x_q, scale)` 하나를 쓰게 했다. 장벽이 없으면 XLA가 피연산자 형태별(to_q/to_gate vs to_k/to_v)로 양자화를 생산자 융합에 따로 넣어 두 번 계산한다.

HLO 감사(노트북 v6e-1 교차 컴파일, `compile_krea2.py <프리셋> compile_hlo_dir=<dir>`로 실행 파일마다 최적화 HLO 덤프): 블록당 q 경로 HBM 트래픽 1024² 573.6 → 118.0 MB(이상적 read+write의 1.04배), 2048² 2523 → 437 MB. 어텐션 입력 int8 양자화 융합 2 → 1. 블록당 글루 트래픽 합 1.72 → 1.11 GB(1024²), 6.41 → 3.73 GB(2048²). `compile_krea2.py` HBM 피크 17.52 / 21.30 GiB는 그대로다.

v6e-1 실측(스팟, 2026-10-04 11:06~11:26 UTC, 코드 051518b, 이전 측정과 같은 프롬프트·시드, Turbo 8 스텝, W8A8 + hybrid): 시간 측정 디노이즈 1024² 1.12 → 1.05 s(−6.2 %), 2048² 7.11 → 6.49 s(−8.7 %). 워밍된 전체 패스(인코드 + 디노이즈 + VAE) 1024² 1.28 → 1.17 s, 2048² 7.49 → 6.81 s. xprof 이미지당 장치 self time: 1024² 1176.8 → 1106.5 ms — RoPE/norm 융합 80.5 ms와 어텐션 글루 51.1 → 27.5 ms 대신 prep 커널 호출 38.5 ms(448회 = 블록·스텝마다 q와 k, 블록·스텝당 0.17 ms). 2048² 7385 → 6740 ms — 글루 550.4 + 262.5 → 2.1 + 100.8 ms, prep 커널 114.8 ms. matmul, 어텐션 커널(호출당 1.129 ms at 1024², 15.77 ms at 2048²), 활성화 양자화, VAE는 잡음 범위에서 같다.

이미지: 이전 코드와 비트 단위로 같지 않다(PSNR 1024² 14.6 dB — 같은 장면이지만 구도가 바뀌었다; 2048² 22.8 dB). 화질은 같고 아티팩트는 없다(메인 세션 판정, 사용자는 아직 보지 않았다). 새 코드의 두 실행(b, bprof)은 비트 단위로 같다(결정적). 원인은 위의 반올림 차이(f32 한 번 캐스트, f32 스케일)이며, 8 스텝 디노이즈에서 작은 차이가 구도 차이로 커질 수 있다.

AOT: 캐시 메타에 `krea2_attention_glue: r1`(`KREA2_ATTENTION_GLUE_REVISION`, `generate_krea2.attention_glue_aot_meta`)이 들어간다. flash_custom + rotate_half이거나 to_q/k/v/gate 중 하나라도 W8A8인 설정에만 붙고, 그 밖 설정의 핑거프린트와 캐시된 실행 파일은 그대로다. 그래서 버킷 `aot/`의 revision 4 실행 파일은 이 프리셋에서 적중하지 않으며, 사전 컴파일(`pc`)과 업로드(`aotup`)는 아직 다시 돌리지 않았다(검증 세션은 단계마다 세션 안에서 컴파일했다).

남은 것(프로파일 기준 추정, 모두 미구현):

- G3 미리 전치한 int8 가중치, G4 어텐션 커널이 `(B, L, H*D)`를 바로 쓰기: 6.7에서 구현했다.
- E 정적 활성화 스케일: 동적 absmax 양자화 패스(예: down_proj 입력 양자화 블록·스텝당 0.142 ms, HBM 피크 속도)를 없앤다. 약 5~8 %, 보정 데이터가 필요하고 화질 위험이 있다.
- 이전부터 보류된 항목: hybrid용 block_q 재조정(자동 규칙은 아직 flash 기준), v5e에서 hybrid 미측정, 2k 4:3의 고정 block_q 512 / 2048 미측정, block_kv 1024 + VMEM 예산 확장, staged / LoRA + W8A8의 TPU 검증.

### 6.7 마지막 레이아웃 복사 제거: 어텐션 직접 I/O + 전치 저장 int8 q/k 가중치 (2026-10-04)

근거: 6.6 이후 최적화 HLO 감사(v6e-1 교차 컴파일)에서 블록·스텝마다 어텐션 커널 주변에 relayout 패스 네 종류가 남아 있었다. (1) 커널의 `(B, Hq, 128, L)` 출력을 to_gate/to_out 앞에서 `(B, L, Hq*D)`로 바꾸는 전치 복사(1024² 113 MB, 2048² 415 MB), (2) v를 block_kv 배수로 맞추는 head-major pad(33 / 109 MB), (3) to_q/to_k/to_v의 스텝마다 s8 가중치 레이아웃 복사(75.5 + 2 × 18.9 MB). (3)은 이 matmul들이 head-major 출력을 내므로 XLA가 가중치를 `(features, in)` row-major로 원하는데 파라미터가 `(in, features)`로 저장되어 있어서 생겼다.

변경(986d122):

- 커널 I/O 레이아웃(`krea2_attention.kernel_io_layout`): hybrid 변형은 v를 모델 투영 그대로인 패딩 없는 `(B, L, Hkv*128)`로 읽고(BlockSpec `(None, bkv, 128)`, kv 헤드 h // g = 마지막 축의 lane 블록 h // g, 마지막 kv 블록은 부분 블록이며 커널은 유효 행만 읽는다), 출력도 `(B, L, Hq*128)`로 바로 쓴다(BlockSpec `(None, bq, 128)`, (헤드, q 블록)마다 정규화된 타일을 XLU로 한 번 전치, 마지막 q 블록은 마스크된 쓰기). flash 변형(v5e)은 head-major 그대로다. flash_custom 래퍼는 레이아웃에 따라 갈라진다(v / 출력의 평평한 shard_map 스펙, v pad 없음, 출력 복사 없음).
- direct 레이아웃에서는 to_v의 head 레이아웃 rescale(`unflatten`)을 뺐다. 남겨 두면 XLA가 to_v를 head-major로 내고 다시 평평하게 복사한다. 값은 같다.
- `Krea2QuantDense.transposed_kernel`: to_q/to_k(`KREA2_TRANSPOSED_KERNEL_TARGETS`)의 int8 커널을 `(features, in)`으로 저장한다. `quantize_transformer_params`가 호스트에서 양자화와 rotate-half 순열 뒤에 전치한다. int8 × int8 → int32는 정확하므로 값이 같다.
- 리비전: `KREA2_ATTENTION_GLUE_REVISION` 2, `KREA2_TRANSFORMER_QUANT_REVISION` 3(AOT 키; 커널 레이아웃이 모든 traced 그래프에 들어가므로 이제 모든 flash_custom 설정이 글루 키를 갖는다), `KREA2_TRANSFORMER_WEIGHT_QUANT_REVISION` 2(가중치 캐시 키; to_q/to_k 저장 레이아웃이 바뀌었다).

HLO 감사(노트북): 블록당 s8 가중치 복사 3 → 0, 어텐션 출력 복사 1 → 0, v pad 없음, 블록당 글루 트래픽 1105.8 → 846.3 MB(1024²), 3729 → 3092 MB(2048²). HBM 피크 17.52 / 21.30 GiB는 그대로다.

v6e-1 실측(스팟 us-east1-d, 2026-10-04 15:42~16:18 UTC, 코드 986d122, 6.6과 같은 프롬프트·시드, Turbo 8 스텝, W8A8 + hybrid; CPU VM에서 다시 만든 가중치 캐시 적중, 로드 4.4 s): 시간 측정 디노이즈 1024² 1.049 → 0.952 s(−9.2 %), 2048² 6.493 → 6.475 s(−0.3 %). 워밍된 전체 패스 1024² 1.17 → 1.07 s, 2048² 6.81 → 6.79 s. xprof 이미지당 장치 self time:

- 1024² 1106.6 → 1009.8 ms(−8.7 %): 어텐션 글루 27.5 → 0.1 ms, to_q 레이아웃 복사 11.6 → 0 ms, 파라미터 data-formatting 복사 10.6 → 0.2 ms. 예상(약 3.5~4 %)보다 컸던 부분은 matmul이다: W8A8 matmul 합 615.3 → 578.3 ms, 호출당 up_proj 0.746 → 0.708, gate_proj 0.521 → 0.484, to_gate/to_out/to_q 약 −7 %(to_k/to_v는 약 +10 %, 합 2 ms). 원인은 분석하지 않았다. 어텐션 커널 그룹(prep 커널 포함) 291.5 → 270.0 ms.
- 2048² 6739.4 → 6723.5 ms(−0.2 %): 글루와 복사는 1024²처럼 사라졌지만(어텐션 글루 100.8 → 0.5 ms, 복사 −21.6 ms), 전치 저장된 to_q의 matmul이 호출당 0.809 → 1.173 ms로 느려져(+81.7 ms) 이득을 거의 다 먹었다. 어텐션 커널 그룹도 3647.9 → 3683.9 ms(+1 %; 출력 타일 전치 또는 v lane 블록 DMA로 추정, 미확인).

이미지: 6.6(051518b)과 비트 단위로 같다(1024², 2048², 프로파일 실행 모두 sha256 동일).

AOT: 버킷 `aot/`는 이 프리셋에서 적중하지 않는다(글루 r2, quant r3). 검증 세션은 사전 컴파일(`pc`) 도중 선점되어 `pc` / `aotup`은 아직 다시 돌리지 않았다. 버킷 `wcache/`는 새 리비전으로 다시 만들었다(transformer-98775c7d427d).

2048² to_q 회귀의 원인(2026-10-05, 노트북 컴파일 전용 진단: HLO window config, LLO, 두 xprof hlo_stats): 융합 자체(형태, 레이아웃)는 이전과 같고 가중치의 메모리 위치가 다르다. 이전의 스텝마다 레이아웃 복사는 36 MiB s8 가중치를 28개 블록 중 27개에서 VMEM(S(1))에 썼다(사실상 VMEM prefetch). 복사가 없어지자 가중치를 HBM에서 읽고, XLA는 이중 버퍼 가중치 window를 기본 scoped VMEM 한도 32 MiB 안에 넣느라 타일을 6 헤드로 줄인다(12 헤드면 약 31.5 MB). MXU에서 가중치가 고정 피연산자이고 그리드가 헤드 그룹(바깥) × 토큰 타일(안쪽)이라, 101.4 MB int8 활성화 타일을 헤드 그룹마다 HBM에서 다시 읽는다: 8 패스(이전 4), 호출당 HBM 1052 vs 609 MB(+405 MB). 호출당 MXU/VPU 작업량은 같다(정적 추정 697~703 vs 672~685 us). 모델 t = max(k × 정적, HBM MB / 0.89 TB/s)가 네 측정점에 맞는다(2k 새 코드 1182 vs 실측 1184 us). 1024²는 x_q가 VMEM에 있어 패스가 HBM을 다시 읽지 않는다(195 vs 203 us).

scoped VMEM 한도와 to_q(L 16512) 타일의 관계(단독 재현기, 컴파일 전용; 헤드 그룹 / 활성화 패스): 16 MiB 4/12, 32 MiB(기본) 6/8, 36 MiB 8/6, 40 MiB 8/6, 44~48 MiB 6/8, 52~60 MiB 8/6, 64 MiB 12/4, 96 MiB 16/3. 단조롭지 않다: XLA 타일러는 활성화 재읽기를 무시하는 estimated_cycles를 최소화한다. 모델 예측(호출당, 2048²): 36 MiB 약 950 us, 4 패스 약 800~830 us, 현재 1184 us(이전 코드 807 us + 복사 49 us). 다만 v6e VMEM 128 MiB = scoped 한도 + MSA 풀이고, 모든 Krea 2 프로그램이 풀을 거의 다 채운다(MSA high-water 1k/2k에서 qwen3 95.5, text_context 95.6, transformer_step 94.7 / 93.8, VAE 94.6 / 92.7 MiB). 한도를 올리면 MSA 배치가 그만큼 줄어든다.

새 키 `krea2_transformer_scoped_vmem_limit_kib`(KiB, 0 = 컴파일러 기본, 그 밖에는 [16384, 131072]): DiT 블록을 담은 실행 파일(monolithic `transformer_step`, staged `transformer_block`)에만 `compiler_options={"xla_tpu_scoped_vmem_limit_kib": N}`을 준다(`aot_cache.cached_jit(..., compiler_options=)`, jit·AOT `lower().compile()`·직렬화 실행 파일 모두; `compile_krea2.py`도 같은 entry로 컴파일한다). TPU 메시에서만 주며 CPU 백엔드는 이 옵션을 거부한다. N > 0이면 AOT 메타에 키가 붙는다(N = 0 설정의 핑거프린트는 그대로). 기본 설정은 0, v6e-1 프리셋은 36864(36 MiB).

v6e-1 교차 컴파일 확인(128 토큰 텍스트 버킷 = 실행 시 형태, 블록 5 to_q): 2048² 기본 [6 헤드] 8 × 26 스텝 → 36 MiB [8] 6 × 26 → 64 MiB [16] 3 × 12; 1024² 기본 [6] 8 × 3 → 36 MiB [8] 6 × 4. 36 MiB에서 28개 to_q가 모두 바뀌고, 그 밖에 to_k/to_v/to_gate/to_out, gate/up/down_proj 융합도 2048²에서 약 170개, 1024²에서 약 190개 타일이 바뀐다(시간 영향 미확인). transformer_step의 MSA high-water 2048² 92.5 → 84.5(36 MiB) / 61.5 MiB(64 MiB), 1024² 93.8 → 90.8 MiB. HBM 피크 17.51 / 21.30 GiB는 그대로다. qwen3_forward, transformer_text_context, vae_decode의 최적화 HLO는 N = 0과 바이트 단위로 같고 덤프된 tpu_comp_env의 한도도 기본(−1)이다(프로그램별 적용). Pallas 커널의 블록 설정은 그대로다. 주의: 텍스트 512 토큰(L 16896, `compile_krea2.py`의 최악 길이)에서는 36 MiB에서도 2048² to_q가 6 헤드 / 8 패스로 남는다. TPU에서는 아직 측정하지 않았다.
