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
