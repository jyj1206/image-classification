# CIFAR-10 spatial-filter experiments

입력 직후의 depthwise 3×3 spatial operation을 바꿔 CIFAR-10 분류 성능을
비교한다.

## 네트워크

```text
Input 3×32×32
  → Depthwise 3×3 spatial operation
    - 3-kernel group: 3→3
    - 6-kernel group: 3→6
  → Learnable 1×1 projection: 3 또는 6→64
  → BatchNorm → ReLU
  → Stage 1: ResBlock ×2, 64 channels
  → Stage 2: ResBlock ×2, 128 channels, 첫 block stride=2
  → Stage 3: ResBlock ×2, 256 channels, 첫 block stride=2
  → Global Average Pooling
  → Linear(256→10)
```

전체 모델 클래스는 `CIFAR10FilterNet`, residual block 클래스는
`ResBlock`이다.

## 실험 설정

3-kernel group:

- `exp1_random_learnable_3.yaml`
- `exp2_identity_fixed.yaml`
- `exp3_identity_learnable.yaml`
- `exp4_laplacian_fixed.yaml`
- `exp5_laplacian_learnable.yaml`
- `exp6_gaussian_fixed.yaml`
- `exp7_gaussian_learnable.yaml`

6-kernel group:

- `exp8_random_learnable_6.yaml`
- `exp9_sobel_fixed.yaml`
- `exp10_sobel_learnable.yaml`

12-kernel group:

- `exp11_random_learnable_12.yaml`
- `exp12_mixed_learnable.yaml`

Mixed는 각 RGB 채널에 `Sobel-X, Sobel-Y, Laplacian, Gaussian`을 하나씩
적용한다. 두 실험 모두 `Depthwise 3→12 → Pointwise 12→64`이며 random과
mixed handcrafted initialization의 차이만 비교한다.

Sobel 커널 순서는 `R-X, R-Y, G-X, G-Y, B-X, B-Y`이다.

## 설치와 학습

```bash
python -m pip install -r requirements.txt
python train.py --config config/exp1_random_learnable_3.yaml
```

중단된 실행 재개:

```bash
python train.py \
  --config config/exp1_random_learnable_3.yaml \
  --resume results/result_exp1_random_learnable_3_cifar10_YYYYMMDD_HHMMSS
```

기본 학습은 200 epochs와 CrossEntropyLoss를 사용한다. Optimizer는
SGD(lr=0.1, momentum=0.9, weight decay=5e-4, Nesterov)를 사용한다.
첫 5 epochs는 0.01에서 0.1까지 linear warm-up을 적용하고, 나머지 195
epochs는 최저 learning rate 1e-5까지 cosine annealing을 적용한다.

## 체크포인트와 검증

매 epoch가 끝날 때 전체 validation set으로 loss와 accuracy를 계산한다.

- `latest.pth`: 매 epoch 항상 덮어쓴다.
- `best.pth`: validation accuracy가 이전 최고 기록보다 높을 때만 덮어쓴다.
- validation accuracy가 같으면 기존 best checkpoint를 유지한다.
- 전체 학습 후 `best.pth`를 불러와 최종 stem filter를 저장한다.
- Test set은 학습 중 사용하지 않고 `test.py`로 별도 평가한다.

각 결과 디렉터리에는 다음 파일이 저장된다.

- `config.yaml`
- `checkpoints/latest.pth`
- `checkpoints/best.pth`
- `best_metrics.json`
- `metrics.csv`, `history.json`
- `visualizations/training_curves.png`
- `visualizations/kernels/init`, `visualizations/kernels/final`
- `visualizations/feature_maps/init`, `visualizations/feature_maps/final`

## 별도 테스트

```bash
python test.py --checkpoint results/result_exp9_sobel_fixed_cifar10_YYYYMMDD_HHMMSS
```

Test accuracy와 loss를 출력하고 `visualizations/` 아래에 kernel과 feature
map overview 및 PPT용 개별 이미지를 저장한다.

모든 학습 결과의 best checkpoint를 일괄 평가:

```bash
python test_all.py --results-root results
```

`results/**/checkpoints/best.pth`를 모두 찾아 각각의 test 결과를 `runs/`에
저장하고, 전체 accuracy, macro precision, macro recall, macro F1을
`runs/all_test_results_<timestamp>.csv`로 집계한다.

## 추가 ablation config

```text
config/no_batchnorm/             RGB, 모든 BatchNorm 제거, 3/6/12 kernels
config/grayscale/                Grayscale, BatchNorm 유지, 1/2/4 kernels
config/grayscale_no_batchnorm/   Grayscale, 모든 BatchNorm 제거, 1/2/4 kernels
```

각 폴더에는 기존 실험과 대응하는 12개 config가 있다. Grayscale 변환은
depthwise convolution 이전에 적용되며, 입력 채널당 kernel multiplier
1/2/4를 유지한다. Grayscale 시각화 개별 파일은 `_Gray_*` 이름으로
저장된다.

학습 결과는 `results/`, test 결과는 `runs/`에 분리된다.

```text
results/result_<experiment>_cifar10_<timestamp>/
├─ config.yaml
├─ history.json
├─ metrics.csv
├─ best_metrics.json
├─ checkpoints/
│  ├─ latest.pth
│  └─ best.pth
└─ visualizations/
   ├─ training_curves.png
   ├─ kernels/
   │  ├─ init/
   │  │  ├─ kernel_overview.png
   │  │  └─ kernel_[R/G/B]*.png
   │  └─ final/
   │     ├─ kernel_overview.png
   │     └─ kernel_[R/G/B]*.png
   └─ feature_maps/
      ├─ init/
      │  ├─ feature_map_overview.png
      │  └─ feature_map_class##_name_[input/R/G/B/RGB]*.png
      └─ final/
         ├─ feature_map_overview.png
         └─ feature_map_class##_name_[input/R/G/B/RGB]*.png

runs/run_<experiment>_cifar10_<timestamp>/
├─ config.yaml
├─ test_metrics.json
└─ visualizations/
   ├─ kernels/
   │  └─ test/
   │     ├─ kernel_overview.png
   │     └─ kernel_[R/G/B]*.png
   └─ feature_maps/
      └─ test/
         ├─ feature_map_overview.png
         └─ feature_map_class##_name_[input/R/G/B/RGB]*.png
```

Train의 init/final feature map은 validation set에서 CIFAR-10 클래스별 한 장씩
선택한 동일한 10장을 사용한다. Test feature map도 test set에서 클래스별 한
장씩 총 10장을 사용한다.
