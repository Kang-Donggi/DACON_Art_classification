# DACON-ART_classification
### 개요: 
월간 데이콘 예술 작품 화가 분류 AI 경진대회 / 예술 작품을 50명의 화가에 대해 분류하는것이 목적
### 결과: 
-image classification task에서 성능을 높일 수 있는 방법을 적용해보고자 해당 대회에 참여하게 되었음<br/> 
-결과적으로 상위 15% 의 성적을 얻으며 대회 마무리 

### 회고: 
-대회 초반 여러가지 생각해 둔 기법이 있었지만 시간상 문제로 모든것을 다 적용해 보지 못한 점이 아쉽다 
-좀 더 효율적으로 실험을 진행하는 방법을 생각 해 봐야할 것 같다 느낌

### -실험 주요사항
1. classification 수행을 위한 모델으로는 Convnext-base를 선택 하였음 Resnet, Efficientnet 기반 모델을 사용 해 보았지만 Convnext가 가장 높은 성능 보였음 메모리 문제로 Convnext large는 사용하지 못 했고 base모델 사용
2. 전체 train set에서 20%를 validation set으로 사용 Cross validation을 사용 해 보고 싶었지만 학습 시간 문제로 사용하지 못 했다
3. 어떤 Augmentation을 적용하느냐에 따라 성능이 크게 갈렸음 Albumentation 라이브러리를 통해 다양한 augmentation 사용, 최종적으로 적용된 방법은 아해와 같다.
```python
A.Compose([A.RandomCrop(img_size, img_size),
          A.CoarseDropout(p=0.5, max_holes=8, max_height=16, max_width=16, min_holes, min_hights=8, min-width=8),
          A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3),
          A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)])
```
4. 추가 Augmentation으로 Cutmix를 사용하였으며 성능 향상에 큰 효과가 있었다.
```python
def rand_bbox(size, lam): # size : [B, C, W, H]
    W = size[2] # 이미지의 width
    H = size[3] # 이미지의 height
    cut_rat = np.sqrt(1. - lam)  # 패치 크기의 비율 정하기 #np.sqrt= numpy배열의 제곱근
    cut_w = np.int(W * cut_rat)  # 패치의 너비
    cut_h = np.int(H * cut_rat)  # 패치의 높이

    # uniform
    # 기존 이미지의 크기에서 랜덤하게 값을 가져옵니다.(중간 좌표 추출)
    cx = np.random.randint(W) # np.random.randint= 균일 분포의 정수 난수 1개생성
    cy = np.random.randint(H)

    # 패치 부분에 대한 좌표값을 추출합니다.
    bbx1 = np.clip(cx - cut_w // 2, 0, W)  # np.clip(array, min, max)
    bby1 = np.clip(cy - cut_h // 2, 0, H)  # array 내의 element들에 대해서
    bbx2 = np.clip(cx + cut_w // 2, 0, W) # min 값 보다 작은 값들을 min값으로 바꿔주고
    bby2 = np.clip(cy + cut_h // 2, 0, H) # max값 보다 큰 값들을 max값으로 바꿔주는 함수

    return bbx1, bby1, bbx2, bby2
```
