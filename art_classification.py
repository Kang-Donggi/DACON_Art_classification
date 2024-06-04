from PIL import Image
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import os
import cv2

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torchvision.models as models

from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings(action='ignore')

# hyperparameter setting
CFG = {
    'IMG_SIZE':224,
    'EPOCHS':15,   # 학습 횟수
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':32, #24  #16
    'SEED':41
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정

df = pd.read_csv('/content/drive/MyDrive/artclassification/datas/train.csv')

df['artist'].values

df['img_path']=['/content/drive/MyDrive/artclassification/datas'+path[1:]for path in df['img_path']]
df.head()

# label encoding

le = preprocessing.LabelEncoder()
df['artist'] = le.fit_transform(df['artist'].values)  # 카테고리형 수치형으로 변환

train_df, val_df, _, _ = train_test_split(df, df['artist'].values, test_size=0.2, random_state=CFG['SEED'])

def get_data(df, infer=False): 
    if infer:
        return df['img_path'].values
    return df['img_path'].values, df['artist'].values

train_img_paths, train_labels = get_data(train_df)
val_img_paths, val_labels = get_data(val_df)

class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, transforms=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):  # index에 해당하는 입출력 데이터를 반환
        img_path = self.img_paths[index]
        image = cv2.imread(img_path)  # 이미지 읽어오기
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #brg 색상을 rgb로 변경
        if self.transforms is not None:     # transform, Albumentation 적용을 위함
            image = self.transforms(image=image)['image'] # (image(파라미터 명)=image(삽입 이미지 명))
                                                            #['image']를 붙여야 image반환 안 붙이면 dict반환
        if self.labels is not None:
            label = self.labels[index]  # label이 None가 아니라면
            return image, label         # image label 반환
        else:
            return image

    def __len__(self):
        return len(self.img_paths)

train_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE']*2,CFG['IMG_SIZE']*2,),
                            A.RandomCrop(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.CoarseDropout(always_apply=False, p=0.5, max_holes=8, max_height=16, max_width=16, min_holes=1, min_height=8, min_width=8),
                            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2,),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()])

test_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

train_dataset = CustomDataset(train_img_paths, train_labels, train_transform) # train image, label, albmetation결과 저장
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = CustomDataset(val_img_paths, val_labels, test_transform)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

class BaseModel(nn.Module):
    def __init__(self, num_classes=len(le.classes_)):
        super(BaseModel, self).__init__()
        #self.backbone = models.efficientnet_b0(pretrained=True)
        # self.backbone = models.efficientnet_b6(pretrained=True)  # gpu로 돌려보고 아니면 x
        #self.backbone = models.convnext_large(pretrained=True)
        self.backbone = models.convnext_base(pretrained=True)
        # self.backbone = models.convnext_small(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self,x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

# cutmix

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

# train
def train(model, optimizer, train_loader, test_loader, scheduler, device,  beta=1, cutmix_prob=0.5):
    model.to(device) # to.(device) 모델을 gpu에서 연산하도록
    beta=1.0
    criterion = nn.CrossEntropyLoss().to(device) # loss함수 softmax자동 적용(다중 클래스분류에 사용되는 활성화 함수, 각 클래스에 속할 확률 계산)
                                                # 정답과 예측한 값 사이의 entropy를 계산
    best_score = 0
    best_model = None


    for epoch in range(1,CFG["EPOCHS"]+1):
      model.train() # 모델을 학습모드로 변환 / 평가 모드는 model.eval()
      train_loss = []
      for img, label in tqdm(iter(train_loader)): # tqdm으로 감싸 진행률 출력, iter은 순회가능한 객체를 받아 iterator로 변환 iter은 한번 출력하면 값이 사라짐
          img, label = img.float().to(device), label.to(device)
           # img와 label을 설정devlce(여기서는 gpu)로 보냄

          optimizer.zero_grad() # 반복할 때 마다 기울기를 새로 계산하므로 해당 명령으로 초기화

          r = np.random.rand(1)

          if beta>0 and np.random.random()>0.5:  # 여기 들여쓰기 해 볼 것!!!!!
            lam=np.random.beta(beta,beta)
            rand_index=torch.randperm(img.size()[0]).cuda()
            target_a=label
            target_b=label[rand_index]
            bbx1,bby1,bbx2,bby2=rand_bbox(img.size(),lam)
            img[:,:,bbx1:bbx2,bby1:bby2]=img[rand_index,:,bbx1:bbx2,bby1:bby2]

            lam=1-((bbx2-bbx1)*(bby2-bby1)/(img.size()[-1]*img.size()[-2]))


            outputs=model(img)
            loss=criterion(outputs,target_a)*lam+criterion(outputs,target_b)*(1.-lam)
            loss.backward() # Require_grad=True로 설정된 모든 tensor에 대해 gradient를 계산
                            # 역전파에서 gradient를 계산하는 starting point가 loss값이기에 loss변수에 적용
            optimizer.step()

            outputs=model(img)
            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
            loss.backward()

          else:
            outputs=model(img)
            loss=criterion(outputs, label)
            loss.backward()
            optimizer.step()

            criterion(model(img), label).backward()






          # loss = criterion(model_pred, label)  #nn.CrossEntropyLoss에 (예측값, 정답) 전달


          # optimizer.step() # 이전 단계에서 계산된 loss를 통해 파라미터를 최적화(optimize)
                              # 아래에서 optimizer는 adam으로 설정 , step을 통해 parameter업데이트

          train_loss.append(loss.item()) #loss 값 train_loss에 추가

      tr_loss = np.mean(train_loss) # tr_loss=train_loss의 평균

      val_loss, val_score = validation(model, criterion, test_loader, device)

      print(f'Epoch [{epoch}], Train Loss : [{tr_loss:.5f}] Val Loss : [{val_loss:.5f}] Val F1 Score : [{val_score:.5f}]')

      if scheduler is not None:
            scheduler.step()

      if best_score < val_score:
         best_model = model
         best_score = val_score

    return best_model

def competition_metric(true, pred):
    return f1_score(true, pred, average="macro") #f1_score(정답, 예측값, 매크로 평균)

def validation(model, criterion, test_loader, device):
    model.eval() # model.eval() 모델을 평가모드로

    model_preds = []
    true_labels = []

    val_loss = []

    with torch.no_grad():  # autograd를 끔으로써 메모리 사용량 줄이고 연산 속도 높임
        for img, label in tqdm(iter(test_loader)):  # tqdm으로 감싸 진행률 출력, iter은 순회가능한 객체를 받아 iterator로 변환 iter은 한번 출력하면 값이 사라짐
            img, label = img.float().to(device), label.to(device)

            model_pred = model(img)

            loss = criterion(model_pred, label)

            val_loss.append(loss.item())

            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label.detach().cpu().numpy().tolist()
            # detach= tensor에서 이루어진 연산을 추적해 기록(grapg)이 기록에서 도함수계산되고 역전파 이루어짐
            # detach는 연산기록에서 분리한 tensor를 반환
            # cpu()=gpu에 올라간 tensor를 cpu로 복사
            # numpy()= tensor를 numpy로 변환하여 반환 저장공간을 공유하기에 하나 변경시 다른 하나도 변경
            #          cpu에 올라간 tensor만 numpy() 사용가능
            # tolist()= list 변환  사용시 detach()-cpu()-numpy()순서로 사용
    val_f1 = competition_metric(true_labels, model_preds)  # f1스코어 계산
    return np.mean(val_loss), val_f1

model = BaseModel()
model.eval()
optimizer = torch.optim.AdamW(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = None

infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

test_df = pd.read_csv('/content/drive/MyDrive/artclassification/datas/test.csv')
test_df['img_path']=['/content/drive/MyDrive/artclassification/datas'+path[1:]for path in test_df['img_path']]
test_df.head()

test_img_paths = get_data(test_df, infer=True)

test_dataset = CustomDataset(test_img_paths, None, test_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

def inference(model, test_loader, device):
    model.to(device)
    model.eval()

    model_preds = []

    with torch.no_grad():
        for img in tqdm(iter(test_loader)):
            img = img.float().to(device)

            model_pred = model(img)
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()

    print('Done.')
    return model_preds

preds = inference(infer_model, test_loader, device)

preds = le.inverse_transform(preds)

submit['artist'] = preds
submit.head()

submit.to_csv('/content/drive/MyDrive/artclassification/datas/submit8.csv', index=False)
