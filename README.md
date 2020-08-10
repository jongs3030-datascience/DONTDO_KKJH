PROJECT DONTDO
==================================
## 프로젝트 소개
COVID19에 대응하여 질병예방본부가 권장하는 예방대책 중
1) 얼굴만지기
2) 마스크 착용   
두 CASE를 제대로 준수하는지 판단하는 딥러닝 프로젝트

## DONTDO 작동 원리
![DONTDO_STRUCTURE](https://user-images.githubusercontent.com/55820227/89745084-59af0200-daec-11ea-96e4-5c4787e78b66.JPG)

## 활용 모델
1) 아주대학교 DONT OPEN SOURCE : 16가지 행동을 학습시킨 안면인식 모델(FACE TOUCH도 포함)
2) 마스크 착용 식별 모델 : MOBILENETV2 모델 기반의 마스크 착용 VS 미착용을 판단하는 딥러닝 모델
3) MTCNN : FACE CROP을 위한 라이브러리
4) DENTHDEPTH OPEN SOURCE : 단일 이미지 깊이 추정을 위한 오픈소스 모델

## 모델 상세 설명
1) DONT
![image](https://user-images.githubusercontent.com/55820227/89745880-7b5eb800-daf1-11ea-8785-7b6c4f551561.png)
![image](https://user-images.githubusercontent.com/55820227/89745907-9af5e080-daf1-11ea-94f3-0a0b11fa4400.png)

2) DENTHDEPTH & FACE CROP
![image](https://user-images.githubusercontent.com/55820227/89745929-b8c34580-daf1-11ea-937e-c9a60d042590.png)
![image](https://user-images.githubusercontent.com/55820227/89745948-d4c6e700-daf1-11ea-887a-744dcc591e7b.png)

3) MASK DETECTION
![image](https://user-images.githubusercontent.com/55820227/89745960-e90ae400-daf1-11ea-9c03-e1f7aaeab45a.png)
