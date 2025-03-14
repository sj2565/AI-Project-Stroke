# 뇌졸중 예측 모델 설계 및 분석

## 1. 프로젝트 소개
성별, 연령, 다양한 질병 및 결혼 여부 등과 같은 입력 매개 변수를 기반으로 환자가 뇌졸중에 걸릴 가능성을 예측. <br>
예측한 데이터를 기반으로 어떤 데이터가 뇌졸중에 영향을 더 끼치는지 확인할 수 있도록 모델 설계를 진행하여 <br> 
사전에 뇌졸중을 조금이라도 방지하도록 목표를 선정. <br>
입력 데이터 10개를 가지고 Train Data와 Test Data의 비율을 7:3으로 나눠 데이터 분석 및 시각화를 진행했으며, <br>
딥러닝의 ANN(MLP)모델로 학습. <br>
각 노드수나 히든층, 훈련 횟수, 학습률 등 조건을 변경하면서 정확도와 손실을 가지고 가장 적합한 모델을 선정. <br>
이외에도 다양한 데이터 전처리 과정(정규화, PCA분석 등)을 진행. <br>

## 2. 활용된 기술
**언어** : Python <br>
**라이브러리** : Numpy, Pandas, Matplotlib, Scikit-learn, Seaborns, Keras <br>
**IDE** : Python3 IDLE, PyCharm <br>

## 3. 프로젝트 실행 이미지
<img width="500" alt="히스토그램" src="https://github.com/user-attachments/assets/a8cdfaa2-d7fc-407f-adae-53f70460bb6f" />
<p>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Data Histogram</p> <br><br>
<img width="330" alt="스캐터플롯1" src="https://github.com/user-attachments/assets/a1a53cba-32b0-47f2-8725-029b13343d32" /> &emsp;&emsp;
<img width="330" alt="스캐터플롯2" src="https://github.com/user-attachments/assets/610c2a57-857b-4c4e-b996-0a845f35e5c3" /> 
<p>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Scatter Plot1&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Scatter Plot2 </p> <br><br>
<img width="330" alt="스캐터플롯3" src="https://github.com/user-attachments/assets/d5a001f7-79c6-4f6a-a96a-436f09dbd1ad" />
<p>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Scatter Plot3 <br><br>

<img width="330" alt="상관관계" src="https://github.com/user-attachments/assets/30dfd44f-be11-4f61-9228-3ffd9485f989" /> <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;좌석점유 이미지 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;이상값 좌석 이미지<br><br><br> 
<img width="800" alt="이상값" src="https://github.com/user-attachments/assets/732a0738-6e32-4dbf-b688-ed0406b4a4d9" /> <br> AI 모델 이상값 탐지 : distance가 1213.4로 측정되어 이상값이라 판단.














Scatter Plot.
상관관계.
PCA 분석.
Keras MLP 학습.
Hidden Weight 시각화.


## 6. 설치 및 실행
**C프로그램 컴파일**
```bash
# 라즈베리파이에서 컴파일 진행
gcc -o SubwaySensor SubwaySensor.c -lwiringPi
```
**Python 라이브러리 설치**
```bash
python3 -m pip install numpy pandas scipy scikit-learn

# pip 설치 안될 시 가상환경을 통해서 설치
python3 -m venv 설정할 이름

# 가상환경 활성화
source 설정할 이름/bin/activate
```

**Node 서버 가동 및 실행**
```bash
node SubwayServer
```
## 7. 향후 개선할 점
+ C -> Node(서버)로 보낼 때 단순 버퍼 형식이 아니라 UART나 TCP/UDP로 활용.
+ AI 모델 평가를 위해 정확도 및 손실함수를 적용.
+ 전송 데이터를 문자열 -> 바이너리로 바꿔서 데이터 크기 감소 및 전송 속도 향상.
-------------------------------------------------------------------------------------
고려해 볼 만한 사항
+ CRC를 추가하여 비트 검사
+ 메모리 관리를 위해 직접 동적 할당
+ Isolation Forest 외에 다른 모델 추가 (Ensemble Learning)
