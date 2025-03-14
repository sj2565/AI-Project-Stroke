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

### Scatter Plot
<img width="330" alt="스캐터플롯1" src="https://github.com/user-attachments/assets/a1a53cba-32b0-47f2-8725-029b13343d32" /> &emsp;&emsp;
<img width="330" alt="스캐터플롯2" src="https://github.com/user-attachments/assets/610c2a57-857b-4c4e-b996-0a845f35e5c3" /> <br>
(gender, age, ever_married, work_type, stroke)&emsp;&emsp;&emsp;&emsp;(hypertension, bmi, smoking_status, stroke) <br><br><br><br><br>
<img width="330" alt="스캐터플롯3" src="https://github.com/user-attachments/assets/d5a001f7-79c6-4f6a-a96a-436f09dbd1ad" /> <br>
(heart_disease, residence_type, avg_glucose_level, stroke) <br><br>

<img width="330" alt="상관관계" src="https://github.com/user-attachments/assets/996c12ce-34c8-4727-9f5b-bc3cc0cea8e7" /> &emsp;&emsp;&emsp;&emsp;
<img width="330" alt="PCA분석" src="https://github.com/user-attachments/assets/97ac7b8c-64fa-43d8-a46a-0c89955a38d8" /> <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;상관관계 이미지 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;PCA 분석 이미지<br><br><br> 

<img width="450" alt="Keras" src="https://github.com/user-attachments/assets/45864f2c-8944-4d6c-ad34-dd68664edd85" /> <br> 
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Keras MLP 학습 <br><br><br>

<img width="450" alt="가중치시각화" src="https://github.com/user-attachments/assets/b2fbfed3-7519-4341-929e-c793c11bb074" /> <br> 
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Hidden Weight 시각화

## 4. 설치 및 실행
**Python 라이브러리 설치**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn tensorflow
```
**Python 실행**
```bash
python3 final.py
```
## 7. 향후 개선할 점
+ MLP(다층 퍼셉트론)모델 이외에 데이터 양이 적어도 학습할 수 있는 최적화된 모델 적용.
+ 설계한 모델을 바탕으로 실제 프로젝트에 활용하여 실용성 증진.
