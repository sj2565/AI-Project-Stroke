import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from pandas.plotting import scatter_matrix
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# bmi데이터 중 fillna()메소드를 이용해서  null값 처리
mean_age = df['bmi'].mean()
df['bmi'].fillna(mean_age, inplace = True)

# 문자열 데이터 숫자 처리
size_mapping = {'Male' :1,  'Female' :0, 'Other' :2}
df['gender'] = df['gender'].map(size_mapping)
size_mapping = {'Yes' :1,  'No' :0}
df['ever_married'] = df['ever_married'].map(size_mapping)
size_mapping = {'Private' :1,  'Self-employed' :2, 'Govt_job' :3, 'children' :4, 'Never_worked' :5}
df['work_type'] = df['work_type'].map(size_mapping)
size_mapping = {'Urban' :1,  'Rural' :2}
df['Residence_type'] = df['Residence_type'].map(size_mapping)
size_mapping = {'formerly smoked' :1,  'never smoked' :2, 'smokes' :3, 'Unknown' :4}
df['smoking_status'] = df['smoking_status'].map(size_mapping)

# 히스토그램
df.hist(bins=50, figsize=(20, 15))
plt.show()

# 스캐터 플롯 
attributes = ['gender', 'age',  'hypertension',  'heart_disease', 'ever_married', 'work_type',
            'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
scatter_matrix(df[attributes], figsize=(12, 18), alpha = 0.5, diagonal='hist')
plt.show()

# feature간 상관도 시각화
df.rename(columns={'PAY_0':'PAY_1','default payment next month':'default'}, inplace=True)
y_target = df['stroke']
# ID컬럼 Drop
X_features = df.drop(['id'], axis=1)
corr = X_features.corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr, annot=True, fmt='.1g')

# 변수 정규화 작업 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
features = ['gender', 'age',  'hypertension',  'heart_disease', 'ever_married', 'work_type',
            'Residence_type', 'avg_glucose_level', 'bmi',  'smoking_status']
target = ['stroke']

x, y = df.loc[:, features].values, df.loc[:, target].values
oversample = SMOTE()
x, y = oversample.fit_resample(x, y)

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
X_train = MinMaxScaler().fit_transform(X_train)
X_test = MinMaxScaler().fit_transform(X_test)

# 데이터 전처리 
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('Eigenvalues : \n%s' %eigen_vals)
#print('Eigenvector : \n%s' %eigen_vecs)

tot = sum(eigen_vals)
var_exp = [(i / tot) for  i in sorted(eigen_vals, reverse=True)]
#cum_var_exp = np.cumsum(var_exp)
print('Explained_variacne_ratio : \n%s' %var_exp)
# 누적
#print('Explained_variacne_ratio : \n%s' %cum_var_exp)

# eigenvalue 값 그래프 그리기
plt.figure(figsize=(10,8))
plt.plot(range(1, 11), var_exp)
#plt.plot(range(1, 11), eigen_vals)
plt.ylabel('Explained_variacne')
plt.xlabel('features')
plt.show()

# PCA실행
from sklearn.decomposition import PCA
for i in range(1, 11):
    pca = PCA(n_components=i)
    X_train_pca = pca.fit_transform(X_train_std)
    pca_variance_ratio = pca.explained_variance_ratio_
    print(f'{i}개의 차원으로 축소한 결과 {sum(pca_variance_ratio)*100}% 유사')
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1
    print('선택할 차원의 수 : ',  d)
    
#pca = PCA(n_components=10) # 주성분 10개
#X_train_pca = pca.fit_transform(X_train_std)
#X_test_pca = pca.transform(X_test_std)

#plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y, cmap="Set1")
#plt.show()

#print(principalDf)
#print('eigen_value :', pca.explained_variance_)
#print('explained variance ratio :', pca.explained_variance_ratio_)

# mlp 작업
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

#n_features= x.shape[1]
#X_train = X_train.reshape(len(X_train), n_features)
#X_test= X_test.reshape(len(X_test), n_features)

input_shape = 10
model = Sequential() # Sequential 클래스 함수를 사용하여 신경망 객체 생성 
model.add(Dense(7, input_shape=(input_shape, ), kernel_initializer='uniform', activation='relu', bias_initializer='zeros'))
# 첫번째 Dense 레이어 => 입력층으로 10개 뉴런을 입력받아 15개 뉴런을 출력.
# model.add(Dropout(0.1))

model.add(Dense(1, kernel_initializer='uniform', activation = 'sigmoid', bias_initializer='zeros'))
# 두번째 Dense 레이어 => 출력층으로 10개 뉴런을 입력받아 1개 뉴런을 출력.

model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate=0.001), metrics = ['accuracy'])
# loss : 현재 가중치 세트를 평가하는 데 사용한 손실 함수 -> 이진클래스 이므로 binary_crossentropy
# optimizer : 경사 하강법 알고리즘 중 하나인 adam 사용
# metrics : 평가 척도를 나타내며 분류 문제에서는 일반적으로 accuracy 지정

hist=model.fit(X_train, y_train, epochs = 100, batch_size= 10, validation_data=(X_test, y_test), verbose=2)
# epochs : 전체 훈련 데이터셋에 대해 학습 반복 횟수를 지정.
# batch_size : 가중치를 업데이트할 배치 크기

scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
scores1 = model.evaluate(X_train, y_train, verbose=0)
print("%s: %.2f%%" %(model.metrics_names[1], scores1[1]*100))

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.grid()
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.grid()
plt.show()

# Hidden Weight 시각화
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(15, ), learning_rate_init=0.001, #batch_size=1, verbose=2,
                    max_iter=1000, random_state=0)
mlp.fit(X_train, y_train)

print("훈련 세트 정확도: {:.2f}".format(mlp.score(X_train, y_train)))
print("테스트 세트 정확도: {:.2f}".format(mlp.score(X_test, y_test)))

plt.figure(figsize= (8, 8))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')  # coefs : (입력10 x 은닉 가중치15)
plt.yticks(range(10), features)
plt.xlabel("hidden weight")
plt.ylabel("input")
plt.colorbar()
plt.show()

#eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
#               for i in range(len(eigen_vals))]

#eigen_pairs.sort(key=lambda k: k[0], reverse=True)
#w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
#              eigen_pairs[1][1][:, np.newaxis]))

#print('Matrix W: \n', w)
#X_train_std[0].dot(w)
#X_train_pca = X_train_std.dot(w)
#colors = ['r', 'b', 'g']
#markers = ['s', 'x', 'o']

#for l, c, m in zip(np.unique(y_train), colors, markers):
#    plt.scatter(X_train_pca[y_train ==  l, 0],
#                X_train_pca[y_train ==  l, 1],
#                c=c, label = l, marker = m)
#plt.legend(loc='lower left')
#plt.tight_layout()
#plt.show()