
import numpy as np

# 아래에 코드를 작성해주세요.
class SVMClassifier:
    def __init__(self,n_iters=100, lr = 0.0001,random_seed=3, lambda_param=0.01):
   
        self.author = __author__
        self.id = __id__
        self.n_iters = n_iters # 몇 회 반복하여 적절한 값을 찾을지 정하는 파라미터
        self.lr = lr  # 학습률과 관련된 파라미터 
        self.lambda_param = lambda_param
        self.random_seed = random_seed
        np.random.seed(self.random_seed)


    def fit(self, x, y):
        """
        본 함수는 x, y를 활용하여 훈련하는 과정을 코딩하는 부분입니다.
        """
        n_samples, n_features = x.shape #(381,30)

        #y값을 SVM 계산에 활용해주기 위하여 0에 해당하는 y값들을 -1로 변환
        y_ =  y  
        for i in range(len(y_)) :
          if y_[i] == 0 :   
            y_[i] = -1     #y_에서 값이 0인 경우 -1로 바꿔준다.
        
        # 문제 2: w값 초기화, (n_features, )의 크기를 가지는 0과 1사이의 랜덤한 변수 어레이 (필수: 넘파이로 정의해야 함)
        init_w = np.random.rand(n_features) #0과 1사이의 랜덤한 변수를 어레이 형태로 init_w에 담는다. 
        self.w = init_w
        self.b = 0 # b값 초기화

        for _ in range(self.n_iters) : #100
            for i in range(n_samples): #X 한개의 행에 몇개의 요소가 있는가 (381)
                x_i = x[i]
                y_i = y_[i]
                # 문제 3: y(i) * (w · x(i) + b) >= 1 를 만족하는 경우의 의미가 담기도록 if문을 채우세요.
                condition =  x_i[ y_i*(self.w * x_i + self.b)>=1], self.w[ y_i*(self.w * x_i + self.b)>=1] 
                if condition:
                    self.w -= self.lr * (2 *self.lambda_param * self.w) # 문제 4: w에 대하여 Gradient Loss Function 수식을 이용하여 W를 업데이트 하세요.
                    # y(i) * (w · x(i) + b) >= 1 을 만족하는 경우는 Loss 값이 아니다.
                else:
                    # 그 외의 경우
                    self.w -= self.lr * (x_i*y_i - 2 *self.lambda_param * self.w) # 문제 5: w에 대하여 Gradient Loss Function 수식을 이용하여 W를 업데이트 하세요.
                    #False 값일 경우 hindge loss 공식을 이용하여 각 열의 오차를 구한다.
                    self.b -= self.lr * y_i

        return y_, init_w, condition


    def predict(self, x):
        """
        문제 6:
            [n_samples x features]로 구성된 x가 주어졌을 때, fit을 통해 계산된 
            self.W와 self.b를 활용하여 예측값을 계산합니다.

            @args:
                [n_samples x features]의 shape으로 구성된 x
            @returns:
                [n_samples, ]의 shape으로 구성된 예측값 array

            아래의 수식과 수도코드를 참고하여 함수를 완성하면 됩니다.
                approximation = W·X - b
                if approximation >= 0 {
                    output = 1
                }
                else{
                    output = 0
                }
        """
        for i in range(n_samples) :
          approximation = self.w* x[i] - self.b
          y_pred=[]
          if approximation >= 0 :
            output = 1
            y_pred.append(output)
          else :
            output = -1
            y_pred.append(output)
          y_pred = np.array(y_pred)

        return y_pred

    def get_accuracy( y_true, y_pred):
        """
            y_true, y_pred가 들어왔을 때, 정확도를 계산하는 함수.
            sklearn의 accuracy_score 사용 불가능 / sklearn의 accuracy_score 함수를 구현
            넘파이만을 활용하여 정확도 계산 함수를 작성하세요.

            주의사항: 
            라벨의 개수는 0,1로 이진 분류가 아니라, 3개 이상일 수 있음
        """
        len_data=len(y_true)
        True_NP=[]
        for i in range(len_data) :
          if y_true[i] == y_pred[i] :
            True_NP.append(y_true[i])
            acc= len(True_NP)/len_data *100
        return  "정확도는 {0:0.2f}% 입니다.".format(acc)

    def score(self, x, y):
        """
        5fold cross validation 함수를 사용하기 위해 필요한 함수입니다.
        """
        y_pred = self.predict(x)
        return self.get_accuracy(y, y_pred)


    def get_params(self):
        """
        5fold-cross validation을 수행하기 위하여 필요한 함수입니다.
        """
        return {
            "n_iters":self.n_iters,
            "lr" :self.lr,
            "lambda_param" :self.lambda_param,
            "random_seed" :self.random_seed
            }
