
import numpy as np

class LogisticRegression:
    def __init__(self, max_iter=500, penalty="l2", initialize = "one", random_seed = 1213):

        self.author = __author__
        self.id = __id__
        
        self.max_iter = max_iter
        self.penalty = penalty
        self.initialize = initialize
        self.random_seed = random_seed
        self.lr = 0.1
        self.lamb = 0.01
        np.random.seed(self.random_seed)

        if self.penalty not in ["l1", "l2"]:
            raise ValueError("Penalty must be l1 or l2")

        if self.initialize not in ["one", "LeCun", "random"]:
            raise ValueError("Only [LeCun, One, random] Initialization supported")

    def activation(self, z):

        a = 1 / (1 + np.exp(-z))        # 시그모이드 함수 공식 : 1/(1+exp(-x)) = exp(x)/(exp(x)+1)
        return a

    def fwpass(self, x):
        """
        x가 주어졌을 때, 가중치 w와 bias인 b를 적절히 x와 내적하여
        아래의 식을 계산하시요.

        z = w1*x1 + w2*x2 + ... wk*xk + b
        """

        z =x.dot(self.w) + self.b # 이 부분에 w1*x1 + w2*x2 + ... wk*xk + b 의 값을 가지도록 계산하시오. (넘파이 행렬 활용 추천) Code Here!
        #(num_data,) 형태의 데이터가 결과값으로 나온다.
        z = self.activation(z)
        return z


    def bwpass(self, x, err):
        """
        x와 오차값인 err가 들어왔을 때, w와 b에 대한 기울기인 w_grad와 b_grad를 구해서 반환하시오.
        l1, l2을 기반으로한 미분은 https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261
        이 문서를 확인하세요.

        w_grad는 (num_data, num_features)
        b_grad는 (num_data, )

        의 데이터 shape을 가지는 넘파이 어레이 입니다.
        어떠한 방법을 활용해서 w_grad를 계산해도 괜찮으며, 이 부분에서 속도의 차이가 발생함.

        self.lamb을 통해 lambda를 활용하세요.
        """
        if self.penalty == "l1":
            w_grad = 2*(np.dot(x.T,err))/x.shape[0] + self.lamb # 이 부분의 코드를 채우세요. Code Here!
            #l1, dw := 2*x*err + lambda
        elif self.penalty == "l2":
            w_grad = 2*(np.dot(x.T,err))/x.shape[0] - 2*self.lamb*self.initialize_w(x) # 이 부분의 코드를 채우세요. Code Here!
            #l2, dw := 2*x*err + 2*lambda*w
        b_grad = sum(err)/x.shape[0]

        return w_grad, b_grad

    def initialize_w(self, x):
        """
        https://reniew.github.io/13/ 의 LeCun 초기화 수식을 참고하여
        LeCun 가중치 초기화 기법으로 초기 w를 설정할 수 있도록 코딩 (힌트: np.random.uniform 활용)
        동일하게 랜덤한 값으로 w가중치를 초기화 하세요.

        단, numpy 만을 사용하여야 하며, 다른 라이브러리는 사용할 수 없습니다.
        w_library에서 one과 같은 shape이 되도록 다른 값을 설정하세요.
        """
        w_library = {
            "one":np.ones(x.shape[1]),
            "LeCun":np.random.uniform(-np.sqrt(3 / x.shape[0]),np.sqrt(3 / x.shape[0]),x.shape[1]), # LeCun 식을 활용하여 w가중치를 초기화 할 수 있도록 수식을 작성하세요. Code Here!
            # lecun uniform : U(-limit,limit), limit=sqrt(3 / fan-in)
            "random":np.random.rand(x.shape[1]) # 랜덤한 0~1사이의 값으로 w가중치를 초기화 할 수 있도록 수식을 작성하세요. Code Here!
        }

        return w_library[self.initialize]


    def fit(self, x, y):
        """
        실제로 가중치를 초기화 하고, 반복을 진행하며 w, b를 미분하여 계산하는 함수입니다.
        다른 함수를 통하여 계산이 수행됨.
        self.w, self.b 의 업데이트 과정만 코딩하세요.
        """
        self.w = self.initialize_w(x)
        self.b = 0
        for _ in range(self.max_iter):
            z = self.fwpass(x)
            err = -(y - z)
            w_grad, b_grad = self.bwpass(x, err)
            #(712,7) (712,)
            # w를 w_grad를 활용하여 업데이트하시오. Code Here!
            # 각 gradient에 learning_rate을 곱한 후 평균을 활용하여 값을 업데이트
            # 어떠한 방법을 사용해서 업데이트 해도 좋으며, 이 부분에서 속도의 차이가 발생함.
            self.w -=self.lr * w_grad # 이미 만든 함수를 쓰면 된다
            #w := w-lr *dw
            # b를 b_grad를 활용하여 업데이트 하시오. Code Here!
            # 어떠한 방법을 사용해서 업데이트 해도 좋으며, 이 부분에서 속도의 차이가 발생함.
            self.b -= self.lr * b_grad
            #b := b-lr*db

    def predict(self, x):
        """
        test용 x가 주어졌을 때, fwpass를 통과한 값을 기반으로
        0.5이상인 경우 1, 0.5이하인 경우 0을 반환하시오.
        """
        z = self.fwpass(x)

        # Code Here!
        a=np.where(z>=0.5,1,0)
        return a

    def score(self, x, y):
        return np.mean(self.predict(x) == y)
