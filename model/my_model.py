"""
AI 서비스에서 사용할 모델을 정의합니다.
"""

class MyModel():
    def __init__(self, param1: float = 1.0, param2: int = 10):
        """
        모델 초기화 함수

        Args:
            param1 (float): 예시 하이퍼파라미터 1
            param2 (int): 예시 하이퍼파라미터 2
        """
        self.param1 = param1
        self.param2 = param2
        self._is_trained = False

    def train(self, data):
        """
        데이터를 사용해서 모델을 학습하는 함수
        
        Args:
            data (Any): 학습 데이터
        """
        print(f"Training with param1={self.param1}, param2={self.param2}")
        # 학습 로직 구현
        self._is_trained = True

    def predict(self, input_data):
        """
        학습된 모델로 예측을 수행하는 함수
        
        Args:
            input_data (Any): 입력 데이터
        
        Returns:
            Any: 예측 결과
        """
        if not self._is_trained:
            raise RuntimeError("모델이 아직 학습되지 않았습니다. 먼저 train()을 호출하세요.")
        
        # 예측 로직 구현
        result = input_data * self.param1 + self.param2
        return result

    def __call__(self, input_data):
        """
        객체를 함수처럼 호출할 수 있도록 함 (predict를 호출)
        """
        return self.predict(input_data)