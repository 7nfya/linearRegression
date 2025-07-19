from LinearData import singleFeatureX as dataX, singleFeatureY as dataY 

class SingleFeatureLR:
    """
    A simple implementation of linear regression using gradient descent
    for one feature (x) and one target (y).

    Attributes:
        dataX (list[int]): Input feature values (e.g., hours studied).
        dataY (list[int]): Target values (e.g., test scores).
        w (float): Weight or slope of the linear function.
        b (float): Bias or y-intercept of the linear function.
        alpha (float): Learning rate for gradient descent.
        epoch (int): Number of training iterations.
        display (bool): Whether to print cost and parameters during training.
    """

    def __init__(self, dataX : list[int] = dataX, dataY: list[int] = dataY, w : int = 0, b : int = 0, alpha : int = 0.01, epoch : int = 1000, display : bool = False) -> None:
        self.dataX = dataX
        self.dataY = dataY
        self.w = w
        self.b = b
        self.alpha = alpha
        self.epoch = epoch
        self.display = display

    def gradientDescent(self) -> None:
        
        tmpW = self.w
        tmpB = self.b
        m = len(self.dataX)

        for i in range(self.epoch):
            dw = 0
            db = 0
            for j in range(m):
                dw += (tmpW * self.dataX[j] + tmpB - self.dataY[j]) * self.dataX[j]
                db += tmpW * self.dataX[j] + tmpB - self.dataY[j]

            dw *= 2/m
            db *= 2/m
            
            tmpW -= self.alpha * dw
            tmpB -= self.alpha * db

            if not i % 100 and self.display:
                cost = sum([tmpW * self.dataX[j] + tmpB - self.dataY[j] for j in range(m)])**2 / m
                print(f"cost: {cost:.2f}, linear function: f(x): wx + b \nw = {tmpW:.2f}, b = {tmpB:.2f} ")
        
        self.w = tmpW
        self.b = tmpB

    def predict(self, hours : int = 3) -> str:
        return f"-=-=-=-=-=-=-=-=\nnumber of hours : {hours:.2f}\npredicted score: {self.w * hours + self.b:.2f}\n-=-=-=-=-=-=-=-="
    
o1 = SingleFeatureLR(display=True)
o1.gradientDescent()
print(o1.predict())
