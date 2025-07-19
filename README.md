# linearRegression
Linear Regression models I built from scratch, LITERAL SCRATCH.

# Linear Regression AI
This is a simple linear regression algorithm implemented from scratch in Python using gradient descent.
given the number of hours studied, it calculates your score.
## Features
- Manually written training loop
- Cost function (Mean Squared Error)
- Gradient descent optimizer
- No external ML libraries (pure Python not a single library)
  

## Sample Prediction
```python
from LinearData import singleFeatureX, singleFeatureY
print(SingleFeatueLR(singleFeatureX, singleFeatureY, w=0, b=0, alpha=0.01, epoch=1000, display=True).predict(2)
