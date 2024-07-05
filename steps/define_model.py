from zenml.steps import step
from sklearn.linear_model import LogisticRegression

@step
def define_model():
    return LogisticRegression()

