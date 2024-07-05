from zenml.pipelines import pipeline
from steps.ingest_data import ingest_data, IngestDataParams
from steps.clean_and_split_data import clean_and_split_data
from steps.define_model import define_model
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model

@pipeline
def train_pipeline(ingest_data, clean_and_split_data, define_model, train_model, evaluate_model):
    df = ingest_data()
    X_train, X_test, y_train, y_test = clean_and_split_data(df)
    model = define_model()
    trained_model = train_model(model, X_train, y_train)
    evaluation_metrics = evaluate_model(trained_model, X_test, y_test)

