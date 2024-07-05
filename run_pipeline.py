from zenml.client import Client
from pipelines.training_pipeline import train_pipeline
from steps.ingest_data import IngestDataParams, ingest_data
from steps.clean_and_split_data import clean_and_split_data
from steps.define_model import define_model
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model

if __name__ == "__main__":
    uri = Client().active_stack.experiment_tracker.get_tracking_uri()
    print(f"MLflow Tracking URI: {uri}")

    data_path = "./data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    
    # Create parameter instances
    ingest_data_params = IngestDataParams(data_path=data_path)
    
    # Initialize the pipeline
    pipeline_instance = train_pipeline(
        ingest_data=ingest_data(params=ingest_data_params),
        clean_and_split_data=clean_and_split_data(),
        define_model=define_model(),
        train_model=train_model(),
        evaluate_model=evaluate_model()
    )
    
    # Run the pipeline
    pipeline_instance.run()

    print(
        "Now run \n"
        f"    mlflow ui --backend-store-uri '{uri}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs."
    )
