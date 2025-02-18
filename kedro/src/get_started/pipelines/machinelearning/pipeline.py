from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_data, evaluate_model, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data,
                inputs=["diabetes-dev", "params:model_options"],
                outputs="diabetes",
                name="preprocess_data_node"
            ),
            node(
                func=split_data,
                inputs=["diabetes", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["model", "X_test", "y_test"],
                outputs="evaluation-metrics",
                name="evaluate_model_node",
            ),
        ]
    )