import os
import joblib
import pandas as pd
import json
import yaml
from agents import label_workloads_with_gemini
from util import get_logger

# Initialize logger
logger = get_logger("main")


MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../model-pipeline/models')
)
ACTUATOR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../actuator'))


def load_models(models_dir=MODELS_DIR):
    """
    Loads all sklearn models from the given directory.
    Returns a dict of {model_filename: model_object}
    """
    models = {}
    for fname in os.listdir(models_dir):
        fpath = os.path.join(models_dir, fname)
        if os.path.isfile(fpath):
            try:
                model = joblib.load(fpath)
                models[fname] = model
            except Exception as e:
                logger.warning(f"Skipping {fname}: {e}")
    return models


def predict_with_models(input_csv, models=None, output_dir=ACTUATOR_DIR):
    """
    Loads input data from CSV, runs predictions for each model, and writes CSV outputs.
    Each output is named <model_filename>_predictions.csv in the actuator directory.
    """
    if models is None:
        models = load_models()
    df = pd.read_csv(input_csv)
    for model_name, model in models.items():
        try:
            preds = model.predict(df)
            out_df = df.copy()
            out_df['prediction'] = preds
            out_name = f"{model_name}_predictions.csv"
            out_path = os.path.join(output_dir, out_name)
            out_df.to_csv(out_path, index=False)
            logger.info(f"Predictions written to {out_path}")
        except Exception as e:
            logger.error(f"Prediction failed for {model_name}: {e}")


def predict_from_json(
    json_path, model=None, output_csv=os.path.join(ACTUATOR_DIR, 'recommendations.csv')
):
    """
    Reads workload info from JSON, prepares features, uses model to predict migration,
    and writes (workload_id, kind, label) to output_csv.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    # Feature engineering: select and convert relevant columns
    feature_cols = [
        'resources.cpu',
        'resources.memory',
        'pods_total',
        'pods_pending',
        'percent_pending',
        'timestamp',
        'cluster_load',
        'cluster_label',
        'cluster_cpu_capacity',
        'cluster_memory_capacity',
    ]
    X = df[feature_cols].copy()

    # Convert cpu/memory to numeric (e.g., '500m' -> 0.5, '1024Mi' -> 1024)
    def cpu_to_float(cpu):
        if isinstance(cpu, str) and cpu.endswith('m'):
            return float(cpu[:-1]) / 1000.0
        return float(cpu)

    def mem_to_float(mem):
        if isinstance(mem, str) and mem.endswith('Mi'):
            return float(mem[:-2])
        return float(mem)

    X['resources.cpu'] = X['resources.cpu'].apply(cpu_to_float)
    X['resources.memory'] = X['resources.memory'].apply(mem_to_float)
    X['cluster_cpu_capacity'] = X['cluster_cpu_capacity'].apply(cpu_to_float)
    X['cluster_memory_capacity'] = X['cluster_memory_capacity'].apply(mem_to_float)
    # Encode cluster_label
    X['cluster_label'] = X['cluster_label'].map({'private': 0, 'public': 1})
    # If model is not provided, load the first available model
    if model is None:
        models = load_models()
        if not models:
            raise RuntimeError('No models found in models directory')
        model = list(models.values())[0]
    # Predict
    y_pred = model.predict(X)
    # Output (workload_id, kind, label)
    result = df[['workload_id', 'kind']].copy()
    result['label'] = y_pred
    result.to_csv(output_csv, index=False)
    print(f"Recommendations written to {output_csv}")


def load_config(config_path=None):
    """
    Loads configuration from a YAML file
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    import pandas as pd
    
    # Load configuration from YAML file
    config = load_config()
    
    # Get input JSON file path from configuration
    json_input = config.get('data', {}).get('input_json')
    
    if json_input:
        # Make sure the path is absolute or relative to the current directory
        if not os.path.isabs(json_input):
            json_input = os.path.join(os.path.dirname(__file__), json_input)
            
        with open(json_input, 'r') as f:
            workloads = json.load(f)
        labels = label_workloads_with_gemini(workloads)
        # Salvar CSV com (workload_id, kind, label)
        df = pd.DataFrame(workloads)
        result = df[['workload_id', 'kind']].copy()
        result['label'] = labels
        output_csv = os.path.join(ACTUATOR_DIR, 'recommendations.csv')
        result.to_csv(output_csv, index=False)
        logger.info(f"Recommendations written to {output_csv}")
    else:
        logger.error('No input JSON file specified in the configuration')


if __name__ == "__main__":
    main()
