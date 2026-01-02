import os
from src.pipeline.training_pipeline import TrainPipeline
from src.utils.main_utils import load_object
import glob

# 1. Run full pipeline (ingest->validate->transform->train->evaluate->push)
print('Running training pipeline...')
TrainPipeline().run_pipeline()
print('Training pipeline finished.')

# 2. Find latest artifact model file
artifact_base = os.path.join('artifact')
models = glob.glob(os.path.join(artifact_base, '*', 'model_trainer', 'trained_model', 'model.pkl'))
if not models:
    print('No models found in artifact folders.')
else:
    model_path = models[-1]
    print('Loading model from', model_path)
    model_obj = load_object(model_path)
    print('Loaded model object type:', type(model_obj))
    # If it's MyModel wrapper, call predict with a simple DataFrame
    try:
        df = model_obj.trained_model_object  # just to inspect
        print('Inner trained model type:', type(model_obj.trained_model_object))
    except Exception:
        pass

print('Done')
