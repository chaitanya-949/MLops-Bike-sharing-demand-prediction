import dill, os
p = r"artifact\01_01_2026_17_46_27\model_trainer\trained_model\model.pkl"
print('path exists', os.path.exists(p))
obj = None
with open(p,'rb') as f:
    obj = dill.load(f)
print('loaded obj type:', type(obj))
if hasattr(obj, 'trained_model_object'):
    trained = obj.trained_model_object
    print('trained model type:', type(trained))
    print('n_features_in_:', getattr(trained, 'n_features_in_', None))
    try:
        print('feature_importances (len):', len(getattr(trained, 'feature_importances_', [])))
    except Exception:
        pass
else:
    print('object has no trained_model_object attribute, repr:', repr(obj)[:200])
