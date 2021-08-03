
import joblib

# load the model
loaded_model = joblib.load('maison.pkl')

pred = loaded_model.predict([[3000, 234, 3, 2, 4, 2, 3, 4, 2, 1, 1]])
print(pred)
