import pickle

# load the model

loaded_model = pickle.load(open('diabetes.pkl', 'rb'))

pred = loaded_model.predict([[10,20,30,40,50,10,20,10]])
print(pred)

if pred[0] == 1:
    print('The person is diabetic')
else:
    print('The person is not diabetic')    