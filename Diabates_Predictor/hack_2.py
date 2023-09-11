import numpy as np
import pandas as pd
import tensorflow as tf
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os.path

def create_model(my_inputs, my_learning_rate, METRICS,my_outputs):
  model = tf.keras.Model(inputs=my_inputs, outputs=my_outputs)
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),loss=tf.keras.losses.BinaryCrossentropy(),metrics=METRICS)
  return model

def get_outputs_dnn():
  dense_output = tf.keras.layers.Dense(units=16,activation='sigmoid',name='hidden_dense_layer_1')(concatenated_inputs)
  dense_output = tf.keras.layers.Dense(units=10,activation='sigmoid',name='hidden_dense_layer_2')(dense_output)
#   dense_output = tf.keras.layers.Dense(units=8,activation='sigmoid',name='hidden_dense_layer_3')(dense_output)
#   dense_output = tf.keras.layers.Dense(units=8,activation='sigmoid',name='hidden_dense_layer_4')(dense_output)
  dense_output = tf.keras.layers.Dense(units=1,name='dense_output')(dense_output)
  outputs = {
    'dense_output': dense_output
  }

  return outputs

def train_model(model, dataset, epochs, label_name,batch_size,validation_split=0.1):
  features = {name:np.array(value) for name, value in dataset.items()}
  label = np.array(features.pop(label_name))
  history = model.fit(x=features, y=label, batch_size=batch_size,epochs=epochs, shuffle=True, validation_split=validation_split)
  epochs = history.epoch
  hist = pd.DataFrame(history.history)
  print("Defined the create_model and train_model functions.")
  return epochs, hist

def plot_curve(epochs, hist, list_of_metrics):
  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Value")

  for m in list_of_metrics:
    x = hist[m]
    plt.plot(epochs[1:], x[1:], label=m)

  plt.legend()
  print("Defined the plot_curve function.")

data = pd.read_csv(r"C:\Users\priya\Downloads\diabetes_binary_5050split_health_indicators_BRFSS2015.csv\diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
test1_data = pd.read_csv(r"C:\Users\priya\Downloads\diabetes_binary_health_indicators_BRFSS2015.csv\diabetes_binary_health_indicators_BRFSS2015.csv")
data = data.reindex(np.random.permutation(data.index))
data["Diabetes_binary"] = data["Diabetes_binary"].astype(int)
data["HighBP"] = data["HighBP"].astype(int)
data["HighChol"] = data["HighChol"].astype(int)
data["CholCheck"] = data["CholCheck"].astype(int)
data["BMI"] = data["BMI"].astype(int)
data["Smoker"] = data["Smoker"].astype(int)
data["Stroke"] = data["Stroke"].astype(int)
data["HeartDiseaseorAttack"] = data["HeartDiseaseorAttack"].astype(int)
data["PhysActivity"] = data["PhysActivity"].astype(int)
data["Fruits"] = data["Fruits"].astype(int) 
data["Veggies"] = data["Veggies"].astype(int)
data["HvyAlcoholConsump"] = data["HvyAlcoholConsump"].astype(int)
data["AnyHealthcare"] = data["AnyHealthcare"].astype(int)
data["NoDocbcCost"] = data["NoDocbcCost"].astype(int)
data["GenHlth"] = data["GenHlth"].astype(int)
data["MentHlth"] = data["MentHlth"].astype(int)
data["PhysHlth"] = data["PhysHlth"].astype(int)
data["DiffWalk"] = data["DiffWalk"].astype(int)
data["Sex"] = data["Sex"].astype(int)
data["Age"] = data["Age"].astype(int)
data["Education"] = data["Education"].astype(int)
data["Income"] =data["Income"].astype(int)
data.drop_duplicates(inplace = True)
train_df_norm, test_df_norm = train_test_split(data, test_size=0.2)
# train_df_mean = train_df.mean()
# train_df_std = train_df.std()
# train_df_norm = (train_df - train_df_mean)/train_df_std
# test_df_norm = (test_df - train_df_mean) / train_df_std
# train_df_norm['Diabetes_binary']=train_df['Diabetes_binary']
# test_df_norm['Diabetes_binary']=test_df['Diabetes_binary']
# train_df.drop(['Sex', 'Fruits','Veggies','NoDocbcCost','AnyHealthcare','CholCheck'], axis=1,inplace=True)
# test_df.drop(['Sex', 'Fruits','Veggies','NoDocbcCost','AnyHealthcare','CholCheck'], axis=1,inplace=True)
# print(train_df_norm.head())
# print(train_df_norm.describe())
if os.path.isfile('medi_new.h5') is False:
    inputs = {
        'HighBP':
            tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                                name='HighBP'),
        'HighChol':
            tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                                name='HighChol'),
        'GenHlth':
            tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                                name='GenHlth'),
        # 'CholCheck':
        #     tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
        #                           name='CholCheck'),
        'BMI':
            tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                                name='BMI'),                          
        'Smoker':
            tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                                name='Smoker'),
        'Stroke':
            tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                                name='Stroke'),
        'HeartDiseaseorAttack':
            tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                                name='HeartDiseasorAttack'),
        'PhysActivity':
            tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                                name='PhysActivity'),
        # 'Fruits':
        #     tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
        #                           name='Fruits'),
        # 'Veggies':
        #     tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
        #                           name='Veggies'),                          
        'HvyAlcoholConsump':
            tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                                name='HvyAlcoholConsump'),
        # 'AnyHealthcare':
        #     tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
        #                           name='AnyHealthcare'),
        # 'NoDocbcCost':
        #     tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
        #                           name='NoDocbcCost'),
        'MentHlth':
            tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                                name='MentHlth'),
        'PhysHlth':
            tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                                name='PhysHlth'),                          
        'DiffWalk':
            tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                                name='DiffWalk'),
        # 'Sex':
        #     tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
        #                           name='Sex'),
        'Age':
            tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                                name='Age'),
        'Education':
            tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                                name='Education'),                          
        'Income':
            tf.keras.layers.Input(shape=(1,), dtype=tf.float32,
                                name='Income'),
    }
    concatenated_inputs = tf.keras.layers.Concatenate()(inputs.values())

    learning_rate = 0.0001
    epochs = 30
    batch_size = 25
    label_name = "Diabetes_binary"
    validation_split = 0.1
    classification_threshold = 0.53
    dnn_outputs = get_outputs_dnn()
    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy',threshold=classification_threshold),
        tf.keras.metrics.Precision(thresholds=classification_threshold, name='precision'),
        tf.keras.metrics.Recall(thresholds=classification_threshold,name="recall"),
    ]
    my_model=None
    my_model = create_model(inputs, learning_rate, METRICS,dnn_outputs)
    epochs, hist = train_model(my_model, train_df_norm, epochs,label_name, batch_size)
    list_of_metrics_to_plot = ['accuracy','precision','recall']

    plot_curve(epochs, hist, list_of_metrics_to_plot)

    features = {name:np.array(value) for name, value in test_df_norm.items()}
    label = np.array(features.pop(label_name))

    my_model.evaluate(x = features, y = label, batch_size=batch_size)
    print(my_model.summary())
    if os.path.isfile('medi_new.h5') is False:
        my_model.save('medi_new.h5')
else:
    model = tf.keras.models.load_model("medi_new.h5")
    batch_size = 25
    label_name = "Diabetes_binary"

    features = {name:np.array(value) for name, value in test1_data.items()}
    label = np.array(features.pop(label_name))

    model.evaluate(x = features, y = label, batch_size=batch_size)
    HighBP=int(input("Enter HighBP:"))
    HighChol=int(input("Enter HighChol:"))
    GenHlth=int(input("Enter GenHlth:"))
    CholCheck=int(input("Enter CholCheck:"))
    BMI=int(input("Enter BMI:"))
    Smoker=int(input("Enter Smoker:"))
    Stroke=int(input("Enter Stroke:"))
    HeartDiseaseorAttack=int(input("HeartDiseaseorAttack:"))
    PhysActivity=int(input("PhysActivity:"))
    Fruits=int(input("Enter Fruits:"))
    Veggies=int(input("Enter Veggies:"))
    HvyAlcoholConsump=int(input("Enter HvyAlcoholConsump:"))
    AnyHealthcare=int(input("AnyHealthcare:"))
    NoDocbcCost=int(input("Enter NoDocbcCost:"))
    MentHlth=int(input("Enter MentHlth:"))
    PhysHlth=int(input("Enter PhysHlth:"))
    DiffWalk=int(input("Enter DiffWalk:"))
    Sex=int(input("Enter Sex:"))
    Age=int(input("Enter Age:"))
    Education=int(input("Enter Education:"))
    Income=int(input("Enter Income:"))
    input_data = {
    'HighBP': np.array([HighBP]),
    'HighChol': np.array([HighChol]),
    'GenHlth': np.array([GenHlth]),
    'BMI': np.array([BMI]),
    'Smoker': np.array([Smoker]),
    'Stroke': np.array([Stroke]),
    'HeartDiseaseorAttack': np.array([HeartDiseaseorAttack]),
    'PhysActivity': np.array([PhysActivity]),
    'HvyAlcoholConsump': np.array([HvyAlcoholConsump]),
    'MentHlth': np.array([MentHlth]),
    'PhysHlth': np.array([PhysHlth]),
    'DiffWalk': np.array([DiffWalk]),
    'Age': np.array([Age]),
    'Education': np.array([Education]),
    'Income': np.array([Income])
    }
    predictions = model.predict(input_data)
    prediction=predictions['dense_output'][0][0]
    print(prediction)
    if prediction >=0.53:
       print("U may have diabetes")
    else:
       print("No you may not have diabates")