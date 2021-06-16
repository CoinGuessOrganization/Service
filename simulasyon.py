import json
import numpy as np
import pandas as pd
from tensorflow import keras
from csv import writer


df3 =  pd.read_csv('BTC/test_dataset.csv')
testClosingData = df3['Closing Price (USD)'].values
testDate = df3['Date'].values 

tahmindeger = 1000.0
tahminicoin= 0.0
gercekdeger = 1000.0
gercekcoin= 0.0
coin= 0.0
baslangıcdegeri = 9464.2280968215

for i in range(testClosingData.size):
    fields = ["BTC",testDate[i],testClosingData[i]," "," "," "]
    with open('BTC/train_dataset.csv', 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(fields)
        
    df1 =  pd.read_csv('BTC/train_dataset.csv')
    model = keras.models.load_model('BTC/model')
    look_back = 20
    

    df2 =  pd.read_csv('BTC/train_dataset.csv')
    df2.drop(columns=['24h Open (USD)', '24h High (USD)', '24h Low (USD)'], inplace=True)
    dates = df2['Date']
    dates = dates.tolist()

    from datetime import date, timedelta
    
    today = date.today()
   
    
    start_date = (pd.to_datetime(str(dates[len(dates)-1])) + timedelta(days=1))
    start_date = str(start_date).split(" ")[0]
 
    df1['Date'] = pd.to_datetime(df1['Date'])
    df1.set_axis(df1['Date'], inplace=True)
    df1.drop(columns=['24h Open (USD)', '24h High (USD)', '24h Low (USD)'], inplace=True)
    
    close_data1 = df1['Closing Price (USD)'].values
    
    close_data1 = close_data1.reshape((-1))
    
    def predict(num_prediction, model):
        prediction_list = close_data1[-look_back:]
        
        for _ in range(num_prediction):
            x = prediction_list[-look_back:]
            x = x.reshape((1, look_back, 1))
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[look_back-1:]
            
        return prediction_list
        
    def predict_dates(num_prediction):
        last_date = df1['Date'].values[-1]
        prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
        return prediction_dates
    
    num_prediction = 7
    forecast = predict(num_prediction, model)

    lists = forecast.tolist()
    data = close_data1.tolist()
    
    if(i == 0 and (baslangıcdegeri < lists[1])):
            tahminicoin = tahmindeger / baslangıcdegeri
            gercekcoin = gercekdeger / baslangıcdegeri
            tahmindeger= tahmindeger-100.0
            gercekdeger= gercekdeger-100.0
    else:
        print("bugün ",testClosingData[i-1]," tahmin ",lists[1])
        if(testClosingData[i-1] < lists[1] and tahmindeger > 0.0):
            if(tahmindeger - 100.0 >= 0.0):
                tahminicoin += 100.0 / testClosingData[i-1]
                tahmindeger = tahmindeger - 100.0
                print("girdi aldı")
            else:
                tahminicoin += tahmindeger / testClosingData[i-1]
                tahmindeger = 0.0
                print("girdi aldı")
                
        elif(testClosingData[i-1] > lists[1] and tahminicoin > 0.0):
            coin = 100.0 / testClosingData[i-1]
            if(tahminicoin - coin >= 0.0):
                tahmindeger += coin * testClosingData[i-1]
                tahminicoin = tahminicoin - coin
                print("girdi sattı")
            else:
                tahmindeger += tahminicoin * testClosingData[i-1]
                tahminicoin = 0.0
                print("girdi sattı")
                    
        if(testClosingData[i-1] < testClosingData[i] and gercekdeger > 0.0):
            if(gercekdeger - 100.0 >= 0.0):
                gercekcoin += 100.0 / testClosingData[i-1]
                gercekdeger = gercekdeger - 100.0
            else:
                gercekcoin += gercekdeger / testClosingData[i-1]
                gercekdeger = 0.0
                
        elif(testClosingData[i-1] > testClosingData[i] and gercekcoin > 0.0):
            coin = 100.0 / testClosingData[i-1]
            if(gercekcoin - coin >= 0.0):
                gercekdeger += coin * testClosingData[i-1]
                gercekcoin = gercekcoin - coin
            else:
                gercekdeger += gercekcoin * testClosingData[i-1]
                gercekcoin = 0.0

            
    if(i == 29 or i == 89 or i == 179):
            print("Gerçek Değer",gercekdeger)
            print("Tahmini Değer",tahmindeger)
            print("Gerçek Coin",gercekcoin)
            print("Tahmini Coin",tahminicoin)
       
    print(i)

print("Gerçek Değer",gercekdeger)
print("Tahmini Değer",tahmindeger)
print("Gerçek Coin",gercekcoin)
print("Tahmini Coin",tahminicoin)
    