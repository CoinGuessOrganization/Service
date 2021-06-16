
from flask import Flask
import json
import numpy as np
import pandas as pd

app = Flask(__name__)

# Make the WSGI interface available at the top level so wfastcgi can get it.
wsgi_app = app.wsgi_app


@app.route('/')
def hello():
    """Renders a sample page."""
    return "Hello CoinGuess"

@app.route('/flaskweb/api/BTC', methods=['GET'])
def get_BTC():
    from tensorflow import keras
    df1 =  pd.read_csv('BTC/test_dataset.csv')
    model = keras.models.load_model('BTC/model')
    look_back = 20
    

    df2 =  pd.read_csv('BTC/test_dataset.csv')
    df2.drop(columns=['24h Open (USD)', '24h High (USD)', '24h Low (USD)'], inplace=True)
    dates = df2['Date']
    dates = dates.tolist()

    from datetime import date, timedelta
    
    today = date.today()
   
    
    start_date = (pd.to_datetime(str(dates[len(dates)-1])) + timedelta(days=1))
    start_date = str(start_date).split(" ")[0]

    if(start_date != today):
        import requests
        


        response = requests.get(
            'https://rest.coinapi.io/v1/ohlcv/BITSTAMP_SPOT_BTC_USD/history?period_id=1DAY&time_start='+start_date,
            headers={'X-CoinAPI-Key': '1508A524-4CCF-4CCD-9010-D1827096D6D9'},
        )
        
        res = json.loads(response.text)
        
        from csv import writer
        
        for i in res:
            print(i["price_close"])
            fields = ["BTC",i["time_period_start"].split("T")[0],i["price_close"]," "," "," "]

            with open('BTC/test_dataset.csv', 'a+', newline='') as write_obj:
                # Create a writer object from csv module
                csv_writer = writer(write_obj)
                # Add contents of list as last row in the csv file
                csv_writer.writerow(fields)
          
        df1 =  pd.read_csv('BTC/test_dataset.csv')
        df2 =  pd.read_csv('BTC/test_dataset.csv')
        df2.drop(columns=['24h Open (USD)', '24h High (USD)', '24h Low (USD)'], inplace=True)
        dates = df2['Date']
        dates = dates.tolist()
        
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
    
    num_prediction = 30
    forecast = predict(num_prediction, model)

    lists = forecast.tolist()
    data = close_data1.tolist()
    

    
    result = '{"predicts": ' + json.dumps(lists) + ', "historicalData": ' + json.dumps(data[-30:])  +', "dates" :'+ json.dumps(dates[-30:]) + '}'

    return result



@app.route('/flaskweb/api/ETH', methods=['GET'])
def get_ETH():
    from tensorflow import keras
    df1 =  pd.read_csv('ETH/test_dataset.csv')
    model = keras.models.load_model('ETH/model')
    look_back = 20
    
    df2 =  pd.read_csv('ETH/test_dataset.csv')
    df2.drop(columns=['24h Open (USD)', '24h High (USD)', '24h Low (USD)'], inplace=True)
    dates = df2['Date']
    dates = dates.tolist()


    from datetime import date, timedelta
    
    today = date.today()
   
    
    start_date = (pd.to_datetime(str(dates[len(dates)-1])) + timedelta(days=1))
    start_date = str(start_date).split(" ")[0]

    if(start_date != today):
        import requests
        


        response = requests.get(
            'https://rest.coinapi.io/v1/ohlcv/BITSTAMP_SPOT_ETH_USD/history?period_id=1DAY&time_start='+start_date,
            headers={'X-CoinAPI-Key': '1508A524-4CCF-4CCD-9010-D1827096D6D9'},
        )
        
        res = json.loads(response.text)
        
        from csv import writer
        
        for i in res:
            print(i["price_close"])
            fields = ["ETH",i["time_period_start"].split("T")[0],i["price_close"]," "," "," "]

            with open('ETH/test_dataset.csv', 'a+', newline='') as write_obj:
                # Create a writer object from csv module
                csv_writer = writer(write_obj)
                # Add contents of list as last row in the csv file
                csv_writer.writerow(fields)
          
        df1 =  pd.read_csv('ETH/test_dataset.csv')
        df2 =  pd.read_csv('ETH/test_dataset.csv')
        df2.drop(columns=['24h Open (USD)', '24h High (USD)', '24h Low (USD)'], inplace=True)
        dates = df2['Date']
        dates = dates.tolist()
        
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
    
    num_prediction = 30
    forecast = predict(num_prediction, model)

    lists = forecast.tolist()
    data = close_data1.tolist()


    
    result = '{"predicts": ' + json.dumps(lists) + ', "historicalData": ' + json.dumps(data[-30:])  +', "dates" :'+ json.dumps(dates[-30:]) + '}'

    return result

@app.route('/flaskweb/api/XRP', methods=['GET'])
def get_XRP():
    from tensorflow import keras
    df1 =  pd.read_csv('XRP/test_dataset.csv')
    model = keras.models.load_model('XRP/model')
    look_back = 20
    
    df2 =  pd.read_csv('XRP/test_dataset.csv')
    df2.drop(columns=['24h Open (USD)', '24h High (USD)', '24h Low (USD)'], inplace=True)
    dates = df2['Date']
    dates = dates.tolist()

    from datetime import date, timedelta
    
    today = date.today()
   
    
    start_date = (pd.to_datetime(str(dates[len(dates)-1])) + timedelta(days=1))
    start_date = str(start_date).split(" ")[0]

    if(start_date != today):
        import requests
        


        response = requests.get(
            'https://rest.coinapi.io/v1/ohlcv/BITSTAMP_SPOT_XRP_USD/history?period_id=1DAY&time_start='+start_date,
            headers={'X-CoinAPI-Key': '1508A524-4CCF-4CCD-9010-D1827096D6D9'},
        )
        
        res = json.loads(response.text)
        
        from csv import writer
        
        for i in res:
            print(i["price_close"])
            fields = ["XRP",i["time_period_start"].split("T")[0],i["price_close"],"0","0","0"]

            with open('XRP/test_dataset.csv', 'a+', newline='') as write_obj:
                # Create a writer object from csv module
                csv_writer = writer(write_obj)
                # Add contents of list as last row in the csv file
                csv_writer.writerow(fields)
                
        df2 =  pd.read_csv('XRP/test_dataset.csv')
        df2.drop(columns=['24h Open (USD)', '24h High (USD)', '24h Low (USD)'], inplace=True)
        dates = df2['Date']
        dates = dates.tolist()
        df1 =  pd.read_csv('XRP/test_dataset.csv')
        
   
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
    
    num_prediction = 30
    forecast = predict(num_prediction, model)

    lists = forecast.tolist()
    data = close_data1.tolist()


    
    result = '{"predicts": ' + json.dumps(lists) + ', "historicalData": ' + json.dumps(data[-30:])  +', "dates" :'+ json.dumps(dates[-30:]) + '}'

    return result




if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '8000'))
    except ValueError:
        PORT = 8000
    app.run('0.0.0.0', 5821)
