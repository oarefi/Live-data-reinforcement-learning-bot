from config import *

from binance.client import Client
import ta
import numpy as np
import time
import pandas as pd

#for live data
import websocket, json

#variables
symbol = 'BTCUSDT'
cc = 'btcusdt'
interval = '1m'
records = '30000'

socket = f'wss://stream.binance.com:9443/ws/{cc}@kline_{interval}'


#function for historical data download
#create client object for exchange
client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

def GetOHLCVDataHist(symbol, interval, lookback=10):
    data = pd.DataFrame(client.get_historical_klines(symbol, interval, lookback + ' min ago UTC'))
    
    data = data.iloc[:,:6]
    data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    #frame = frame.set_index('Time')
    data['Date'] = pd.to_datetime(data['Date'], unit='ms')
    data['Date'] = data['Date'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    data[['Open', 'High', 'Low', 'Close', 'Volume']] = data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float).round(3)
    return data


#function for live data download
def GetOHLCVDataLive(ws, message):
    #print(message)
    json_message = json.loads(message)
    candle = json_message['k']
    
    is_candle_closed = candle['x']
    close = candle['c']
    high = candle['h']
    low = candle['l']
    volume = candle['v']
    
    #print(candle)
    
    if is_candle_closed == True:
        l=[]
        for k in candle.keys():
            l.append(candle[k])
            candle[k] = l
            l=[]
        df = pd.DataFrame(candle) 
        df = df[['t','o','c','l','h','v']]
        df.rename(columns = {'t':'Date', 'o':'Open', 'c':'Close', 'l':'Low', 'h':'High','v':'Volume'}, inplace = True)
        df = df[['Date','Open','High','Low','Close','Volume']]
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df['Date'] = df['Date'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float).round(3)
        
        df.to_csv('/Users/asagnihotri/RL-Bitcoin-trading-bot/RL-Bitcoin-trading-bot_4/BTCUSDT_1m_50000__20220317.csv', mode='a', index=False, header=False)
        #print(df)

def GetOHLCVDataLiveOnClose(ws, close_status_code, close_msg):
    print("### closed ###")


#Call historical
df = GetOHLCVDataHist(symbol, interval, records)

df.to_csv('/Users/asagnihotri/RL-Bitcoin-trading-bot/RL-Bitcoin-trading-bot_4/BTCUSDT_1m_50000__20220317.csv', index=False)

#Call live
ws = websocket.WebSocketApp(socket ,on_message=GetOHLCVDataLive, on_close=GetOHLCVDataLiveOnClose)
ws.run_forever()