import time
import pandas as pd
import matplotlib

import warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

'''
while True:
    with open('/Users/asagnihotri/RL-Bitcoin-trading-bot/RL-Bitcoin-trading-bot_4/BTCUSDT_1m_50000__20220317.csv','r') as file:
        data = file.readlines() 
    lastRow = data[-1]
    print(lastRow)
    time.sleep(60)
'''

#create empty file with header
headerList = ['Date', 'Open', 'Close', 'Low', 'High', 'Volumn','Decision']
df=pd.DataFrame(columns=headerList)
df.to_csv('/Users/asagnihotri/RL-Bitcoin-trading-bot/RL-Bitcoin-trading-bot_4/BTCUSDT_1m_50000__20220317_supervised.csv', index=False)


def addDecisionColumn(entry, lookback, open_position = False):
    while True:
        buytime = 0
        input = pd.read_csv('/Users/asagnihotri/RL-Bitcoin-trading-bot/RL-Bitcoin-trading-bot_4/unlabelled_live_data.csv')
        latest_rec = input.iloc[-1:]
        #print(latest_rec)

        lookbackperiod = input.iloc[lookback:]
        cumret = (lookbackperiod.Close.pct_change() +1).cumprod() -1
        if not open_position:
            print(cumret[cumret.last_valid_index()])
            if cumret[cumret.last_valid_index()] > entry:
                latest_rec['Decision'] = 'Buy'
                buytime = latest_rec['Date']
                latest_rec.to_csv('/Users/asagnihotri/RL-Bitcoin-trading-bot/RL-Bitcoin-trading-bot_4/labelled_live_data.csv', mode='a', index=False, header=False)
                open_position = True
                break
        
        latest_rec['Decision'] = 'Hold'
        latest_rec.to_csv('/Users/asagnihotri/RL-Bitcoin-trading-bot/RL-Bitcoin-trading-bot_4/labelled_live_data.csv', mode='a', index=False, header=False)
        
        print(latest_rec)
        
        time.sleep(60)
                
                
    if open_position:
        while True:
            input = pd.read_csv('/Users/asagnihotri/RL-Bitcoin-trading-bot/RL-Bitcoin-trading-bot_4/unlabelled_live_data.csv')
            latest_rec = input.iloc[-1:]
            #print(latest_rec)

            sincebuy = input.loc[latest_rec['Date'] > buytime]

            if len(sincebuy) > 1:
                sincebuyret = (sincebuy.Close.pct_change() +1).cumprod() -1
                last_entry = sincebuyret[sincebuyret.last_valid_index()]
                if last_entry > 0.0015 or last_entry < -0.0015:
                    latest_rec['Decision'] = 'Sell'
                    latest_rec.to_csv('/Users/asagnihotri/RL-Bitcoin-trading-bot/RL-Bitcoin-trading-bot_4/labelled_live_data.csv', mode='a', index=False, header=False)
                    break
                    
        latest_rec['Decision'] = 'Hold'
        latest_rec.to_csv('/Users/asagnihotri/RL-Bitcoin-trading-bot/RL-Bitcoin-trading-bot_4/labelled_live_data.csv', mode='a', index=False, header=False)
        
        print(latest_rec)
        
        time.sleep(60)

if __name__ == "__main__":
    addDecisionColumn(0.001, 50)
