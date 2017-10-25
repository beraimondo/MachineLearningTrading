""" connect to oanda API ------------------------------------------------------------------------------------------- """
import oandapy as opy

oanda = opy.API(environment="practice",
                access_token="4d608abb7f46813447188432866d6698-7176a6a2d4443df441a471f2da62dd2f")


""" acquire data --------------------------------------------------------------------------------------------------- """
import pandas as pd

dataH = oanda.get_history(instrument="GBP_USD",  # our instrument ISOcurrency1_ISOcurrency2
                         start="2017-01-02",  # start data aaaa/mm/dd
                         end="2017-10-02",  # end date aaaa/mm/dd
                         granularity="H1")  # Candles: M=Monthly W= Weekly D=daily H1=Hourly M15=15minutes

dfH = pd.DataFrame(dataH["candles"]).set_index("time")

dfH.index = pd.DatetimeIndex(dfH.index)

dfH.info()


""" Print data ----------------------------------------------------------------------------------------------------- """
desired_width = 350
pd.set_option('display.width', desired_width)

print("H1 dataframe")
print(dfH)


#add columns with daily highest high, lowest low and close
# (1 of 2) resample to Business day, find the key daily value and reindex to hourly
dfDay = pd.DataFrame()
dfDay['HH'] = dfH['highBid'].resample('B').max().shift(1).reindex(dfH.index).fillna(method="ffill") #ffill fills values forward, bfill backward
dfDay['LL'] = dfH['lowAsk'].resample('B').min().shift(1).reindex(dfH.index).fillna(method="ffill")
dfDay['C'] = dfH['closeBid'].resample('B').last().shift(1).reindex(dfH.index).fillna(method="ffill")

print('dfDay key values')
print(dfDay)

# (2 of 2) concatenate the dataframes
dfH = pd.concat([dfH, dfDay], axis=1)
print('append dfDay to dfH')
print(dfH)


#visually represent the data series
import matplotlib.pyplot as plt

plt.subplot(2,1,1)
plt.plot(dfH.index, dfH['closeAsk'], label='close')
plt.plot(dfH.index, dfH['LL'], label='LL (previuos day)')
plt.plot(dfH.index, dfH['HH'], label='HH (previous day)')

plt.xlabel('time')
plt.ylabel('price')

plt.title("test plot")



""" ------------------------------------
backtester
 ----------------------------------- """

#(1 of 5) daily resample


dfP = pd.DataFrame()
dfP['HH'] = dfH['highBid'].resample('B').max()
dfP['LL'] = dfH['lowAsk'].resample('B').min()
dfP['C'] = dfH['closeBid'].resample('B').last()


# (2 of 5) daily gains
cumgain = 1

for i in range(0, len(dfP)):
    if i == 0:
        dfP.ix[i, 'breakout'] = 0
    elif dfP.ix[i, 'HH'] > dfP.ix[i-1, 'HH']:
        dfP.ix[i, 'breakout'] = 1
    elif dfP.ix[i, 'LL'] < dfP.ix[i-1, 'LL']:
        dfP.ix[i, 'breakout'] = -1
    else:
        dfP.ix[i, 'breakout'] = 0

for i in range(0, len(dfP)):
    if  dfP.ix[i, 'breakout'] == 1:
        dfP.ix[i, 'gain'] = dfP.ix[i, 'C'] - dfP.ix[i-1, 'HH']
    if dfP.ix[i, 'breakout'] == -1:
        dfP.ix[i, 'gain'] = dfP.ix[i-1, 'LL'] - dfP.ix[i, 'C']
    if dfP.ix[i, 'breakout'] == 0:
        dfP.ix[i, 'gain'] = 0

# (3 of 5) cumulative gains
for i in range(0, len(dfP)):
    cumgain += dfP.ix[i, 'gain']
    dfP.ix[i, 'cumgain'] = cumgain  #cumulative gains

# (4 of 5) reindex from daily to hourly
dfP['cumgain'].reindex(dfH.index).fillna(method="ffill")


# (5 of 5) plot graphs and stats
plt.subplot(2,1,2)
plt.plot(dfP.index, dfP['cumgain'], label='portfolio value')

plt.xlabel('time')
plt.ylabel('cumulative gains')

plt.title("gains")


plt.legend()
plt.savefig('kata03.png')

plt.show(block=True)
plt.interactive(False)

print(dfP)

initialCapital = 1
portfolioValue = initialCapital + dfP['gain'].sum()
print(portfolioValue)