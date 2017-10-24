""" connect to oanda API ------------------------------------------------------------------------------------------- """
import oandapy as opy

oanda = opy.API(environment="practice",
                access_token="4d608abb7f46813447188432866d6698-7176a6a2d4443df441a471f2da62dd2f")


""" acquire data --------------------------------------------------------------------------------------------------- """
import pandas as pd

dataH = oanda.get_history(instrument="GBP_USD",  # our instrument ISOcurrency1_ISOcurrency2
                         start="2008-07-02",  # start data aaaa/mm/dd
                         end="2008-07-04",  # end date aaaa/mm/dd
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

