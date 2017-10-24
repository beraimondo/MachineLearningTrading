""" connect to oanda API ------------------------------------------------------------------------------------------- """
import oandapy as opy

oanda = opy.API(environment="practice",
                access_token="4d608abb7f46813447188432866d6698-7176a6a2d4443df441a471f2da62dd2f")


""" acquire data --------------------------------------------------------------------------------------------------- """
import pandas as pd

data = oanda.get_history(instrument="GBP_USD",  # our instrument ISOcurrency1_ISOcurrency2
                         start="2008-01-01",  # start data aaaa/mm/dd
                         end="2017-10-17",  # end date aaaa/mm/dd
                         granularity="D")  # Candles: M=Monthly W= Weekly D=daily H1=Hourly M15=15minutes

df = pd.DataFrame(data["candles"]).set_index("time")

df.index = pd.DatetimeIndex(df.index)

df.info()



""" Print data ----------------------------------------------------------------------------------------------------- """
#console settings for a user-friendly display of the tables
desired_width = 350
pd.set_option('display.width', desired_width)

print(df)