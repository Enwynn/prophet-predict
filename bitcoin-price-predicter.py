import pandas as pd
from prophet import Prophet

# Link below is to a guide on multivariate modelling using additional regression
# https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#additional-regressors


christmas = pd.DataFrame({
    'holiday': 'christmas',
    'ds': pd.to_datetime(['2016-12-25', '2017-12-25', '2018-12-25', '2019-12-25', '2020-12-25', '2021-12-25',
                          '2022-12-25', '2023-12-25', '2024-12-25', '2025-12-25']),
    'lower_window': -1,
    'upper_window': 1
})

new_years = pd.DataFrame({
    'holiday': 'new_years',
    'ds': pd.to_datetime(['2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01',
                          '2022-01-01', '2023-01-01', '2024-01-01', '2025-01-01']),
    'lower_window': -1,
    'upper_window': 0
})

chinese_new_year = pd.DataFrame({
    'holiday': 'chinese_new_year',
    'ds': pd.to_datetime(['2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01',
                          '2022-01-01', '2023-01-01', '2024-01-01', '2025-01-01']),
    'lower_window': 0,
    'upper_window': 23
})

bitcoin_pizza_day = pd.DataFrame({
    'holiday': 'bitcoin_pizza_day',
    'ds': pd.to_datetime(['2016-05-22', '2017-05-22', '2018-05-22', '2019-05-22', '2020-05-22', '2021-05-22',
                          '2022-05-22', '2023-05-22', '2024-05-22', '2025-05-22']),
    'lower_window': 0,
    'upper_window': 0
})

df = pd.read_csv('BTC-USD.csv')

df['ds'] = pd.to_datetime(df['Date'])
df['y'] = df['Close']

df.head()

holidays = pd.concat([christmas, new_years, chinese_new_year, bitcoin_pizza_day])

m = Prophet(holidays=holidays)
m.fit(df)

future = m.make_future_dataframe(periods=365 * 5)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig2 = m.plot_components(forecast)
fig2.show()
