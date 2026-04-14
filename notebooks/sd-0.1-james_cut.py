# %%
import pandas as pd
import seaborn as sns
from housing_projections.load import load_csv_s3

completions = load_csv_s3('housing-projections', 'lsoa_completions_time_series_pivot.csv')

london_lsoas = completions['LSOA Cd'].unique()


# %%

dwellings_2011 = pd.read_csv("https://ukds-ckan.s3.eu-west-1.amazonaws.com/DWLTYP/DWLTYP_LSOADZ_England_Northern_Ireland_Scotland_Wales_Descriptions.csv")

dwellings_2021 = pd.read_excel('https://ukds-ckan.s3.eu-west-1.amazonaws.com/2021/ONS/number-of-dwellings/RM204-Number-Of-Dwellings-2021-lsoa-ONS.xlsx', sheet_name='Dataset')




# %%

dwellings_london_2011 = dwellings_2011[dwellings_2011['GEO_CODE'].isin(london_lsoas)]

dwellings_london_2021 = dwellings_2021[dwellings_2021['Lower layer Super Output Areas Code'].isin(london_lsoas)]


# %%

dwellings_london_2011.rename(columns={
    'GEO_CODE':'LSOA_ID',
    'Dwellings : Total\ Dwellings - Unit : Dwellings': 'dwellings'
    }, inplace=True)

dwellings_london_2021.rename(columns={
    'Lower layer Super Output Areas Code':'LSOA_ID',
    'Observation': 'dwellings'
    }, inplace=True)

completions.rename(columns={
    'LSOA Cd': 'LSOA_ID',
}, inplace=True)

# %%

total_dwellings_census = dwellings_london_2021.join(
    dwellings_london_2011.set_index ('LSOA_ID'), 
    on='LSOA_ID',
    lsuffix='_2021',
    rsuffix='_2011',
    how='inner'
)[['LSOA_ID', 'dwellings_2011', 'dwellings_2021']]

dwellings = total_dwellings_census.join(
    completions.set_index('LSOA_ID'),
    on='LSOA_ID',
    how='inner'
)

# %%

dwellings['total_completions'] = dwellings.drop(columns={'LSOA_ID', 'dwellings_2011', 'dwellings_2021'})[['2011/12', '2012/13', '2013/14', '2014/15', '2015/16', '2016/17', '2017/18', '2018/19', '2019/20', '2020/21']].sum(axis=1)

dwellings['total_completions'] = dwellings.drop(columns={'LSOA_ID', 'dwellings_2011', 'dwellings_2021'}).sum(axis=1)

dwellings['change_2011_2021'] = dwellings['dwellings_2021'] - dwellings['dwellings_2011']

# %%

fgrid=sns.lmplot(dwellings, x='change_2011_2021', y='total_completions')
fgrid.set(ylim=(-200,5000))

min_val = min(dwellings['change_2011_2021'].min(), 0)
max_val = max(dwellings['change_2011_2021'].max(), 4000)

fgrid.set(xlim=(min_val, max_val))
fgrid.set(ylim=(min_val, max_val))

fgrid.set(xlabel="Census dwellings change 2011-2021")
fgrid.set(ylabel="Net PLD completions 2011-2021")

fgrid.set(aspect='equal', adjustable='box')

fgrid.ax.axline((0, 0), slope=1, color='k', ls='--')
fgrid.ax.grid(True, axis='both', ls=':')


# %%
