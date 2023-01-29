import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

def main():
    df = pd.read_csv("https://data.lacity.org/api/views/az43-p47q/rows.csv")
    df = df[['Latitude', 'Longitude']]
    print(df)

if __name__=='__main__':
    main()