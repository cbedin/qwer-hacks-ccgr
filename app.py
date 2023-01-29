import geopandas as gpd
import os
import pandas as pd
# from shapely.geometry import Point

from flask import Flask, request, redirect, render_template
from twilio.twiml.messaging_response import MessagingResponse


app = Flask(__name__)

"""
@app.route("/sms", methods=['GET', 'POST'])
def sms_reply():
    # Start our TwiML response
    resp = MessagingResponse()

    # Add a message
    resp.message("The Robots are coming! Head for the hills!")

    return str(resp)

if __name__ == "__main__":
    app.run(debug=True)
"""

@app.route('/hacks')
def hacks():
  context = {
    "hello": "Hello, world!",
    "value": 42,
  }
  return render_template("hacks.html", **context)

@app.route('/maptest')
def maptest():
  # nybb = geopandas.read_file(geopandas.datasets.get_path('nybb'))
  # world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
  # cities = geopandas.read_file(geopandas.datasets.get_path('naturalearth_cities'))

  df = pd.read_csv("/Users/richardkhillah/Developer/qwerhacks/qwer-hacks-ccgr/311_Homeless_Encampments_Requests_1.csv")

  gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs=4326)

  m = gdf.explore()

  print(df)
  print(gdf.head())
  print(gdf.dtypes)
  print('end')
  
  spath = "maps/base_map.html"

  m.save( os.path.join('static', spath) )

  context = {
    "map_url": spath,
  }

  return render_template('maptest.html', **context)

if __name__ == "__main__":
  app.run()