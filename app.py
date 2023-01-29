import geopandas as gpd
import os
import pandas as pd
# from shapely.geometry import Point

from flask import Flask, request, redirect, render_template
from twilio.twiml.messaging_response import MessagingResponse
import requests
import json
import re
import csv
import datetime
import os
from run_model import update_predictions

SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard to guess string'

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('layout.html')

@app.route("/sms", methods=['POST'])
def sms_reply():
    with open('static/sighting.json', 'r') as f:
        sighting = json.load(f)

    if request.values.get('Body', ''):
        m = re.fullmatch(r'.*google.com.*/(-?\d+\.\d+)\+(-?\d+\.\d+)/.*', request.values.get('Body'))
        if m:
            sighting['Latitude'] = m.group(1)
            sighting['Longitude'] = m.group(2)
        else:
            return "Invalid message"
    elif request.values.get('MediaUrl0', ''):
        img_url = request.values.get('MediaUrl0', '')
        file_name = f"img{hash(img_url)}.jpeg"
        img_data = requests.get(img_url).content
        if not os.path.exists('static/imgs'):
            os.makedirs("static/imgs")
        with open(f'static/imgs/{file_name}', 'wb') as handler:
            handler.write(img_data)
        sighting['File_Name'] = file_name
    else:
        return "Invalid message"

    if 'File_Name' in sighting and 'Latitude' in sighting:
        sighting['Time'] = datetime.datetime.now()
        sighting['Class'] = 'U'
        fields = ['File_Name', 'Latitude', 'Longitude', 'Time', 'Class']
        if not os.path.exists('static/img_data.csv'):
            with open('static/img_data.csv', 'w+') as img_data_file:
                img_data_writer = csv.DictWriter(img_data_file, fieldnames=fields)
                img_data_writer.writeheader()
        with open('static/img_data.csv', 'a') as img_data_file:
            fields = ['File_Name', 'Latitude', 'Longitude', 'Time', 'Class']
            img_data_writer = csv.DictWriter(img_data_file, fieldnames=fields)
            img_data_writer.writerow(sighting)
        sighting = dict()
        update_predictions()

    with open('static/sighting.json', 'w') as f:
        json.dump(sighting, f)

    return "Message processed"

@app.route('/hacks')
def hacks():
  context = {
    "hello": "Hello, world!",
    "value": 42,
  }
  return render_template("hacks.html", **context)

@app.route('/maptest', methods=['GET', 'POST'])
def maptest():
  DATASET_LOOKUP = {  
    'Encampment Reports': '/Users/richardkhillah/Developer/qwerhacks/qwer-hacks-ccgr/311_Homeless_Encampments_Requests_1.csv',
    'Police Sightings': '/Users/richardkhillah/Developer/qwerhacks/qwer-hacks-ccgr/static/img_data.csv',
  }

  dataset_dropdown = {
    0: 'Encampment Reports',
    1: 'Police Sightings',
  }

  timeframe_dropdown = {
    0: "1 Month",
    1: "3 Month",
    2: "6 Month",
    3: "12 Month",
  }

  # Default
  dataset = DATASET_LOOKUP['Encampment Reports']
  
  if request.method == "POST":
    param = request.form['dataset']
    try:
      key = dataset_dropdown[int(param)]
      print(key)
      if key in DATASET_LOOKUP:
        dataset = DATASET_LOOKUP[key]
    
      print(dataset)

    except:
      print('failing with big depression')

  df = pd.read_csv(dataset)

  # manipulation for dataset = img_data.csv
  if dataset == DATASET_LOOKUP['Police Sightings']:
    df = df[df['Class'] == 1]

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
    "dataset_dropdown": dataset_dropdown,
    "timeframe_dropdown": timeframe_dropdown,
  }

  return render_template('maptest.html', **context)

if __name__ == "__main__":
  app.run()