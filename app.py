from flask import Flask, request, redirect
from twilio.twiml.messaging_response import MessagingResponse
import requests
import json
import re
import csv
import datetime
import os

app = Flask(__name__)

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
        with open(f'static/imgs/{file_name}', 'wb') as handler:
            handler.write(img_data)
        sighting['File_Name'] = file_name
    else:
        return "Invalid message"

    if 'File_Name' in sighting and 'Latitude' in sighting:
        sighting['Time'] = datetime.datetime.now()
        sighting['Class'] = 'U'
        if not os.path.exists('static/imgs'):
            os.makedirs("static/imgs")
        with open('static/img_data.csv', 'a+') as img_data_file:
            fields = ['File_Name', 'Latitude', 'Longitude', 'Time', 'Class']
            img_data_writer = csv.DictWriter(img_data_file, fieldnames=fields)
            img_data_writer.writerow(sighting)
        sighting = dict()

    with open('static/sighting.json', 'w') as f:
        json.dump(sighting, f)

    return "Message processed"

@app.route("/")
def hello():
  return "Hello World!"

if __name__ == "__main__":
  app.run()