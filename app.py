from flask import Flask, request, redirect
from twilio.twiml.messaging_response import MessagingResponse
import requests
import json
import re
import csv
import datetime

app = Flask(__name__)

@app.route("/sms", methods=['GET', 'POST'])
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
        img_hash = hash(img_url)
        img_data = requests.get(img_url).content
        with open(f'static/imgs/{img_hash}.jpeg', 'wb') as handler:
            handler.write(img_data)
        sighting['Img_Hash'] = img_hash
    else:
        return "Invalid message"

    if 'Img_Hash' in sighting and 'Latitude' in sighting:
        sighting['Time'] = datetime.datetime.now()
        sighting['Class'] = 'U'
        with open('static/img_data.csv', 'a') as img_data_file:
            img_data_writer = csv.DictWriter(img_data_file, fieldnames=['Img_Hash', 'Latitude', 'Longitude', 'Time', 'Class'])
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