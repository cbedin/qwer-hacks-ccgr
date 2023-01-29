# Spotlight: Mapping police presence in relation to houseless folk 

## Description

Police violently engage with houseless folks in LA, evicting them and destroying their possessions Unhoused LGBTQ+ people are disproportionately at risk of police violence, especially Black folk, as well as others who are marginalized in multiple ways. There already exists “crime trackers” that purport to create safer neighborhoods by documenting criminalized activity. We turn the idea on its head by tracking police presence, for the benefit of individuals who are disproportionately at risk of harm from police contact. This project maps two components:
1. Areas where police have been called to investigate homeless encampments – through recorded LA city data
2. Current police presence – through a photo text-in service


## How to install and run locally

- Run flask run
- Make accounts through Twilio and ngrock (temporary server, as it is not currently hosted on a server). Use this tutorial to link: https://www.twilio.com/docs/sms/tutorials/how-to-receive-and-reply-python

## How to use
- Text the number set up through Twilio an image and a pin of your location (in either order) in order for it to be validated (through a classifier) and mapped 
- Use map drop down box to either look at reports to police of homeless encampments; or live updates regarding police sitings in your area

## How we built it
*Front end* 
We used bootstrap CSS, html and geopandas for map visualization

*Back end* 
Mapping: dataframes Text service:
- Receives text messages through Twilio
- Convolutional neural network classifier with PyTorch to verify submitted photos as police/not police

## Where things are currently at/need for future development
- It isn't hosted on a server
- The classification model currently classifies pics as 'car'/'not car' rather than 'police'/'not police' and needs to be trained on police data
- The classifier currently only can deal with (32 x 32 x 3) images. Needs to be changed so that it can cope with images of various dimentions
- The ability to filter for time period (last month, last 6 months, etc) is not yet fully built out
- Need for 


## to develop:

```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```


## Credits

Richard Khillah (he/him, UCLA): rkhillah@ucla.edu
Cooper Bedin (any/all, UCSB): cbedin@ucsb.edu
Cedar Brown (they/he, UCSB): cedarbrown@ucsb.edu
Gabriel Sanchez (he/him, UCLA): gabrielsanchez7@g.ucla.edu
