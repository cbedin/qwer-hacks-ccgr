# Spotlight: Mapping police presence in relation to houseless folk 

## Description

Police violently engage with houseless folks in LA, evicting them and destroying their possessions Unhoused LGBTQ+ people are disproportionately at risk of police violence, especially Black folk, as well as others who are marginalized in multiple ways. There already exists “crime trackers” that purport to create safer neighborhoods by documenting criminalized activity. We turn the idea on its head by tracking police presence, for the benefit of individuals who are disproportionately at risk of harm from police contact. This project maps two components:
1. Areas where police have been called to investigate homeless encampments – through recorded LA city data
2. Current police presence – through a photo text-in service


## How to install and run
tbc

## How to use
tbc

## How we built it
*Front end* 
We used bootstrap CSS, html and geopandas for map visualization
*Back end* 
Mapping: dataframes Text service:
- Receives text messages through Twilio
- Convolutional neural network classifier with PyTorch to verify submitted photos as police/not police

## Challenges we ran into
- connecting sms messaging into web app
- integrating neural network model into web app
- generating interactive map
- having to move between languages and therefore sorting out the dependencies
- plotting data
- data consistency
- sourcing data of police interactions
- generating training data for the classification model (we didn't end up to being able to scrape enough police image data in the time frame to train a model, so we used a generic dataset)
- creating an application that seemly combines all these features


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
