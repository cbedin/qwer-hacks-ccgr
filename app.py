from flask import Flask, request, redirect
# from flask_ngrok import run_with_ngrok
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)
# run_with_ngrok(app)

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

@app.route("/", methods=['GET', 'POST'])
def hello():
  return "Hello World!"

if __name__ == "__main__":
  app.run()