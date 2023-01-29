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
#APPLICATION NAME - DECORATOR
@app.route("/")
def hello():

  context={
    "hello": "Bobby brown",
    "number42": 42,
  }
  return render_template("hacks.html", **context)

if __name__ == "__main__":
  app.run()