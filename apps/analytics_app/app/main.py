import dash
import dash_auth

import flask
import base64

from app import *
import app_analytics

dash_app.layout = app_analytics.layout
# dash_app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

if __name__ == '__main__':
    dash_app.run_server(debug=True)