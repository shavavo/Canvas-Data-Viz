import dash
import dash_auth
import appCredentials
import flask

app = flask.Flask(__name__)

dash_app = dash.Dash(__name__, server = app, url_base_pathname = '/')
dash_app.config.update({
        # 'routes_pathname_prefix': '/',
        # 'requests_pathname_prefix': 'analytics/',
        'suppress_callback_exceptions': True
})

auth = dash_auth.BasicAuth(
    dash_app,
    appCredentials.VALID_USERNAME_PASSWORD_PAIRS
)
