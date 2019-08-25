import dash
import dash_auth
import appCredentials
import flask

app = flask.Flask(__name__)

dash_app = dash.Dash(__name__, server = app, url_base_pathname = '/', serve_locally=True)
dash_app.config.update({
        # 'routes_pathname_prefix': '/',
        # 'requests_pathname_prefix': 'predictive/',
        'suppress_callback_exceptions': True
})


auth = dash_auth.BasicAuth(
    dash_app,
    appCredentials.VALID_USERNAME_PASSWORD_PAIRS
)

tabs_styles = {
    'height': '5vh',
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}