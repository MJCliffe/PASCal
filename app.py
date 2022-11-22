from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import plotly
import plotly.express as px
import json
import os

app = Flask(__name__)


@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['POST'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('hello.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))

@app.route('/test')
def test():
    print('Request for test page received')
    long_df = px.data.medals_long()
    fig = px.bar(long_df, x="nation", y="count", color="medal", title="Long-Form Input")
    #fig = px.bar(df, x='Fruit', y='Amount', color='City', 
    # barmode='group')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('test.html', graphJSON=graphJSON) 
   #return render_template('test.html', name = 'MJC_test_page')

#    # Generate the figure **without using pyplot**.
 #   fig = Figure()
 #   ax = fig.subplots()
 #   ax.plot([1, 2])
 #   # Save it to a temporary buffer.
 #   buf = BytesIO()
 #   fig.savefig(buf, format="png")
 #   # Embed the result in the html output.
 #   data = base64.b64encode(buf.getbuffer()).decode("ascii")
 #   return f"<img src='data:image/png;base64,{data}'/>"

if __name__ == '__main__':
   app.run()