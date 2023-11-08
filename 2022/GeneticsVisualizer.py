import pandas as pd
import plotly.express as px


f = open("predictions/genetics/visualization.json", "r")
df = pd.read_json(f)
f.close()

fig = px.scatter(df, x="parameter", y="value", animation_frame="generation", range_y=[0,5.5], color="parameter")
fig.show()