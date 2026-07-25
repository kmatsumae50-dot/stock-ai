import plotly.graph_objects as go
import yfinance as yf

ticker = yf.Ticker("7203.T")  # Toyota
df = ticker.history(period="3mo")

fig = go.Figure()

fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Toyota",
    )
)

fig.update_layout(
    title="Toyota Stock Price",
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
)

fig.show()
