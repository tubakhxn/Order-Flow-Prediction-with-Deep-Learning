import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from io import StringIO
import time

# =============================
# CONFIGURATION
# =============================
st.set_page_config(
    page_title="Order Flow Prediction with Deep Learning",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# =============================
# SIDEBAR
# =============================
st.sidebar.title("Order Flow Prediction")
st.sidebar.markdown("---")
lookback = st.sidebar.slider("Lookback Window (steps)", 10, 100, 50, 5)
pred_horizon = st.sidebar.slider("Prediction Horizon (steps)", 1, 20, 5, 1)
model_type = st.sidebar.selectbox("Model Type", ["CNN", "LSTM"])

# =============================
# DATA PIPELINE
# =============================
def generate_synthetic_orderbook(n_steps=500, n_levels=20):
    np.random.seed(42)
    prices = np.linspace(100, 110, n_levels)
    times = np.arange(n_steps)
    orderbook = []
    for t in times:
        bid_vol = np.abs(np.random.normal(10, 5, n_levels))
        ask_vol = np.abs(np.random.normal(10, 5, n_levels))
        orderbook.append({
            'time': t,
            'bid_prices': prices,
            'ask_prices': prices,
            'bid_volumes': bid_vol,
            'ask_volumes': ask_vol
        })
    return orderbook

def parse_csv(file):
    df = pd.read_csv(file)
    # Expect columns: time, bid_prices, ask_prices, bid_volumes, ask_volumes
    orderbook = []
    for _, row in df.iterrows():
        orderbook.append({
            'time': row['time'],
            'bid_prices': np.array(eval(row['bid_prices'])),
            'ask_prices': np.array(eval(row['ask_prices'])),
            'bid_volumes': np.array(eval(row['bid_volumes'])),
            'ask_volumes': np.array(eval(row['ask_volumes']))
        })
    return orderbook

st.markdown("# Order Flow Prediction with Deep Learning")
st.markdown("---")

st.markdown("### Data Input")
data_source = st.radio("Select Data Source", ["Upload CSV", "Synthetic Generator"])
orderbook = generate_synthetic_orderbook(n_steps=500, n_levels=20)
st.success("Synthetic order book data generated automatically.")

# =============================
# FEATURE ENGINEERING
# =============================
def build_features(orderbook, lookback, pred_horizon):
    n = len(orderbook)
    n_levels = len(orderbook[0]['bid_prices'])
    X = []
    imbalance = []
    volume_delta = []
    labels = []
    for i in range(lookback, n - pred_horizon):
        # Build tensor: [lookback, 2, n_levels]
        tensor = np.zeros((lookback, 2, n_levels))
        for j in range(lookback):
            ob = orderbook[i - lookback + j]
            tensor[j, 0, :] = ob['bid_volumes']
            tensor[j, 1, :] = ob['ask_volumes']
        X.append(tensor)
        # Imbalance
        imb = np.sum(ob['bid_volumes']) - np.sum(ob['ask_volumes'])
        imbalance.append(imb)
        # Volume delta
        vdelta = np.sum(ob['bid_volumes']) + np.sum(ob['ask_volumes'])
        volume_delta.append(vdelta)
        # Label: future return
        future_ob = orderbook[i + pred_horizon]
        mid_now = (np.mean(ob['bid_prices']) + np.mean(ob['ask_prices'])) / 2
        mid_future = (np.mean(future_ob['bid_prices']) + np.mean(future_ob['ask_prices'])) / 2
        ret = mid_future - mid_now
        labels.append(1 if ret > 0 else 0)
    X = np.array(X)
    imbalance = np.array(imbalance)
    volume_delta = np.array(volume_delta)
    labels = np.array(labels)
    return X, imbalance, volume_delta, labels

with st.spinner("Building features..."):
    X, imbalance, volume_delta, labels = build_features(orderbook, lookback, pred_horizon)

# =============================
# MODEL DEFINITION
# =============================
class CNNOrderFlow(nn.Module):
    def __init__(self, lookback, n_levels):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), padding=1)
        self.fc1 = nn.Linear(32 * lookback * n_levels, 64)  # Placeholder, will fix below
        self.fc2 = nn.Linear(64, 2)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        # x shape: [batch, 32, lookback, n_levels]
        x = x.view(x.size(0), 32, -1)
        x = x.mean(dim=1)  # [batch, lookback*n_levels]
        # Fix fc1 input size
        if not hasattr(self, 'fc1_fixed'):
            self.fc1 = nn.Linear(x.size(1), 64).to(x.device)
            self.fc1_fixed = True
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LSTMOrderFlow(nn.Module):
    def __init__(self, lookback, n_levels):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2*n_levels, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 2)
    def forward(self, x):
        # x: [batch, lookback, 2, n_levels]
        x = x.view(x.size(0), x.size(1), -1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# =============================
# TRAINING
# =============================
def train_model(X, y, model_type, lookback, n_levels, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)
    if model_type == "CNN":
        model = CNNOrderFlow(lookback, n_levels).to(device)
        X_tensor = X_tensor.permute(0,2,1,3)  # [batch, 2, lookback, n_levels]
    else:
        model = LSTMOrderFlow(lookback, n_levels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    # Prediction
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor)
        probs = torch.softmax(preds, dim=1).cpu().numpy()
        pred_labels = np.argmax(probs, axis=1)
        acc = accuracy_score(y, pred_labels)
    return model, probs, pred_labels, acc, losses

with st.spinner("Training model..."):
    n_levels = X.shape[2]
    model, probs, pred_labels, acc, losses = train_model(X, labels, model_type, lookback, n_levels, epochs=15)

# =============================
# METRICS
# =============================
st.markdown("---")
st.markdown("## Model Metrics")
col1, col2, col3 = st.columns([1,1,1])
col1.metric("Accuracy", f"{acc*100:.2f}%")
col2.metric("Upward Pressure %", f"{np.mean(pred_labels)*100:.2f}%")
col3.metric("Samples", f"{len(labels)}")

# =============================
# VISUALIZATION
# =============================
st.markdown("---")
st.markdown("## Visualizations")

# 1) 3D Order Book Surface
st.markdown("### 3D Order Book Surface")
ob_surface = orderbook[-100:]
surface_X = []
surface_Y = []
surface_Z = []
for ob in ob_surface:
    surface_X.append(ob['bid_prices'])
    surface_Y.append([ob['time']]*len(ob['bid_prices']))
    surface_Z.append(ob['bid_volumes'])
X_surf = np.array(surface_X)
Y_surf = np.array(surface_Y)
Z_surf = np.array(surface_Z)
fig_surface = go.Figure(data=[go.Surface(
    x=X_surf,
    y=Y_surf,
    z=Z_surf,
    colorscale='Viridis',
    showscale=True,
    lighting=dict(ambient=0.7, diffuse=0.8, specular=0.5, roughness=0.9),
    opacity=0.95
)])
fig_surface.update_layout(
    title="Order Book Bid Volume Surface",
    autosize=True,
    template="plotly_dark",
    width=1200,
    height=600,
    margin=dict(l=0, r=0, b=0, t=40),
    scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
    font=dict(family="Quant", size=16, color="#fff"),
)
st.plotly_chart(fig_surface, use_container_width=True)

# 2) Order Flow Imbalance Heatmap
st.markdown("### Order Flow Imbalance Heatmap")
imbalance_matrix = np.array([ob['bid_volumes'] - ob['ask_volumes'] for ob in ob_surface])
fig_heatmap = px.imshow(
    imbalance_matrix.T,
    aspect="auto",
    color_continuous_scale="RdBu",
    origin="lower",
    labels=dict(x="Time", y="Price Level", color="Imbalance"),
)
fig_heatmap.update_layout(
    template="plotly_dark",
    width=1200,
    height=400,
    font=dict(family="Quant", size=16, color="#fff"),
    margin=dict(l=0, r=0, b=0, t=40),
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# 3) Prediction Probability Time Series with Glow
st.markdown("### Prediction Probability Time Series")
time_axis = np.arange(len(probs))
fig_prob = go.Figure()
fig_prob.add_trace(go.Scatter(
    x=time_axis,
    y=probs[:,1],
    mode="lines",
    line=dict(width=4, color="#00ffe7", shape="spline", dash="solid"),
    name="Upward Pressure",
    opacity=0.9,
    hoverinfo="x+y"
))
# Glow effect
for w in [8, 12, 18]:
    fig_prob.add_trace(go.Scatter(
        x=time_axis,
        y=probs[:,1],
        mode="lines",
        line=dict(width=w, color="rgba(0,255,231,0.08)", shape="spline"),
        name="",
        opacity=0.2,
        hoverinfo="skip",
        showlegend=False
    ))
fig_prob.update_layout(
    template="plotly_dark",
    width=1200,
    height=400,
    font=dict(family="Quant", size=16, color="#fff"),
    margin=dict(l=0, r=0, b=0, t=40),
    xaxis_title="Time",
    yaxis_title="Upward Pressure Probability",
)
st.plotly_chart(fig_prob, use_container_width=True)

# 4) Feature Importance / Activation Visualization
st.markdown("### Feature Importance / Activation Visualization")
if model_type == "CNN":
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).permute(0,2,1,3)
        act = model.conv1(X_tensor)
        act_mean = act.mean(dim=0).cpu().numpy()
    fig_act = px.imshow(
        act_mean[0],
        aspect="auto",
        color_continuous_scale="Viridis",
        origin="lower",
        labels=dict(x="Price Level", y="Lookback", color="Activation"),
    )
    fig_act.update_layout(
        template="plotly_dark",
        width=1200,
        height=400,
        font=dict(family="Quant", size=16, color="#fff"),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    st.plotly_chart(fig_act, use_container_width=True)
else:
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        X_flat = X_tensor.view(X_tensor.size(0), X_tensor.size(1), -1)
        out, (hn, cn) = model.lstm(X_flat)
        importance = hn[-1].cpu().numpy()
    fig_imp = go.Figure()
    fig_imp.add_trace(go.Bar(
        x=[f"Feature {i+1}" for i in range(len(importance))],
        y=importance,
        marker_color="#00ffe7"
    ))
    fig_imp.update_layout(
        template="plotly_dark",
        width=1200,
        height=400,
        font=dict(family="Quant", size=16, color="#fff"),
        margin=dict(l=0, r=0, b=0, t=40),
        xaxis_title="LSTM Hidden Features",
        yaxis_title="Activation Value",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("---")
st.markdown("### End of Report")
