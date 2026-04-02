# OptiPrice India — Deployment Guide

## System Requirements

- **Python**: 3.10 or higher
- **RAM**: Minimum 4GB (8GB recommended for Neural Network training)
- **Storage**: ~500MB for dependencies + model weights
- **OS**: Linux, macOS, or Windows
- **Internet**: Required for Yahoo Finance data fetching

## Installation Steps

### 1. Navigate to Project Directory

```bash
cd "/home/bankai/Desktop/Quants work/Options_Pricing"
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
```

### 3. Activate Virtual Environment

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

**CPU-only (recommended for most users):**
```bash
pip install -r requirements.txt
```

**GPU-accelerated (NVIDIA CUDA 12.6):**
```bash
pip install -r requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

## Running the Application

### Local Development

```bash
streamlit run app.py
```

The application will start at `http://localhost:8501`

### Custom Port

```bash
streamlit run app.py --server.port 8080
```

### Network Access (LAN)

```bash
streamlit run app.py --server.address 0.0.0.0
```

Access from other devices on your network at `http://<your-ip>:8501`

## Configuration

### Streamlit Theme

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#448aff"
backgroundColor = "#0a0a0f"
secondaryBackgroundColor = "#12121a"
textColor = "#e8e8f0"
font = "monospace"

[server]
maxUploadSize = 50
enableCORS = false
enableXsrfProtection = true
```

### Application Constants

Edit `src/utils/config.py` to customize:

- `DEFAULT_RISK_FREE_RATE`: Default risk-free rate (7% for Indian 10Y G-Sec)
- `CACHE_TTL_MARKET_DATA`: Yahoo Finance cache duration (300 seconds)
- `NN_ENSEMBLE_SIZE`: Number of neural networks in ensemble (5)
- `BAYESIAN_SAMPLES`: Posterior samples for Bayesian model (1000)
- `TREE_STEPS`: Default binomial tree steps (100)

## Production Deployment

### Option 1: Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click

**Advantages:**
- Free hosting
- Automatic HTTPS
- Easy updates via git push
- Built-in analytics

### Option 2: Docker Container

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

Build and run:

```bash
docker build -t optiprice-india .
docker run -p 8501:8501 optiprice-india
```

### Option 3: VPS/Cloud Server

**Using systemd (Linux):**

Create `/etc/systemd/system/optiprice.service`:

```ini
[Unit]
Description=OptiPrice India Streamlit App
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/Options_Pricing
Environment="PATH=/path/to/Options_Pricing/.venv/bin"
ExecStart=/path/to/Options_Pricing/.venv/bin/streamlit run app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable optiprice
sudo systemctl start optiprice
```

### Option 4: Nginx Reverse Proxy

Configure Nginx to proxy Streamlit:

```nginx
server {
    listen 80;
    server_name optiprice.yourdomain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Performance Optimization

### 1. Enable Caching

Already implemented via `@st.cache_data` and `@st.cache_resource` decorators.

### 2. Pre-train Neural Networks

For frequently used stocks, pre-train and save weights:

```python
from src.models.neural_ensemble import NeuralEnsemble
from src.models.heston import calibrate_heston, generate_training_data

# Load historical data
symbol = "RELIANCE"
hist = fetch_historical(symbol)

# Calibrate Heston and generate training data
params = calibrate_heston(hist["Close"])
X_train, y_train = generate_training_data(params, n_paths=50000)

# Train ensemble
ensemble = NeuralEnsemble()
ensemble.train(X_train, y_train)
ensemble.save_weights(f"data/models/{symbol}_ensemble.pt")
```

### 3. Database Caching (Advanced)

For high-traffic deployments, use Redis for caching:

```python
import redis
import streamlit as st

@st.cache_resource
def get_redis_client():
    return redis.Redis(host='localhost', port=6379, db=0)
```

## Monitoring

### Application Logs

Streamlit logs to stdout. Redirect to file:

```bash
streamlit run app.py > logs/app.log 2>&1
```

### Health Check Endpoint

Streamlit provides `/_stcore/health` endpoint for monitoring.

### Metrics

Track usage via Streamlit Cloud analytics or custom logging:

```python
import logging

logging.basicConfig(
    filename='logs/usage.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Log each calculation
logging.info(f"Priced {option_type} option on {symbol}")
```

## Security

### API Rate Limiting

Yahoo Finance has rate limits. Aggressive caching (5 min TTL) mitigates this.

### Input Validation

All user inputs are validated:
- Strike price: > 0
- Time to expiry: 1-3650 days
- Volatility: 1-200%
- Risk-free rate: 0-15%

### HTTPS

Always use HTTPS in production. Streamlit Cloud provides this automatically.

## Troubleshooting

### Yahoo Finance Errors

**Issue**: `Could not fetch data for SYMBOL`

**Solutions:**
- Check internet connection
- Verify symbol is correct (NSE format)
- Wait 5 minutes for cache to expire
- Try a different stock

### Neural Network Training Timeout

**Issue**: Training takes > 60 seconds

**Solutions:**
- Use pre-trained weights for Nifty 50 stocks
- Reduce `NN_EPOCHS` in config (default: 50)
- Use GPU acceleration
- Increase `NN_MAX_TRAINING_TIME` limit

### Memory Issues

**Issue**: Application crashes with large datasets

**Solutions:**
- Reduce `BAYESIAN_SAMPLES` (default: 1000)
- Reduce `TREE_STEPS` (default: 100)
- Limit historical data period
- Increase server RAM

### Port Already in Use

**Issue**: `Address already in use`

**Solutions:**
```bash
# Find process using port 8501
lsof -i :8501

# Kill process
kill -9 <PID>

# Or use different port
streamlit run app.py --server.port 8080
```

## Backup & Recovery

### Data Backup

Backup critical files:
- `data/nifty500.csv` — Stock universe
- `data/models/*.pt` — Pre-trained weights
- `.streamlit/config.toml` — Configuration
- `src/utils/config.py` — Constants

### Version Control

Use git for code versioning:

```bash
git init
git add .
git commit -m "Initial deployment"
git remote add origin <your-repo-url>
git push -u origin main
```

## Scaling

### Horizontal Scaling

Deploy multiple instances behind a load balancer:

```nginx
upstream optiprice {
    server 127.0.0.1:8501;
    server 127.0.0.1:8502;
    server 127.0.0.1:8503;
}

server {
    location / {
        proxy_pass http://optiprice;
    }
}
```

### Vertical Scaling

For single-instance performance:
- Increase RAM for larger caches
- Use GPU for faster NN inference
- Enable multi-threading in PyTorch

## Maintenance

### Update Dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Update Nifty 500 List

Periodically refresh `data/nifty500.csv` with latest constituents.

### Clear Cache

```bash
# Clear Streamlit cache
rm -rf ~/.streamlit/cache

# Clear model weights
rm -rf data/models/*.pt
```

## Support

For deployment issues:
- Check logs: `logs/app.log`
- Review Streamlit docs: [docs.streamlit.io](https://docs.streamlit.io)
- GitHub Issues: [repository-url]/issues

---

**Deployed with ❤️ for the Indian derivatives market**
