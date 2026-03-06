# Concept Cartographer - Deployment Guide

## Quick Deploy (50 minutes)

### 1. Setup on EC2 (15 min)

```bash
# SSH into your EC2
ssh -i your-key.pem ec2-user@your-instance

# Create project directory
mkdir -p ~/concept-cartographer
cd ~/concept-cartographer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upload files (run from your LOCAL machine)
scp -i your-key.pem concept_cartographer.py ec2-user@your-instance:~/concept-cartographer/
scp -i your-key.pem requirements.txt ec2-user@your-instance:~/concept-cartographer/

# Back on EC2: Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (5 min)

```bash
# Create .env file
cat > .env << 'EOF'
OPENAI_API_KEY=your_openai_key_here
EOF

chmod 600 .env
```

### 3. Test Locally (5 min)

```bash
python concept_cartographer.py
```

Look for the public URL: `https://xxxxx.gradio.live`

Test with a question like: "Explain how neural networks learn"

### 4. Create Systemd Service (15 min)

```bash
sudo nano /etc/systemd/system/concept-cartographer.service
```

Paste:

```ini
[Unit]
Description=Concept Cartographer - Interactive Ontology Builder
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/concept-cartographer
Environment="PATH=/home/ec2-user/concept-cartographer/venv/bin"
EnvironmentFile=/home/ec2-user/concept-cartographer/.env
ExecStart=/home/ec2-user/concept-cartographer/venv/bin/python concept_cartographer.py

Restart=on-failure
RestartSec=10s

MemoryMax=1.5G
CPUQuota=50%

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable concept-cartographer.service
sudo systemctl start concept-cartographer.service
sudo systemctl status concept-cartographer.service
```

### 5. Get Your Public URL (5 min)

```bash
# Watch logs for the Gradio URL
sudo journalctl -u concept-cartographer.service -f

# Look for: "Running on public URL: https://xxxxx.gradio.live"
```

### 6. Management Commands

```bash
# Restart after code changes
sudo systemctl restart concept-cartographer.service

# View logs
sudo journalctl -u concept-cartographer.service -n 100

# Stop service
sudo systemctl stop concept-cartographer.service
```

