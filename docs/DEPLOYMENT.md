# Deployment Guide

## Overview

This guide covers deploying MASB-Alt in production environments, including containerization, scaling, monitoring, and security best practices.

## Table of Contents

1. [Deployment Options](#deployment-options)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Cloud Deployments](#cloud-deployments)
5. [Production Configuration](#production-configuration)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Security Considerations](#security-considerations)
8. [Scaling Guidelines](#scaling-guidelines)

## Deployment Options

### 1. Single Server Deployment
- Best for: Small teams, proof of concept
- Requirements: 8GB RAM, 4 CPU cores minimum
- Capacity: ~100 evaluations/hour

### 2. Distributed Deployment
- Best for: Production use, high availability
- Requirements: Multiple servers, load balancer
- Capacity: 1000+ evaluations/hour

### 3. Cloud Native Deployment
- Best for: Enterprise, auto-scaling needs
- Requirements: Kubernetes cluster
- Capacity: Unlimited with auto-scaling

## Docker Deployment

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data logs reports

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "masb_orchestrator.py", "monitor"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  masb-api:
    build: .
    container_name: masb-api
    command: python api_server.py
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - COHERE_API_KEY=${COHERE_API_KEY}
      - DATABASE_URL=sqlite:///data/masb_alt.db
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./reports:/app/reports
    restart: unless-stopped
    networks:
      - masb-network

  masb-dashboard:
    build: .
    container_name: masb-dashboard
    command: python monitoring_dashboard.py
    ports:
      - "8501:8501"
    environment:
      - DATABASE_URL=sqlite:///data/masb_alt.db
    volumes:
      - ./data:/app/data:ro
      - ./reports:/app/reports:ro
    depends_on:
      - masb-api
    restart: unless-stopped
    networks:
      - masb-network

  masb-worker:
    build: .
    container_name: masb-worker
    command: python -m celery -A masb_tasks worker --loglevel=info
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - COHERE_API_KEY=${COHERE_API_KEY}
      - CELERY_BROKER_URL=redis://redis:6379/0
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - redis
    restart: unless-stopped
    networks:
      - masb-network
    deploy:
      replicas: 3

  redis:
    image: redis:7-alpine
    container_name: masb-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped
    networks:
      - masb-network

  nginx:
    image: nginx:alpine
    container_name: masb-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - masb-api
      - masb-dashboard
    restart: unless-stopped
    networks:
      - masb-network

networks:
  masb-network:
    driver: bridge

volumes:
  redis-data:
```

### Build and Run

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Kubernetes Deployment

### Deployment Manifest

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: masb-api
  labels:
    app: masb-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: masb-api
  template:
    metadata:
      labels:
        app: masb-api
    spec:
      containers:
      - name: masb-api
        image: masb-alt:latest
        command: ["python", "api_server.py"]
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: masb-secrets
              key: openai-api-key
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: masb-secrets
              key: anthropic-api-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: masb-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: masb-api-service
spec:
  selector:
    app: masb-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: masb-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: masb-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace masb-alt

# Create secrets
kubectl create secret generic masb-secrets \
  --from-literal=openai-api-key=$OPENAI_API_KEY \
  --from-literal=anthropic-api-key=$ANTHROPIC_API_KEY \
  -n masb-alt

# Apply configurations
kubectl apply -f k8s/ -n masb-alt

# Check status
kubectl get all -n masb-alt

# View logs
kubectl logs -f deployment/masb-api -n masb-alt
```

## Cloud Deployments

### AWS Deployment

```yaml
# cloudformation/masb-stack.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'MASB-Alt Production Stack'

Parameters:
  InstanceType:
    Type: String
    Default: t3.large
    Description: EC2 instance type

Resources:
  MASBInstance:
    Type: AWS::EC2::Instance
    Properties:
      InstanceType: !Ref InstanceType
      ImageId: ami-0c55b159cbfafe1f0  # Ubuntu 22.04
      SecurityGroups:
        - !Ref MASBSecurityGroup
      UserData:
        Fn::Base64: !Sub |
          #!/bin/bash
          apt-get update
          apt-get install -y docker.io docker-compose
          git clone https://github.com/masb-alt/masb-alt.git
          cd masb-alt
          docker-compose up -d

  MASBSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for MASB-Alt
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 8000
          ToPort: 8000
          CidrIp: 10.0.0.0/8
        - IpProtocol: tcp
          FromPort: 8501
          ToPort: 8501
          CidrIp: 10.0.0.0/8
```

### Google Cloud Platform

```yaml
# terraform/gcp.tf
provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_compute_instance" "masb_instance" {
  name         = "masb-alt-instance"
  machine_type = "n1-standard-4"
  zone         = "${var.region}-a"

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 100
    }
  }

  network_interface {
    network = "default"
    access_config {
      // Ephemeral IP
    }
  }

  metadata_startup_script = file("startup.sh")

  tags = ["masb-api", "masb-dashboard"]
}

resource "google_compute_firewall" "masb_firewall" {
  name    = "masb-firewall"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["80", "443", "8000", "8501"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["masb-api", "masb-dashboard"]
}
```

### Azure Deployment

```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "vmSize": {
      "type": "string",
      "defaultValue": "Standard_D4s_v3"
    }
  },
  "resources": [
    {
      "type": "Microsoft.Compute/virtualMachines",
      "apiVersion": "2021-03-01",
      "name": "masb-alt-vm",
      "location": "[resourceGroup().location]",
      "properties": {
        "hardwareProfile": {
          "vmSize": "[parameters('vmSize')]"
        },
        "storageProfile": {
          "imageReference": {
            "publisher": "Canonical",
            "offer": "0001-com-ubuntu-server-focal",
            "sku": "20_04-lts-gen2",
            "version": "latest"
          }
        }
      }
    }
  ]
}
```

## Production Configuration

### Environment Variables

```bash
# .env.production
# API Keys (use secrets management in production)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
COHERE_API_KEY=

# Database
DATABASE_URL=postgresql://user:pass@localhost/masb_prod
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=50

# API Settings
API_RATE_LIMIT=1000/hour
API_TIMEOUT=120
MAX_CONCURRENT_REQUESTS=20

# Monitoring
SENTRY_DSN=https://your-sentry-dsn
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret
ALLOWED_HOSTS=masb.yourdomain.com
SECURE_SSL_REDIRECT=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/masb/app.log
LOG_MAX_SIZE=100M
LOG_BACKUP_COUNT=10

# Performance
WORKER_PROCESSES=4
WORKER_THREADS=8
REQUEST_TIMEOUT=300
KEEPALIVE_TIMEOUT=65
```

### Nginx Configuration

```nginx
# nginx.conf
upstream masb_api {
    least_conn;
    server masb-api-1:8000 max_fails=3 fail_timeout=30s;
    server masb-api-2:8000 max_fails=3 fail_timeout=30s;
    server masb-api-3:8000 max_fails=3 fail_timeout=30s;
}

upstream masb_dashboard {
    server masb-dashboard:8501;
}

server {
    listen 80;
    server_name masb.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name masb.yourdomain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # API endpoints
    location /api {
        proxy_pass http://masb_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        
        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }

    # Dashboard
    location / {
        proxy_pass http://masb_dashboard;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Static files
    location /static {
        alias /var/www/masb/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # Security headers
    add_header X-Frame-Options "DENY";
    add_header X-Content-Type-Options "nosniff";
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
}
```

## Monitoring and Logging

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'masb-api'
    static_configs:
      - targets: ['masb-api:9090']
    
  - job_name: 'masb-dashboard'
    static_configs:
      - targets: ['masb-dashboard:9091']
      
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:9121']
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "MASB-Alt Production Metrics",
    "panels": [
      {
        "title": "API Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Evaluation Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(evaluation_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Model Performance",
        "targets": [
          {
            "expr": "avg(alignment_score) by (model)"
          }
        ]
      }
    ]
  }
}
```

### ELK Stack Configuration

```yaml
# logstash.conf
input {
  file {
    path => "/var/log/masb/*.log"
    start_position => "beginning"
    codec => multiline {
      pattern => "^\d{4}-\d{2}-\d{2}"
      negate => true
      what => "previous"
    }
  }
}

filter {
  grok {
    match => {
      "message" => "%{TIMESTAMP_ISO8601:timestamp} - %{LOGLEVEL:level} - %{GREEDYDATA:message}"
    }
  }
  
  if [level] == "ERROR" {
    mutate {
      add_tag => [ "error", "alert" ]
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "masb-logs-%{+YYYY.MM.dd}"
  }
}
```

## Security Considerations

### API Authentication

```python
# auth_middleware.py
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### Database Encryption

```python
# Encrypt sensitive data at rest
from cryptography.fernet import Fernet

def encrypt_api_key(api_key: str) -> str:
    cipher = Fernet(ENCRYPTION_KEY)
    return cipher.encrypt(api_key.encode()).decode()

def decrypt_api_key(encrypted_key: str) -> str:
    cipher = Fernet(ENCRYPTION_KEY)
    return cipher.decrypt(encrypted_key.encode()).decode()
```

### Network Security

```bash
# iptables rules
iptables -A INPUT -p tcp --dport 22 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT
iptables -A INPUT -p tcp --dport 8000 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 8501 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -j DROP
```

## Scaling Guidelines

### Horizontal Scaling

1. **API Servers**: Add more instances behind load balancer
2. **Workers**: Increase worker replicas for parallel processing
3. **Database**: Consider PostgreSQL with read replicas
4. **Cache**: Implement Redis cluster for distributed caching

### Vertical Scaling

1. **Memory**: Increase for larger evaluation batches
2. **CPU**: More cores for concurrent evaluations
3. **Storage**: Fast SSDs for database performance
4. **Network**: Higher bandwidth for API traffic

### Performance Tuning

```python
# Async connection pooling
async_engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Caching strategy
@cache(expire=3600)
async def get_evaluation_results(dataset_id: str):
    # Expensive operation cached for 1 hour
    pass
```

## Backup and Recovery

### Automated Backups

```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/masb"

# Database backup
pg_dump $DATABASE_URL > $BACKUP_DIR/db_$DATE.sql

# Data files backup
tar -czf $BACKUP_DIR/data_$DATE.tar.gz /app/data

# Upload to S3
aws s3 cp $BACKUP_DIR/db_$DATE.sql s3://masb-backups/
aws s3 cp $BACKUP_DIR/data_$DATE.tar.gz s3://masb-backups/

# Cleanup old backups
find $BACKUP_DIR -mtime +30 -delete
```

### Disaster Recovery

1. **RPO** (Recovery Point Objective): 1 hour
2. **RTO** (Recovery Time Objective): 4 hours
3. **Backup retention**: 30 days
4. **Geographic replication**: Multi-region backups

---

For support with production deployments, contact the MASB-Alt team or consult the community forums.