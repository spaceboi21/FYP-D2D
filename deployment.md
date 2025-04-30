# Data2Dash: Production Deployment Documentation

## Table of Contents
1. [Deployment Overview](#deployment-overview)
2. [Docker Containerization](#docker-containerization)
3. [AWS Infrastructure Setup](#aws-infrastructure-setup)
4. [Production Server Configuration](#production-server-configuration)
5. [Security Considerations](#security-considerations)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)
7. [Scaling and Performance](#scaling-and-performance)

## Deployment Overview

Data2Dash is deployed using a containerized approach with Docker, running on AWS infrastructure. The deployment stack includes:
- Docker containers for application isolation
- AWS EC2 for hosting
- Nginx as a reverse proxy and load balancer
- Gunicorn as the WSGI server
- SSL/TLS for secure communication
- Automated deployment pipeline

## Docker Containerization

### 1. Dockerfile Configuration

```dockerfile
# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV FLASK_APP=app/new_ver.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8050

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "app.new_ver:server"]
```

### 2. Docker Compose Configuration

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8050:8050"
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=postgresql://user:password@db:5432/datadash
    depends_on:
      - db
    volumes:
      - ./app:/app/app
      - ./logs:/app/logs

  db:
    image: postgres:13
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=datadash

  nginx:
    image: nginx:latest
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - web

volumes:
  postgres_data:
```

## AWS Infrastructure Setup

### 1. EC2 Instance Configuration

- **Instance Type**: t2.medium (2 vCPU, 4GB RAM)
- **AMI**: Ubuntu Server 20.04 LTS
- **Security Groups**:
  - Inbound: HTTP (80), HTTPS (443), SSH (22)
  - Outbound: All traffic

### 2. AWS Services Integration

- **Route 53**: Domain name management
- **ACM**: SSL certificate management
- **CloudWatch**: Logging and monitoring
- **S3**: Static file storage
- **RDS**: Managed PostgreSQL database

### 3. AWS Deployment Steps

1. **Launch EC2 Instance**
   ```bash
   # Update system
   sudo apt-get update
   sudo apt-get upgrade -y

   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh

   # Install Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

2. **Configure Domain and SSL**
   ```bash
   # Install Certbot
   sudo apt-get install certbot python3-certbot-nginx

   # Obtain SSL certificate
   sudo certbot --nginx -d yourdomain.com
   ```

## Production Server Configuration

### 1. Nginx Configuration

```nginx
# /etc/nginx/nginx.conf
user www-data;
worker_processes auto;
pid /run/nginx.pid;

events {
    worker_connections 768;
}

http {
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;

    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;

    gzip on;
    gzip_disable "msie6";

    server {
        listen 80;
        server_name yourdomain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl;
        server_name yourdomain.com;

        ssl_certificate /etc/nginx/ssl/fullchain.pem;
        ssl_certificate_key /etc/nginx/ssl/privkey.pem;

        location / {
            proxy_pass http://web:8050;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /static/ {
            alias /app/static/;
            expires 30d;
        }
    }
}
```

### 2. Gunicorn Configuration

```python
# gunicorn_config.py
bind = "0.0.0.0:8050"
workers = 4
worker_class = "gthread"
threads = 2
timeout = 120
keepalive = 5
max_requests = 1000
max_requests_jitter = 50
```

### 3. Environment Configuration

```bash
# .env.production
FLASK_APP=app/new_ver.py
FLASK_ENV=production
DATABASE_URL=postgresql://user:password@db:5432/datadash
SECRET_KEY=your-secret-key
OPENAI_API_KEY=your-openai-key
```

## Security Considerations

### 1. Network Security
- Implemented AWS Security Groups
- Configured Nginx with SSL/TLS
- Enabled HTTP/2
- Set up WAF rules

### 2. Application Security
- Implemented CSRF protection
- Configured secure session management
- Enabled CORS with proper restrictions
- Implemented rate limiting

### 3. Data Security
- Encrypted database connections
- Implemented secure file upload handling
- Configured proper file permissions
- Regular security updates

## Monitoring and Maintenance

### 1. Logging Configuration

```python
# logging_config.py
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(app):
    # File handler
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=1024 * 1024 * 100,  # 100MB
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s'
    ))
    console_handler.setLevel(logging.INFO)
    app.logger.addHandler(console_handler)
    
    app.logger.setLevel(logging.INFO)
    app.logger.info('Data2Dash startup')
```

### 2. Monitoring Setup
- Configured CloudWatch alarms
- Set up health checks
- Implemented error tracking
- Regular backup schedule

## Scaling and Performance

### 1. Horizontal Scaling
- Implemented load balancing with Nginx
- Configured multiple Gunicorn workers
- Set up database connection pooling
- Implemented caching strategies

### 2. Performance Optimization
- Enabled Gzip compression
- Configured static file caching
- Implemented database indexing
- Optimized query performance

### 3. Auto-scaling Configuration
```json
{
    "AutoScalingGroupName": "datadash-asg",
    "LaunchConfigurationName": "datadash-lc",
    "MinSize": 2,
    "MaxSize": 5,
    "DesiredCapacity": 2,
    "DefaultCooldown": 300,
    "AvailabilityZones": ["us-east-1a", "us-east-1b"],
    "LoadBalancerNames": ["datadash-elb"],
    "HealthCheckType": "ELB",
    "HealthCheckGracePeriod": 300
}
```

## Deployment Process

### 1. Initial Deployment
```bash
# Clone repository
git clone https://github.com/yourusername/datadash.git
cd datadash

# Build and start containers
docker-compose build
docker-compose up -d

# Run database migrations
docker-compose exec web flask db upgrade
```

### 2. Continuous Deployment
```yaml
# .github/workflows/deploy.yml
name: Deploy to AWS

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1
      
      - name: Build, tag, and push image to Amazon ECR
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: datadash
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
      
      - name: Deploy to EC2
        run: |
          ssh ubuntu@your-ec2-instance "cd /app && \
          docker-compose pull && \
          docker-compose up -d && \
          docker-compose exec web flask db upgrade"
```

This documentation provides a comprehensive guide to deploying Data2Dash in a production environment using Docker, AWS, Nginx, and Gunicorn. The setup ensures high availability, security, and scalability while maintaining optimal performance. 