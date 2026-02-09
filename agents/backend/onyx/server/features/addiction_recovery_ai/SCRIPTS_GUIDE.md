# Scripts Guide - Addiction Recovery AI

## ✅ Scripts Structure

### Scripts Organization

Scripts are organized in the `scripts/` directory:

```
scripts/
├── train_model.py        # ✅ Model training script
├── inference_server.py   # ✅ Inference server script
├── deploy.py             # ✅ Deployment script
└── verify_refactoring.py # ✅ Refactoring verification script
```

## 📋 Script Descriptions

### `scripts/train_model.py` - Model Training
- **Purpose**: Train ML models
- **Usage**:
```bash
python scripts/train_model.py --config config/model_config.yaml
```

**Features:**
- Model training pipeline
- Configuration support
- Checkpoint management
- Experiment tracking

### `scripts/inference_server.py` - Inference Server
- **Purpose**: Run inference server
- **Usage**:
```bash
python scripts/inference_server.py --port 8000
```

**Features:**
- Fast inference server
- Model loading
- Request handling
- Performance monitoring

### `scripts/deploy.py` - Deployment
- **Purpose**: Deploy application
- **Usage**:
```bash
python scripts/deploy.py --environment production
```

**Features:**
- Environment configuration
- Deployment automation
- Health checks
- Rollback support

### `scripts/verify_refactoring.py` - Refactoring Verification
- **Purpose**: Verify refactoring changes
- **Usage**:
```bash
python scripts/verify_refactoring.py
```

**Features:**
- Import verification
- Structure validation
- Deprecated file detection
- Migration status

## 📝 Usage Examples

### Training Models
```bash
# Basic training
python scripts/train_model.py

# With configuration
python scripts/train_model.py --config config/model_config.yaml

# With specific model
python scripts/train_model.py --model llm_coach
```

### Running Inference Server
```bash
# Default port
python scripts/inference_server.py

# Custom port
python scripts/inference_server.py --port 8080

# With model
python scripts/inference_server.py --model sentiment_analyzer
```

### Deployment
```bash
# Development
python scripts/deploy.py --environment development

# Production
python scripts/deploy.py --environment production

# With verification
python scripts/deploy.py --environment production --verify
```

### Verifying Refactoring
```bash
# Full verification
python scripts/verify_refactoring.py

# Check specific component
python scripts/verify_refactoring.py --component api

# Generate report
python scripts/verify_refactoring.py --report
```

## 🎯 Quick Reference

| Script | Purpose | Common Use Cases |
|--------|---------|------------------|
| `train_model.py` | Model training | Training new models, fine-tuning |
| `inference_server.py` | Inference server | Running inference API |
| `deploy.py` | Deployment | Deploying to environments |
| `verify_refactoring.py` | Refactoring verification | Validating refactoring changes |

## 📚 Additional Resources

- See `README.md` for project overview
- See `INSTALLATION.md` for installation instructions
- See `AWS_DEPLOYMENT.md` for AWS deployment
- See `REFACTORING_STATUS.md` for refactoring status






