# 🚀 Opus Clip Enhanced - Quick Start Guide

## 🎯 **What's New**

This enhanced version now includes the core features that make Opus Clip a market leader:

- ✅ **Content Curation Engine (ClipGenius™)** - Automatically finds engaging moments
- ✅ **Speaker Tracking System** - Keeps speakers perfectly framed
- ✅ **B-roll Integration** - Adds relevant visuals automatically
- ✅ **Platform Optimization** - Adapts for TikTok, YouTube, Instagram, etc.

## 🚀 **Quick Start**

### 1. **Install Dependencies**

```bash
pip install -r requirements_opus_clip.txt
```

### 2. **Start the Enhanced API**

```bash
python enhanced_api.py
```

The API will be available at `http://localhost:8000`

### 3. **Process Your First Video**

#### **Option A: Full Opus Clip Pipeline**

```python
import requests

# Full Opus Clip processing
response = requests.post("http://localhost:8000/opus-clip/process", json={
    "video_path": "/path/to/your/video.mp4",
    "content_text": "Your video content description here",
    "target_platform": "tiktok",
    "enable_content_curation": True,
    "enable_speaker_tracking": True,
    "enable_broll_integration": True,
    "max_clips": 5,
    "quality": "high"
})

job_id = response.json()["job_id"]
print(f"Processing started: {job_id}")
```

#### **Option B: Individual Features**

```python
# Content Curation Only
response = requests.post("http://localhost:8000/content-curation/analyze", json={
    "video_path": "/path/to/your/video.mp4",
    "analysis_depth": "high"
})

# Speaker Tracking Only
response = requests.post("http://localhost:8000/speaker-tracking/track", json={
    "video_path": "/path/to/your/video.mp4",
    "target_resolution": [1080, 1920]
})

# B-roll Integration Only
response = requests.post("http://localhost:8000/broll-integration/integrate", json={
    "video_path": "/path/to/your/video.mp4",
    "content_text": "Your content description"
})
```

### 4. **Check Job Status**

```python
# Check processing status
response = requests.get(f"http://localhost:8000/jobs/{job_id}")
status = response.json()["status"]

if status == "completed":
    clips = response.json()["clips"]
    print(f"Generated {len(clips)} viral clips!")
```

### 5. **Download Processed Clips**

```python
# Download a specific clip
response = requests.get(f"http://localhost:8000/download/{job_id}/0")
with open("viral_clip_0.mp4", "wb") as f:
    f.write(response.content)
```

## 🎬 **Demo Script**

Run the comprehensive demo to see all features in action:

```bash
python opus_clip_demo.py
```

This will demonstrate:
- Content curation analysis
- Speaker tracking
- B-roll integration
- Full pipeline processing

## 📊 **API Endpoints**

### **Main Processing**
- `POST /opus-clip/process` - Full Opus Clip pipeline
- `GET /jobs/{job_id}` - Check job status
- `GET /download/{job_id}/{clip_index}` - Download processed clip

### **Individual Features**
- `POST /content-curation/analyze` - Content curation only
- `POST /speaker-tracking/track` - Speaker tracking only
- `POST /broll-integration/integrate` - B-roll integration only

### **System**
- `GET /health` - Health check
- `GET /health/detailed` - Detailed system status
- `GET /jobs` - List all jobs

## 🎯 **Platform-Specific Optimization**

The system automatically optimizes for different platforms:

### **TikTok** (Default)
- Aspect ratio: 9:16 (vertical)
- Resolution: 1080x1920
- Duration: 8-15 seconds
- Format: MP4

### **YouTube Shorts**
- Aspect ratio: 9:16 (vertical)
- Resolution: 1080x1920
- Duration: 8-60 seconds
- Format: MP4

### **Instagram Reels**
- Aspect ratio: 9:16 (vertical)
- Resolution: 1080x1920
- Duration: 5-30 seconds
- Format: MP4

### **Instagram Posts**
- Aspect ratio: 1:1 (square)
- Resolution: 1080x1080
- Duration: 5-30 seconds
- Format: MP4

### **YouTube**
- Aspect ratio: 16:9 (horizontal)
- Resolution: 1920x1080
- Duration: 10-60 seconds
- Format: MP4

## 🔧 **Configuration Options**

### **Content Curation**
```python
{
    "analysis_depth": "high",  # low, medium, high
    "target_duration": 12.0,   # seconds
    "engagement_threshold": 0.7
}
```

### **Speaker Tracking**
```python
{
    "target_resolution": [1080, 1920],
    "tracking_quality": "high",  # low, medium, high
    "enable_auto_framing": True
}
```

### **B-roll Integration**
```python
{
    "broll_types": ["stock_footage", "ai_generated", "graphics"],
    "max_suggestions": 3,
    "confidence_threshold": 0.7
}
```

## 📈 **Performance Tips**

1. **Use High-Quality Source Videos**: Better input = better output
2. **Provide Content Text**: Improves B-roll suggestions
3. **Choose Appropriate Platform**: Optimizes for target audience
4. **Use High Quality Setting**: Better results but slower processing
5. **Process in Batches**: More efficient for multiple videos

## 🐛 **Troubleshooting**

### **Common Issues**

1. **Video Not Found**
   - Check file path is correct
   - Ensure video format is supported (MP4, MOV, AVI)

2. **Processing Fails**
   - Check video file is not corrupted
   - Ensure sufficient disk space
   - Check system resources

3. **Low Quality Results**
   - Use higher quality source video
   - Increase quality setting
   - Check video resolution

4. **Slow Processing**
   - Reduce video resolution
   - Use lower quality setting
   - Check system performance

### **Debug Mode**

Enable detailed logging:

```python
import structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
```

## 🎉 **Success Metrics**

After processing, you'll get:

- **Clips Generated**: Number of viral clips created
- **Success Rate**: Percentage of successful processing
- **Processing Time**: Total time taken
- **Quality Score**: Overall quality assessment
- **Viral Potential**: Estimated viral potential

## 🚀 **Next Steps**

1. **Try the Demo**: Run `python opus_clip_demo.py`
2. **Process Your Videos**: Use the API endpoints
3. **Customize Settings**: Adjust configuration for your needs
4. **Scale Up**: Deploy to production for high-volume processing

## 📞 **Support**

- **Documentation**: Check the implementation status
- **Issues**: Report bugs and feature requests
- **Community**: Join discussions about video processing

---

**🎬 Ready to create viral content? Let's go! 🚀**


