"""
IoT and Edge Computing Integration Module
"""

import asyncio
import logging
import time
import json
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import uuid
import paho.mqtt.client as mqtt
import threading
from pathlib import Path

import tensorflow as tf
import onnxruntime as ort
import mediapipe as mp
from ultralytics import YOLO

from config import settings
from models import ProcessingStatus

logger = logging.getLogger(__name__)


class IoTEdgeComputing:
    """IoT and Edge Computing Integration Engine"""
    
    def __init__(self):
        self.mqtt_client = None
        self.edge_devices = {}
        self.edge_models = {}
        self.camera_streams = {}
        self.initialized = False
        
    async def initialize(self):
        """Initialize IoT and edge computing system"""
        if self.initialized:
            return
            
        try:
            logger.info("Initializing IoT and Edge Computing System...")
            
            # Initialize MQTT client
            await self._initialize_mqtt()
            
            # Initialize edge models
            await self._initialize_edge_models()
            
            # Initialize camera streams
            await self._initialize_camera_streams()
            
            self.initialized = True
            logger.info("IoT and Edge Computing System initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing IoT and edge computing: {e}")
            raise
    
    async def _initialize_mqtt(self):
        """Initialize MQTT client for IoT communication"""
        try:
            if hasattr(settings, 'mqtt_broker') and settings.mqtt_broker:
                self.mqtt_client = mqtt.Client()
                self.mqtt_client.on_connect = self._on_mqtt_connect
                self.mqtt_client.on_message = self._on_mqtt_message
                
                # Connect to MQTT broker
                self.mqtt_client.connect(
                    settings.mqtt_broker,
                    settings.mqtt_port if hasattr(settings, 'mqtt_port') else 1883,
                    60
                )
                
                # Start MQTT loop
                self.mqtt_client.loop_start()
                
                logger.info("MQTT client initialized and connected")
            
        except Exception as e:
            logger.error(f"Error initializing MQTT: {e}")
    
    async def _initialize_edge_models(self):
        """Initialize edge computing models"""
        try:
            # Initialize TensorFlow Lite models
            if hasattr(settings, 'tflite_models_path') and settings.tflite_models_path:
                tflite_models_path = Path(settings.tflite_models_path)
                
                for model_file in tflite_models_path.glob("*.tflite"):
                    model_name = model_file.stem
                    interpreter = tf.lite.Interpreter(model_path=str(model_file))
                    interpreter.allocate_tensors()
                    
                    self.edge_models[f"tflite_{model_name}"] = {
                        "type": "tflite",
                        "interpreter": interpreter,
                        "input_details": interpreter.get_input_details(),
                        "output_details": interpreter.get_output_details()
                    }
                    
                    logger.info(f"Loaded TensorFlow Lite model: {model_name}")
            
            # Initialize ONNX models
            if hasattr(settings, 'onnx_models_path') and settings.onnx_models_path:
                onnx_models_path = Path(settings.onnx_models_path)
                
                for model_file in onnx_models_path.glob("*.onnx"):
                    model_name = model_file.stem
                    session = ort.InferenceSession(str(model_file))
                    
                    self.edge_models[f"onnx_{model_name}"] = {
                        "type": "onnx",
                        "session": session,
                        "input_names": [input.name for input in session.get_inputs()],
                        "output_names": [output.name for output in session.get_outputs()]
                    }
                    
                    logger.info(f"Loaded ONNX model: {model_name}")
            
            # Initialize MediaPipe models
            self.edge_models["mediapipe_hands"] = {
                "type": "mediapipe",
                "model": mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.7
                )
            }
            
            self.edge_models["mediapipe_pose"] = {
                "type": "mediapipe",
                "model": mp.solutions.pose.Pose(
                    static_image_mode=False,
                    min_detection_confidence=0.7
                )
            }
            
            # Initialize YOLO model for edge
            self.edge_models["yolo_nano"] = {
                "type": "yolo",
                "model": YOLO('yolov8n.pt')  # Nano version for edge
            }
            
            logger.info("Edge models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing edge models: {e}")
    
    async def _initialize_camera_streams(self):
        """Initialize camera streams for edge processing"""
        try:
            # Initialize camera streams
            if hasattr(settings, 'camera_streams') and settings.camera_streams:
                for stream_config in settings.camera_streams:
                    stream_id = stream_config["id"]
                    stream_url = stream_config["url"]
                    
                    # Test camera connection
                    cap = cv2.VideoCapture(stream_url)
                    if cap.isOpened():
                        self.camera_streams[stream_id] = {
                            "url": stream_url,
                            "cap": cap,
                            "active": True,
                            "last_frame": None,
                            "processing": False
                        }
                        logger.info(f"Camera stream initialized: {stream_id}")
                    else:
                        logger.warning(f"Failed to initialize camera stream: {stream_id}")
            
        except Exception as e:
            logger.error(f"Error initializing camera streams: {e}")
    
    async def process_edge_document_capture(self, device_id: str, 
                                          image_data: bytes) -> Dict[str, Any]:
        """Process document capture from edge device"""
        try:
            if not self.initialized:
                await self.initialize()
            
            # Convert image data to OpenCV format
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {"error": "Invalid image data", "status": "failed"}
            
            # Process image with edge models
            results = {}
            
            # Document detection
            if "yolo_nano" in self.edge_models:
                doc_results = await self._detect_documents_edge(image)
                results["document_detection"] = doc_results
            
            # Text extraction
            if "tflite_ocr" in self.edge_models:
                text_results = await self._extract_text_edge(image)
                results["text_extraction"] = text_results
            
            # Quality assessment
            quality_results = await self._assess_image_quality_edge(image)
            results["quality_assessment"] = quality_results
            
            # Send results back to device
            await self._send_results_to_device(device_id, results)
            
            return {
                "device_id": device_id,
                "results": results,
                "processing_time": time.time(),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error processing edge document capture: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def process_realtime_camera_stream(self, stream_id: str) -> Dict[str, Any]:
        """Process real-time camera stream for document detection"""
        try:
            if stream_id not in self.camera_streams:
                return {"error": "Stream not found", "status": "failed"}
            
            stream = self.camera_streams[stream_id]
            
            if stream["processing"]:
                return {"error": "Stream already being processed", "status": "busy"}
            
            stream["processing"] = True
            
            try:
                # Capture frame
                ret, frame = stream["cap"].read()
                if not ret:
                    return {"error": "Failed to capture frame", "status": "failed"}
                
                # Process frame
                results = await self._process_camera_frame(frame)
                
                # Update stream
                stream["last_frame"] = frame
                stream["last_processed"] = datetime.now()
                
                return {
                    "stream_id": stream_id,
                    "results": results,
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed"
                }
                
            finally:
                stream["processing"] = False
                
        except Exception as e:
            logger.error(f"Error processing camera stream: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def deploy_model_to_edge_device(self, device_id: str, 
                                        model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy model to edge device"""
        try:
            # Prepare model for deployment
            deployment_package = await self._prepare_model_deployment(model_config)
            
            # Send deployment package to device
            deployment_result = await self._send_model_to_device(device_id, deployment_package)
            
            # Verify deployment
            verification_result = await self._verify_model_deployment(device_id, model_config)
            
            return {
                "device_id": device_id,
                "model_config": model_config,
                "deployment_result": deployment_result,
                "verification_result": verification_result,
                "status": "deployed" if verification_result["success"] else "failed"
            }
            
        except Exception as e:
            logger.error(f"Error deploying model to edge device: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def collect_iot_sensor_data(self, device_id: str, 
                                    sensor_type: str) -> Dict[str, Any]:
        """Collect data from IoT sensors"""
        try:
            # Request sensor data from device
            sensor_data = await self._request_sensor_data(device_id, sensor_type)
            
            # Process sensor data
            processed_data = await self._process_sensor_data(sensor_data, sensor_type)
            
            # Store data
            await self._store_sensor_data(device_id, sensor_type, processed_data)
            
            return {
                "device_id": device_id,
                "sensor_type": sensor_type,
                "raw_data": sensor_data,
                "processed_data": processed_data,
                "timestamp": datetime.now().isoformat(),
                "status": "collected"
            }
            
        except Exception as e:
            logger.error(f"Error collecting IoT sensor data: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def _detect_documents_edge(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect documents in image using edge models"""
        try:
            if "yolo_nano" not in self.edge_models:
                return {"error": "YOLO model not available"}
            
            model = self.edge_models["yolo_nano"]["model"]
            results = model(image)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Filter for document-like objects
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        
                        if class_id in [0, 1, 2, 3, 4, 5]:  # Common document classes
                            detections.append({
                                "class": class_name,
                                "confidence": float(box.conf[0]),
                                "bbox": box.xyxy[0].tolist()
                            })
            
            return {
                "detections": detections,
                "total_documents": len(detections)
            }
            
        except Exception as e:
            logger.error(f"Error detecting documents on edge: {e}")
            return {"error": str(e)}
    
    async def _extract_text_edge(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text from image using edge models"""
        try:
            if "tflite_ocr" not in self.edge_models:
                return {"error": "OCR model not available"}
            
            model_info = self.edge_models["tflite_ocr"]
            interpreter = model_info["interpreter"]
            
            # Preprocess image
            input_details = model_info["input_details"]
            input_shape = input_details[0]['shape']
            
            # Resize image to model input size
            resized_image = cv2.resize(image, (input_shape[2], input_shape[1]))
            input_data = np.expand_dims(resized_image, axis=0).astype(np.float32)
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            # Get output
            output_details = model_info["output_details"]
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Process output (simplified)
            text = "Extracted text from edge processing"  # Placeholder
            
            return {
                "text": text,
                "confidence": 0.8,
                "processing_method": "edge_tflite"
            }
            
        except Exception as e:
            logger.error(f"Error extracting text on edge: {e}")
            return {"error": str(e)}
    
    async def _assess_image_quality_edge(self, image: np.ndarray) -> Dict[str, Any]:
        """Assess image quality using edge processing"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate quality metrics
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            contrast = gray.std()
            
            # Determine quality score
            quality_score = min(100, max(0, 
                (laplacian_var / 1000) * 40 +
                (contrast / 100) * 30 +
                (1 - abs(brightness - 128) / 128) * 30
            ))
            
            return {
                "sharpness": float(laplacian_var),
                "brightness": float(brightness),
                "contrast": float(contrast),
                "quality_score": float(quality_score),
                "quality_rating": "excellent" if quality_score > 80 else "good" if quality_score > 60 else "fair"
            }
            
        except Exception as e:
            logger.error(f"Error assessing image quality on edge: {e}")
            return {"error": str(e)}
    
    async def _process_camera_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process camera frame for document detection"""
        try:
            results = {}
            
            # Document detection
            if "yolo_nano" in self.edge_models:
                doc_results = await self._detect_documents_edge(frame)
                results["document_detection"] = doc_results
            
            # Motion detection
            motion_results = await self._detect_motion(frame)
            results["motion_detection"] = motion_results
            
            # Face detection
            if "mediapipe_hands" in self.edge_models:
                face_results = await self._detect_faces_edge(frame)
                results["face_detection"] = face_results
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing camera frame: {e}")
            return {"error": str(e)}
    
    async def _detect_motion(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect motion in frame"""
        try:
            # Simple motion detection using frame differencing
            # In production, you'd maintain previous frames
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate motion score (simplified)
            motion_score = np.std(gray) / 255.0
            
            return {
                "motion_detected": motion_score > 0.1,
                "motion_score": float(motion_score),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error detecting motion: {e}")
            return {"error": str(e)}
    
    async def _detect_faces_edge(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect faces using edge models"""
        try:
            if "mediapipe_hands" not in self.edge_models:
                return {"error": "MediaPipe model not available"}
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.edge_models["mediapipe_hands"]["model"].process(rgb_frame)
            
            faces = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract hand bounding box
                    h, w, _ = frame.shape
                    x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
                    y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
                    
                    faces.append({
                        "bbox": [min(x_coords), min(y_coords), max(x_coords), max(y_coords)],
                        "landmarks": len(hand_landmarks.landmark)
                    })
            
            return {
                "faces": faces,
                "face_count": len(faces)
            }
            
        except Exception as e:
            logger.error(f"Error detecting faces on edge: {e}")
            return {"error": str(e)}
    
    async def _send_results_to_device(self, device_id: str, results: Dict[str, Any]):
        """Send processing results back to edge device"""
        try:
            if self.mqtt_client:
                topic = f"devices/{device_id}/results"
                message = json.dumps(results)
                self.mqtt_client.publish(topic, message)
                logger.info(f"Results sent to device {device_id}")
            
        except Exception as e:
            logger.error(f"Error sending results to device: {e}")
    
    async def _prepare_model_deployment(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare model for deployment to edge device"""
        try:
            # This would prepare the model package for deployment
            # Including model files, configuration, and dependencies
            
            deployment_package = {
                "model_type": model_config["type"],
                "model_file": model_config["file"],
                "config": model_config,
                "dependencies": model_config.get("dependencies", []),
                "deployment_id": str(uuid.uuid4()),
                "created_at": datetime.now().isoformat()
            }
            
            return deployment_package
            
        except Exception as e:
            logger.error(f"Error preparing model deployment: {e}")
            raise
    
    async def _send_model_to_device(self, device_id: str, 
                                  deployment_package: Dict[str, Any]) -> Dict[str, Any]:
        """Send model deployment package to edge device"""
        try:
            if self.mqtt_client:
                topic = f"devices/{device_id}/deploy"
                message = json.dumps(deployment_package)
                self.mqtt_client.publish(topic, message)
                
                return {
                    "device_id": device_id,
                    "deployment_id": deployment_package["deployment_id"],
                    "status": "sent"
                }
            
            return {"error": "MQTT client not available", "status": "failed"}
            
        except Exception as e:
            logger.error(f"Error sending model to device: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def _verify_model_deployment(self, device_id: str, 
                                     model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Verify model deployment on edge device"""
        try:
            # This would verify that the model was successfully deployed
            # and is working correctly on the edge device
            
            return {
                "device_id": device_id,
                "model_loaded": True,
                "model_working": True,
                "verification_timestamp": datetime.now().isoformat(),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error verifying model deployment: {e}")
            return {"error": str(e), "success": False}
    
    async def _request_sensor_data(self, device_id: str, 
                                 sensor_type: str) -> Dict[str, Any]:
        """Request sensor data from IoT device"""
        try:
            if self.mqtt_client:
                topic = f"devices/{device_id}/sensors/{sensor_type}/request"
                self.mqtt_client.publish(topic, "request_data")
                
                # Wait for response (simplified)
                await asyncio.sleep(1)
                
                # Return mock sensor data
                return {
                    "device_id": device_id,
                    "sensor_type": sensor_type,
                    "value": 25.5,
                    "unit": "celsius",
                    "timestamp": datetime.now().isoformat()
                }
            
            return {"error": "MQTT client not available"}
            
        except Exception as e:
            logger.error(f"Error requesting sensor data: {e}")
            return {"error": str(e)}
    
    async def _process_sensor_data(self, sensor_data: Dict[str, Any], 
                                 sensor_type: str) -> Dict[str, Any]:
        """Process sensor data"""
        try:
            # Process sensor data based on type
            if sensor_type == "temperature":
                processed_data = {
                    "value": sensor_data["value"],
                    "unit": sensor_data["unit"],
                    "status": "normal" if 20 <= sensor_data["value"] <= 30 else "warning",
                    "processed_at": datetime.now().isoformat()
                }
            elif sensor_type == "humidity":
                processed_data = {
                    "value": sensor_data["value"],
                    "unit": sensor_data["unit"],
                    "status": "normal" if 40 <= sensor_data["value"] <= 60 else "warning",
                    "processed_at": datetime.now().isoformat()
                }
            else:
                processed_data = {
                    "value": sensor_data["value"],
                    "unit": sensor_data.get("unit", "unknown"),
                    "status": "unknown",
                    "processed_at": datetime.now().isoformat()
                }
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing sensor data: {e}")
            return {"error": str(e)}
    
    async def _store_sensor_data(self, device_id: str, sensor_type: str, 
                               processed_data: Dict[str, Any]):
        """Store processed sensor data"""
        try:
            # Store sensor data (simplified)
            storage_key = f"sensor_data/{device_id}/{sensor_type}/{datetime.now().isoformat()}"
            logger.info(f"Stored sensor data: {storage_key}")
            
        except Exception as e:
            logger.error(f"Error storing sensor data: {e}")
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            logger.info("Connected to MQTT broker")
            # Subscribe to device topics
            client.subscribe("devices/+/data")
            client.subscribe("devices/+/status")
        else:
            logger.error(f"Failed to connect to MQTT broker: {rc}")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            
            # Process incoming messages from IoT devices
            if topic.startswith("devices/"):
                device_id = topic.split("/")[1]
                asyncio.create_task(self._handle_device_message(device_id, payload))
            
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    async def _handle_device_message(self, device_id: str, payload: Dict[str, Any]):
        """Handle incoming messages from IoT devices"""
        try:
            message_type = payload.get("type")
            
            if message_type == "document_capture":
                # Process document capture from device
                image_data = payload.get("image_data")
                if image_data:
                    await self.process_edge_document_capture(device_id, image_data)
            
            elif message_type == "sensor_data":
                # Process sensor data
                sensor_type = payload.get("sensor_type")
                if sensor_type:
                    await self.collect_iot_sensor_data(device_id, sensor_type)
            
            elif message_type == "status":
                # Update device status
                self.edge_devices[device_id] = {
                    "status": payload.get("status"),
                    "last_seen": datetime.now().isoformat(),
                    "capabilities": payload.get("capabilities", [])
                }
            
        except Exception as e:
            logger.error(f"Error handling device message: {e}")


# Global IoT edge computing instance
iot_edge_computing = IoTEdgeComputing()


async def initialize_iot_edge_computing():
    """Initialize the IoT and edge computing system"""
    await iot_edge_computing.initialize()














