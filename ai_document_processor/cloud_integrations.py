"""
Cloud Service Integrations Module
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import uuid
import json
import os
import io

# AWS SDK
import boto3
from botocore.exceptions import ClientError

# Google Cloud SDK
from google.cloud import storage, vision, documentai
from google.cloud.exceptions import NotFound

# Azure SDK
from azure.storage.blob import BlobServiceClient
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.exceptions import ResourceNotFoundError

# OpenAI
import openai

from config import settings
from models import ProcessingStatus

logger = logging.getLogger(__name__)


class CloudIntegrations:
    """Cloud Service Integrations Engine"""
    
    def __init__(self):
        self.aws_clients = {}
        self.gcp_clients = {}
        self.azure_clients = {}
        self.openai_client = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize cloud service integrations"""
        if self.initialized:
            return
            
        try:
            logger.info("Initializing Cloud Service Integrations...")
            
            # Initialize AWS clients
            await self._initialize_aws_clients()
            
            # Initialize Google Cloud clients
            await self._initialize_gcp_clients()
            
            # Initialize Azure clients
            await self._initialize_azure_clients()
            
            # Initialize OpenAI client
            await self._initialize_openai_client()
            
            self.initialized = True
            logger.info("Cloud Service Integrations initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing cloud integrations: {e}")
            raise
    
    async def _initialize_aws_clients(self):
        """Initialize AWS service clients"""
        try:
            # Check if AWS credentials are configured
            if hasattr(settings, 'aws_access_key_id') and settings.aws_access_key_id:
                # Initialize S3 client
                self.aws_clients['s3'] = boto3.client(
                    's3',
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key,
                    region_name=settings.aws_region
                )
                
                # Initialize Textract client
                self.aws_clients['textract'] = boto3.client(
                    'textract',
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key,
                    region_name=settings.aws_region
                )
                
                # Initialize Comprehend client
                self.aws_clients['comprehend'] = boto3.client(
                    'comprehend',
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key,
                    region_name=settings.aws_region
                )
                
                logger.info("AWS clients initialized successfully")
            else:
                logger.info("AWS credentials not configured, skipping AWS initialization")
                
        except Exception as e:
            logger.error(f"Error initializing AWS clients: {e}")
    
    async def _initialize_gcp_clients(self):
        """Initialize Google Cloud service clients"""
        try:
            # Check if GCP credentials are configured
            if hasattr(settings, 'gcp_project_id') and settings.gcp_project_id:
                # Initialize Cloud Storage client
                self.gcp_clients['storage'] = storage.Client(project=settings.gcp_project_id)
                
                # Initialize Vision API client
                self.gcp_clients['vision'] = vision.ImageAnnotatorClient()
                
                # Initialize Document AI client
                self.gcp_clients['documentai'] = documentai.DocumentProcessorServiceClient()
                
                logger.info("Google Cloud clients initialized successfully")
            else:
                logger.info("GCP credentials not configured, skipping GCP initialization")
                
        except Exception as e:
            logger.error(f"Error initializing GCP clients: {e}")
    
    async def _initialize_azure_clients(self):
        """Initialize Azure service clients"""
        try:
            # Check if Azure credentials are configured
            if hasattr(settings, 'azure_connection_string') and settings.azure_connection_string:
                # Initialize Blob Storage client
                self.azure_clients['blob'] = BlobServiceClient.from_connection_string(
                    settings.azure_connection_string
                )
                
                # Initialize Form Recognizer client
                self.azure_clients['form_recognizer'] = DocumentAnalysisClient(
                    endpoint=settings.azure_form_recognizer_endpoint,
                    credential=settings.azure_form_recognizer_key
                )
                
                logger.info("Azure clients initialized successfully")
            else:
                logger.info("Azure credentials not configured, skipping Azure initialization")
                
        except Exception as e:
            logger.error(f"Error initializing Azure clients: {e}")
    
    async def _initialize_openai_client(self):
        """Initialize OpenAI client"""
        try:
            if hasattr(settings, 'openai_api_key') and settings.openai_api_key:
                openai.api_key = settings.openai_api_key
                self.openai_client = openai
                logger.info("OpenAI client initialized successfully")
            else:
                logger.info("OpenAI API key not configured, skipping OpenAI initialization")
                
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
    
    # AWS Services
    async def aws_textract_analyze_document(self, document_path: str, 
                                          analysis_type: str = "text") -> Dict[str, Any]:
        """Analyze document using AWS Textract"""
        try:
            if 'textract' not in self.aws_clients:
                raise ValueError("AWS Textract client not initialized")
            
            # Read document
            with open(document_path, 'rb') as document:
                document_bytes = document.read()
            
            # Analyze document
            if analysis_type == "text":
                response = self.aws_clients['textract'].detect_document_text(
                    Document={'Bytes': document_bytes}
                )
            elif analysis_type == "tables":
                response = self.aws_clients['textract'].analyze_document(
                    Document={'Bytes': document_bytes},
                    FeatureTypes=['TABLES']
                )
            elif analysis_type == "forms":
                response = self.aws_clients['textract'].analyze_document(
                    Document={'Bytes': document_bytes},
                    FeatureTypes=['FORMS']
                )
            else:
                response = self.aws_clients['textract'].analyze_document(
                    Document={'Bytes': document_bytes},
                    FeatureTypes=['TABLES', 'FORMS']
                )
            
            # Process response
            result = {
                "analysis_type": analysis_type,
                "blocks": response.get('Blocks', []),
                "document_metadata": response.get('DocumentMetadata', {}),
                "status": "completed"
            }
            
            return result
            
        except ClientError as e:
            logger.error(f"AWS Textract error: {e}")
            return {"error": str(e), "status": "failed"}
        except Exception as e:
            logger.error(f"Error analyzing document with AWS Textract: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def aws_comprehend_analyze_text(self, text: str, 
                                        analysis_type: str = "sentiment") -> Dict[str, Any]:
        """Analyze text using AWS Comprehend"""
        try:
            if 'comprehend' not in self.aws_clients:
                raise ValueError("AWS Comprehend client not initialized")
            
            # Analyze text based on type
            if analysis_type == "sentiment":
                response = self.aws_clients['comprehend'].detect_sentiment(
                    Text=text,
                    LanguageCode='en'
                )
            elif analysis_type == "entities":
                response = self.aws_clients['comprehend'].detect_entities(
                    Text=text,
                    LanguageCode='en'
                )
            elif analysis_type == "key_phrases":
                response = self.aws_clients['comprehend'].detect_key_phrases(
                    Text=text,
                    LanguageCode='en'
                )
            elif analysis_type == "language":
                response = self.aws_clients['comprehend'].detect_dominant_language(
                    Text=text
                )
            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
            
            return {
                "analysis_type": analysis_type,
                "result": response,
                "status": "completed"
            }
            
        except ClientError as e:
            logger.error(f"AWS Comprehend error: {e}")
            return {"error": str(e), "status": "failed"}
        except Exception as e:
            logger.error(f"Error analyzing text with AWS Comprehend: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def aws_s3_upload_document(self, file_path: str, bucket_name: str, 
                                   object_key: str) -> Dict[str, Any]:
        """Upload document to AWS S3"""
        try:
            if 's3' not in self.aws_clients:
                raise ValueError("AWS S3 client not initialized")
            
            # Upload file
            self.aws_clients['s3'].upload_file(file_path, bucket_name, object_key)
            
            # Get object URL
            object_url = f"https://{bucket_name}.s3.amazonaws.com/{object_key}"
            
            return {
                "bucket": bucket_name,
                "object_key": object_key,
                "object_url": object_url,
                "status": "uploaded"
            }
            
        except ClientError as e:
            logger.error(f"AWS S3 error: {e}")
            return {"error": str(e), "status": "failed"}
        except Exception as e:
            logger.error(f"Error uploading document to AWS S3: {e}")
            return {"error": str(e), "status": "failed"}
    
    # Google Cloud Services
    async def gcp_vision_analyze_image(self, image_path: str, 
                                     analysis_type: str = "text_detection") -> Dict[str, Any]:
        """Analyze image using Google Cloud Vision API"""
        try:
            if 'vision' not in self.gcp_clients:
                raise ValueError("Google Cloud Vision client not initialized")
            
            # Read image
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            
            # Analyze image based on type
            if analysis_type == "text_detection":
                response = self.gcp_clients['vision'].text_detection(image=image)
            elif analysis_type == "label_detection":
                response = self.gcp_clients['vision'].label_detection(image=image)
            elif analysis_type == "face_detection":
                response = self.gcp_clients['vision'].face_detection(image=image)
            elif analysis_type == "object_localization":
                response = self.gcp_clients['vision'].object_localization(image=image)
            else:
                response = self.gcp_clients['vision'].text_detection(image=image)
            
            # Process response
            result = {
                "analysis_type": analysis_type,
                "annotations": [],
                "status": "completed"
            }
            
            if analysis_type == "text_detection":
                result["annotations"] = [annotation.description for annotation in response.text_annotations]
            elif analysis_type == "label_detection":
                result["annotations"] = [{"description": label.description, "score": label.score} 
                                       for label in response.label_annotations]
            elif analysis_type == "face_detection":
                result["annotations"] = [{"joy": face.joy_likelihood, "sorrow": face.sorrow_likelihood}
                                       for face in response.face_annotations]
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing image with Google Cloud Vision: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def gcp_documentai_process_document(self, document_path: str, 
                                            processor_id: str) -> Dict[str, Any]:
        """Process document using Google Cloud Document AI"""
        try:
            if 'documentai' not in self.gcp_clients:
                raise ValueError("Google Cloud Document AI client not initialized")
            
            # Read document
            with open(document_path, 'rb') as document:
                document_bytes = document.read()
            
            # Create document
            raw_document = documentai.RawDocument(
                content=document_bytes,
                mime_type='application/pdf'
            )
            
            # Process document
            request = documentai.ProcessRequest(
                name=processor_id,
                raw_document=raw_document
            )
            
            response = self.gcp_clients['documentai'].process_document(request=request)
            
            # Process response
            result = {
                "processor_id": processor_id,
                "document_text": response.document.text,
                "entities": [{"type": entity.type_, "value": entity.mention_text} 
                           for entity in response.document.entities],
                "status": "completed"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document with Google Cloud Document AI: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def gcp_storage_upload_document(self, file_path: str, bucket_name: str, 
                                        object_name: str) -> Dict[str, Any]:
        """Upload document to Google Cloud Storage"""
        try:
            if 'storage' not in self.gcp_clients:
                raise ValueError("Google Cloud Storage client not initialized")
            
            # Upload file
            bucket = self.gcp_clients['storage'].bucket(bucket_name)
            blob = bucket.blob(object_name)
            
            blob.upload_from_filename(file_path)
            
            # Get object URL
            object_url = f"https://storage.googleapis.com/{bucket_name}/{object_name}"
            
            return {
                "bucket": bucket_name,
                "object_name": object_name,
                "object_url": object_url,
                "status": "uploaded"
            }
            
        except Exception as e:
            logger.error(f"Error uploading document to Google Cloud Storage: {e}")
            return {"error": str(e), "status": "failed"}
    
    # Azure Services
    async def azure_form_recognizer_analyze_document(self, document_path: str, 
                                                   model_id: str = "prebuilt-document") -> Dict[str, Any]:
        """Analyze document using Azure Form Recognizer"""
        try:
            if 'form_recognizer' not in self.azure_clients:
                raise ValueError("Azure Form Recognizer client not initialized")
            
            # Read document
            with open(document_path, 'rb') as document:
                document_bytes = document.read()
            
            # Analyze document
            poller = self.azure_clients['form_recognizer'].begin_analyze_document(
                model_id, document_bytes
            )
            result = poller.result()
            
            # Process response
            analysis_result = {
                "model_id": model_id,
                "pages": len(result.pages),
                "tables": len(result.tables),
                "key_value_pairs": len(result.key_value_pairs),
                "entities": len(result.entities),
                "status": "completed"
            }
            
            # Extract text content
            if result.content:
                analysis_result["content"] = result.content
            
            # Extract tables
            if result.tables:
                analysis_result["tables"] = [
                    {
                        "row_count": table.row_count,
                        "column_count": table.column_count,
                        "cells": len(table.cells)
                    }
                    for table in result.tables
                ]
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing document with Azure Form Recognizer: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def azure_blob_upload_document(self, file_path: str, container_name: str, 
                                       blob_name: str) -> Dict[str, Any]:
        """Upload document to Azure Blob Storage"""
        try:
            if 'blob' not in self.azure_clients:
                raise ValueError("Azure Blob Storage client not initialized")
            
            # Upload file
            blob_client = self.azure_clients['blob'].get_blob_client(
                container=container_name, blob=blob_name
            )
            
            with open(file_path, 'rb') as data:
                blob_client.upload_blob(data)
            
            # Get object URL
            object_url = blob_client.url
            
            return {
                "container": container_name,
                "blob_name": blob_name,
                "object_url": object_url,
                "status": "uploaded"
            }
            
        except Exception as e:
            logger.error(f"Error uploading document to Azure Blob Storage: {e}")
            return {"error": str(e), "status": "failed"}
    
    # OpenAI Services
    async def openai_analyze_text(self, text: str, analysis_type: str = "sentiment") -> Dict[str, Any]:
        """Analyze text using OpenAI API"""
        try:
            if not self.openai_client:
                raise ValueError("OpenAI client not initialized")
            
            # Create prompt based on analysis type
            if analysis_type == "sentiment":
                prompt = f"Analyze the sentiment of the following text and provide a score from -1 (very negative) to 1 (very positive):\n\n{text}"
            elif analysis_type == "summary":
                prompt = f"Provide a concise summary of the following text:\n\n{text}"
            elif analysis_type == "keywords":
                prompt = f"Extract the most important keywords from the following text:\n\n{text}"
            elif analysis_type == "classification":
                prompt = f"Classify the following text into one of these categories: Business, Technical, Academic, Legal, Personal:\n\n{text}"
            else:
                prompt = f"Analyze the following text:\n\n{text}"
            
            # Call OpenAI API
            response = self.openai_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that analyzes text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            result = {
                "analysis_type": analysis_type,
                "result": response.choices[0].message.content,
                "usage": response.usage,
                "status": "completed"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing text with OpenAI: {e}")
            return {"error": str(e), "status": "failed"}
    
    # Utility Methods
    async def get_cloud_service_status(self) -> Dict[str, Any]:
        """Get status of all cloud services"""
        try:
            status = {
                "aws": {
                    "s3": "s3" in self.aws_clients,
                    "textract": "textract" in self.aws_clients,
                    "comprehend": "comprehend" in self.aws_clients
                },
                "gcp": {
                    "storage": "storage" in self.gcp_clients,
                    "vision": "vision" in self.gcp_clients,
                    "documentai": "documentai" in self.gcp_clients
                },
                "azure": {
                    "blob": "blob" in self.azure_clients,
                    "form_recognizer": "form_recognizer" in self.azure_clients
                },
                "openai": self.openai_client is not None
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting cloud service status: {e}")
            return {"error": str(e)}
    
    async def test_cloud_connections(self) -> Dict[str, Any]:
        """Test connections to all cloud services"""
        try:
            results = {}
            
            # Test AWS connections
            if 's3' in self.aws_clients:
                try:
                    # Test S3 connection
                    self.aws_clients['s3'].list_buckets()
                    results["aws_s3"] = "connected"
                except Exception as e:
                    results["aws_s3"] = f"error: {str(e)}"
            
            # Test GCP connections
            if 'storage' in self.gcp_clients:
                try:
                    # Test GCS connection
                    list(self.gcp_clients['storage'].list_buckets())
                    results["gcp_storage"] = "connected"
                except Exception as e:
                    results["gcp_storage"] = f"error: {str(e)}"
            
            # Test Azure connections
            if 'blob' in self.azure_clients:
                try:
                    # Test Azure Blob connection
                    self.azure_clients['blob'].get_account_information()
                    results["azure_blob"] = "connected"
                except Exception as e:
                    results["azure_blob"] = f"error: {str(e)}"
            
            # Test OpenAI connection
            if self.openai_client:
                try:
                    # Test OpenAI connection
                    self.openai_client.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=1
                    )
                    results["openai"] = "connected"
                except Exception as e:
                    results["openai"] = f"error: {str(e)}"
            
            return results
            
        except Exception as e:
            logger.error(f"Error testing cloud connections: {e}")
            return {"error": str(e)}


# Global cloud integrations instance
cloud_integrations = CloudIntegrations()


async def initialize_cloud_integrations():
    """Initialize the cloud integrations"""
    await cloud_integrations.initialize()














