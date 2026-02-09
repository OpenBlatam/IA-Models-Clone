"""
S3 Service para almacenamiento de archivos de audio
"""

import logging
from typing import Optional, BinaryIO, Dict, Any
from io import BytesIO
import json

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import ClientError
    from botocore.config import Config
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not available. Install with: pip install boto3")


class S3Service:
    """
    Servicio para interactuar con S3
    Optimizado para archivos de audio
    """
    
    def __init__(
        self,
        bucket_name: str,
        region_name: Optional[str] = None,
        endpoint_url: Optional[str] = None
    ):
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for S3Service")
        
        self.bucket_name = bucket_name
        
        # Configuración optimizada para S3
        config = Config(
            max_pool_connections=50,
            retries={'max_attempts': 3, 'mode': 'standard'}
        )
        
        self.s3_client = boto3.client(
            's3',
            region_name=region_name,
            endpoint_url=endpoint_url,
            config=config
        )
        
        self.s3_resource = boto3.resource(
            's3',
            region_name=region_name,
            endpoint_url=endpoint_url
        )
    
    def upload_file(
        self,
        file_obj: BinaryIO,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        acl: Optional[str] = None
    ) -> bool:
        """
        Sube un archivo a S3
        
        Args:
            file_obj: Objeto de archivo o bytes
            key: Clave (path) en S3
            content_type: Tipo de contenido (ej: 'audio/mpeg')
            metadata: Metadatos adicionales
            acl: ACL (ej: 'public-read', 'private')
            
        Returns:
            True si fue exitoso
        """
        try:
            extra_args = {}
            
            if content_type:
                extra_args['ContentType'] = content_type
            
            if metadata:
                extra_args['Metadata'] = metadata
            
            if acl:
                extra_args['ACL'] = acl
            
            # Si es bytes, convertir a BytesIO
            if isinstance(file_obj, bytes):
                file_obj = BytesIO(file_obj)
            
            self.s3_client.upload_fileobj(
                file_obj,
                self.bucket_name,
                key,
                ExtraArgs=extra_args if extra_args else None
            )
            
            logger.info(f"File uploaded to S3: s3://{self.bucket_name}/{key}")
            return True
        except ClientError as e:
            logger.error(f"S3 upload error: {e}")
            raise
    
    def download_file(self, key: str) -> bytes:
        """
        Descarga un archivo de S3
        
        Args:
            key: Clave del archivo en S3
            
        Returns:
            Contenido del archivo como bytes
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body'].read()
        except ClientError as e:
            logger.error(f"S3 download error: {e}")
            raise
    
    def delete_file(self, key: str) -> bool:
        """Elimina un archivo de S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            logger.info(f"File deleted from S3: s3://{self.bucket_name}/{key}")
            return True
        except ClientError as e:
            logger.error(f"S3 delete error: {e}")
            raise
    
    def get_presigned_url(
        self,
        key: str,
        expiration: int = 3600,
        http_method: str = 'GET'
    ) -> str:
        """
        Genera una URL pre-firmada para acceso temporal
        
        Args:
            key: Clave del archivo
            expiration: Tiempo de expiración en segundos
            http_method: Método HTTP ('GET' o 'PUT')
            
        Returns:
            URL pre-firmada
        """
        try:
            params = {
                'Bucket': self.bucket_name,
                'Key': key
            }
            
            url = self.s3_client.generate_presigned_url(
                'get_object' if http_method == 'GET' else 'put_object',
                Params=params,
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            logger.error(f"S3 presigned URL error: {e}")
            raise
    
    def list_files(self, prefix: str = "", max_keys: int = 1000) -> list:
        """
        Lista archivos en S3 con un prefijo
        
        Args:
            prefix: Prefijo para filtrar
            max_keys: Número máximo de resultados
            
        Returns:
            Lista de claves
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except ClientError as e:
            logger.error(f"S3 list error: {e}")
            raise
    
    def file_exists(self, key: str) -> bool:
        """Verifica si un archivo existe en S3"""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise
    
    def get_file_metadata(self, key: str) -> Dict[str, Any]:
        """Obtiene metadatos de un archivo"""
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return {
                'content_type': response.get('ContentType'),
                'content_length': response.get('ContentLength'),
                'last_modified': response.get('LastModified'),
                'metadata': response.get('Metadata', {}),
                'etag': response.get('ETag')
            }
        except ClientError as e:
            logger.error(f"S3 metadata error: {e}")
            raise















