import pandas as pd
import json
from datetime import datetime, timedelta
import boto3
from botocore.exceptions import ClientError
import logging
from pathlib import Path
import os
from typing import Dict, List, Optional
import shutil
import gzip

logger = logging.getLogger(__name__)

class PredictionBackupManager:
    def __init__(
        self,
        local_backup_dir: str = "backups",
        s3_bucket: Optional[str] = None,
        backup_frequency_hours: int = 24,
        retention_days: int = 90
    ):
        self.local_backup_dir = Path(local_backup_dir)
        self.s3_bucket = s3_bucket
        self.backup_frequency_hours = backup_frequency_hours
        self.retention_days = retention_days
        self.last_backup_time = None
        
        # Create local backup directory if it doesn't exist
        self.local_backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize S3 client if bucket is provided
        self.s3_client = boto3.client('s3') if s3_bucket else None

    def needs_backup(self) -> bool:
        """Check if it's time for a new backup."""
        if not self.last_backup_time:
            return True
        
        time_since_last = datetime.now() - self.last_backup_time
        return time_since_last.total_seconds() >= self.backup_frequency_hours * 3600

    def create_backup(self, predictions: List[Dict]) -> str:
        """Create a backup of predictions."""
        try:
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.local_backup_dir / f"predictions_backup_{timestamp}.json.gz"
            
            # Compress and save predictions
            with gzip.open(backup_file, 'wt') as f:
                json.dump(predictions, f)
            
            self.last_backup_time = datetime.now()
            logger.info(f"Created backup: {backup_file}")
            
            # Upload to S3 if configured
            if self.s3_bucket:
                self._upload_to_s3(backup_file)
            
            return str(backup_file)
        
        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}")
            raise

    def _upload_to_s3(self, file_path: Path) -> None:
        """Upload backup file to S3."""
        try:
            s3_key = f"predictions/{file_path.name}"
            self.s3_client.upload_file(
                str(file_path),
                self.s3_bucket,
                s3_key
            )
            logger.info(f"Uploaded backup to S3: {s3_key}")
        
        except ClientError as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            raise

    def cleanup_old_backups(self) -> None:
        """Remove backups older than retention period."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            # Clean local backups
            for backup_file in self.local_backup_dir.glob("predictions_backup_*.json.gz"):
                # Extract timestamp from filename
                timestamp_str = backup_file.stem.split('_')[2]
                backup_date = datetime.strptime(timestamp_str, '%Y%m%d')
                
                if backup_date < cutoff_date:
                    backup_file.unlink()
                    logger.info(f"Deleted old backup: {backup_file}")
            
            # Clean S3 backups if configured
            if self.s3_bucket:
                self._cleanup_s3_backups(cutoff_date)
        
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {str(e)}")
            raise

    def _cleanup_s3_backups(self, cutoff_date: datetime) -> None:
        """Remove old backups from S3."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix="predictions/"
            )
            
            for obj in response.get('Contents', []):
                # Extract timestamp from key
                key = obj['Key']
                if not key.endswith('.json.gz'):
                    continue
                
                timestamp_str = key.split('_')[2]
                backup_date = datetime.strptime(timestamp_str, '%Y%m%d')
                
                if backup_date < cutoff_date:
                    self.s3_client.delete_object(
                        Bucket=self.s3_bucket,
                        Key=key
                    )
                    logger.info(f"Deleted old S3 backup: {key}")
        
        except ClientError as e:
            logger.error(f"Error cleaning up S3 backups: {str(e)}")
            raise

    def restore_from_backup(
        self,
        backup_file: Optional[str] = None,
        date: Optional[datetime] = None
    ) -> List[Dict]:
        """Restore predictions from backup."""
        try:
            if backup_file:
                return self._restore_from_file(Path(backup_file))
            
            if date:
                return self._restore_from_date(date)
            
            # If neither specified, restore from latest backup
            return self._restore_latest()
        
        except Exception as e:
            logger.error(f"Error restoring from backup: {str(e)}")
            raise

    def _restore_from_file(self, backup_file: Path) -> List[Dict]:
        """Restore from specific backup file."""
        with gzip.open(backup_file, 'rt') as f:
            return json.load(f)

    def _restore_from_date(self, date: datetime) -> List[Dict]:
        """Restore backup closest to specified date."""
        date_str = date.strftime('%Y%m%d')
        backup_files = list(self.local_backup_dir.glob("predictions_backup_*.json.gz"))
        
        if not backup_files:
            raise FileNotFoundError("No backup files found")
        
        # Find closest backup
        closest_file = min(
            backup_files,
            key=lambda x: abs(
                datetime.strptime(x.stem.split('_')[2][:8], '%Y%m%d') - date
            )
        )
        
        return self._restore_from_file(closest_file)

    def _restore_latest(self) -> List[Dict]:
        """Restore from latest backup."""
        backup_files = list(self.local_backup_dir.glob("predictions_backup_*.json.gz"))
        
        if not backup_files:
            raise FileNotFoundError("No backup files found")
        
        latest_file = max(backup_files, key=lambda x: x.stat().st_mtime)
        return self._restore_from_file(latest_file)

    def list_backups(self) -> List[Dict]:
        """List all available backups."""
        backups = []
        
        # List local backups
        for backup_file in self.local_backup_dir.glob("predictions_backup_*.json.gz"):
            stats = backup_file.stat()
            backups.append({
                'filename': backup_file.name,
                'size': stats.st_size,
                'created': datetime.fromtimestamp(stats.st_mtime),
                'location': 'local'
            })
        
        # List S3 backups if configured
        if self.s3_bucket:
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.s3_bucket,
                    Prefix="predictions/"
                )
                
                for obj in response.get('Contents', []):
                    backups.append({
                        'filename': obj['Key'].split('/')[-1],
                        'size': obj['Size'],
                        'created': obj['LastModified'],
                        'location': 's3'
                    })
            
            except ClientError as e:
                logger.error(f"Error listing S3 backups: {str(e)}")
        
        return sorted(backups, key=lambda x: x['created'], reverse=True)

# Example usage:
# backup_manager = PredictionBackupManager(
#     local_backup_dir="backups",
#     s3_bucket="my-prediction-backups",
#     backup_frequency_hours=24,
#     retention_days=90
# ) 