"""
Sync Cache

Tracks external_id ↔ lifeops_id mappings for bidirectional sync.
"""

import sqlite3
from datetime import datetime
from typing import Optional, List, Tuple
from pathlib import Path

from loguru import logger


class SyncCache:
    """
    Cache for tracking sync mappings between external services and LifeOps.
    
    Prevents duplicate syncs and enables bidirectional updates.
    """
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._create_tables()
        
        logger.info(f"[SYNC-CACHE] Initialized at {db_path}")
    
    def _create_tables(self):
        """Create database schema."""
        cursor = self.conn.cursor()
        
        # Sync mapping table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sync_mapping (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                external_service TEXT NOT NULL,
                external_id TEXT NOT NULL,
                lifeops_event_id TEXT NOT NULL,
                last_synced TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                sync_count INTEGER DEFAULT 1,
                metadata TEXT,
                UNIQUE(external_service, external_id)
            )
        """)
        
        # Sync history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sync_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                external_service TEXT NOT NULL,
                sync_type TEXT NOT NULL,
                items_synced INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                details TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_external_service_id 
            ON sync_mapping(external_service, external_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_lifeops_id 
            ON sync_mapping(lifeops_event_id)
        """)
        
        self.conn.commit()
    
    def store_mapping(
        self,
        service: str,
        external_id: str,
        lifeops_id: str,
        metadata: Optional[str] = None
    ):
        """
        Store a sync mapping.
        
        Args:
            service: External service name ('todoist', 'gcal', 'notion')
            external_id: ID in external service
            lifeops_id: ID in LifeOps timeline
            metadata: Optional JSON metadata
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO sync_mapping 
            (external_service, external_id, lifeops_event_id, metadata)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(external_service, external_id) 
            DO UPDATE SET
                lifeops_event_id = excluded.lifeops_event_id,
                last_synced = CURRENT_TIMESTAMP,
                sync_count = sync_count + 1,
                metadata = excluded.metadata
        """, (service, external_id, lifeops_id, metadata))
        
        self.conn.commit()
        
        logger.debug(
            f"[SYNC-CACHE] Stored mapping: {service}:{external_id} → {lifeops_id}"
        )
    
    def get_lifeops_id(
        self,
        service: str,
        external_id: str
    ) -> Optional[str]:
        """
        Get LifeOps ID for an external ID.
        
        Args:
            service: External service name
            external_id: ID in external service
            
        Returns:
            LifeOps event ID or None if not found
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT lifeops_event_id 
            FROM sync_mapping
            WHERE external_service = ? AND external_id = ?
        """, (service, external_id))
        
        result = cursor.fetchone()
        return result[0] if result else None
    
    def get_external_id(
        self,
        service: str,
        lifeops_id: str
    ) -> Optional[str]:
        """
        Get external ID for a LifeOps ID.
        
        Args:
            service: External service name
            lifeops_id: LifeOps event ID
            
        Returns:
            External ID or None if not found
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT external_id 
            FROM sync_mapping
            WHERE external_service = ? AND lifeops_event_id = ?
        """, (service, lifeops_id))
        
        result = cursor.fetchone()
        return result[0] if result else None
    
    def is_synced(
        self,
        service: str,
        external_id: str
    ) -> bool:
        """
        Check if an external item has been synced.
        
        Args:
            service: External service name
            external_id: ID in external service
            
        Returns:
            True if already synced
        """
        return self.get_lifeops_id(service, external_id) is not None
    
    def get_all_mappings(
        self,
        service: Optional[str] = None
    ) -> List[Tuple[str, str, str]]:
        """
        Get all sync mappings.
        
        Args:
            service: Optional service filter
            
        Returns:
            List of (service, external_id, lifeops_id) tuples
        """
        cursor = self.conn.cursor()
        
        if service:
            cursor.execute("""
                SELECT external_service, external_id, lifeops_event_id
                FROM sync_mapping
                WHERE external_service = ?
                ORDER BY last_synced DESC
            """, (service,))
        else:
            cursor.execute("""
                SELECT external_service, external_id, lifeops_event_id
                FROM sync_mapping
                ORDER BY last_synced DESC
            """)
        
        return cursor.fetchall()
    
    def delete_mapping(
        self,
        service: str,
        external_id: str
    ):
        """
        Delete a sync mapping.
        
        Args:
            service: External service name
            external_id: ID in external service
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            DELETE FROM sync_mapping
            WHERE external_service = ? AND external_id = ?
        """, (service, external_id))
        
        self.conn.commit()
        
        logger.debug(
            f"[SYNC-CACHE] Deleted mapping: {service}:{external_id}"
        )
    
    def record_sync(
        self,
        service: str,
        sync_type: str,
        items_synced: int,
        errors: int = 0,
        details: Optional[str] = None
    ):
        """
        Record a sync operation in history.
        
        Args:
            service: External service name
            sync_type: Type of sync ('pull', 'push', 'bidirectional')
            items_synced: Number of items synced
            errors: Number of errors
            details: Optional details/error messages
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO sync_history
            (external_service, sync_type, items_synced, errors, details)
            VALUES (?, ?, ?, ?, ?)
        """, (service, sync_type, items_synced, errors, details))
        
        self.conn.commit()
        
        logger.info(
            f"[SYNC-CACHE] Recorded sync: {service} {sync_type} "
            f"({items_synced} items, {errors} errors)"
        )
    
    def get_sync_history(
        self,
        service: Optional[str] = None,
        limit: int = 100
    ) -> List[dict]:
        """
        Get sync history.
        
        Args:
            service: Optional service filter
            limit: Max number of records
            
        Returns:
            List of sync history records
        """
        cursor = self.conn.cursor()
        
        if service:
            cursor.execute("""
                SELECT external_service, sync_type, items_synced, 
                       errors, timestamp, details
                FROM sync_history
                WHERE external_service = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (service, limit))
        else:
            cursor.execute("""
                SELECT external_service, sync_type, items_synced, 
                       errors, timestamp, details
                FROM sync_history
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
        
        return [
            {
                'service': row[0],
                'sync_type': row[1],
                'items_synced': row[2],
                'errors': row[3],
                'timestamp': row[4],
                'details': row[5],
            }
            for row in rows
        ]
    
    def get_stats(self) -> dict:
        """Get sync cache statistics."""
        cursor = self.conn.cursor()
        
        # Total mappings
        cursor.execute("SELECT COUNT(*) FROM sync_mapping")
        total_mappings = cursor.fetchone()[0]
        
        # Mappings by service
        cursor.execute("""
            SELECT external_service, COUNT(*) 
            FROM sync_mapping 
            GROUP BY external_service
        """)
        by_service = dict(cursor.fetchall())
        
        # Total syncs
        cursor.execute("SELECT COUNT(*) FROM sync_history")
        total_syncs = cursor.fetchone()[0]
        
        # Last sync
        cursor.execute("""
            SELECT external_service, timestamp 
            FROM sync_history 
            ORDER BY timestamp DESC 
            LIMIT 1
        """)
        last_sync = cursor.fetchone()
        
        return {
            'total_mappings': total_mappings,
            'mappings_by_service': by_service,
            'total_syncs': total_syncs,
            'last_sync': {
                'service': last_sync[0] if last_sync else None,
                'timestamp': last_sync[1] if last_sync else None,
            }
        }
    
    def close(self):
        """Close database connection."""
        self.conn.close()
        logger.info("[SYNC-CACHE] Closed")
