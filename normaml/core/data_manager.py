"""
Data management with zero-copy operations through shared memory.

This module implements the SharedMemoryDataManager that provides efficient
data sharing across processes and containers using shared memory, memory mapping,
and Apache Arrow for serialization.
"""

import os
import tempfile
import mmap
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from multiprocessing import shared_memory
import threading

import polars as pl
import pyarrow as pa
import numpy as np

from .interfaces import DataManager, SharedMemoryHandle, DataManagerError, SharedMemoryError


logger = logging.getLogger(__name__)


class SharedMemoryDataManager(DataManager):
    """
    Data manager that uses shared memory for zero-copy data operations.
    
    This implementation provides efficient data sharing across processes and containers
    using POSIX shared memory, memory-mapped files, and Apache Arrow serialization.
    
    Features:
    - Zero-copy data access through shared memory
    - Memory-mapped file support for container sharing
    - Apache Arrow IPC for efficient serialization
    - Automatic cleanup of resources
    - Thread-safe operations
    """
    
    def __init__(self, temp_dir: Optional[str] = None, max_memory_mb: int = 1024):
        """
        Initialize the SharedMemoryDataManager.
        
        Args:
            temp_dir: Directory for temporary files. If None, creates a new temp dir.
            max_memory_mb: Maximum memory to use for shared memory blocks in MB.
        """
        self.shared_blocks: Dict[str, shared_memory.SharedMemory] = {}
        self.data_registry: Dict[str, SharedMemoryHandle] = {}
        self.memory_mapped_files: Dict[str, mmap.mmap] = {}
        self.file_handles: Dict[str, Any] = {}
        
        # Setup temporary directory
        if temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="normaml_shm_")
            self._owns_temp_dir = True
        else:
            self.temp_dir = temp_dir
            self._owns_temp_dir = False
            os.makedirs(self.temp_dir, exist_ok=True)
        
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_usage = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Initialized SharedMemoryDataManager with temp_dir={self.temp_dir}, "
                   f"max_memory_mb={max_memory_mb}")
    
    def load_data(self, source: str, **kwargs) -> pl.LazyFrame:
        """
        Lazy loading of data from various sources.
        
        Args:
            source: Path to data file (CSV, Parquet, etc.)
            **kwargs: Additional parameters for Polars scan functions
            
        Returns:
            Lazy frame for deferred execution
            
        Raises:
            DataManagerError: If source cannot be loaded
        """
        try:
            source_path = Path(source)
            
            if not source_path.exists():
                raise DataManagerError(f"Data source not found: {source}")
            
            # Choose appropriate scanner based on file extension
            suffix = source_path.suffix.lower()
            
            if suffix == '.csv':
                return pl.scan_csv(source, **kwargs)
            elif suffix == '.parquet':
                return pl.scan_parquet(source, **kwargs)
            elif suffix == '.json':
                # For JSON, we need to load first then convert to lazy
                df = pl.read_json(source, **kwargs)
                return df.lazy()
            elif suffix == '.xlsx':
                # For Excel, we need to load first then convert to lazy
                df = pl.read_excel(source, **kwargs)
                return df.lazy()
            else:
                # Try CSV as fallback
                logger.warning(f"Unknown file extension {suffix}, trying CSV scanner")
                return pl.scan_csv(source, **kwargs)
                
        except Exception as e:
            raise DataManagerError(f"Failed to load data from {source}: {e}") from e
    
    def create_shared_memory_view(self, data: pl.DataFrame, name: str) -> SharedMemoryHandle:
        """
        Create shared memory representation of data for zero-copy access.
        
        Args:
            data: DataFrame to be shared
            name: Unique identifier for the shared memory segment
            
        Returns:
            Handle for accessing the shared memory segment
            
        Raises:
            SharedMemoryError: If shared memory creation fails
            DataManagerError: If memory limit exceeded
        """
        with self._lock:
            try:
                # Check if name already exists
                if name in self.shared_blocks:
                    logger.warning(f"Shared memory block '{name}' already exists, cleaning up old one")
                    self._cleanup_single_block(name)
                
                # Convert to Arrow table for efficient serialization
                arrow_table = data.to_arrow()
                
                # Serialize using Arrow IPC format
                sink = pa.BufferOutputStream()
                with pa.ipc.new_stream(sink, arrow_table.schema) as writer:
                    writer.write_table(arrow_table)
                
                buffer_bytes = sink.getvalue().to_pybytes()
                buffer_size = len(buffer_bytes)
                
                # Check memory limits
                if self.current_memory_usage + buffer_size > self.max_memory_bytes:
                    raise DataManagerError(
                        f"Memory limit exceeded. Current: {self.current_memory_usage}, "
                        f"Requested: {buffer_size}, Max: {self.max_memory_bytes}"
                    )
                
                # Create shared memory block
                shm_name = f"normaml_{name}_{os.getpid()}"
                try:
                    shm = shared_memory.SharedMemory(
                        create=True,
                        size=buffer_size,
                        name=shm_name
                    )
                except FileExistsError:
                    # If name exists, try with timestamp
                    import time
                    shm_name = f"normaml_{name}_{os.getpid()}_{int(time.time())}"
                    shm = shared_memory.SharedMemory(
                        create=True,
                        size=buffer_size,
                        name=shm_name
                    )
                
                # Copy data to shared memory
                shm.buf[:buffer_size] = buffer_bytes
                
                # Store references
                self.shared_blocks[name] = shm
                self.current_memory_usage += buffer_size
                
                # Create handle
                handle = SharedMemoryHandle(
                    name=shm.name,
                    size=buffer_size,
                    dtype="arrow_ipc",
                    shape=arrow_table.shape
                )
                
                self.data_registry[name] = handle
                
                logger.info(f"Created shared memory block '{name}' with size {buffer_size} bytes")
                return handle
                
            except Exception as e:
                raise SharedMemoryError(f"Failed to create shared memory view for '{name}': {e}") from e
    
    def load_from_shared_memory(self, handle: SharedMemoryHandle) -> pl.LazyFrame:
        """
        Load data from shared memory using handle.
        
        Args:
            handle: Shared memory handle containing metadata
            
        Returns:
            Lazy frame containing the shared data
            
        Raises:
            SharedMemoryError: If shared memory access fails
        """
        try:
            # Connect to existing shared memory
            shm = shared_memory.SharedMemory(name=handle.name)
            
            # Read data as Arrow IPC
            buffer = pa.py_buffer(shm.buf[:handle.size])
            reader = pa.ipc.open_stream(buffer)
            arrow_table = reader.read_all()
            
            # Convert to Polars LazyFrame
            polars_df = pl.from_arrow(arrow_table)
            
            # Don't close the shared memory here as it might be used by others
            
            logger.debug(f"Loaded data from shared memory '{handle.name}' with shape {arrow_table.shape}")
            return polars_df.lazy()
            
        except Exception as e:
            raise SharedMemoryError(f"Failed to load from shared memory '{handle.name}': {e}") from e
    
    def create_arrow_buffer(self, data: pl.DataFrame) -> pa.Buffer:
        """
        Create Arrow buffer for inter-process data exchange.
        
        Args:
            data: DataFrame to convert to Arrow buffer
            
        Returns:
            Arrow buffer containing serialized data
        """
        try:
            arrow_table = data.to_arrow()
            sink = pa.BufferOutputStream()
            
            with pa.ipc.new_stream(sink, arrow_table.schema) as writer:
                writer.write_table(arrow_table)
            
            return sink.getvalue()
            
        except Exception as e:
            raise DataManagerError(f"Failed to create Arrow buffer: {e}") from e
    
    def share_across_containers(self, data_id: str, container_ids: List[str]) -> Dict[str, Any]:
        """
        Prepare data for sharing across Docker containers.
        
        Args:
            data_id: Identifier of the data to share
            container_ids: List of container IDs that need access
            
        Returns:
            Configuration for container data sharing
            
        Raises:
            DataManagerError: If data not found or sharing setup fails
        """
        with self._lock:
            try:
                if data_id not in self.data_registry:
                    raise DataManagerError(f"Data '{data_id}' not found in registry")
                
                handle = self.data_registry[data_id]
                
                # Create memory-mapped file for container sharing
                mmap_filename = f"{data_id}.mmap"
                mmap_path = os.path.join(self.temp_dir, mmap_filename)
                
                # Copy data from shared memory to memory-mapped file
                if data_id in self.shared_blocks:
                    shm = self.shared_blocks[data_id]
                    
                    with open(mmap_path, 'wb') as f:
                        f.write(shm.buf[:handle.size])
                    
                    # Create memory-mapped view
                    with open(mmap_path, 'r+b') as f:
                        mm = mmap.mmap(f.fileno(), 0)
                        self.memory_mapped_files[data_id] = mm
                        self.file_handles[data_id] = f
                else:
                    raise DataManagerError(f"Shared memory block for '{data_id}' not found")
                
                # Prepare container configuration
                container_config = {
                    'type': 'memory_mapped_file',
                    'path': mmap_path,
                    'handle': handle.to_dict(),
                    'volume_mount': f"{mmap_path}:/shared_data/{mmap_filename}:ro",
                    'container_ids': container_ids,
                    'environment_vars': {
                        'SHARED_DATA_PATH': f'/shared_data/{mmap_filename}',
                        'SHARED_DATA_HANDLE': str(handle.to_dict())
                    }
                }
                
                logger.info(f"Prepared data '{data_id}' for sharing across {len(container_ids)} containers")
                return container_config
                
            except Exception as e:
                raise DataManagerError(f"Failed to setup container sharing for '{data_id}': {e}") from e
    
    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get information about current memory usage.
        
        Returns:
            Dictionary with memory usage statistics
        """
        with self._lock:
            utilization_percent = int((self.current_memory_usage / self.max_memory_bytes) * 100) if self.max_memory_bytes > 0 else 0
            return {
                'total_shared_memory_bytes': self.current_memory_usage,
                'total_shared_memory_mb': self.current_memory_usage // (1024 * 1024),
                'active_blocks': len(self.shared_blocks),
                'registered_datasets': len(self.data_registry),
                'memory_mapped_files': len(self.memory_mapped_files),
                'max_memory_bytes': self.max_memory_bytes,
                'memory_utilization_percent': utilization_percent
            }
    
    def _cleanup_single_block(self, name: str) -> None:
        """
        Clean up a single shared memory block.
        
        Args:
            name: Name of the block to clean up
        """
        try:
            if name in self.shared_blocks:
                shm = self.shared_blocks[name]
                handle = self.data_registry.get(name)
                
                if handle:
                    self.current_memory_usage -= handle.size
                
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass  # Already unlinked
                
                del self.shared_blocks[name]
                
                if name in self.data_registry:
                    del self.data_registry[name]
                
                logger.debug(f"Cleaned up shared memory block '{name}'")
            
            # Clean up memory-mapped file if exists
            if name in self.memory_mapped_files:
                mm = self.memory_mapped_files[name]
                mm.close()
                del self.memory_mapped_files[name]
                
                if name in self.file_handles:
                    fh = self.file_handles[name]
                    fh.close()
                    del self.file_handles[name]
                
                # Remove the actual file
                mmap_path = os.path.join(self.temp_dir, f"{name}.mmap")
                if os.path.exists(mmap_path):
                    os.remove(mmap_path)
                
                logger.debug(f"Cleaned up memory-mapped file for '{name}'")
                
        except Exception as e:
            logger.error(f"Error cleaning up block '{name}': {e}")
    
    def cleanup(self) -> None:
        """Clean up all shared memory resources and temporary files."""
        with self._lock:
            logger.info("Starting cleanup of SharedMemoryDataManager")
            
            # Clean up all shared memory blocks
            for name in list(self.shared_blocks.keys()):
                self._cleanup_single_block(name)
            
            # Clean up remaining memory-mapped files
            for name in list(self.memory_mapped_files.keys()):
                try:
                    mm = self.memory_mapped_files[name]
                    mm.close()
                    del self.memory_mapped_files[name]
                    
                    if name in self.file_handles:
                        fh = self.file_handles[name]
                        fh.close()
                        del self.file_handles[name]
                except Exception as e:
                    logger.error(f"Error closing memory-mapped file '{name}': {e}")
            
            # Clean up temporary directory if we own it
            if self._owns_temp_dir and os.path.exists(self.temp_dir):
                try:
                    shutil.rmtree(self.temp_dir, ignore_errors=True)
                    logger.info(f"Removed temporary directory: {self.temp_dir}")
                except Exception as e:
                    logger.error(f"Error removing temporary directory: {e}")
            
            # Reset counters
            self.current_memory_usage = 0
            
            logger.info("SharedMemoryDataManager cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
    
    def __del__(self):
        """Destructor with cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during destruction