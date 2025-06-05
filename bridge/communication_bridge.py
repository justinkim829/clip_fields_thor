"""
CLIP-Fields Thor Integration Bridge
===================================

This module implements the ZeroRPC-based communication bridge that enables
the dual-process architecture for integrating CLIP-Fields with AI2-THOR.

The bridge provides a clean API abstraction that hides the complexity of
inter-process communication while maintaining high performance suitable
for real-time navigation control.
"""

import zerorpc
import numpy as np
import cv2
import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
import threading
import queue
import msgpack
import gzip
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """Structured observation data from AI2-THOR."""
    rgb: np.ndarray  # RGB image (H, W, 3)
    depth: np.ndarray  # Depth image (H, W)
    pose: np.ndarray  # 4x4 transformation matrix
    timestamp: float
    camera_intrinsics: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class SemanticQuery:
    """Semantic query for spatial search."""
    text: str
    spatial_bounds: Optional[Tuple[float, float, float, float, float, float]] = None
    resolution: float = 0.05  # 5cm resolution
    max_points: int = 1000


@dataclass
class QueryResult:
    """Result of semantic spatial query."""
    query: str
    probability_map: np.ndarray  # 3D probability distribution
    spatial_coords: np.ndarray  # Corresponding 3D coordinates
    max_prob_location: Tuple[float, float, float]
    confidence: float
    processing_time: float


class DataCompressor:
    """Handles efficient compression of observation data for transmission."""
    
    @staticmethod
    def compress_rgb(rgb: np.ndarray, quality: int = 85) -> bytes:
        """Compress RGB image using JPEG."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', rgb, encode_param)
        return encoded_img.tobytes()
    
    @staticmethod
    def decompress_rgb(data: bytes) -> np.ndarray:
        """Decompress RGB image from JPEG."""
        nparr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    @staticmethod
    def compress_depth(depth: np.ndarray) -> bytes:
        """Compress depth image using lossless compression."""
        # Convert to 16-bit integers (millimeter precision)
        depth_mm = (depth * 1000).astype(np.uint16)
        # Use gzip compression
        return gzip.compress(depth_mm.tobytes())
    
    @staticmethod
    def decompress_depth(data: bytes) -> np.ndarray:
        """Decompress depth image."""
        decompressed = gzip.decompress(data)
        depth_mm = np.frombuffer(decompressed, dtype=np.uint16)
        # Reshape and convert back to meters
        return (depth_mm.astype(np.float32) / 1000.0)
    
    @staticmethod
    def compress_observation(obs: Observation) -> Dict[str, Any]:
        """Compress complete observation for transmission."""
        return {
            'rgb': DataCompressor.compress_rgb(obs.rgb),
            'depth': DataCompressor.compress_depth(obs.depth),
            'pose': obs.pose.tobytes(),
            'pose_shape': obs.pose.shape,
            'timestamp': obs.timestamp,
            'camera_intrinsics': obs.camera_intrinsics,
            'metadata': obs.metadata
        }
    
    @staticmethod
    def decompress_observation(data: Dict[str, Any]) -> Observation:
        """Decompress observation from transmission format."""
        rgb = DataCompressor.decompress_rgb(data['rgb'])
        depth = DataCompressor.decompress_depth(data['depth'])
        pose = np.frombuffer(data['pose'], dtype=np.float32).reshape(data['pose_shape'])
        
        return Observation(
            rgb=rgb,
            depth=depth,
            pose=pose,
            timestamp=data['timestamp'],
            camera_intrinsics=data['camera_intrinsics'],
            metadata=data['metadata']
        )


class PerformanceMonitor:
    """Monitors system performance and provides optimization feedback."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.query_times = []
        self.update_times = []
        self.communication_times = []
        self.lock = threading.Lock()
    
    def record_query_time(self, time_ms: float):
        """Record semantic query processing time."""
        with self.lock:
            self.query_times.append(time_ms)
            if len(self.query_times) > self.window_size:
                self.query_times.pop(0)
    
    def record_update_time(self, time_ms: float):
        """Record field update processing time."""
        with self.lock:
            self.update_times.append(time_ms)
            if len(self.update_times) > self.window_size:
                self.update_times.pop(0)
    
    def record_communication_time(self, time_ms: float):
        """Record communication latency."""
        with self.lock:
            self.communication_times.append(time_ms)
            if len(self.communication_times) > self.window_size:
                self.communication_times.pop(0)
    
    def get_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        with self.lock:
            stats = {}
            if self.query_times:
                stats['avg_query_time'] = np.mean(self.query_times)
                stats['max_query_time'] = np.max(self.query_times)
            if self.update_times:
                stats['avg_update_time'] = np.mean(self.update_times)
                stats['max_update_time'] = np.max(self.update_times)
            if self.communication_times:
                stats['avg_comm_time'] = np.mean(self.communication_times)
                stats['max_comm_time'] = np.max(self.communication_times)
            return stats


class BridgeInterface(ABC):
    """Abstract interface for the communication bridge."""
    
    @abstractmethod
    def push_observation(self, observation: Observation) -> bool:
        """Push new observation to semantic memory."""
        pass
    
    @abstractmethod
    def query_semantic_field(self, query: SemanticQuery) -> QueryResult:
        """Query semantic field for spatial-semantic search."""
        pass
    
    @abstractmethod
    def get_field_status(self) -> Dict[str, Any]:
        """Get current status of semantic field."""
        pass
    
    @abstractmethod
    def reset_field(self, spatial_bounds: Tuple[float, float, float, float, float, float]) -> bool:
        """Reset semantic field for new scene."""
        pass


class CLIPFieldsClient(BridgeInterface):
    """Client-side interface for communicating with CLIP-Fields memory process."""
    
    def __init__(self, server_address: str = "tcp://127.0.0.1:4242"):
        self.server_address = server_address
        self.client = None
        self.performance_monitor = PerformanceMonitor()
        self.connect()
    
    def connect(self):
        """Establish connection to CLIP-Fields server."""
        try:
            self.client = zerorpc.Client()
            self.client.connect(self.server_address)
            logger.info(f"Connected to CLIP-Fields server at {self.server_address}")
        except Exception as e:
            logger.error(f"Failed to connect to CLIP-Fields server: {e}")
            raise
    
    def disconnect(self):
        """Close connection to server."""
        if self.client:
            self.client.close()
            self.client = None
    
    def push_observation(self, observation: Observation) -> bool:
        """Push new observation to semantic memory."""
        start_time = time.time()
        try:
            # Compress observation for transmission
            compressed_obs = DataCompressor.compress_observation(observation)
            
            # Send to server
            result = self.client.push_observation(compressed_obs)
            
            # Record performance
            comm_time = (time.time() - start_time) * 1000
            self.performance_monitor.record_communication_time(comm_time)
            
            return result
        except Exception as e:
            logger.error(f"Failed to push observation: {e}")
            return False
    
    def query_semantic_field(self, query: SemanticQuery) -> QueryResult:
        """Query semantic field for spatial-semantic search."""
        start_time = time.time()
        try:
            # Send query to server
            query_dict = asdict(query)
            result_dict = self.client.query_semantic_field(query_dict)
            
            # Decompress result
            probability_map = np.frombuffer(
                result_dict['probability_map'], 
                dtype=np.float32
            ).reshape(result_dict['prob_map_shape'])
            
            spatial_coords = np.frombuffer(
                result_dict['spatial_coords'],
                dtype=np.float32
            ).reshape(result_dict['coords_shape'])
            
            result = QueryResult(
                query=result_dict['query'],
                probability_map=probability_map,
                spatial_coords=spatial_coords,
                max_prob_location=tuple(result_dict['max_prob_location']),
                confidence=result_dict['confidence'],
                processing_time=result_dict['processing_time']
            )
            
            # Record performance
            comm_time = (time.time() - start_time) * 1000
            self.performance_monitor.record_communication_time(comm_time)
            self.performance_monitor.record_query_time(result.processing_time)
            
            return result
        except Exception as e:
            logger.error(f"Failed to query semantic field: {e}")
            raise
    
    def get_field_status(self) -> Dict[str, Any]:
        """Get current status of semantic field."""
        try:
            return self.client.get_field_status()
        except Exception as e:
            logger.error(f"Failed to get field status: {e}")
            return {}
    
    def reset_field(self, spatial_bounds: Tuple[float, float, float, float, float, float]) -> bool:
        """Reset semantic field for new scene."""
        try:
            return self.client.reset_field(spatial_bounds)
        except Exception as e:
            logger.error(f"Failed to reset field: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get client-side performance statistics."""
        return self.performance_monitor.get_stats()


class AsyncObservationBuffer:
    """Asynchronous buffer for handling observation updates without blocking navigation."""
    
    def __init__(self, max_size: int = 100):
        self.buffer = queue.Queue(maxsize=max_size)
        self.processing_thread = None
        self.running = False
        self.client = None
    
    def start(self, client: CLIPFieldsClient):
        """Start asynchronous processing."""
        self.client = client
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_observations)
        self.processing_thread.start()
        logger.info("Started asynchronous observation processing")
    
    def stop(self):
        """Stop asynchronous processing."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
        logger.info("Stopped asynchronous observation processing")
    
    def add_observation(self, observation: Observation) -> bool:
        """Add observation to processing buffer."""
        try:
            self.buffer.put_nowait(observation)
            return True
        except queue.Full:
            logger.warning("Observation buffer full, dropping observation")
            return False
    
    def _process_observations(self):
        """Background thread for processing observations."""
        while self.running:
            try:
                # Get observation with timeout
                observation = self.buffer.get(timeout=1.0)
                
                # Process observation
                if self.client:
                    success = self.client.push_observation(observation)
                    if not success:
                        logger.warning("Failed to process observation")
                
                self.buffer.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing observation: {e}")


class CoordinateTransformer:
    """Handles coordinate system transformations between Unity and NeRF conventions."""
    
    @staticmethod
    def unity_to_nerf_pose(unity_pose: np.ndarray) -> np.ndarray:
        """Transform Unity pose to NeRF coordinate system.
        
        Unity: Left-handed, Y-up
        NeRF: Right-handed, Z-up
        """
        # Unity to NeRF transformation matrix
        # Swap Y and Z axes, negate X to change handedness
        transform = np.array([
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        return transform @ unity_pose @ np.linalg.inv(transform)
    
    @staticmethod
    def nerf_to_unity_pose(nerf_pose: np.ndarray) -> np.ndarray:
        """Transform NeRF pose to Unity coordinate system."""
        # Inverse transformation
        transform = np.array([
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        return np.linalg.inv(transform) @ nerf_pose @ transform
    
    @staticmethod
    def unity_to_nerf_point(unity_point: np.ndarray) -> np.ndarray:
        """Transform 3D point from Unity to NeRF coordinates."""
        if unity_point.shape[-1] == 3:
            # Add homogeneous coordinate
            unity_point = np.concatenate([unity_point, np.ones((*unity_point.shape[:-1], 1))], axis=-1)
        
        transform = np.array([
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        nerf_point = (transform @ unity_point.T).T
        return nerf_point[..., :3]  # Remove homogeneous coordinate
    
    @staticmethod
    def nerf_to_unity_point(nerf_point: np.ndarray) -> np.ndarray:
        """Transform 3D point from NeRF to Unity coordinates."""
        if nerf_point.shape[-1] == 3:
            # Add homogeneous coordinate
            nerf_point = np.concatenate([nerf_point, np.ones((*nerf_point.shape[:-1], 1))], axis=-1)
        
        transform = np.array([
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        unity_point = (np.linalg.inv(transform) @ nerf_point.T).T
        return unity_point[..., :3]  # Remove homogeneous coordinate


if __name__ == "__main__":
    # Example usage and testing
    logger.info("CLIP-Fields Thor Integration Bridge initialized")
    
    # Test coordinate transformations
    unity_pose = np.eye(4, dtype=np.float32)
    unity_pose[:3, 3] = [1, 2, 3]  # Translation
    
    nerf_pose = CoordinateTransformer.unity_to_nerf_pose(unity_pose)
    recovered_pose = CoordinateTransformer.nerf_to_unity_pose(nerf_pose)
    
    print("Unity pose:")
    print(unity_pose)
    print("NeRF pose:")
    print(nerf_pose)
    print("Recovered pose:")
    print(recovered_pose)
    print("Transformation error:", np.max(np.abs(unity_pose - recovered_pose)))

