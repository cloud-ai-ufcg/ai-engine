"""
Cluster configuration management utilities.

This module provides utilities for loading and managing multi-cluster configurations
from config.yaml, supporting flexible cluster definitions with profiles and metadata.
"""

from typing import Dict, List, Any, Optional
from .util import load_config, get_logger

logger = get_logger("cluster_config")


class ClusterConfig:
    """Represents a single cluster configuration."""
    
    def __init__(self, cluster_id: str, cluster_profile: str, cluster_label: str = None, cluster_description: str = ""):
        """
        Initialize a cluster configuration.
        
        Args:
            cluster_id: Unique identifier for the cluster (e.g., "aws-3241")
            cluster_profile: Profile/category for grouping (e.g., "private", "public")
            cluster_label: Label used in monitoring data (for backward compatibility)
            cluster_description: Human-readable description
        """
        self.cluster_id = cluster_id
        self.cluster_profile = cluster_profile
        self.cluster_label = cluster_label or cluster_id
        self.cluster_description = cluster_description
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cluster config to dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "cluster_profile": self.cluster_profile,
            "cluster_label": self.cluster_label,
            "cluster_description": self.cluster_description,
        }
    
    def __repr__(self) -> str:
        return f"ClusterConfig(id={self.cluster_id}, profile={self.cluster_profile}, label={self.cluster_label})"


class ClusterManager:
    """Manages all cluster configurations and provides lookup utilities."""
    
    def __init__(self):
        """Initialize cluster manager and load configurations."""
        self.clusters: List[ClusterConfig] = []
        self.clusters_by_id: Dict[str, ClusterConfig] = {}
        self.clusters_by_label: Dict[str, ClusterConfig] = {}
        self.clusters_by_profile: Dict[str, List[ClusterConfig]] = {}
        self._load_clusters()
    
    def _load_clusters(self) -> None:
        """Load clusters from configuration."""
        config = load_config()
        clusters_config = config.get("clusters", {})
        
        if not clusters_config.get("enabled", False):
            logger.warning("Cluster support not enabled in configuration")
            return
        
        definitions = clusters_config.get("definitions", [])
        
        if not definitions:
            logger.warning("No cluster definitions found in configuration")
            return
        
        for cluster_def in definitions:
            try:
                cluster = ClusterConfig(
                    cluster_id=cluster_def.get("cluster_id"),
                    cluster_profile=cluster_def.get("cluster_profile"),
                    cluster_label=cluster_def.get("cluster_label"),
                    cluster_description=cluster_def.get("cluster_description", ""),
                )
                
                # Validate required fields
                if not cluster.cluster_id or not cluster.cluster_profile:
                    logger.warning(f"Skipping cluster definition with missing required fields: {cluster_def}")
                    continue
                
                self.clusters.append(cluster)
                self.clusters_by_id[cluster.cluster_id] = cluster
                self.clusters_by_label[cluster.cluster_label] = cluster
                
                # Index by profile
                if cluster.cluster_profile not in self.clusters_by_profile:
                    self.clusters_by_profile[cluster.cluster_profile] = []
                self.clusters_by_profile[cluster.cluster_profile].append(cluster)
            
            except Exception as e:
                logger.error(f"Error loading cluster definition {cluster_def}: {e}")
        
        logger.info(f"Loaded {len(self.clusters)} cluster(s): {', '.join([c.cluster_id for c in self.clusters])}")
    
    def get_cluster_by_id(self, cluster_id: str) -> Optional[ClusterConfig]:
        """Get cluster configuration by cluster ID."""
        return self.clusters_by_id.get(cluster_id)
    
    def get_cluster_by_label(self, cluster_label: str) -> Optional[ClusterConfig]:
        """Get cluster configuration by cluster label (monitoring data identifier)."""
        return self.clusters_by_label.get(cluster_label)
    
    def get_clusters_by_profile(self, profile: str) -> List[ClusterConfig]:
        """Get all clusters for a specific profile."""
        return self.clusters_by_profile.get(profile, [])
    
    def get_all_clusters(self) -> List[ClusterConfig]:
        """Get all cluster configurations."""
        return self.clusters
    
    def get_cluster_ids(self) -> List[str]:
        """Get list of all cluster IDs."""
        return [c.cluster_id for c in self.clusters]
    
    def get_cluster_labels(self) -> List[str]:
        """Get list of all cluster labels (for monitoring data filtering)."""
        return [c.cluster_label for c in self.clusters]
    
    def get_profiles(self) -> List[str]:
        """Get list of all unique cluster profiles."""
        return list(self.clusters_by_profile.keys())
    
    def get_cluster_count(self) -> int:
        """Get total number of configured clusters."""
        return len(self.clusters)
    
    def resolve_cluster_label_to_id(self, cluster_label: str) -> Optional[str]:
        """
        Resolve a cluster label (from monitoring data) to cluster ID.
        
        Args:
            cluster_label: The cluster label from monitoring data
        
        Returns:
            The cluster ID, or None if not found
        """
        cluster = self.get_cluster_by_label(cluster_label)
        return cluster.cluster_id if cluster else None
    
    def resolve_cluster_id_to_label(self, cluster_id: str) -> Optional[str]:
        """
        Resolve a cluster ID to its monitoring label.
        
        Args:
            cluster_id: The cluster ID
        
        Returns:
            The cluster label for monitoring data, or None if not found
        """
        cluster = self.get_cluster_by_id(cluster_id)
        return cluster.cluster_label if cluster else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all clusters to dictionary representation."""
        return {
            c.cluster_id: c.to_dict() for c in self.clusters
        }


# Global cluster manager instance
_cluster_manager: Optional[ClusterManager] = None


def get_cluster_manager() -> ClusterManager:
    """Get the global cluster manager instance."""
    global _cluster_manager
    if _cluster_manager is None:
        _cluster_manager = ClusterManager()
    return _cluster_manager


def reset_cluster_manager() -> None:
    """Reset the global cluster manager (useful for testing)."""
    global _cluster_manager
    _cluster_manager = None
