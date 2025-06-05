#!/usr/bin/env python3
"""
Semantic Field Visualization Script
==================================

This script visualizes the learned CLIP-Fields semantic field,
showing spatial-semantic associations and field evolution.
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "bridge"))

from communication_bridge import CLIPFieldsClient, SemanticQuery

# Configure matplotlib
plt.switch_backend('Agg')

class SemanticFieldVisualizer:
    """Visualizes CLIP-Fields semantic field."""
    
    def __init__(self, checkpoint_path: str = None):
        self.checkpoint_path = checkpoint_path
        self.client = None
        
        # Visualization parameters
        self.resolution = 0.1  # Spatial resolution for visualization
        self.field_bounds = (-6.0, 6.0, -6.0, 6.0)  # Default bounds
        self.height_slice = 1.0  # Height at which to slice the field
        
    def connect_to_server(self):
        """Connect to CLIP-Fields server."""
        print("Connecting to CLIP-Fields server...")
        self.client = CLIPFieldsClient()
        
        try:
            status = self.client.get_field_status()
            print(f"Connected successfully: {status}")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def query_semantic_field_2d(self, query_text: str, bounds: Tuple[float, float, float, float] = None) -> np.ndarray:
        """Query semantic field over a 2D grid."""
        if bounds is None:
            bounds = self.field_bounds
        
        x_min, x_max, z_min, z_max = bounds
        
        # Create 2D grid
        x_coords = np.arange(x_min, x_max, self.resolution)
        z_coords = np.arange(z_min, z_max, self.resolution)
        
        # Query field at each point
        confidence_map = np.zeros((len(z_coords), len(x_coords)))
        
        print(f"Querying semantic field for '{query_text}' over {len(x_coords)}x{len(z_coords)} grid...")
        
        for i, z in enumerate(z_coords):
            for j, x in enumerate(x_coords):
                try:
                    # Create 3D point
                    points = [(x, self.height_slice, z)]
                    
                    # Query semantic field
                    query = SemanticQuery(
                        text=query_text,
                        spatial_points=points,
                        resolution=self.resolution
                    )
                    
                    result = self.client.query_semantic_field(query)
                    
                    if result and len(result.confidences) > 0:
                        confidence_map[i, j] = result.confidences[0]
                    else:
                        confidence_map[i, j] = 0.0
                        
                except Exception as e:
                    print(f"Query failed at ({x:.1f}, {z:.1f}): {e}")
                    confidence_map[i, j] = 0.0
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(z_coords)} rows completed")
        
        return confidence_map, x_coords, z_coords
    
    def visualize_single_query(self, query_text: str, output_path: str = None, scene: str = "FloorPlan1"):
        """Visualize semantic field for a single query."""
        print(f"Visualizing semantic field for: '{query_text}'")
        
        # Query semantic field
        confidence_map, x_coords, z_coords = self.query_semantic_field_2d(query_text)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Create custom colormap
        colors = ['white', 'lightblue', 'blue', 'darkblue', 'red']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('semantic', colors, N=n_bins)
        
        # Plot confidence map
        im = ax.imshow(
            confidence_map,
            extent=[x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],
            origin='lower',
            cmap=cmap,
            vmin=0,
            vmax=1,
            alpha=0.8
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Semantic Confidence', rotation=270, labelpad=20)
        
        # Add contour lines
        contour_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        contours = ax.contour(
            x_coords, z_coords, confidence_map,
            levels=contour_levels,
            colors='black',
            alpha=0.5,
            linewidths=1
        )
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
        
        # Formatting
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Z Position (m)')
        ax.set_title(f'Semantic Field Visualization\nQuery: "{query_text}"\nScene: {scene}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add statistics
        max_confidence = np.max(confidence_map)
        mean_confidence = np.mean(confidence_map)
        high_conf_area = np.sum(confidence_map > 0.5) * (self.resolution ** 2)
        
        stats_text = f'Max: {max_confidence:.3f}\nMean: {mean_confidence:.3f}\nHigh Conf Area: {high_conf_area:.1f}m²'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
        
        return confidence_map, x_coords, z_coords
    
    def visualize_multiple_queries(self, queries: List[str], output_path: str = None, scene: str = "FloorPlan1"):
        """Visualize semantic field for multiple queries."""
        print(f"Visualizing semantic field for {len(queries)} queries")
        
        # Query all fields
        query_results = {}
        for query in queries:
            print(f"Processing query: '{query}'")
            confidence_map, x_coords, z_coords = self.query_semantic_field_2d(query)
            query_results[query] = confidence_map
        
        # Create multi-panel visualization
        n_queries = len(queries)
        cols = min(3, n_queries)
        rows = (n_queries + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_queries == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Custom colormap
        colors = ['white', 'lightblue', 'blue', 'darkblue', 'red']
        cmap = LinearSegmentedColormap.from_list('semantic', colors, N=100)
        
        for i, query in enumerate(queries):
            ax = axes[i]
            confidence_map = query_results[query]
            
            # Plot confidence map
            im = ax.imshow(
                confidence_map,
                extent=[x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],
                origin='lower',
                cmap=cmap,
                vmin=0,
                vmax=1,
                alpha=0.8
            )
            
            # Add contours
            contours = ax.contour(
                x_coords, z_coords, confidence_map,
                levels=[0.3, 0.6, 0.9],
                colors='black',
                alpha=0.5,
                linewidths=0.8
            )
            
            # Formatting
            ax.set_title(f'"{query}"', fontsize=10)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # Add max confidence annotation
            max_conf = np.max(confidence_map)
            ax.text(0.02, 0.98, f'Max: {max_conf:.2f}', transform=ax.transAxes,
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Hide unused subplots
        for i in range(n_queries, len(axes)):
            axes[i].set_visible(False)
        
        # Add overall title
        fig.suptitle(f'Semantic Field Comparison - Scene: {scene}', fontsize=14)
        
        # Add shared colorbar
        cbar = fig.colorbar(im, ax=axes[:n_queries], shrink=0.8, aspect=20)
        cbar.set_label('Semantic Confidence', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Multi-query visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
        
        return query_results
    
    def visualize_field_evolution(self, query_text: str, checkpoint_dir: str, output_path: str = None):
        """Visualize how semantic field evolves during training."""
        print(f"Visualizing field evolution for: '{query_text}'")
        
        checkpoint_dir = Path(checkpoint_dir)
        
        # Find checkpoint files
        checkpoint_files = sorted(checkpoint_dir.glob("*_ep_*.json"))
        
        if not checkpoint_files:
            print("No checkpoint files found for evolution visualization")
            return
        
        print(f"Found {len(checkpoint_files)} checkpoints")
        
        # Select subset of checkpoints for visualization
        if len(checkpoint_files) > 6:
            # Select evenly spaced checkpoints
            indices = np.linspace(0, len(checkpoint_files)-1, 6, dtype=int)
            selected_files = [checkpoint_files[i] for i in indices]
        else:
            selected_files = checkpoint_files
        
        # Query field at each checkpoint (this would require loading different models)
        # For now, we'll simulate evolution
        evolution_data = self._simulate_field_evolution(query_text, len(selected_files))
        
        # Create evolution visualization
        n_checkpoints = len(selected_files)
        cols = min(3, n_checkpoints)
        rows = (n_checkpoints + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_checkpoints == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Custom colormap
        colors = ['white', 'lightblue', 'blue', 'darkblue', 'red']
        cmap = LinearSegmentedColormap.from_list('semantic', colors, N=100)
        
        for i, (checkpoint_file, confidence_map) in enumerate(zip(selected_files, evolution_data)):
            ax = axes[i]
            
            # Extract episode number from filename
            episode_num = self._extract_episode_number(checkpoint_file.name)
            
            # Create coordinate grids
            x_coords = np.arange(self.field_bounds[0], self.field_bounds[1], self.resolution)
            z_coords = np.arange(self.field_bounds[2], self.field_bounds[3], self.resolution)
            
            # Plot confidence map
            im = ax.imshow(
                confidence_map,
                extent=[x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],
                origin='lower',
                cmap=cmap,
                vmin=0,
                vmax=1,
                alpha=0.8
            )
            
            # Add contours
            contours = ax.contour(
                x_coords, z_coords, confidence_map,
                levels=[0.3, 0.6, 0.9],
                colors='black',
                alpha=0.5,
                linewidths=0.8
            )
            
            # Formatting
            ax.set_title(f'Episode {episode_num}', fontsize=10)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # Add statistics
            max_conf = np.max(confidence_map)
            mean_conf = np.mean(confidence_map)
            ax.text(0.02, 0.98, f'Max: {max_conf:.2f}\nMean: {mean_conf:.2f}', 
                   transform=ax.transAxes, verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Hide unused subplots
        for i in range(n_checkpoints, len(axes)):
            axes[i].set_visible(False)
        
        # Add overall title
        fig.suptitle(f'Semantic Field Evolution\nQuery: "{query_text}"', fontsize=14)
        
        # Add shared colorbar
        cbar = fig.colorbar(im, ax=axes[:n_checkpoints], shrink=0.8, aspect=20)
        cbar.set_label('Semantic Confidence', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Evolution visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _simulate_field_evolution(self, query_text: str, n_checkpoints: int) -> List[np.ndarray]:
        """Simulate field evolution (placeholder for actual checkpoint loading)."""
        # Create coordinate grids
        x_coords = np.arange(self.field_bounds[0], self.field_bounds[1], self.resolution)
        z_coords = np.arange(self.field_bounds[2], self.field_bounds[3], self.resolution)
        
        evolution_data = []
        
        for i in range(n_checkpoints):
            # Simulate progressive learning
            progress = (i + 1) / n_checkpoints
            
            # Create base field with some structure
            x_grid, z_grid = np.meshgrid(x_coords, z_coords)
            
            # Simulate object locations based on query
            if 'table' in query_text.lower():
                # Tables typically in center areas
                center_x, center_z = 0.0, 0.0
                confidence_map = np.exp(-((x_grid - center_x)**2 + (z_grid - center_z)**2) / (4 * progress))
            elif 'kitchen' in query_text.lower():
                # Kitchen typically in one corner
                center_x, center_z = -3.0, -3.0
                confidence_map = np.exp(-((x_grid - center_x)**2 + (z_grid - center_z)**2) / (3 * progress))
            elif 'sofa' in query_text.lower():
                # Sofa typically along walls
                center_x, center_z = 2.0, 2.0
                confidence_map = np.exp(-((x_grid - center_x)**2 + (z_grid - center_z)**2) / (3 * progress))
            else:
                # Generic object
                center_x, center_z = 1.0, -1.0
                confidence_map = np.exp(-((x_grid - center_x)**2 + (z_grid - center_z)**2) / (2 * progress))
            
            # Add noise and normalize
            noise = np.random.normal(0, 0.1 * (1 - progress), confidence_map.shape)
            confidence_map = np.clip(confidence_map + noise, 0, 1)
            
            # Scale by progress (field gets stronger over time)
            confidence_map *= progress
            
            evolution_data.append(confidence_map)
        
        return evolution_data
    
    def _extract_episode_number(self, filename: str) -> int:
        """Extract episode number from checkpoint filename."""
        try:
            # Look for pattern like "ep_123" in filename
            import re
            match = re.search(r'ep_(\d+)', filename)
            if match:
                return int(match.group(1))
            else:
                return 0
        except:
            return 0
    
    def create_query_comparison_grid(self, queries: List[str], output_path: str = None):
        """Create a grid comparison of different semantic queries."""
        print(f"Creating query comparison grid for {len(queries)} queries")
        
        # Standard object queries for comparison
        if not queries:
            queries = [
                "dining table with chairs",
                "comfortable sofa",
                "kitchen counter",
                "bed in bedroom",
                "television screen",
                "refrigerator door",
                "bathroom sink",
                "office desk",
                "bookshelf with books"
            ]
        
        # Query all fields
        query_results = {}
        max_confidence = 0
        
        for query in queries:
            print(f"Processing: '{query}'")
            confidence_map, x_coords, z_coords = self.query_semantic_field_2d(query)
            query_results[query] = confidence_map
            max_confidence = max(max_confidence, np.max(confidence_map))
        
        # Create grid visualization
        n_queries = len(queries)
        cols = 3
        rows = (n_queries + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # Custom colormap
        colors = ['white', 'lightblue', 'blue', 'darkblue', 'red']
        cmap = LinearSegmentedColormap.from_list('semantic', colors, N=100)
        
        for i, query in enumerate(queries):
            row, col = i // cols, i % cols
            ax = axes[row, col]
            
            confidence_map = query_results[query]
            
            # Plot with consistent scale
            im = ax.imshow(
                confidence_map,
                extent=[x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],
                origin='lower',
                cmap=cmap,
                vmin=0,
                vmax=max_confidence,
                alpha=0.8
            )
            
            # Add contours
            if np.max(confidence_map) > 0.1:
                contours = ax.contour(
                    x_coords, z_coords, confidence_map,
                    levels=[0.1, 0.3, 0.5],
                    colors='black',
                    alpha=0.5,
                    linewidths=0.8
                )
            
            # Formatting
            ax.set_title(f'"{query}"', fontsize=10, wrap=True)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # Add confidence stats
            max_conf = np.max(confidence_map)
            mean_conf = np.mean(confidence_map[confidence_map > 0.01])
            ax.text(0.02, 0.98, f'Max: {max_conf:.2f}\nMean: {mean_conf:.2f}', 
                   transform=ax.transAxes, verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Hide unused subplots
        for i in range(n_queries, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].set_visible(False)
        
        # Add overall title
        fig.suptitle('Semantic Field Query Comparison', fontsize=16)
        
        # Add shared colorbar
        cbar = fig.colorbar(im, ax=axes, shrink=0.6, aspect=30)
        cbar.set_label('Semantic Confidence', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Query comparison grid saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main function for semantic field visualization."""
    parser = argparse.ArgumentParser(description="Semantic Field Visualization")
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--query', type=str, help='Semantic query to visualize')
    parser.add_argument('--queries', type=str, help='Comma-separated list of queries')
    parser.add_argument('--scene', type=str, default='FloorPlan1', help='Scene name')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--mode', type=str, choices=['single', 'multiple', 'evolution', 'grid'],
                       default='single', help='Visualization mode')
    parser.add_argument('--checkpoint-dir', type=str, help='Directory with checkpoints for evolution')
    parser.add_argument('--resolution', type=float, default=0.1, help='Spatial resolution')
    parser.add_argument('--bounds', type=str, help='Field bounds as x_min,x_max,z_min,z_max')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = SemanticFieldVisualizer(args.checkpoint)
    
    # Set parameters
    if args.resolution:
        visualizer.resolution = args.resolution
    
    if args.bounds:
        bounds = [float(x) for x in args.bounds.split(',')]
        if len(bounds) == 4:
            visualizer.field_bounds = tuple(bounds)
    
    # Connect to server
    if not visualizer.connect_to_server():
        print("Failed to connect to CLIP-Fields server")
        return 1
    
    try:
        if args.mode == 'single':
            if not args.query:
                args.query = "dining table with chairs"
                print(f"No query specified, using default: '{args.query}'")
            
            visualizer.visualize_single_query(args.query, args.output, args.scene)
            
        elif args.mode == 'multiple':
            if args.queries:
                queries = [q.strip() for q in args.queries.split(',')]
            else:
                queries = [
                    "dining table with chairs",
                    "comfortable sofa",
                    "kitchen counter",
                    "bed in bedroom"
                ]
                print(f"No queries specified, using defaults: {queries}")
            
            visualizer.visualize_multiple_queries(queries, args.output, args.scene)
            
        elif args.mode == 'evolution':
            if not args.query:
                args.query = "dining table with chairs"
            if not args.checkpoint_dir:
                args.checkpoint_dir = "checkpoints/"
            
            visualizer.visualize_field_evolution(args.query, args.checkpoint_dir, args.output)
            
        elif args.mode == 'grid':
            if args.queries:
                queries = [q.strip() for q in args.queries.split(',')]
            else:
                queries = []  # Will use defaults
            
            visualizer.create_query_comparison_grid(queries, args.output)
        
        print("✓ Visualization completed successfully!")
        
    except KeyboardInterrupt:
        print("Visualization interrupted by user")
        return 1
    except Exception as e:
        print(f"Visualization failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

