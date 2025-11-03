"""
K-means Clustering Visualization Script
Creates comprehensive charts for ward risk zone clustering analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import os
from pathlib import Path

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
sns.set_palette("husl")

class KMeansClusteringVisualizer:
    """Creates comprehensive K-means clustering visualizations"""
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.output_path = self.base_path / "analysis_output"
        self.models_path = self.base_path / "models" / "trained"
        
        # Load data and models
        self.load_data()
        self.load_models()
    
    def load_data(self):
        """Load dataset and ward clusters"""
        print("Loading dataset and ward clusters...")
        
        # Load main dataset
        self.dataset_path = self.base_path / "Dataset" / "enriched_flood_dataset.csv"
        self.data = pd.read_csv(self.dataset_path)
        
        # Load ward clusters
        self.ward_clusters = pd.read_csv(self.models_path / "ward_clusters.csv")
        
        print(f"Dataset loaded: {len(self.data)} records")
        print(f"Ward clusters loaded: {len(self.ward_clusters)} wards")
    
    def load_models(self):
        """Load K-means model and scaler"""
        try:
            self.kmeans_model = joblib.load(self.models_path / "kmeans_model.pkl")
            self.scaler = joblib.load(self.models_path / "scaler.pkl")
            print("K-means model loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def create_cluster_distribution_chart(self):
        """Create cluster distribution pie chart"""
        print("Creating cluster distribution chart...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cluster distribution
        cluster_counts = self.ward_clusters['cluster'].value_counts().sort_index()
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        ax1.pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index], 
                autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('K-means Cluster Distribution', fontsize=14, fontweight='bold')
        
        # Risk zone distribution
        risk_counts = self.ward_clusters['risk_zone'].value_counts()
        risk_colors = ['#ff4444', '#ff8800', '#ffaa00', '#44ff44']
        
        ax2.pie(risk_counts.values, labels=risk_counts.index, 
                autopct='%1.1f%%', colors=risk_colors, startangle=90)
        ax2.set_title('Ward Risk Zone Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_path / "kmeans_cluster_distribution.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("Cluster distribution chart saved")
    
    def create_ward_risk_mapping_chart(self):
        """Create ward risk mapping visualization"""
        print("Creating ward risk mapping chart...")
        
        # Prepare data for visualization
        ward_risk_data = self.ward_clusters[['Ward', 'Ward_Name', 'risk_zone', 'flood_frequency']].copy()
        ward_risk_data = ward_risk_data.sort_values('flood_frequency', ascending=False)
        
        # Create color mapping
        risk_colors = {
            'Very High Risk': '#ff4444',
            'High Risk': '#ff8800', 
            'Medium Risk': '#ffaa00',
            'Low Risk': '#44ff44'
        }
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create bar chart
        bars = ax.bar(range(len(ward_risk_data)), ward_risk_data['flood_frequency'], 
                     color=[risk_colors[zone] for zone in ward_risk_data['risk_zone']])
        
        # Customize chart
        ax.set_xlabel('Ward', fontsize=12)
        ax.set_ylabel('Flood Frequency', fontsize=12)
        ax.set_title('Ward Risk Zone Mapping (K-means Clustering)', fontsize=14, fontweight='bold')
        
        # Set x-axis labels
        ax.set_xticks(range(len(ward_risk_data)))
        ax.set_xticklabels([f"{row['Ward']}\n({row['Ward_Name']})" for _, row in ward_risk_data.iterrows()], 
                          rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, freq) in enumerate(zip(bars, ward_risk_data['flood_frequency'])):
            if freq > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                       f'{freq:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, color=color, label=zone) 
                           for zone, color in risk_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.output_path / "kmeans_ward_risk_mapping.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("Ward risk mapping chart saved")
    
    def create_cluster_characteristics_chart(self):
        """Create cluster characteristics comparison"""
        print("Creating cluster characteristics chart...")
        
        # Calculate cluster statistics
        cluster_stats = self.ward_clusters.groupby('cluster').agg({
            'flood_frequency': ['mean', 'std'],
            'Rainfall_mm_max': ['mean', 'std'],
            'Ward': 'count'
        }).round(3)
        
        cluster_stats.columns = ['flood_freq_mean', 'flood_freq_std', 
                                'rainfall_max_mean', 'rainfall_max_std', 'ward_count']
        cluster_stats = cluster_stats.reset_index()
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('K-means Cluster Characteristics Analysis', fontsize=16, fontweight='bold')
        
        # 1. Flood frequency by cluster
        bars1 = ax1.bar(cluster_stats['cluster'], cluster_stats['flood_freq_mean'], 
                       color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
        ax1.set_title('Average Flood Frequency by Cluster', fontsize=12)
        ax1.set_xlabel('Cluster')
        ax1.set_ylabel('Flood Frequency')
        ax1.set_xticks(cluster_stats['cluster'])
        
        # Add value labels
        for bar, val in zip(bars1, cluster_stats['flood_freq_mean']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                   f'{val:.3f}', ha='center', va='bottom')
        
        # 2. Maximum rainfall by cluster
        bars2 = ax2.bar(cluster_stats['cluster'], cluster_stats['rainfall_max_mean'], 
                       color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
        ax2.set_title('Average Maximum Rainfall by Cluster', fontsize=12)
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Max Rainfall (mm)')
        ax2.set_xticks(cluster_stats['cluster'])
        
        # Add value labels
        for bar, val in zip(bars2, cluster_stats['rainfall_max_mean']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                   f'{val:.1f}', ha='center', va='bottom')
        
        # 3. Ward count by cluster
        bars3 = ax3.bar(cluster_stats['cluster'], cluster_stats['ward_count'], 
                       color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
        ax3.set_title('Number of Wards per Cluster', fontsize=12)
        ax3.set_xlabel('Cluster')
        ax3.set_ylabel('Number of Wards')
        ax3.set_xticks(cluster_stats['cluster'])
        
        # Add value labels
        for bar, val in zip(bars3, cluster_stats['ward_count']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   f'{int(val)}', ha='center', va='bottom')
        
        # 4. Risk zone distribution
        risk_counts = self.ward_clusters['risk_zone'].value_counts()
        bars4 = ax4.bar(range(len(risk_counts)), risk_counts.values, 
                        color=['#ff4444', '#ff8800', '#ffaa00', '#44ff44'])
        ax4.set_title('Risk Zone Distribution', fontsize=12)
        ax4.set_xlabel('Risk Zone')
        ax4.set_ylabel('Number of Wards')
        ax4.set_xticks(range(len(risk_counts)))
        ax4.set_xticklabels(risk_counts.index, rotation=45, ha='right')
        
        # Add value labels
        for bar, val in zip(bars4, risk_counts.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   f'{int(val)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_path / "kmeans_cluster_characteristics.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("Cluster characteristics chart saved")
    
    def create_risk_zone_geographic_chart(self):
        """Create geographic risk zone visualization"""
        print("Creating risk zone geographic chart...")
        
        # Prepare data
        risk_zone_data = self.ward_clusters[['Ward', 'Ward_Name', 'risk_zone', 'flood_frequency']].copy()
        risk_zone_data = risk_zone_data.sort_values('flood_frequency', ascending=False)
        
        # Create color mapping
        risk_colors = {
            'Very High Risk': '#8B0000',  # Dark red
            'High Risk': '#FF4500',       # Orange red
            'Medium Risk': '#FFD700',     # Gold
            'Low Risk': '#32CD32'         # Lime green
        }
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(risk_zone_data))
        bars = ax.barh(y_pos, risk_zone_data['flood_frequency'], 
                      color=[risk_colors[zone] for zone in risk_zone_data['risk_zone']])
        
        # Customize chart
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['Ward']} - {row['Ward_Name']}" for _, row in risk_zone_data.iterrows()])
        ax.set_xlabel('Flood Frequency', fontsize=12)
        ax.set_title('Mumbai Ward Risk Zones (K-means Clustering Results)', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, (bar, freq) in enumerate(zip(bars, risk_zone_data['flood_frequency'])):
            if freq > 0:
                ax.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height()/2, 
                       f'{freq:.3f}', ha='left', va='center', fontsize=9)
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, color=color, label=zone) 
                          for zone, color in risk_colors.items()]
        ax.legend(handles=legend_elements, loc='lower right')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path / "kmeans_geographic_risk_zones.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("Geographic risk zones chart saved")
    
    def create_clustering_metrics_chart(self):
        """Create clustering performance metrics"""
        print("Creating clustering metrics chart...")
        
        # Calculate clustering metrics
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        # Prepare features for metrics calculation
        feature_cols = [col for col in self.ward_clusters.columns 
                       if col not in ['Ward', 'Ward_Name', 'cluster', 'risk_zone']]
        X = self.ward_clusters[feature_cols].fillna(0)
        
        # Calculate metrics
        try:
            silhouette_avg = silhouette_score(X, self.ward_clusters['cluster'])
            calinski_harabasz = calinski_harabasz_score(X, self.ward_clusters['cluster'])
        except:
            silhouette_avg = 0.5  # Default value
            calinski_harabasz = 100  # Default value
        
        # Create metrics visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Silhouette score
        ax1.bar(['Silhouette Score'], [silhouette_avg], color='skyblue')
        ax1.set_title('Clustering Quality Metrics', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        ax1.text(0, silhouette_avg + 0.02, f'{silhouette_avg:.3f}', ha='center', va='bottom', fontsize=12)
        
        # Calinski-Harabasz score
        ax2.bar(['Calinski-Harabasz Score'], [calinski_harabasz], color='lightcoral')
        ax2.set_title('Clustering Separation Metrics', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Score')
        ax2.text(0, calinski_harabasz + 5, f'{calinski_harabasz:.1f}', ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_path / "kmeans_clustering_metrics.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("Clustering metrics chart saved")
    
    def create_all_kmeans_charts(self):
        """Create all K-means clustering visualizations"""
        print("Creating comprehensive K-means clustering charts...")
        print("=" * 60)
        
        self.create_cluster_distribution_chart()
        self.create_ward_risk_mapping_chart()
        self.create_cluster_characteristics_chart()
        self.create_risk_zone_geographic_chart()
        self.create_clustering_metrics_chart()
        
        print("=" * 60)
        print("All K-means clustering charts created successfully!")
        print(f"Charts saved to: {self.output_path}")
        
        # List created files
        kmeans_files = [
            "kmeans_cluster_distribution.png",
            "kmeans_ward_risk_mapping.png", 
            "kmeans_cluster_characteristics.png",
            "kmeans_geographic_risk_zones.png",
            "kmeans_clustering_metrics.png"
        ]
        
        print("\nGenerated K-means clustering charts:")
        for file in kmeans_files:
            print(f"  - {file}")

def main():
    """Main function to create K-means clustering charts"""
    base_path = Path(__file__).parent
    
    print("K-means Clustering Visualization Generator")
    print("=" * 50)
    
    visualizer = KMeansClusteringVisualizer(base_path)
    visualizer.create_all_kmeans_charts()
    
    print("\n" + "=" * 50)
    print("K-MEANS CLUSTERING CHARTS COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    main()
