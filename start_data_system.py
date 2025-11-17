#!/usr/bin/env python3
"""
DATA System Quick Start
========================

Simple script to start the DATA (Distributed Abductive Technical Agent) system.
"""

import asyncio
import sys
from pathlib import Path
from loguru import logger

# Add singularis to path
sys.path.insert(0, str(Path(__file__).parent))

from singularis.data import DATASystem


async def main():
    """Main entry point"""
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║          DATA - Distributed Abductive Technical Agent        ║
    ║                                                              ║
    ║              Proto-AGI System for Singularis                 ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    logger.info("Initializing DATA system...")
    
    try:
        # Initialize DATA system
        data = DATASystem(config_path="config/data_config.yaml")
        init_success = await data.initialize()
        
        if not init_success:
            logger.error("Failed to initialize DATA system")
            return 1
        
        logger.success("DATA system initialized successfully!\n")
        
        # Print system status
        metrics = data.get_metrics()
        print("System Status:")
        print(f"  ✓ Active nodes: {metrics['active_nodes']}")
        print(f"  ✓ Available experts: {metrics['available_experts']}")
        print(f"  ✓ Workspace capacity: 7 items")
        print()
        
        # Example queries
        logger.info("Running example queries...\n")
        
        queries = [
            {
                "query": "Explain the concept of emergence in complex systems",
                "context": {"domain": "philosophy", "depth": "intermediate"}
            },
            {
                "query": "What is the most efficient way to sort a large dataset?",
                "context": {"domain": "technical", "depth": "expert"}
            },
            {
                "query": "How do I improve my sleep quality based on patterns?",
                "context": {"domain": "life_ops", "depth": "practical"}
            }
        ]
        
        for i, query_data in enumerate(queries, 1):
            print(f"\n{'─'*60}")
            print(f"Query {i}: {query_data['query']}")
            print(f"{'─'*60}\n")
            
            result = await data.process_query(
                query=query_data["query"],
                context=query_data["context"],
                priority=0.7
            )
            
            if result['success']:
                print(f"✓ Experts used: {', '.join(result['expert_sources'])}")
                print(f"✓ Latency: {result['latency_ms']:.1f}ms")
                print(f"\nResponse:")
                print(f"{result['content']}\n")
            else:
                print(f"✗ Query failed: {result.get('error', 'Unknown error')}\n")
        
        # Final metrics
        print(f"\n{'═'*60}")
        print("Final System Metrics:")
        print(f"{'═'*60}\n")
        
        final_metrics = data.get_metrics()
        print(f"  - Queries processed: {final_metrics['queries_processed']}")
        print(f"  - Expert calls: {final_metrics['expert_calls']}")
        print(f"  - Workspace broadcasts: {final_metrics['workspace_broadcasts']}")
        print(f"  - Average latency: {final_metrics['avg_latency_ms']:.1f}ms")
        
        # Workspace metrics
        ws_metrics = data.global_workspace.get_metrics()
        print(f"\n  Workspace:")
        print(f"  - Acceptance rate: {ws_metrics['acceptance_rate']:.1%}")
        print(f"  - Total submissions: {ws_metrics['total_submissions']}")
        print(f"  - Current size: {ws_metrics['current_workspace_size']}")
        
        # Node status
        cluster_status = data.node_manager.get_cluster_status()
        print(f"\n  Cluster:")
        print(f"  - Active nodes: {cluster_status['active_nodes']}/{cluster_status['total_nodes']}")
        print(f"  - Utilization: {cluster_status['utilization']:.1%}")
        
        print(f"\n{'═'*60}\n")
        
        # Shutdown
        logger.info("Shutting down DATA system...")
        await data.shutdown()
        logger.success("DATA system shutdown complete")
        
        print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║              DATA System Demo Complete                       ║
    ║                                                              ║
    ║     For more examples, see: examples/data_example.py         ║
    ║     For documentation, see: DATA_README.md                   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
        """)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\nShutdown requested by user")
        if 'data' in locals():
            await data.shutdown()
        return 130
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

