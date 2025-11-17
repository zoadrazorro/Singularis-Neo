"""
DATA System Usage Examples
===========================

Demonstrates how to use the DATA (Distributed Abductive Technical Agent) system.
"""

import asyncio
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from singularis.data import DATASystem, DATAConfig
from singularis.data.singularis_integration import (
    DATAConsciousnessIntegration,
    LifeOpsIntegration,
    SkyrimAGIIntegration
)
from loguru import logger


async def example_1_basic_usage():
    """Example 1: Basic DATA system usage"""
    print("\n" + "="*60)
    print("Example 1: Basic DATA System Usage")
    print("="*60 + "\n")
    
    # Initialize DATA system
    config_path = "config/data_config.yaml"
    data = DATASystem(config_path=config_path)
    
    await data.initialize()
    
    # Process a simple query
    result = await data.process_query(
        query="What are the key principles of quantum computing?",
        context={"domain": "technical", "depth": "intermediate"},
        priority=0.7
    )
    
    print(f"✓ Success: {result['success']}")
    print(f"✓ Experts used: {', '.join(result['expert_sources'])}")
    print(f"✓ Latency: {result['latency_ms']:.1f}ms")
    print(f"\nResponse:\n{result['content']}\n")
    
    # Get metrics
    metrics = data.get_metrics()
    print(f"System metrics:")
    print(f"  - Queries processed: {metrics['queries_processed']}")
    print(f"  - Average latency: {metrics['avg_latency_ms']:.1f}ms")
    print(f"  - Active nodes: {metrics['active_nodes']}")
    
    await data.shutdown()


async def example_2_expert_routing():
    """Example 2: Routing to specific experts"""
    print("\n" + "="*60)
    print("Example 2: Expert-Specific Routing")
    print("="*60 + "\n")
    
    data = DATASystem(config_path="config/data_config.yaml")
    await data.initialize()
    
    # Query for reasoning expert
    print("Routing to reasoning expert...")
    reasoning_result = await data.expert_router.execute_expert(
        expert_name="reasoning_expert",
        query="Solve: If all A are B, and all B are C, what can we conclude about A and C?",
        context={"domain": "logic"}
    )
    print(f"Reasoning Expert: {reasoning_result['response']}\n")
    
    # Query for creativity expert
    print("Routing to creativity expert...")
    creativity_result = await data.expert_router.execute_expert(
        expert_name="creativity_expert",
        query="Generate a creative metaphor for distributed computing",
        context={"domain": "creative"}
    )
    print(f"Creativity Expert: {creativity_result['response']}\n")
    
    # Get routing stats
    stats = data.expert_router.get_routing_stats()
    print(f"Routing statistics:")
    print(f"  - Total routings: {stats['total_routings']}")
    print(f"  - Expert load: {stats['expert_load']}")
    
    await data.shutdown()


async def example_3_global_workspace():
    """Example 3: Global Workspace interaction"""
    print("\n" + "="*60)
    print("Example 3: Global Workspace Theory")
    print("="*60 + "\n")
    
    data = DATASystem(config_path="config/data_config.yaml")
    await data.initialize()
    
    # Submit items to workspace
    print("Submitting items to Global Workspace...\n")
    
    items = [
        ("perception", "Visual input: red traffic light", 0.9),
        ("memory", "Recall: red means stop", 0.8),
        ("reasoning", "Inference: should stop vehicle", 0.85),
        ("action", "Plan: apply brakes", 0.9),
    ]
    
    for source, content, priority in items:
        success = data.global_workspace.submit_to_workspace(
            content=content,
            source=source,
            priority=priority,
            metadata={"example": "traffic_light"}
        )
        print(f"  {'✓' if success else '✗'} Submitted from {source} (priority={priority})")
    
    # Wait for attention competition
    await asyncio.sleep(0.5)
    
    # Get workspace contents
    workspace_contents = data.global_workspace.get_workspace_contents()
    print(f"\nWorkspace contents ({len(workspace_contents)} items):")
    for item in workspace_contents:
        print(f"  - {item.source}: {item.content} (priority={item.priority:.2f})")
    
    # Get metrics
    ws_metrics = data.global_workspace.get_metrics()
    print(f"\nWorkspace metrics:")
    print(f"  - Acceptance rate: {ws_metrics['acceptance_rate']:.1%}")
    print(f"  - Total submissions: {ws_metrics['total_submissions']}")
    print(f"  - Broadcasts: {ws_metrics['broadcasts']}")
    
    await data.shutdown()


async def example_4_consciousness_integration():
    """Example 4: Integration with Singularis Consciousness"""
    print("\n" + "="*60)
    print("Example 4: Consciousness Layer Integration")
    print("="*60 + "\n")
    
    try:
        from singularis.consciousness.unified_consciousness_layer import UnifiedConsciousnessLayer
        
        # Initialize both systems
        data = DATASystem(config_path="config/data_config.yaml")
        consciousness = UnifiedConsciousnessLayer()
        
        # Create integration
        integration = DATAConsciousnessIntegration(
            data_system=data,
            consciousness=consciousness,
            enable_fallback=True
        )
        
        await integration.initialize()
        
        # Process query using hybrid approach
        print("Processing query with hybrid approach...\n")
        
        result = await integration.process_query_hybrid(
            query="Analyze the relationship between sleep quality and productivity",
            context={"domain": "life_ops", "use_distributed": True}
        )
        
        print(f"✓ Routing: {result['routing']}")
        print(f"✓ Experts: {', '.join(result['expert_sources'])}")
        print(f"\nResponse:\n{result['content']}\n")
        
        # Get integration stats
        stats = integration.get_integration_stats()
        print(f"Integration stats:")
        print(f"  - DATA available: {stats['data_available']}")
        print(f"  - Fallback count: {stats['fallback_count']}")
        
        await integration.shutdown()
        
    except ImportError:
        print("⚠ Consciousness layer not available, skipping integration example")


async def example_5_node_management():
    """Example 5: Node management and monitoring"""
    print("\n" + "="*60)
    print("Example 5: Node Management")
    print("="*60 + "\n")
    
    data = DATASystem(config_path="config/data_config.yaml")
    await data.initialize()
    
    # Get cluster status
    cluster_status = data.node_manager.get_cluster_status()
    
    print("Cluster Status:")
    print(f"  - Total nodes: {cluster_status['total_nodes']}")
    print(f"  - Active nodes: {cluster_status['active_nodes']}")
    print(f"  - Utilization: {cluster_status['utilization']:.1%}\n")
    
    print("Node Details:")
    for node_id, node_status in cluster_status['nodes'].items():
        print(f"\n  {node_id}:")
        print(f"    - Status: {node_status['status']}")
        print(f"    - Load: {node_status['current_load']:.1%}")
        print(f"    - Healthy: {node_status['is_healthy']}")
    
    # Get node info
    print("\nDetailed Node Info (node_a):")
    node_info = data.node_manager.get_node_info("node_a")
    if node_info:
        config = node_info['config']
        print(f"  - Role: {config['role']}")
        print(f"  - GPU Count: {config['gpu_count']}")
        print(f"  - VRAM: {config['vram_gb']}GB")
        print(f"  - Capabilities: {', '.join(config['capabilities'])}")
    
    await data.shutdown()


async def main():
    """Run all examples"""
    logger.remove()  # Remove default logger
    logger.add(sys.stdout, level="INFO", format="<level>{message}</level>")
    
    try:
        await example_1_basic_usage()
        await example_2_expert_routing()
        await example_3_global_workspace()
        await example_4_consciousness_integration()
        await example_5_node_management()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

