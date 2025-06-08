"""
FastMCP endpoints for Rhetor.

This module provides FastAPI endpoints for MCP (Model Context Protocol) integration,
allowing external systems to interact with Rhetor LLM management and prompt engineering capabilities.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from tekton.models import TektonBaseModel
import asyncio

from tekton.mcp.fastmcp.server import FastMCPServer
from tekton.mcp.fastmcp.utils.endpoints import add_mcp_endpoints
from tekton.mcp.fastmcp.exceptions import FastMCPError

from rhetor.core.mcp.tools import (
    llm_management_tools,
    prompt_engineering_tools,
    context_management_tools,
    ai_orchestration_tools
)
from rhetor.core.mcp.capabilities import (
    LLMManagementCapability,
    PromptEngineeringCapability,
    ContextManagementCapability,
    AIOrchestrationCapability
)


class MCPRequest(TektonBaseModel):
    """Request model for MCP tool execution."""
    tool_name: str
    arguments: Dict[str, Any]


class MCPResponse(TektonBaseModel):
    """Response model for MCP tool execution."""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Create FastMCP server instance
fastmcp_server = FastMCPServer(
    name="rhetor",
    version="0.1.0",
    description="Rhetor LLM Management and Prompt Engineering MCP Server"
)

# Register capabilities and tools
fastmcp_server.register_capability(LLMManagementCapability())
fastmcp_server.register_capability(PromptEngineeringCapability())
fastmcp_server.register_capability(ContextManagementCapability())
fastmcp_server.register_capability(AIOrchestrationCapability())

# Import the actual tool functions to populate tool lists
from rhetor.core.mcp.tools import (
    get_available_models, set_default_model, get_model_capabilities,
    test_model_connection, get_model_performance, manage_model_rotation,
    create_prompt_template, optimize_prompt, validate_prompt_syntax,
    get_prompt_history, analyze_prompt_performance, manage_prompt_library,
    analyze_context_usage, optimize_context_window, track_context_history,
    compress_context, list_ai_specialists, activate_ai_specialist,
    send_message_to_specialist, orchestrate_team_chat,
    get_specialist_conversation_history, configure_ai_orchestration
)

# Register all tools with their metadata
if hasattr(get_available_models, '_mcp_tool_meta'):
    fastmcp_server.register_tool(get_available_models._mcp_tool_meta)


# Create router for MCP endpoints
mcp_router = APIRouter(prefix="/api/mcp/v2")

# Add standard MCP endpoints using shared utilities
add_mcp_endpoints(mcp_router, fastmcp_server)


# Additional Rhetor-specific MCP endpoints
@mcp_router.get("/llm-status")
async def get_llm_status() -> Dict[str, Any]:
    """
    Get overall LLM system status.
    
    Returns:
        Dictionary containing LLM system status and capabilities
    """
    try:
        # Mock LLM status - real implementation would check actual LLM client
        return {
            "success": True,
            "status": "operational",
            "service": "rhetor-llm-management",
            "capabilities": [
                "llm_management",
                "prompt_engineering", 
                "context_management",
                "ai_orchestration"
            ],
            "available_providers": 4,  # Would query actual providers
            "mcp_tools": len(llm_management_tools + prompt_engineering_tools + context_management_tools + ai_orchestration_tools),
            "llm_engine_status": "ready",
            "message": "Rhetor LLM management system is operational"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get LLM status: {str(e)}")


@mcp_router.post("/execute-llm-workflow")
async def execute_llm_workflow(
    workflow_name: str,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute a predefined LLM management workflow.
    
    Args:
        workflow_name: Name of the workflow to execute
        parameters: Parameters for the workflow
        
    Returns:
        Dictionary containing workflow execution results
    """
    try:
        # Define available workflows
        workflows = {
            "model_optimization": _model_optimization_workflow,
            "prompt_optimization": _prompt_optimization_workflow,
            "context_analysis": _context_analysis_workflow,
            "multi_model_comparison": _multi_model_comparison_workflow
        }
        
        if workflow_name not in workflows:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown workflow: {workflow_name}. Available workflows: {list(workflows.keys())}"
            )
        
        # Execute the workflow
        result = await workflows[workflow_name](parameters)
        
        return {
            "success": True,
            "workflow": workflow_name,
            "result": result,
            "message": f"LLM workflow '{workflow_name}' executed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")


# ============================================================================
# Workflow Implementations
# ============================================================================

async def _model_optimization_workflow(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Model optimization workflow including performance testing and routing."""
    from rhetor.core.mcp.tools import (
        get_available_models, test_model_connection, 
        get_model_performance, set_default_model
    )
    
    # Extract parameters
    task_type = parameters.get("task_type", "general")
    performance_criteria = parameters.get("performance_criteria", ["speed", "quality"])
    
    # Step 1: Get available models
    models_result = await get_available_models()
    
    # Step 2: Test connections for all models
    connection_results = []
    if models_result["success"]:
        for provider_id, provider_info in models_result["providers"].items():
            for model in provider_info.get("models", []):
                test_result = await test_model_connection(
                    provider_id=provider_id,
                    model_id=model["id"]
                )
                connection_results.append({
                    "provider": provider_id,
                    "model": model["id"],
                    "connected": test_result["success"],
                    "response_time": test_result.get("response_time", 0)
                })
    
    # Step 3: Get performance metrics for connected models
    performance_results = []
    for conn_result in connection_results:
        if conn_result["connected"]:
            perf_result = await get_model_performance(
                provider_id=conn_result["provider"],
                model_id=conn_result["model"],
                task_type=task_type
            )
            performance_results.append({
                "provider": conn_result["provider"],
                "model": conn_result["model"],
                "performance": perf_result.get("metrics", {}),
                "recommended": perf_result.get("recommended", False)
            })
    
    # Step 4: Select optimal model based on criteria
    optimal_model = None
    best_score = 0
    
    for perf_result in performance_results:
        score = 0
        metrics = perf_result["performance"]
        
        if "speed" in performance_criteria:
            score += metrics.get("speed_score", 0) * 0.4
        if "quality" in performance_criteria:
            score += metrics.get("quality_score", 0) * 0.4
        if "cost" in performance_criteria:
            score += metrics.get("cost_score", 0) * 0.2
        
        if score > best_score:
            best_score = score
            optimal_model = perf_result
    
    # Step 5: Set optimal model as default if found
    optimization_applied = False
    if optimal_model:
        set_result = await set_default_model(
            provider_id=optimal_model["provider"],
            model_id=optimal_model["model"]
        )
        optimization_applied = set_result["success"]
    
    return {
        "models_tested": len(connection_results),
        "models_connected": len([r for r in connection_results if r["connected"]]),
        "performance_analyzed": len(performance_results),
        "optimal_model": optimal_model,
        "optimization_applied": optimization_applied,
        "workflow_summary": {
            "task_type": task_type,
            "criteria": performance_criteria,
            "best_score": best_score,
            "optimization_confidence": "high" if optimal_model else "low"
        }
    }


async def _prompt_optimization_workflow(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Prompt optimization workflow including template creation and validation."""
    from rhetor.core.mcp.tools import (
        create_prompt_template, optimize_prompt, 
        validate_prompt_syntax, analyze_prompt_performance
    )
    
    # Extract parameters
    base_prompt = parameters.get("base_prompt", "")
    task_context = parameters.get("task_context", {})
    optimization_goals = parameters.get("optimization_goals", ["clarity", "effectiveness"])
    
    # Step 1: Create initial template
    template_result = await create_prompt_template(
        name=f"optimized_prompt_{task_context.get('task_type', 'general')}",
        template=base_prompt,
        variables=list(task_context.keys()),
        description="Auto-generated optimized prompt template"
    )
    
    # Step 2: Optimize the prompt
    optimization_result = None
    if template_result["success"]:
        optimization_result = await optimize_prompt(
            template_id=template_result["template"]["template_id"],
            optimization_goals=optimization_goals,
            context=task_context
        )
    
    # Step 3: Validate syntax of optimized prompt
    validation_result = None
    if optimization_result and optimization_result["success"]:
        validation_result = await validate_prompt_syntax(
            prompt_text=optimization_result["optimized_prompt"],
            template_variables=list(task_context.keys())
        )
    
    # Step 4: Analyze performance of optimized prompt
    performance_result = None
    if validation_result and validation_result["success"]:
        performance_result = await analyze_prompt_performance(
            prompt_text=optimization_result["optimized_prompt"],
            test_contexts=[task_context],
            metrics_to_analyze=["clarity", "specificity", "effectiveness"]
        )
    
    return {
        "template_creation": template_result,
        "prompt_optimization": optimization_result,
        "syntax_validation": validation_result,
        "performance_analysis": performance_result,
        "workflow_summary": {
            "optimization_goals": optimization_goals,
            "improvements_made": len(optimization_result.get("improvements", [])) if optimization_result else 0,
            "validation_passed": validation_result["success"] if validation_result else False,
            "optimization_confidence": "high" if all([template_result["success"], optimization_result and optimization_result["success"]]) else "medium"
        }
    }


async def _context_analysis_workflow(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Context analysis workflow including usage tracking and optimization."""
    from rhetor.core.mcp.tools import (
        analyze_context_usage, optimize_context_window,
        track_context_history, compress_context
    )
    
    # Extract parameters
    context_id = parameters.get("context_id", "default")
    analysis_period = parameters.get("analysis_period", "last_week")
    optimization_target = parameters.get("optimization_target", "efficiency")
    
    # Step 1: Analyze context usage patterns
    usage_result = await analyze_context_usage(
        context_id=context_id,
        time_period=analysis_period,
        include_metrics=True
    )
    
    # Step 2: Track context history for patterns
    history_result = await track_context_history(
        context_id=context_id,
        analysis_depth="detailed",
        include_token_counts=True
    )
    
    # Step 3: Optimize context window if needed
    optimization_result = None
    if usage_result["success"] and usage_result["analysis"]["optimization_needed"]:
        optimization_result = await optimize_context_window(
            context_id=context_id,
            optimization_strategy=optimization_target,
            preserve_recent_messages=True
        )
    
    # Step 4: Compress context if size is an issue
    compression_result = None
    if history_result["success"] and history_result["metrics"]["total_tokens"] > 8000:
        compression_result = await compress_context(
            context_id=context_id,
            compression_ratio=0.7,
            preserve_key_information=True
        )
    
    return {
        "usage_analysis": usage_result,
        "history_tracking": history_result,
        "context_optimization": optimization_result,
        "context_compression": compression_result,
        "workflow_summary": {
            "analysis_period": analysis_period,
            "optimization_applied": optimization_result["success"] if optimization_result else False,
            "compression_applied": compression_result["success"] if compression_result else False,
            "space_saved": compression_result.get("space_saved_percent", 0) if compression_result else 0,
            "analysis_confidence": "high" if usage_result["success"] else "medium"
        }
    }


async def _multi_model_comparison_workflow(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Multi-model comparison workflow for evaluating different models on the same task."""
    from rhetor.core.mcp.tools import (
        get_available_models, get_model_capabilities,
        test_model_connection, get_model_performance
    )
    
    # Extract parameters
    task_description = parameters.get("task_description", "")
    test_prompts = parameters.get("test_prompts", [])
    comparison_metrics = parameters.get("comparison_metrics", ["speed", "quality", "cost"])
    
    # Step 1: Get all available models
    models_result = await get_available_models()
    
    # Step 2: Get capabilities for each model
    capability_results = []
    if models_result["success"]:
        for provider_id, provider_info in models_result["providers"].items():
            for model in provider_info.get("models", []):
                cap_result = await get_model_capabilities(
                    provider_id=provider_id,
                    model_id=model["id"]
                )
                capability_results.append({
                    "provider": provider_id,
                    "model": model["id"],
                    "capabilities": cap_result.get("capabilities", {}),
                    "suitable": cap_result.get("suitable_for_task", False)
                })
    
    # Step 3: Test performance for suitable models
    performance_comparisons = []
    suitable_models = [r for r in capability_results if r["suitable"]]
    
    for model_info in suitable_models:
        perf_result = await get_model_performance(
            provider_id=model_info["provider"],
            model_id=model_info["model"],
            task_type="comparison",
            test_prompts=test_prompts
        )
        performance_comparisons.append({
            "provider": model_info["provider"],
            "model": model_info["model"],
            "performance": perf_result.get("metrics", {}),
            "test_results": perf_result.get("test_results", [])
        })
    
    # Step 4: Rank models by specified metrics
    model_rankings = []
    for comparison in performance_comparisons:
        total_score = 0
        metrics = comparison["performance"]
        
        for metric in comparison_metrics:
            score = metrics.get(f"{metric}_score", 0)
            weight = 1.0 / len(comparison_metrics)  # Equal weighting
            total_score += score * weight
        
        model_rankings.append({
            "provider": comparison["provider"],
            "model": comparison["model"],
            "total_score": total_score,
            "individual_scores": {metric: metrics.get(f"{metric}_score", 0) for metric in comparison_metrics}
        })
    
    # Sort by total score
    model_rankings.sort(key=lambda x: x["total_score"], reverse=True)
    
    return {
        "models_evaluated": len(capability_results),
        "suitable_models": len(suitable_models),
        "performance_comparisons": performance_comparisons,
        "model_rankings": model_rankings,
        "recommended_model": model_rankings[0] if model_rankings else None,
        "workflow_summary": {
            "task_description": task_description,
            "comparison_metrics": comparison_metrics,
            "test_prompts_count": len(test_prompts),
            "comparison_confidence": "high" if len(model_rankings) >= 2 else "medium"
        }
    }


# Export the router
__all__ = ["mcp_router", "fastmcp_server"]