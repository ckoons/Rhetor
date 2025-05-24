"""
MCP tools for Rhetor.

This module implements the actual MCP tools that provide Rhetor's LLM management,
prompt engineering, and context management functionality.
"""

import json
import time
from typing import Dict, Any, List, Optional
from tekton.mcp.fastmcp.decorators import mcp_tool


# ============================================================================
# LLM Management Tools
# ============================================================================

@mcp_tool
async def get_available_models() -> Dict[str, Any]:
    """
    Get all available LLM models and providers.
    
    Returns:
        Dictionary containing available models organized by provider
    """
    try:
        # Mock implementation - replace with actual LLM client integration
        providers = {
            "ollama": {
                "available": True,
                "models": [
                    {"id": "llama2", "name": "Llama 2", "size": "7B", "capabilities": ["chat", "completion"]},
                    {"id": "codellama", "name": "Code Llama", "size": "13B", "capabilities": ["code", "completion"]},
                    {"id": "mistral", "name": "Mistral", "size": "7B", "capabilities": ["chat", "completion"]}
                ]
            },
            "anthropic": {
                "available": True,
                "models": [
                    {"id": "claude-3-haiku", "name": "Claude 3 Haiku", "size": "medium", "capabilities": ["chat", "analysis"]},
                    {"id": "claude-3-sonnet", "name": "Claude 3 Sonnet", "size": "large", "capabilities": ["chat", "reasoning"]}
                ]
            },
            "openai": {
                "available": True,
                "models": [
                    {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "size": "medium", "capabilities": ["chat", "completion"]},
                    {"id": "gpt-4", "name": "GPT-4", "size": "large", "capabilities": ["chat", "reasoning", "analysis"]}
                ]
            }
        }
        
        return {
            "success": True,
            "providers": providers,
            "total_providers": len(providers),
            "total_models": sum(len(p["models"]) for p in providers.values()),
            "message": "Successfully retrieved available models"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get available models: {str(e)}",
            "providers": {}
        }


@mcp_tool
async def set_default_model(provider_id: str, model_id: str) -> Dict[str, Any]:
    """
    Set the default model for LLM operations.
    
    Args:
        provider_id: ID of the provider
        model_id: ID of the model
        
    Returns:
        Dictionary containing operation result
    """
    try:
        # Mock implementation - replace with actual configuration update
        valid_providers = ["ollama", "anthropic", "openai"]
        if provider_id not in valid_providers:
            return {
                "success": False,
                "error": f"Invalid provider: {provider_id}. Valid providers: {valid_providers}"
            }
        
        # Simulate setting the default model
        return {
            "success": True,
            "previous_default": {"provider": "ollama", "model": "llama2"},
            "new_default": {"provider": provider_id, "model": model_id},
            "message": f"Default model set to {provider_id}/{model_id}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to set default model: {str(e)}"
        }


@mcp_tool
async def get_model_capabilities(provider_id: str, model_id: str) -> Dict[str, Any]:
    """
    Get capabilities and specifications for a specific model.
    
    Args:
        provider_id: ID of the provider
        model_id: ID of the model
        
    Returns:
        Dictionary containing model capabilities
    """
    try:
        # Mock implementation - replace with actual capability detection
        model_capabilities = {
            "ollama/llama2": {
                "max_tokens": 4096,
                "supports_streaming": True,
                "supports_functions": False,
                "languages": ["en", "es", "fr"],
                "specialties": ["general", "conversation"],
                "cost_per_token": 0.0
            },
            "anthropic/claude-3-sonnet": {
                "max_tokens": 200000,
                "supports_streaming": True,
                "supports_functions": True,
                "languages": ["en", "es", "fr", "de", "it"],
                "specialties": ["reasoning", "analysis", "code", "creative"],
                "cost_per_token": 0.00003
            },
            "openai/gpt-4": {
                "max_tokens": 8192,
                "supports_streaming": True,
                "supports_functions": True,
                "languages": ["en", "es", "fr", "de", "it", "pt"],
                "specialties": ["reasoning", "analysis", "code", "creative"],
                "cost_per_token": 0.00006
            }
        }
        
        model_key = f"{provider_id}/{model_id}"
        capabilities = model_capabilities.get(model_key, {
            "max_tokens": 2048,
            "supports_streaming": False,
            "supports_functions": False,
            "languages": ["en"],
            "specialties": ["general"],
            "cost_per_token": 0.0
        })
        
        return {
            "success": True,
            "provider": provider_id,
            "model": model_id,
            "capabilities": capabilities,
            "suitable_for_task": True,  # Would be determined based on task requirements
            "message": f"Retrieved capabilities for {provider_id}/{model_id}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get model capabilities: {str(e)}"
        }


@mcp_tool
async def test_model_connection(provider_id: str, model_id: str) -> Dict[str, Any]:
    """
    Test connection to a specific model.
    
    Args:
        provider_id: ID of the provider
        model_id: ID of the model
        
    Returns:
        Dictionary containing connection test results
    """
    try:
        # Mock implementation - replace with actual connection test
        start_time = time.time()
        
        # Simulate connection test with some randomness for realism
        import random
        success = random.choice([True, True, True, False])  # 75% success rate
        response_time = random.uniform(0.1, 2.0)
        
        end_time = time.time()
        
        if success:
            return {
                "success": True,
                "provider": provider_id,
                "model": model_id,
                "response_time": response_time,
                "connection_quality": "good" if response_time < 1.0 else "slow",
                "test_prompt_response": "Hello! I'm ready to assist.",
                "message": f"Successfully connected to {provider_id}/{model_id}"
            }
        else:
            return {
                "success": False,
                "provider": provider_id,
                "model": model_id,
                "error": "Connection timeout or model unavailable",
                "message": f"Failed to connect to {provider_id}/{model_id}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"Connection test failed: {str(e)}"
        }


@mcp_tool
async def get_model_performance(
    provider_id: str, 
    model_id: str, 
    task_type: str = "general",
    test_prompts: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Get performance metrics for a specific model.
    
    Args:
        provider_id: ID of the provider
        model_id: ID of the model
        task_type: Type of task to evaluate performance for
        test_prompts: Optional list of prompts to test with
        
    Returns:
        Dictionary containing performance metrics
    """
    try:
        # Mock implementation - replace with actual performance testing
        import random
        
        # Generate realistic performance metrics
        speed_score = random.uniform(60, 95)
        quality_score = random.uniform(70, 98)
        cost_score = random.uniform(50, 90)
        consistency_score = random.uniform(75, 95)
        
        # Adjust scores based on provider/model characteristics
        if provider_id == "ollama":
            cost_score += 20  # Local models are cheaper
            speed_score = min(speed_score + 10, 100)  # Often faster locally
        elif provider_id == "anthropic":
            quality_score = min(quality_score + 15, 100)  # High quality
        elif provider_id == "openai":
            consistency_score = min(consistency_score + 10, 100)  # Very consistent
        
        # Test results for provided prompts
        test_results = []
        if test_prompts:
            for i, prompt in enumerate(test_prompts[:3]):  # Limit to 3 tests
                test_results.append({
                    "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                    "response_time": random.uniform(0.5, 3.0),
                    "quality_rating": random.uniform(3.5, 5.0),
                    "token_efficiency": random.uniform(0.7, 0.95)
                })
        
        metrics = {
            "speed_score": round(speed_score, 1),
            "quality_score": round(quality_score, 1),
            "cost_score": round(cost_score, 1),
            "consistency_score": round(consistency_score, 1),
            "overall_score": round((speed_score + quality_score + cost_score + consistency_score) / 4, 1)
        }
        
        return {
            "success": True,
            "provider": provider_id,
            "model": model_id,
            "task_type": task_type,
            "metrics": metrics,
            "test_results": test_results,
            "recommended": metrics["overall_score"] > 80,
            "message": f"Performance analysis completed for {provider_id}/{model_id}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Performance analysis failed: {str(e)}"
        }


@mcp_tool
async def manage_model_rotation(
    rotation_strategy: str = "round_robin",
    models: Optional[List[Dict[str, str]]] = None,
    criteria: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Manage automatic model rotation for load balancing or optimization.
    
    Args:
        rotation_strategy: Strategy for model rotation
        models: List of models to rotate between
        criteria: Criteria for rotation decisions
        
    Returns:
        Dictionary containing rotation configuration result
    """
    try:
        # Mock implementation - replace with actual rotation management
        if not models:
            models = [
                {"provider": "ollama", "model": "llama2"},
                {"provider": "anthropic", "model": "claude-3-haiku"},
                {"provider": "openai", "model": "gpt-3.5-turbo"}
            ]
        
        strategies = ["round_robin", "performance_based", "cost_optimized", "random"]
        if rotation_strategy not in strategies:
            return {
                "success": False,
                "error": f"Invalid rotation strategy: {rotation_strategy}. Valid strategies: {strategies}"
            }
        
        # Configure rotation
        rotation_config = {
            "strategy": rotation_strategy,
            "models": models,
            "criteria": criteria or {"prioritize": "cost", "fallback": "speed"},
            "rotation_interval": "per_request" if rotation_strategy == "round_robin" else "adaptive",
            "health_check_interval": 300  # 5 minutes
        }
        
        return {
            "success": True,
            "rotation_config": rotation_config,
            "active_models": len(models),
            "current_model": models[0] if models else None,
            "message": f"Model rotation configured with {rotation_strategy} strategy"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Model rotation setup failed: {str(e)}"
        }


# ============================================================================
# Prompt Engineering Tools  
# ============================================================================

@mcp_tool
async def create_prompt_template(
    name: str,
    template: str,
    variables: List[str],
    description: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create a new prompt template.
    
    Args:
        name: Name of the template
        template: Template content with variable placeholders
        variables: List of variables used in the template
        description: Optional description
        tags: Optional tags for categorization
        
    Returns:
        Dictionary containing created template information
    """
    try:
        import uuid
        from datetime import datetime
        
        # Generate template ID
        template_id = str(uuid.uuid4())[:8]
        
        # Create template object
        template_obj = {
            "template_id": template_id,
            "name": name,
            "template": template,
            "variables": variables,
            "description": description or "",
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "usage_count": 0
        }
        
        # Mock storage - replace with actual template storage
        return {
            "success": True,
            "template": template_obj,
            "message": f"Template '{name}' created successfully with ID {template_id}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Template creation failed: {str(e)}"
        }


@mcp_tool
async def optimize_prompt(
    template_id: str,
    optimization_goals: List[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Optimize a prompt for better performance.
    
    Args:
        template_id: ID of the template to optimize
        optimization_goals: Goals for optimization
        context: Context information for optimization
        
    Returns:
        Dictionary containing optimized prompt
    """
    try:
        if not optimization_goals:
            optimization_goals = ["clarity", "effectiveness"]
        
        # Mock optimization - replace with actual optimization logic
        improvements = []
        if "clarity" in optimization_goals:
            improvements.extend([
                "Added specific instructions for better clarity",
                "Restructured for logical flow"
            ])
        if "effectiveness" in optimization_goals:
            improvements.extend([
                "Added examples for better context",
                "Optimized for target model capabilities"
            ])
        if "brevity" in optimization_goals:
            improvements.extend([
                "Removed redundant phrases",
                "Condensed instructions"
            ])
        
        # Generate optimized prompt (mock)
        optimized_prompt = """You are an expert assistant. Please provide a comprehensive and accurate response to the following request:

{user_query}

Requirements:
- Be specific and detailed
- Provide examples when relevant
- Structure your response clearly
- Focus on practical, actionable advice

Context: {context}"""
        
        return {
            "success": True,
            "template_id": template_id,
            "optimization_goals": optimization_goals,
            "optimized_prompt": optimized_prompt,
            "improvements": improvements,
            "optimization_score": 8.5,  # Mock score
            "message": f"Prompt optimized successfully with {len(improvements)} improvements"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Prompt optimization failed: {str(e)}"
        }


@mcp_tool
async def validate_prompt_syntax(
    prompt_text: str,
    template_variables: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate prompt syntax and structure.
    
    Args:
        prompt_text: The prompt text to validate
        template_variables: Expected template variables
        
    Returns:
        Dictionary containing validation results
    """
    try:
        validation_results = {
            "syntax_valid": True,
            "template_variables_found": [],
            "missing_variables": [],
            "undefined_variables": [],
            "suggestions": []
        }
        
        # Check for template variables (simple implementation)
        import re
        found_variables = re.findall(r'\{(\w+)\}', prompt_text)
        validation_results["template_variables_found"] = found_variables
        
        if template_variables:
            # Check for missing variables
            missing = set(template_variables) - set(found_variables)
            validation_results["missing_variables"] = list(missing)
            
            # Check for undefined variables
            undefined = set(found_variables) - set(template_variables)
            validation_results["undefined_variables"] = list(undefined)
        
        # Generate suggestions
        if validation_results["missing_variables"]:
            validation_results["suggestions"].append("Consider adding placeholders for missing variables")
        if validation_results["undefined_variables"]:
            validation_results["suggestions"].append("Remove or define undefined variables")
        if len(prompt_text) > 2000:
            validation_results["suggestions"].append("Consider shortening prompt for better token efficiency")
        
        # Overall validation status
        is_valid = (
            validation_results["syntax_valid"] and
            not validation_results["missing_variables"] and
            not validation_results["undefined_variables"]
        )
        
        return {
            "success": True,
            "validation_passed": is_valid,
            "results": validation_results,
            "message": "Validation completed" if is_valid else "Validation found issues"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Prompt validation failed: {str(e)}"
        }


@mcp_tool
async def get_prompt_history(
    template_id: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Get prompt usage history.
    
    Args:
        template_id: Filter by specific template
        user_id: Filter by specific user
        limit: Maximum number of results
        
    Returns:
        Dictionary containing prompt history
    """
    try:
        from datetime import datetime, timedelta
        import random
        
        # Generate mock history
        history_entries = []
        for i in range(min(limit, 5)):  # Generate up to 5 mock entries
            history_entries.append({
                "usage_id": f"usage_{i+1}",
                "template_id": template_id or f"template_{i+1}",
                "user_id": user_id or f"user_{random.randint(1,3)}",
                "timestamp": (datetime.now() - timedelta(hours=i*2)).isoformat(),
                "prompt_text": f"Sample prompt {i+1}...",
                "variables_used": {"query": f"example query {i+1}", "context": f"context {i+1}"},
                "response_quality": random.uniform(3.5, 5.0),
                "success": True
            })
        
        return {
            "success": True,
            "history": history_entries,
            "total_count": len(history_entries),
            "filters": {
                "template_id": template_id,
                "user_id": user_id,
                "limit": limit
            },
            "message": f"Retrieved {len(history_entries)} history entries"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get prompt history: {str(e)}"
        }


@mcp_tool
async def analyze_prompt_performance(
    prompt_text: str,
    test_contexts: List[Dict[str, Any]],
    metrics_to_analyze: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze prompt performance across different contexts.
    
    Args:
        prompt_text: The prompt to analyze
        test_contexts: List of contexts to test against
        metrics_to_analyze: Specific metrics to focus on
        
    Returns:
        Dictionary containing performance analysis
    """
    try:
        if not metrics_to_analyze:
            metrics_to_analyze = ["clarity", "specificity", "effectiveness"]
        
        import random
        
        # Analyze each metric
        analysis_results = {}
        for metric in metrics_to_analyze:
            # Mock analysis scores
            scores = [random.uniform(3.0, 5.0) for _ in test_contexts]
            analysis_results[metric] = {
                "average_score": sum(scores) / len(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "consistency": 1.0 - (max(scores) - min(scores)) / 5.0  # Normalized consistency
            }
        
        # Overall analysis
        overall_score = sum(
            analysis_results[metric]["average_score"] 
            for metric in metrics_to_analyze
        ) / len(metrics_to_analyze)
        
        # Generate recommendations
        recommendations = []
        for metric in metrics_to_analyze:
            if analysis_results[metric]["average_score"] < 3.5:
                recommendations.append(f"Consider improving {metric} - current score is below average")
            if analysis_results[metric]["consistency"] < 0.7:
                recommendations.append(f"Improve consistency for {metric} across different contexts")
        
        return {
            "success": True,
            "prompt_analyzed": prompt_text[:100] + "..." if len(prompt_text) > 100 else prompt_text,
            "contexts_tested": len(test_contexts),
            "metrics": analysis_results,
            "overall_score": round(overall_score, 2),
            "recommendations": recommendations,
            "message": f"Performance analysis completed for {len(test_contexts)} contexts"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Performance analysis failed: {str(e)}"
        }


@mcp_tool
async def manage_prompt_library(
    action: str,
    template_id: Optional[str] = None,
    category: Optional[str] = None,
    search_term: Optional[str] = None
) -> Dict[str, Any]:
    """
    Manage the prompt template library.
    
    Args:
        action: Action to perform (list, search, categorize, delete)
        template_id: Specific template ID for operations
        category: Category for filtering or categorization
        search_term: Search term for finding templates
        
    Returns:
        Dictionary containing library management results
    """
    try:
        # Mock library data
        mock_templates = [
            {
                "template_id": "tmpl_001",
                "name": "Code Review Template",
                "category": "development",
                "tags": ["code", "review", "quality"],
                "usage_count": 45,
                "created_at": "2024-01-15T10:00:00Z"
            },
            {
                "template_id": "tmpl_002",
                "name": "Data Analysis Template",
                "category": "analysis",
                "tags": ["data", "analysis", "insights"],
                "usage_count": 32,
                "created_at": "2024-01-20T14:30:00Z"
            },
            {
                "template_id": "tmpl_003",
                "name": "Creative Writing Template",
                "category": "creative",
                "tags": ["writing", "creative", "story"],
                "usage_count": 18,
                "created_at": "2024-02-01T09:15:00Z"
            }
        ]
        
        if action == "list":
            # Filter by category if provided
            if category:
                filtered_templates = [t for t in mock_templates if t["category"] == category]
            else:
                filtered_templates = mock_templates
            
            return {
                "success": True,
                "action": "list",
                "templates": filtered_templates,
                "total_count": len(filtered_templates),
                "filters": {"category": category},
                "message": f"Listed {len(filtered_templates)} templates"
            }
        
        elif action == "search":
            # Search by name or tags
            if search_term:
                search_results = []
                for template in mock_templates:
                    if (search_term.lower() in template["name"].lower() or
                        any(search_term.lower() in tag.lower() for tag in template["tags"])):
                        search_results.append(template)
            else:
                search_results = mock_templates
            
            return {
                "success": True,
                "action": "search",
                "search_term": search_term,
                "results": search_results,
                "result_count": len(search_results),
                "message": f"Found {len(search_results)} matching templates"
            }
        
        elif action == "categorize":
            if not template_id or not category:
                return {
                    "success": False,
                    "error": "Both template_id and category required for categorization"
                }
            
            return {
                "success": True,
                "action": "categorize",
                "template_id": template_id,
                "new_category": category,
                "message": f"Template {template_id} categorized as {category}"
            }
        
        elif action == "delete":
            if not template_id:
                return {
                    "success": False,
                    "error": "template_id required for deletion"
                }
            
            return {
                "success": True,
                "action": "delete",
                "template_id": template_id,
                "message": f"Template {template_id} deleted successfully"
            }
        
        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}. Valid actions: list, search, categorize, delete"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Library management failed: {str(e)}"
        }


# ============================================================================
# Context Management Tools
# ============================================================================

@mcp_tool
async def analyze_context_usage(
    context_id: str,
    time_period: str = "last_week",
    include_metrics: bool = True
) -> Dict[str, Any]:
    """
    Analyze context usage patterns and efficiency.
    
    Args:
        context_id: ID of the context to analyze
        time_period: Time period for analysis
        include_metrics: Whether to include detailed metrics
        
    Returns:
        Dictionary containing context usage analysis
    """
    try:
        import random
        from datetime import datetime, timedelta
        
        # Mock usage data
        usage_stats = {
            "total_messages": random.randint(50, 200),
            "total_tokens": random.randint(5000, 50000),
            "avg_message_length": random.randint(50, 300),
            "context_switches": random.randint(5, 25),
            "peak_usage_period": "14:00-16:00",
            "efficiency_score": random.uniform(0.6, 0.9)
        }
        
        # Detailed metrics if requested
        detailed_metrics = {}
        if include_metrics:
            detailed_metrics = {
                "token_distribution": {
                    "system_tokens": int(usage_stats["total_tokens"] * 0.1),
                    "user_tokens": int(usage_stats["total_tokens"] * 0.6),  
                    "assistant_tokens": int(usage_stats["total_tokens"] * 0.3)
                },
                "message_types": {
                    "questions": random.randint(20, 60),
                    "commands": random.randint(10, 30),
                    "clarifications": random.randint(5, 20)
                },
                "response_quality": {
                    "helpful_responses": random.uniform(0.8, 0.95),
                    "relevant_responses": random.uniform(0.85, 0.98),
                    "clear_responses": random.uniform(0.75, 0.92)
                }
            }
        
        # Analysis insights
        insights = []
        if usage_stats["efficiency_score"] < 0.7:
            insights.append("Context efficiency could be improved")
        if usage_stats["avg_message_length"] > 200:
            insights.append("Consider breaking down longer messages")
        if usage_stats["context_switches"] > 15:
            insights.append("High number of context switches detected")
        
        # Determine if optimization is needed
        optimization_needed = (
            usage_stats["efficiency_score"] < 0.7 or
            usage_stats["total_tokens"] > 40000 or
            usage_stats["context_switches"] > 20
        )
        
        return {
            "success": True,
            "context_id": context_id,
            "time_period": time_period,
            "analysis": {
                "usage_stats": usage_stats,
                "detailed_metrics": detailed_metrics,
                "insights": insights,
                "optimization_needed": optimization_needed,
                "analysis_date": datetime.now().isoformat()
            },
            "message": f"Context usage analysis completed for {context_id}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Context usage analysis failed: {str(e)}"
        }


@mcp_tool
async def optimize_context_window(
    context_id: str,
    optimization_strategy: str = "efficiency",
    preserve_recent_messages: bool = True
) -> Dict[str, Any]:
    """
    Optimize the context window for better performance.
    
    Args:
        context_id: ID of the context to optimize
        optimization_strategy: Strategy for optimization
        preserve_recent_messages: Whether to preserve recent messages
        
    Returns:
        Dictionary containing optimization results
    """
    try:
        import random
        
        strategies = ["efficiency", "relevance", "recency", "importance"]
        if optimization_strategy not in strategies:
            return {
                "success": False,
                "error": f"Invalid optimization strategy: {optimization_strategy}. Valid strategies: {strategies}"
            }
        
        # Mock current context state
        current_state = {
            "total_messages": random.randint(50, 150),
            "total_tokens": random.randint(8000, 25000),
            "oldest_message_age": random.randint(7, 30)  # days
        }
        
        # Simulate optimization
        optimized_state = {
            "total_messages": max(20, int(current_state["total_messages"] * 0.7)),
            "total_tokens": max(2000, int(current_state["total_tokens"] * 0.6)),
            "optimization_ratio": random.uniform(0.3, 0.5)
        }
        
        # Optimization actions taken
        actions_taken = []
        if optimization_strategy == "efficiency":
            actions_taken.extend([
                "Removed redundant messages",
                "Compressed repetitive content",
                "Merged similar queries"
            ])
        elif optimization_strategy == "relevance":
            actions_taken.extend([
                "Removed off-topic messages",
                "Prioritized task-relevant content",
                "Filtered low-quality responses"
            ])
        elif optimization_strategy == "recency":
            actions_taken.extend([
                "Removed oldest messages beyond window",
                "Preserved recent conversation flow",
                "Maintained context continuity"
            ])
        
        # Calculate improvements
        token_reduction = current_state["total_tokens"] - optimized_state["total_tokens"]
        efficiency_improvement = (token_reduction / current_state["total_tokens"]) * 100
        
        return {
            "success": True,
            "context_id": context_id,
            "optimization_strategy": optimization_strategy,
            "current_state": current_state,
            "optimized_state": optimized_state,
            "improvements": {
                "tokens_reduced": token_reduction,
                "efficiency_improvement_percent": round(efficiency_improvement, 1),
                "messages_preserved": optimized_state["total_messages"],
                "actions_taken": actions_taken
            },
            "preserved_recent": preserve_recent_messages,
            "message": f"Context window optimized using {optimization_strategy} strategy"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Context optimization failed: {str(e)}"
        }


@mcp_tool
async def track_context_history(
    context_id: str,
    analysis_depth: str = "standard",
    include_token_counts: bool = True
) -> Dict[str, Any]:
    """
    Track and analyze context history patterns.
    
    Args:
        context_id: ID of the context to track
        analysis_depth: Depth of analysis (standard, detailed, comprehensive)
        include_token_counts: Whether to include token count information
        
    Returns:
        Dictionary containing context history tracking
    """
    try:
        import random
        from datetime import datetime, timedelta
        
        depths = ["standard", "detailed", "comprehensive"]
        if analysis_depth not in depths:
            return {
                "success": False,
                "error": f"Invalid analysis depth: {analysis_depth}. Valid depths: {depths}"
            }
        
        # Generate mock history data
        num_sessions = random.randint(3, 10)
        history_data = {
            "context_id": context_id,
            "tracking_period": "last_30_days",
            "total_sessions": num_sessions,
            "sessions": []
        }
        
        total_tokens = 0
        for i in range(num_sessions):
            session_tokens = random.randint(500, 5000)
            total_tokens += session_tokens
            
            session = {
                "session_id": f"session_{i+1}",
                "start_time": (datetime.now() - timedelta(days=i*3)).isoformat(),
                "duration_minutes": random.randint(15, 120),
                "message_count": random.randint(10, 50),
                "tokens_used": session_tokens if include_token_counts else None,
                "topics": [f"topic_{j+1}" for j in range(random.randint(1, 4))]
            }
            
            if analysis_depth in ["detailed", "comprehensive"]:
                session.update({
                    "avg_response_time": random.uniform(0.5, 3.0),
                    "user_satisfaction": random.uniform(3.5, 5.0),
                    "context_switches": random.randint(0, 5),
                    "error_count": random.randint(0, 2)
                })
            
            if analysis_depth == "comprehensive":
                session.update({
                    "detailed_metrics": {
                        "query_complexity": random.uniform(0.3, 0.9),
                        "response_relevance": random.uniform(0.8, 0.98),
                        "conversation_flow": random.uniform(0.7, 0.95),
                        "knowledge_utilization": random.uniform(0.6, 0.9)
                    }
                })
            
            history_data["sessions"].append(session)
        
        # Calculate summary metrics
        metrics = {
            "total_tokens": total_tokens if include_token_counts else None,
            "avg_session_duration": sum(s["duration_minutes"] for s in history_data["sessions"]) / num_sessions,
            "total_messages": sum(s["message_count"] for s in history_data["sessions"]),
            "context_retention_score": random.uniform(0.7, 0.95)
        }
        
        if analysis_depth in ["detailed", "comprehensive"]:
            metrics.update({
                "avg_response_time": sum(s.get("avg_response_time", 0) for s in history_data["sessions"]) / num_sessions,
                "avg_satisfaction": sum(s.get("user_satisfaction", 0) for s in history_data["sessions"]) / num_sessions,
                "total_errors": sum(s.get("error_count", 0) for s in history_data["sessions"])
            })
        
        return {
            "success": True,
            "history": history_data,
            "metrics": metrics,
            "analysis_depth": analysis_depth,
            "recommendations": [
                "Consider regular context cleanup",
                "Monitor token usage trends",
                "Track user satisfaction patterns"
            ],
            "message": f"Context history tracking completed with {analysis_depth} analysis"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Context history tracking failed: {str(e)}"
        }


@mcp_tool
async def compress_context(
    context_id: str,
    compression_ratio: float = 0.7,
    preserve_key_information: bool = True
) -> Dict[str, Any]:
    """
    Compress context to reduce token usage while preserving important information.
    
    Args:
        context_id: ID of the context to compress
        compression_ratio: Target compression ratio (0.0 to 1.0)
        preserve_key_information: Whether to preserve key information
        
    Returns:
        Dictionary containing compression results
    """
    try:
        import random
        
        if not 0.0 <= compression_ratio <= 1.0:
            return {
                "success": False,
                "error": "Compression ratio must be between 0.0 and 1.0"
            }
        
        # Mock current context state
        original_state = {
            "total_messages": random.randint(80, 200),
            "total_tokens": random.randint(15000, 50000),
            "unique_topics": random.randint(5, 15),
            "key_information_items": random.randint(10, 30)
        }
        
        # Calculate compression targets
        target_tokens = int(original_state["total_tokens"] * compression_ratio)
        target_messages = int(original_state["total_messages"] * (compression_ratio * 1.2))  # Slightly higher ratio for messages
        
        # Simulate compression process
        compression_techniques = []
        if preserve_key_information:
            compression_techniques.extend([
                "Semantic summarization",
                "Key information extraction",
                "Redundancy removal"
            ])
        else:
            compression_techniques.extend([
                "Simple truncation",
                "Token-based compression",
                "Message filtering"
            ])
        
        # Compression results
        compressed_state = {
            "total_messages": max(20, target_messages),
            "total_tokens": max(2000, target_tokens),
            "preserved_topics": original_state["unique_topics"] if preserve_key_information else max(1, int(original_state["unique_topics"] * 0.6)),
            "preserved_key_info": original_state["key_information_items"] if preserve_key_information else max(1, int(original_state["key_information_items"] * 0.4))
        }
        
        # Calculate actual compression achieved
        actual_token_ratio = compressed_state["total_tokens"] / original_state["total_tokens"]
        actual_message_ratio = compressed_state["total_messages"] / original_state["total_messages"]
        
        # Quality metrics
        quality_metrics = {
            "information_retention": random.uniform(0.8, 0.95) if preserve_key_information else random.uniform(0.5, 0.7),
            "context_coherence": random.uniform(0.7, 0.9),
            "semantic_integrity": random.uniform(0.75, 0.92) if preserve_key_information else random.uniform(0.4, 0.6)
        }
        
        return {
            "success": True,
            "context_id": context_id,
            "compression_config": {
                "target_ratio": compression_ratio,
                "preserve_key_info": preserve_key_information,
                "techniques_used": compression_techniques
            },
            "original_state": original_state,
            "compressed_state": compressed_state,
            "compression_results": {
                "tokens_reduced": original_state["total_tokens"] - compressed_state["total_tokens"],
                "messages_reduced": original_state["total_messages"] - compressed_state["total_messages"],
                "actual_token_ratio": round(actual_token_ratio, 3),
                "actual_message_ratio": round(actual_message_ratio, 3),
                "space_saved_percent": round((1 - actual_token_ratio) * 100, 1)
            },
            "quality_metrics": quality_metrics,
            "recommendations": [
                "Review compressed context for completeness",
                "Monitor conversation quality after compression",
                "Consider periodic re-compression"
            ],
            "message": f"Context compressed successfully, {round((1 - actual_token_ratio) * 100, 1)}% space saved"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Context compression failed: {str(e)}"
        }


# ============================================================================
# Tool Collections
# ============================================================================

# LLM Management Tools
# Tools are automatically registered by @mcp_tool decorator
llm_management_tools = []

# Prompt Engineering Tools
prompt_engineering_tools = []

# Context Management Tools  
context_management_tools = []


__all__ = [
    "llm_management_tools",
    "prompt_engineering_tools", 
    "context_management_tools",
    "get_available_models",
    "set_default_model",
    "get_model_capabilities",
    "test_model_connection",
    "get_model_performance",
    "manage_model_rotation",
    "create_prompt_template",
    "optimize_prompt",
    "validate_prompt_syntax",
    "get_prompt_history",
    "analyze_prompt_performance",
    "manage_prompt_library",
    "analyze_context_usage",
    "optimize_context_window",
    "track_context_history",
    "compress_context"
]