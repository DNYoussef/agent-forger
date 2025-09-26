"""
Pydantic models for Agent Forge Pipeline API

Type-safe request/response models for all pipeline operations.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class PipelinePhase(str, Enum):
    """Available pipeline phases."""
    COGNATE = "cognate"
    EVOMERGE = "evomerge"
    QUIETSTAR = "quietstar"
    BITNET = "bitnet"
    TRAINING = "training"
    BAKING = "baking"
    ADAS = "adas"
    COMPRESSION = "compression"


class PipelineStatus(str, Enum):
    """Pipeline execution status."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SwarmTopology(str, Enum):
    """Swarm coordination topologies."""
    HIERARCHICAL = "hierarchical"
    MESH = "mesh"
    STAR = "star"
    RING = "ring"


# Request Models

class PipelineStartRequest(BaseModel):
    """Request to start pipeline execution."""
    phases: List[PipelinePhase] = Field(
        description="Phases to execute in order",
        example=["cognate", "evomerge", "quietstar"]
    )
    config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Phase-specific configuration overrides"
    )
    enable_monitoring: bool = Field(
        default=True,
        description="Enable real-time monitoring"
    )
    enable_checkpoints: bool = Field(
        default=True,
        description="Save checkpoints between phases"
    )
    swarm_topology: SwarmTopology = Field(
        default=SwarmTopology.HIERARCHICAL,
        description="Swarm coordination topology"
    )
    max_agents: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Maximum number of agents"
    )


class PipelineControlRequest(BaseModel):
    """Request to control pipeline execution."""
    action: str = Field(
        description="Control action: pause, resume, stop, cancel",
        regex="^(pause|resume|stop|cancel)$"
    )
    session_id: str = Field(
        description="Pipeline session ID"
    )
    force: bool = Field(
        default=False,
        description="Force action even if unsafe"
    )


class PhaseConfigRequest(BaseModel):
    """Request to configure a specific phase."""
    phase: PipelinePhase
    config: Dict[str, Any] = Field(
        description="Phase configuration parameters"
    )

    @validator('config')
    def validate_config(cls, v, values):
        """Validate phase-specific configuration."""
        phase = values.get('phase')

        # Phase-specific validation
        if phase == PipelinePhase.COGNATE:
            required = ['base_models', 'init_strategy']
        elif phase == PipelinePhase.EVOMERGE:
            required = ['population_size', 'generations']
        elif phase == PipelinePhase.QUIETSTAR:
            required = ['thought_length', 'num_thoughts']
        elif phase == PipelinePhase.TRAINING:
            required = ['batch_size', 'learning_rate']
        else:
            required = []

        missing = [key for key in required if key not in v]
        if missing:
            raise ValueError(f"Missing required config for {phase}: {missing}")

        return v


class CheckpointRequest(BaseModel):
    """Request to save/load checkpoint."""
    session_id: str
    checkpoint_name: Optional[str] = None
    include_model_state: bool = True
    include_swarm_state: bool = True


class PresetRequest(BaseModel):
    """Request to save/load configuration preset."""
    preset_name: str
    config: Optional[Dict[str, Any]] = None


# Response Models

class AgentStatus(BaseModel):
    """Agent status information."""
    agent_id: str
    role: str
    phase: int
    state: str
    current_task: Optional[str] = None
    memory_usage_mb: float
    cpu_usage_percent: float
    uptime_seconds: float


class PhaseMetrics(BaseModel):
    """Metrics for a specific phase."""
    phase: str
    status: PipelineStatus
    progress_percent: float = Field(ge=0, le=100)
    duration_seconds: float
    memory_usage_mb: float
    gpu_utilization_percent: Optional[float] = None
    custom_metrics: Dict[str, float] = Field(default_factory=dict)
    error_message: Optional[str] = None


class PipelineStatusResponse(BaseModel):
    """Current pipeline status."""
    session_id: str
    status: PipelineStatus
    current_phase: Optional[PipelinePhase] = None
    phases_completed: List[PipelinePhase]
    phases_remaining: List[PipelinePhase]
    total_progress_percent: float = Field(ge=0, le=100)
    elapsed_seconds: float
    estimated_remaining_seconds: Optional[float] = None
    phase_metrics: List[PhaseMetrics]
    agent_count: int
    active_agents: List[AgentStatus]
    last_checkpoint: Optional[str] = None


class PipelineResultResponse(BaseModel):
    """Final pipeline execution result."""
    session_id: str
    success: bool
    phases_executed: List[PipelinePhase]
    phases_successful: List[PipelinePhase]
    phases_failed: List[PipelinePhase]
    total_duration_seconds: float
    final_metrics: Dict[str, Any]
    model_path: Optional[str] = None
    artifacts: Dict[str, str] = Field(default_factory=dict)
    error_summary: Optional[str] = None


class SwarmStatusResponse(BaseModel):
    """Swarm coordination status."""
    topology: SwarmTopology
    total_agents: int
    active_agents: int
    idle_agents: int
    agents: List[AgentStatus]
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    communication_latency_ms: float


class HealthCheckResponse(BaseModel):
    """API health check response."""
    status: str
    version: str
    uptime_seconds: float
    active_sessions: int
    total_memory_mb: float
    available_memory_mb: float


class ConfigPresetResponse(BaseModel):
    """Configuration preset response."""
    preset_name: str
    config: Dict[str, Any]
    created_at: datetime
    last_used: Optional[datetime] = None
    usage_count: int = 0


class CheckpointResponse(BaseModel):
    """Checkpoint save/load response."""
    checkpoint_id: str
    session_id: str
    phase: Optional[PipelinePhase] = None
    timestamp: datetime
    size_mb: float
    model_included: bool
    swarm_included: bool
    path: str


class ExecutionHistoryResponse(BaseModel):
    """Pipeline execution history entry."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: PipelineStatus
    phases: List[PipelinePhase]
    success: bool
    duration_seconds: Optional[float] = None
    final_metrics: Optional[Dict[str, Any]] = None


# WebSocket Event Models

class WebSocketEvent(BaseModel):
    """Base WebSocket event."""
    event_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)


class PipelineProgressEvent(WebSocketEvent):
    """Pipeline progress update event."""
    event_type: str = "pipeline_progress"
    phase: PipelinePhase
    progress_percent: float
    metrics: Dict[str, float]


class AgentUpdateEvent(WebSocketEvent):
    """Agent status update event."""
    event_type: str = "agent_update"
    agent_id: str
    state: str
    task: Optional[str] = None


class PhaseCompletionEvent(WebSocketEvent):
    """Phase completion event."""
    event_type: str = "phase_complete"
    phase: PipelinePhase
    success: bool
    duration_seconds: float
    metrics: Dict[str, Any]


class ErrorEvent(WebSocketEvent):
    """Error event."""
    event_type: str = "error"
    error_code: str
    error_message: str
    phase: Optional[PipelinePhase] = None
    recoverable: bool = False


class MetricsStreamEvent(WebSocketEvent):
    """Real-time metrics stream event."""
    event_type: str = "metrics"
    metrics: Dict[str, float]
    phase: Optional[PipelinePhase] = None


# Quality Gate Models

class QualityGateResult(BaseModel):
    """Quality gate validation result."""
    gate_name: str
    passed: bool
    score: float = Field(ge=0, le=1)
    threshold: float
    details: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)


class TheaterDetectionResult(BaseModel):
    """Theater detection analysis result."""
    theater_detected: bool
    confidence: float = Field(ge=0, le=1)
    theater_score: float = Field(ge=0, le=1)
    indicators: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)


class QualityGateResponse(BaseModel):
    """Quality gate validation response."""
    phase: PipelinePhase
    all_gates_passed: bool
    gates_passed: int
    gates_failed: int
    gate_results: List[QualityGateResult]
    theater_detection: Optional[TheaterDetectionResult] = None
    blocking_failures: List[str] = Field(default_factory=list)
    can_proceed: bool


# Error Response Models

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    error_code: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)