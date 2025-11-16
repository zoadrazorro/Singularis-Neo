"""
Singularis Beta v2 - Configuration Template

This file provides several pre-defined configuration presets for running the
Singularis Beta v2 AGI. Users can copy this file to create their own custom
configurations or use the provided `get_config` function to load a preset
by name.

Usage:
    1. Copy this file: `cp beta_v2_config_template.py my_config.py`
    2. Edit `my_config.py` to customize the settings.
    3. Import your custom config in the main run script, or use `get_config('preset_name')`.
"""

from singularis.skyrim.config import SkyrimConfig


def create_standard_config() -> SkyrimConfig:
    """Creates the standard, balanced performance configuration.

    This configuration is suitable for most general use cases and is a good
    starting point for typical 30-60 minute sessions. It enables key features
    like voice, video, and the GPT-5 orchestrator without being excessively
    demanding on API limits.

    Returns:
        A `SkyrimConfig` object with the standard settings.
    """
    config = SkyrimConfig()
    
    # Timing
    config.cycle_interval = 2.0  # seconds between cycles
    
    # Features
    config.enable_voice = True
    config.enable_video_interpreter = True
    config.use_wolfram_telemetry = True
    config.use_gpt5_orchestrator = True
    config.use_double_helix = True
    
    # Voice settings
    config.voice_type = "NOVA"  # Alloy, Echo, Fable, Onyx, Nova, Shimmer
    config.voice_min_priority = "HIGH"  # CRITICAL, HIGH, MEDIUM, LOW
    
    # Video settings
    config.video_interpretation_mode = "COMPREHENSIVE"
    config.video_frame_rate = 0.5  # frames per second
    
    # API limits
    config.gemini_rpm_limit = 15
    config.openai_rpm_limit = 50
    
    # Experts
    config.num_gemini_experts = 1
    config.num_openai_experts = 2
    
    # Verbose
    config.verbose = True
    
    return config


def create_fast_config() -> SkyrimConfig:
    """Creates a fast configuration with minimal features for quick testing.

    This preset is ideal for development, debugging, and running quick tests,
    as it disables expensive features like voice and video and uses a faster
    cycle time.

    Returns:
        A `SkyrimConfig` object with settings optimized for speed.
    """
    config = SkyrimConfig()
    
    # Timing - faster cycles
    config.cycle_interval = 1.0
    
    # Disable expensive features
    config.enable_voice = False
    config.enable_video_interpreter = False
    config.use_wolfram_telemetry = False
    
    # Minimal experts
    config.num_gemini_experts = 1
    config.num_openai_experts = 1
    
    # Verbose for debugging
    config.verbose = True
    
    return config


def create_conservative_config() -> SkyrimConfig:
    """Creates a conservative configuration designed for low API usage.

    This is a good choice for users on free API tiers or for running very long
    sessions where minimizing cost is a priority. It uses a slower cycle time
    and lower API rate limits.

    Returns:
        A `SkyrimConfig` object with settings optimized for low resource consumption.
    """
    config = SkyrimConfig()
    
    # Timing - slower to reduce API calls
    config.cycle_interval = 5.0
    
    # Reduced API limits
    config.gemini_rpm_limit = 10
    config.openai_rpm_limit = 30
    
    # Minimal experts
    config.num_gemini_experts = 1
    config.num_openai_experts = 1
    
    # Keep essential features
    config.enable_voice = False
    config.enable_video_interpreter = False
    config.use_wolfram_telemetry = True  # Only every 20 cycles
    
    config.verbose = False  # Reduce console spam
    
    return config


def create_premium_config() -> SkyrimConfig:
    """Creates a premium configuration with all features enabled for the full experience.

    This preset is intended for users with paid API tiers and high rate limits.
    It enables all advanced features, a faster video frame rate, and more
    expert models for a more responsive and capable AGI.

    Returns:
        A `SkyrimConfig` object with all features enabled.
    """
    config = SkyrimConfig()
    
    # Timing - responsive
    config.cycle_interval = 1.5
    
    # All features enabled
    config.enable_voice = True
    config.enable_video_interpreter = True
    config.use_wolfram_telemetry = True
    config.use_gpt5_orchestrator = True
    config.use_double_helix = True
    config.self_improvement_gating = True
    
    # Voice settings - high quality
    config.voice_type = "NOVA"
    config.voice_min_priority = "MEDIUM"  # Speak more often
    
    # Video settings - comprehensive
    config.video_interpretation_mode = "COMPREHENSIVE"
    config.video_frame_rate = 1.0  # 1 FPS
    
    # Higher API limits (requires paid tier)
    config.gemini_rpm_limit = 30
    config.openai_rpm_limit = 100
    
    # More experts
    config.num_gemini_experts = 2
    config.num_openai_experts = 3
    
    # Verbose
    config.verbose = True
    
    return config


def create_research_config() -> SkyrimConfig:
    """Creates a research-oriented configuration for maximum data collection.

    This preset enables all features that contribute to data generation and logging,
    such as comprehensive video interpretation and telemetry. It disables voice
    to ensure cleaner data logs for analysis.

    Returns:
        A `SkyrimConfig` object optimized for research and data collection.
    """
    config = SkyrimConfig()
    
    # Timing - balanced
    config.cycle_interval = 3.0
    
    # Wolfram for analysis
    config.use_wolfram_telemetry = True
    
    # All systems for complete data
    config.use_gpt5_orchestrator = True
    config.use_double_helix = True
    config.self_improvement_gating = True
    
    # Video for visual analysis
    config.enable_video_interpreter = True
    config.video_interpretation_mode = "COMPREHENSIVE"
    config.video_frame_rate = 0.5
    
    # Voice can be disabled for cleaner data
    config.enable_voice = False
    
    # Moderate API usage
    config.gemini_rpm_limit = 15
    config.openai_rpm_limit = 50
    
    # Verbose for detailed logs
    config.verbose = True
    
    return config


def create_silent_config() -> SkyrimConfig:
    """Creates a silent configuration with no voice or video output.

    This preset is suitable for running the AGI in the background or on a server
    where audio-visual features are not needed. It keeps the core reasoning
    systems active but disables all interactive sensory outputs.

    Returns:
        A `SkyrimConfig` object for silent operation.
    """
    config = SkyrimConfig()
    
    # Timing
    config.cycle_interval = 2.5
    
    # Core systems only
    config.enable_voice = False
    config.enable_video_interpreter = False
    config.use_wolfram_telemetry = True
    config.use_gpt5_orchestrator = True
    
    # Normal API limits
    config.gemini_rpm_limit = 15
    config.openai_rpm_limit = 50
    
    # Quiet operation
    config.verbose = False
    
    return config


# A dictionary mapping preset names to their creation functions.
CONFIGS = {
    'standard': create_standard_config,
    'fast': create_fast_config,
    'conservative': create_conservative_config,
    'premium': create_premium_config,
    'research': create_research_config,
    'silent': create_silent_config,
}


def get_config(preset: str = 'standard') -> SkyrimConfig:
    """Retrieves a `SkyrimConfig` object for a named preset.

    Args:
        preset: The name of the preset to load. Must be one of 'standard',
                'fast', 'conservative', 'premium', 'research', or 'silent'.

    Returns:
        A `SkyrimConfig` instance with the specified preset's settings.

    Raises:
        ValueError: If the specified preset name is unknown.
    """
    if preset not in CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(CONFIGS.keys())}")
    
    return CONFIGS[preset]()


if __name__ == '__main__':
    # Test and print all available configurations
    print("Testing configuration presets...\n")
    
    for name, create_func in CONFIGS.items():
        print(f"[{name.upper()}]")
        config = create_func()
        print(f"  Cycle interval: {config.cycle_interval}s")
        print(f"  Voice: {config.enable_voice}")
        print(f"  Video: {config.enable_video_interpreter}")
        print(f"  Wolfram: {config.use_wolfram_telemetry}")
        print(f"  Gemini RPM: {config.gemini_rpm_limit}")
        print()
    
    print("All presets validated ✓")
