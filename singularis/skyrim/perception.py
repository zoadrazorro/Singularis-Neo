"""
Skyrim Perception Layer

Handles all perception for the Skyrim environment:
1. Screen capture → CLIP embeddings
2. Scene classification (indoor/outdoor, combat/dialogue, etc.)
3. Game state reading (via mods or OCR)
4. Object detection and NPC tracking

Design principles:
- Perception grounded in actual gameplay visuals
- CLIP vision provides semantic understanding of game scenes
- Real-time processing for gameplay decisions
- Scene classification enables context-appropriate actions
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from PIL import Image
import time
from .action_affordances import ActionAffordanceSystem
from .enhanced_vision import EnhancedVision

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("Warning: mss not installed. Screen capture will use dummy mode.")
    print("Install with: pip install mss")


class SceneType(Enum):
    """Types of scenes in Skyrim."""
    OUTDOOR_WILDERNESS = "outdoor_wilderness"
    OUTDOOR_CITY = "outdoor_city"
    INDOOR_DUNGEON = "indoor_dungeon"
    INDOOR_BUILDING = "indoor_building"
    COMBAT = "combat"
    DIALOGUE = "dialogue"
    INVENTORY = "inventory"
    MAP = "map"
    UNKNOWN = "unknown"


@dataclass
class GameState:
    """
    Current game state.

    This can be populated from:
    1. Game state API (via SKSE mods)
    2. OCR from screen
    3. Memory reading (advanced)
    """
    # Player stats
    health: float = 100.0
    magicka: float = 100.0
    stamina: float = 100.0
    level: int = 1

    # Position
    position: Optional[Tuple[float, float, float]] = None
    location_name: str = "Unknown"

    # Environment
    time_of_day: float = 12.0  # Hour (0-24)
    weather: str = "clear"

    # NPCs nearby
    nearby_npcs: List[str] = None

    # Inventory (simplified)
    gold: int = 0
    inventory_items: List[str] = None

    # Quest state
    active_quests: List[str] = None

    # Combat state
    in_combat: bool = False
    enemies_nearby: int = 0
    
    # Dialogue state
    in_dialogue: bool = False
    
    # Menu state
    in_menu: bool = False
    menu_type: str = ""
    
    # Action layer awareness
    current_action_layer: str = "Exploration"
    available_actions: List[str] = None
    layer_transition_reason: str = ""

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.nearby_npcs is None:
            self.nearby_npcs = []
        if self.inventory_items is None:
            self.inventory_items = []
        if self.active_quests is None:
            self.active_quests = []
        if self.available_actions is None:
            self.available_actions = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for world model."""
        return {
            'health': self.health,
            'magicka': self.magicka,
            'stamina': self.stamina,
            'level': self.level,
            'position': self.position,
            'location': self.location_name,
            'time': self.time_of_day,
            'in_combat': self.in_combat,
            'enemies_nearby': self.enemies_nearby,
            'in_dialogue': self.in_dialogue,
            'in_menu': self.in_menu,
            'menu_type': self.menu_type,
            'nearby_npcs': self.nearby_npcs,
            'gold': self.gold,
            'quest_count': len(self.active_quests),
            'current_action_layer': self.current_action_layer,
            'available_actions': self.available_actions,
            'layer_transition_reason': self.layer_transition_reason,
        }


class SkyrimPerception:
    """
    Perception layer for Skyrim.
    """

    def __init__(
        self,
        vision_module=None,
        screen_region: Optional[Dict[str, int]] = None,
        use_game_api: bool = False
    ):
        self._vision_module = vision_module
        # Screen capture
        try:
            import mss
            self.sct = mss.mss()
        except ImportError:
            self.sct = None
        # Screen region
        if screen_region is None:
            if self.sct:
                self.screen_region = self.sct.monitors[1]
            else:
                self.screen_region = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        else:
            self.screen_region = screen_region
        # Game API
        self.use_game_api = use_game_api
        self._game_api = None
        # Scene classification candidates (improved for better accuracy)
        self.scene_candidates = [
            "outdoor wilderness with mountains, trees, and sky visible in first-person view",
            "city or town with stone buildings, NPCs walking, and cobblestone streets",
            "dark dungeon cave with stone walls, torches, and shadows - first person gameplay view",
            "indoor building interior with wooden furniture, fireplace, and NPCs - tavern or house",
            "active combat with ENEMY characters attacking the player, multiple enemy health bars visible, taking damage - NOT just player weapon visible in view",
            "dialogue conversation showing NPC face close-up with dialogue options at bottom of screen",
            "inventory menu UI showing item grid, character model on left, and equipment slots with detailed stats",
            "world map interface showing roads, cities, and location markers with compass rose",
        ]
        # Object detection candidates
        self.object_candidates = [
            "person", "warrior", "mage", "dragon", "guard",
            "sword", "bow", "staff", "potion", "chest",
            "door", "lever", "book", "gold", "armor",
        ]
        # Perception history
        self.perception_history: List[Dict[str, Any]] = []
        self._last_scene_type: SceneType = SceneType.UNKNOWN
        self._current_visual_embedding: Optional[np.ndarray] = None
        self._last_simulation_time: float = time.time()

        # Simulated state placeholders (used when HUD data unavailable)
        self._simulated_health: float = 100.0
        self._simulated_magicka: float = 100.0
        self._simulated_stamina: float = 100.0
        self._simulated_gold: int = 120
        self._simulated_enemy_count: int = 0
        self._simulated_progress: float = 0.0
        self._simulated_locations: int = 0
        self._simulated_quests: int = 0
        self._simulated_npcs_met: int = 0
        self._simulated_mechanics: int = 0
        self._simulated_equipment_quality: float = 0.35
        self._simulated_carry_weight: float = 160.0
        self._simulated_max_carry_weight: float = 300.0
        self._simulated_combat_score: float = 0.5
        self._simulated_stealth_score: float = 0.5
        self._simulated_social_score: float = 0.5
        self._simulated_player_level: int = 1
        self._simulated_skill_level: float = 15.0
        
        # Action affordance system
        self.affordance_system = ActionAffordanceSystem()
        
        # Current controller reference for layer awareness
        self._controller = None

        # Optional OCR-assisted HUD reader
        self.enhanced_vision: Optional[EnhancedVision] = None
        self.gemini_analyzer: Optional[Any] = None


    def set_enhanced_vision(self, enhanced_vision: EnhancedVision) -> None:
        """Attach enhanced vision helper."""

        self.enhanced_vision = enhanced_vision

    def set_gemini_analyzer(self, analyzer: Any) -> None:
        """Attach optional Gemini-based vision analyzer."""

        self.gemini_analyzer = analyzer

    def detect_collision(self, threshold: float = 0.01, window: int = 3) -> bool:
        """
        Detect visual collision by checking if the visual embedding has not changed significantly for several frames.
        Args:
            threshold: Cosine distance threshold for considering "no movement" (collision)
            window: Number of consecutive frames to check
        Returns:
            True if collision likely, False otherwise
        """
        if len(self.perception_history) < window:
            return False
        import numpy as np
        recent = self.perception_history[-window:]
        diffs = [
            np.linalg.norm(recent[i]['visual_embedding'] - recent[i-1]['visual_embedding'])
            for i in range(1, window)
        ]
        return all(d < threshold for d in diffs)

    def detect_visual_stuckness(self, window: int = 8, similarity_threshold: float = 0.9985) -> bool:
        """
        Check if visual embedding hasn't changed (stuck/collision).
        
        Args:
            window: Number of recent frames to check
            similarity_threshold: Cosine similarity threshold for "stuck"
            
        Returns:
            True if visually stuck, False otherwise
        """
        if len(self.perception_history) < window:
            return False
        
        recent = self.perception_history[-window:]
        embeddings = [p['visual_embedding'] for p in recent]
        
        # Check cosine similarity between consecutive frames
        similarities = []
        for i in range(1, len(embeddings)):
            # Compute cosine similarity (1 - cosine distance)
            dot_product = np.dot(embeddings[i-1], embeddings[i])
            norm_a = np.linalg.norm(embeddings[i-1])
            norm_b = np.linalg.norm(embeddings[i])
            
            if norm_a == 0 or norm_b == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm_a * norm_b)
            
            similarities.append(similarity)
        
        # If all very similar (>threshold), probably stuck
        # Also require minimum movement threshold to avoid false positives during menus/dialogue
        stuck = (all(s > similarity_threshold for s in similarities) and 
                len(similarities) >= 7)  # Need at least 7 consecutive similar frames
        
        if stuck:
            print(f"[VISUAL] Detected stuckness: similarities = {[f'{s:.4f}' for s in similarities]}")
        
        return stuck

    def set_controller(self, controller):
        """Set controller reference for layer awareness."""
        self._controller = controller

    def _initialize_game_api(self):
        """Initialize game state API (via mods)."""
        # This would connect to SKSE (Skyrim Script Extender) + Python bridge
        # For now, stub implementation
        print("Game API not yet implemented - using screen capture only")
        self._game_api = None

    def _ensure_vision_loaded(self):
        """Lazy load vision module."""
        if self._vision_module is None:
            from ..world_model import VisionModule
            self._vision_module = VisionModule(model_name="ViT-B/32")

    def capture_screen(self) -> Image.Image:
        """
        Capture current screen.

        Returns:
            PIL Image of screen
        """
        if self.sct is None:
            # Dummy image for testing
            return Image.new('RGB', (800, 600), color=(73, 109, 137))

        # Capture screen region
        screenshot = self.sct.grab(self.screen_region)

        # Convert to PIL Image
        img = Image.frombytes(
            'RGB',
            (screenshot.width, screenshot.height),
            screenshot.rgb
        )

        return img

    def classify_scene(self, image: Image.Image) -> Tuple[SceneType, Dict[str, float]]:
        """
        Classify scene type using CLIP.

        Args:
            image: Screen capture

        Returns:
            (scene_type, probabilities)
        """
        self._ensure_vision_loaded()

        # Zero-shot classification
        probs = self._vision_module.zero_shot_classify(
            image,
            candidates=self.scene_candidates
        )

        # Map to SceneType
        scene_mapping = {
            0: SceneType.OUTDOOR_WILDERNESS,
            1: SceneType.OUTDOOR_CITY,
            2: SceneType.INDOOR_DUNGEON,
            3: SceneType.INDOOR_BUILDING,
            4: SceneType.COMBAT,
            5: SceneType.DIALOGUE,
            6: SceneType.INVENTORY,
            7: SceneType.MAP,
        }

        # Get highest probability scene
        max_idx = max(range(len(self.scene_candidates)),
                     key=lambda i: probs[self.scene_candidates[i]])

        scene_type = scene_mapping.get(max_idx, SceneType.UNKNOWN)
        self._last_scene_type = scene_type

        return scene_type, probs

    def detect_objects(self, image: Image.Image, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Detect objects in scene using CLIP zero-shot.

        Args:
            image: Screen capture
            top_k: Number of top objects to return

        Returns:
            List of (object, confidence)
        """
        self._ensure_vision_loaded()

        # Zero-shot classification for objects
        probs = self._vision_module.zero_shot_classify(
            image,
            candidates=self.object_candidates
        )

        # Sort by probability
        sorted_objects = sorted(
            probs.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_objects[:top_k]

    def read_game_state(self, screenshot: Optional[Image.Image] = None) -> GameState:
        """
        Read current game state.

        Returns:
            GameState with current status
        """
        if self.use_game_api and self._game_api:
            return self._read_from_api()
        return self._read_from_screen(screenshot)

    def _read_from_api(self) -> GameState:
        """Read game state from API."""
        # Stub - would call SKSE Python bridge
        return GameState()

    def _read_from_screen(self, screenshot: Optional[Image.Image]) -> GameState:
        """Read game state from screen (heuristics with optional OCR)."""
        # Get current layer from controller if available
        current_layer = "Exploration"  # Default
        if self._controller and hasattr(self._controller, 'active_layer'):
            current_layer = self._controller.active_layer or "Exploration"
        
        hud_info: Dict[str, Any] = {}
        if self.enhanced_vision and screenshot is not None:
            hud_info = self.enhanced_vision.extract_hud_info(screenshot)

        # Enhanced Skyrim-specific state detection
        game_state_dict = self._detect_skyrim_state(hud_info)
        
        # Get available actions for current layer
        available_actions = self.affordance_system.get_available_actions(
            current_layer, 
            game_state_dict
        )
        
        return GameState(
            health=game_state_dict.get('health', 100.0),
            magicka=game_state_dict.get('magicka', 100.0),
            stamina=game_state_dict.get('stamina', 100.0),
            level=game_state_dict.get('level', 1),
            location_name=game_state_dict.get('location_name', "Skyrim"),
            gold=game_state_dict.get('gold', 100),
            in_combat=game_state_dict.get('in_combat', False),
            enemies_nearby=game_state_dict.get('enemies_nearby', 0),
            in_dialogue=game_state_dict.get('in_dialogue', False),
            in_menu=game_state_dict.get('in_menu', False),
            menu_type=game_state_dict.get('menu_type', ''),
            nearby_npcs=game_state_dict.get('nearby_npcs', []),
            current_action_layer=current_layer,
            available_actions=[a.name for a in available_actions],
            layer_transition_reason=game_state_dict.get('layer_transition_reason', "")
        )

    def _detect_skyrim_state(self, hud_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect Skyrim-specific game state from screen analysis.
        This would use OCR, color detection, and UI element recognition.
        """
        hud_info = hud_info or {}
        scene = self._last_scene_type

        inferred_in_menu, inferred_menu_type = self._detect_menu_state()
        scene_forces_menu = scene in (SceneType.INVENTORY, SceneType.MAP)
        in_menu = hud_info.get('in_menu', False) or inferred_in_menu or scene_forces_menu
        menu_type = hud_info.get('menu_type', inferred_menu_type)
        if scene_forces_menu:
            menu_type = 'inventory' if scene == SceneType.INVENTORY else 'map'

        in_dialogue = hud_info.get('in_dialogue', False) or scene == SceneType.DIALOGUE or self._detect_dialogue_state()
        hud_combat = hud_info.get('in_combat')
        in_combat = bool(hud_combat) if hud_combat is not None else (scene == SceneType.COMBAT or self._detect_combat_state())

        motion_score = self._estimate_motion_score(self._current_visual_embedding)
        allow_simulated_core = 'health_percent' not in hud_info
        self._update_simulated_state(motion_score, allow_core_stats=allow_simulated_core)

        if allow_simulated_core:
            health = self._simulated_health
            magicka = self._simulated_magicka
            stamina = self._simulated_stamina
        else:
            health = hud_info.get('health_percent', self._simulated_health)
            magicka = hud_info.get('magicka_percent', self._simulated_magicka)
            stamina = hud_info.get('stamina_percent', self._simulated_stamina)
            # Keep simulated values roughly aligned with actual data
            self._simulated_health = health
            self._simulated_magicka = magicka
            self._simulated_stamina = stamina

        location = hud_info.get('location') or self._detect_location()
        if not hud_info.get('location'):
            if scene == SceneType.OUTDOOR_CITY:
                location = "Whiterun"
            elif scene == SceneType.OUTDOOR_WILDERNESS:
                location = "Pine Forest"
            elif scene == SceneType.INDOOR_DUNGEON:
                location = "Nordic Ruin"
            elif scene == SceneType.INDOOR_BUILDING:
                location = "Tavern Interior"

        enemies_nearby = self._simulated_enemy_count if in_combat else max(0, min(3, int(round(motion_score * 2))))

        nearby_npc_count = 0
        if scene in (SceneType.OUTDOOR_CITY, SceneType.DIALOGUE, SceneType.INDOOR_BUILDING):
            nearby_npc_count = max(1, int(round(self._simulated_social_score * 3)))
        nearby_npcs = [f"NPC_{i+1}" for i in range(min(nearby_npc_count, 4))]

        state = {
            'health': health,
            'magicka': magicka,
            'stamina': stamina,
            'level': self._simulated_player_level,
            'average_skill_level': self._simulated_skill_level,
            'location_name': location,
            'gold': self._simulated_gold,
            'in_combat': in_combat,
            'enemies_nearby': enemies_nearby,
            'in_dialogue': in_dialogue,
            'in_menu': in_menu,
            'menu_type': menu_type,
            'nearby_npcs': nearby_npcs,
            'layer_transition_reason': self._determine_layer_transition_reason(in_combat, in_menu, in_dialogue),
            'scene': scene.value,
            'movement_score': motion_score,
            'completed_quests': self._simulated_quests,
            'locations_discovered': self._simulated_locations,
            'npcs_met': self._simulated_npcs_met,
            'mechanics_learned': self._simulated_mechanics,
            'equipment_quality': self._simulated_equipment_quality,
            'carry_weight': self._simulated_carry_weight,
            'max_carry_weight': self._simulated_max_carry_weight,
            'combat_win_rate': self._simulated_combat_score,
            'stealth_success_rate': self._simulated_stealth_score,
            'persuasion_success_rate': self._simulated_social_score,
        }

        return state

    def _detect_menu_state(self) -> Tuple[bool, str]:
        """
        Detect if currently in a menu and which menu type.
        
        Returns:
            (in_menu, menu_type) where menu_type is one of: 'inventory', 'map', 'magic', 'skills', ''
        """
        # TODO: Implement actual menu detection using screen analysis
        # Would look for inventory UI, map UI, skills UI, etc.
        # For now, infer from recent scene classification if available
        if len(self.perception_history) > 0:
            last_scene = self.perception_history[-1].get('scene_type', SceneType.UNKNOWN)
            if last_scene == SceneType.INVENTORY:
                return True, 'inventory'
            elif last_scene == SceneType.MAP:
                return True, 'map'
        return False, ''
    
    def _detect_dialogue_state(self) -> bool:
        """Detect if currently in dialogue with an NPC."""
        # TODO: Implement actual dialogue detection using screen analysis
        # Would look for dialogue UI, conversation options, NPC portraits
        # For now, infer from recent scene classification
        if len(self.perception_history) > 0:
            last_scene = self.perception_history[-1].get('scene_type', SceneType.UNKNOWN)
            if last_scene == SceneType.DIALOGUE:
                return True
        return False

    def _detect_combat_state(self) -> bool:
        """Detect if currently in combat (would analyze combat UI)."""
        # TODO: Implement actual combat detection
        # Would look for:
        # - Red enemy health bars
        # - Combat music indicators
        # - Weapon drawn state
        # - Enemy targeting reticles
        
        # For now, default to NOT in combat
        # Real implementation would analyze screen for combat indicators
        return False

    def _detect_location(self) -> str:
        """Detect current location (would use OCR on location text)."""
        # TODO: Implement actual location detection using OCR
        # Would read the location text that appears when entering new areas
        
        # For now, return a stable default location
        # Real implementation would use OCR to read location name from screen
        return "Skyrim"

    def _detect_nearby_npcs(self) -> List[str]:
        """Detect nearby NPCs (would analyze screen for NPC indicators)."""
        # TODO: Implement actual NPC detection
        # Would look for:
        # - NPC name tags
        # - Character models
        # - Dialogue prompts
        
        # For now, return empty list
        # Real implementation would detect NPC name tags on screen
        return []

    def _estimate_motion_score(self, current_embedding: Optional[np.ndarray], window: int = 4) -> float:
        """Estimate how much the scene is changing using visual embeddings."""
        if current_embedding is None or not self.perception_history:
            return 0.5

        similarities: List[float] = []
        for past in self.perception_history[-window:]:
            prev_embedding = past.get('visual_embedding')
            if prev_embedding is None:
                continue
            denom = np.linalg.norm(prev_embedding) * np.linalg.norm(current_embedding)
            if denom == 0:
                continue
            similarity = float(np.dot(prev_embedding, current_embedding) / denom)
            similarities.append(similarity)

        if not similarities:
            return 0.5

        avg_similarity = sum(similarities) / len(similarities)
        avg_similarity = max(-1.0, min(1.0, avg_similarity))
        motion = max(0.0, min(1.0, 1.0 - avg_similarity))
        return motion

    def _update_simulated_state(self, motion_score: float, allow_core_stats: bool) -> None:
        """Update simulated game metrics when HUD data is missing."""
        now = time.time()
        elapsed = max(0.1, min(5.0, now - self._last_simulation_time))
        self._last_simulation_time = now

        scene = self._last_scene_type

        if allow_core_stats:
            if scene == SceneType.COMBAT:
                damage_factor = (0.6 + 0.8 * (1.0 - motion_score)) * elapsed
                self._simulated_health = max(20.0, self._simulated_health - damage_factor * 3.5)
                self._simulated_stamina = max(5.0, self._simulated_stamina - damage_factor * 4.5)
                self._simulated_magicka = max(5.0, self._simulated_magicka - damage_factor * 2.0)
                self._simulated_enemy_count = max(1, min(5, int(round(1 + motion_score * 3))))
            else:
                regen_factor = (0.4 + 0.6 * motion_score) * elapsed
                self._simulated_health = min(100.0, self._simulated_health + regen_factor * 3.0)
                self._simulated_stamina = min(100.0, self._simulated_stamina + (2.8 - motion_score) * elapsed * 2.0)
                self._simulated_magicka = min(100.0, self._simulated_magicka + 2.2 * elapsed)
                self._simulated_enemy_count = 0
        else:
            if scene == SceneType.COMBAT:
                self._simulated_enemy_count = max(1, min(5, int(round(1 + motion_score * 3))))
            else:
                self._simulated_enemy_count = 0

        if scene in (SceneType.OUTDOOR_WILDERNESS, SceneType.INDOOR_DUNGEON):
            if motion_score > 0.25:
                self._simulated_progress = min(1.0, self._simulated_progress + 0.002 * elapsed * (0.6 + motion_score))
                self._simulated_locations = min(343, self._simulated_locations + int(max(0, elapsed // 1)))
                self._simulated_equipment_quality = min(1.0, self._simulated_equipment_quality + 0.0008 * elapsed)
                self._simulated_gold = min(20000, self._simulated_gold + int(3 * elapsed * (0.5 + motion_score)))
        elif scene == SceneType.OUTDOOR_CITY:
            if motion_score > 0.2:
                self._simulated_social_score = min(0.95, self._simulated_social_score + 0.012 * elapsed)
                self._simulated_npcs_met = min(100, self._simulated_npcs_met + int(max(0, elapsed // 1)))
                self._simulated_progress = min(1.0, self._simulated_progress + 0.001 * elapsed)
        elif scene == SceneType.INDOOR_BUILDING:
            if motion_score > 0.2:
                self._simulated_mechanics = min(50, self._simulated_mechanics + int(max(0, elapsed // 1)))
                self._simulated_progress = min(1.0, self._simulated_progress + 0.0012 * elapsed)
        elif scene == SceneType.DIALOGUE:
            self._simulated_social_score = min(0.96, self._simulated_social_score + 0.02 * elapsed)
            self._simulated_npcs_met = min(100, self._simulated_npcs_met + int(max(1, elapsed // 0.5)))
        elif scene == SceneType.MAP:
            self._simulated_progress = min(1.0, self._simulated_progress + 0.0008 * elapsed)

        self._simulated_carry_weight = max(60.0, min(
            self._simulated_max_carry_weight,
            self._simulated_carry_weight + (1.5 if scene == SceneType.OUTDOOR_WILDERNESS else -0.8) * elapsed
        ))

        if scene == SceneType.COMBAT:
            self._simulated_combat_score = max(0.25, min(0.95, self._simulated_combat_score + (0.04 if motion_score > 0.4 else -0.05)))
            self._simulated_stealth_score = max(0.3, self._simulated_stealth_score - 0.01)
        else:
            self._simulated_combat_score = min(0.9, self._simulated_combat_score + 0.01)
            self._simulated_stealth_score = min(0.9, self._simulated_stealth_score + 0.005)

        self._simulated_progress = max(0.0, min(1.0, self._simulated_progress))
        self._simulated_player_level = max(1, min(81, int(1 + self._simulated_progress * 40)))
        self._simulated_skill_level = min(100.0, 15.0 + self._simulated_progress * 70.0)
        self._simulated_quests = min(100, max(self._simulated_quests, int(self._simulated_progress * 60)))
        self._simulated_mechanics = min(50, self._simulated_mechanics)
        self._simulated_locations = min(343, self._simulated_locations)
        self._simulated_npcs_met = min(100, self._simulated_npcs_met)
        self._simulated_social_score = max(0.3, min(0.96, self._simulated_social_score))

    def _determine_layer_transition_reason(self, in_combat: bool, in_menu: bool, in_dialogue: bool = False) -> str:
        """Determine why a layer transition might be needed."""
        if in_combat:
            return "Combat detected - consider Combat layer"
        elif in_dialogue:
            return "Dialogue detected - consider Dialogue layer"
        elif in_menu:
            return "Menu open - consider Menu layer"
        else:
            return ""

    async def perceive(self) -> Dict[str, Any]:
        """
        Complete perception cycle.

        Returns:
            Dict with:
                - visual_embedding: CLIP embedding of screen
                - scene_type: Classified scene type
                - scene_probs: Scene probabilities
                - objects: Detected objects
                - game_state: Current game state
                - timestamp: Time of perception
        """
        timestamp = time.time()

        # 1. Capture screen
        screen = self.capture_screen()

        # 2. Encode with CLIP
        self._ensure_vision_loaded()
        visual_embedding = self._vision_module.encode_image(screen)
        self._current_visual_embedding = visual_embedding

        # 3. Classify scene
        scene_type, scene_probs = self.classify_scene(screen)

        # 4. Detect objects
        objects = self.detect_objects(screen, top_k=5)

        # 5. Read game state
        game_state = self.read_game_state(screen)

        # 6. Package perception
        perception = {
            'visual_embedding': visual_embedding,
            'scene_type': scene_type,
            'scene_probs': scene_probs,
            'objects': objects,
            'game_state': game_state,
            'timestamp': timestamp,
            'screenshot': screen,  # Include screenshot for VL models
        }

        # 7. Add to history
        self.perception_history.append(perception)
        if len(self.perception_history) > 100:
            self.perception_history = self.perception_history[-100:]

        return perception

    def get_temporal_context(self, window: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent perception history.

        Args:
            window: Number of recent perceptions

        Returns:
            List of recent perceptions
        """
        return self.perception_history[-window:]

    def detect_change(self) -> Dict[str, Any]:
        """
        Detect significant changes in perception.

        Returns:
            Dict with detected changes
        """
        if len(self.perception_history) < 2:
            return {'changed': False}

        prev = self.perception_history[-2]
        curr = self.perception_history[-1]

        changes = {
            'changed': False,
            'scene_changed': prev['scene_type'] != curr['scene_type'],
            'combat_started': (not prev['game_state'].in_combat and
                             curr['game_state'].in_combat),
            'combat_ended': (prev['game_state'].in_combat and
                           not curr['game_state'].in_combat),
            'layer_changed': (prev['game_state'].current_action_layer != 
                            curr['game_state'].current_action_layer),
            'actions_changed': (set(prev['game_state'].available_actions) != 
                              set(curr['game_state'].available_actions)),
        }

        changes['changed'] = any([
            changes['scene_changed'],
            changes['combat_started'],
            changes['combat_ended'],
            changes['layer_changed'],
            changes['actions_changed']
        ])

        return changes

    def get_stats(self) -> Dict[str, Any]:
        """Get perception statistics."""
        return {
            'screen_region': self.screen_region,
            'using_game_api': self.use_game_api,
            'perception_history_size': len(self.perception_history),
            'vision_loaded': self._vision_module is not None,
            'mss_available': MSS_AVAILABLE,
        }


# Example usage
if __name__ == "__main__":
    print("Testing Skyrim Perception...")

    perception = SkyrimPerception()

    # Test screen capture
    print("\n1. Capturing screen...")
    screen = perception.capture_screen()
    print(f"   ✓ Captured: {screen.size}")

    # Test perception cycle
    print("\n2. Running perception cycle...")
    import asyncio

    async def test():
        result = await perception.perceive()
        print(f"   Scene type: {result['scene_type'].value}")
        print(f"   Top objects: {result['objects'][:3]}")
        print(f"   Game state: {result['game_state'].to_dict()}")

    asyncio.run(test())

    # Stats
    print(f"\n3. Stats: {perception.get_stats()}")

    print("\n✓ Perception tests complete")
