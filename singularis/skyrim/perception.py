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
        # Scene classification candidates
        self.scene_candidates = [
            "outdoor wilderness with mountains and trees",
            "city or town with buildings and NPCs",
            "dark dungeon or cave interior",
            "indoor building like tavern or house",
            "combat scene with enemies fighting",
            "dialogue conversation with NPC",
            "inventory menu screen",
            "map view showing locations",
        ]
        # Object detection candidates
        self.object_candidates = [
            "person", "warrior", "mage", "dragon", "guard",
            "sword", "bow", "staff", "potion", "chest",
            "door", "lever", "book", "gold", "armor",
        ]
        # Perception history
        self.perception_history: List[Dict[str, Any]] = []
        
        # Action affordance system
        self.affordance_system = ActionAffordanceSystem()
        
        # Current controller reference for layer awareness
        self._controller = None


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

    def read_game_state(self) -> GameState:
        """
        Read current game state.

        Returns:
            GameState with current status
        """
        if self.use_game_api and self._game_api:
            # Read from game API
            return self._read_from_api()
        else:
            # Read from screen (OCR or heuristics)
            return self._read_from_screen()

    def _read_from_api(self) -> GameState:
        """Read game state from API."""
        # Stub - would call SKSE Python bridge
        return GameState()

    def _read_from_screen(self) -> GameState:
        """Read game state from screen (heuristics with Skyrim-specific detection)."""
        # Get current layer from controller if available
        current_layer = "Exploration"  # Default
        if self._controller and hasattr(self._controller, 'active_layer'):
            current_layer = self._controller.active_layer or "Exploration"
        
        # Enhanced Skyrim-specific state detection
        game_state_dict = self._detect_skyrim_state()
        
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

    def _detect_skyrim_state(self) -> Dict[str, Any]:
        """
        Detect Skyrim-specific game state from screen analysis.
        This would use OCR, color detection, and UI element recognition.
        """
        # Detect if in menu (would check for UI elements)
        in_menu, menu_type = self._detect_menu_state()
        
        # Detect dialogue state
        in_dialogue = self._detect_dialogue_state()
        
        # Detect combat state (would check for combat UI, red health bars, etc.)
        in_combat = self._detect_combat_state()
        
        # Detect location (would use OCR on location text)
        location = self._detect_location()
        
        # Use more stable state values (would come from actual game state reading)
        state = {
            'health': 100.0,  # Would read from health bar
            'magicka': 100.0,  # Would read from magicka bar
            'stamina': 100.0,  # Would read from stamina bar
            'level': 1,  # Would read from character stats
            'location_name': location,
            'gold': 100,  # Would read from inventory
            'in_combat': in_combat,
            'enemies_nearby': 0,  # Would detect from enemy health bars
            'in_dialogue': in_dialogue,
            'in_menu': in_menu,
            'menu_type': menu_type,
            'nearby_npcs': self._detect_nearby_npcs(),
            'layer_transition_reason': self._determine_layer_transition_reason(in_combat, in_menu, in_dialogue)
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

        # 3. Classify scene
        scene_type, scene_probs = self.classify_scene(screen)

        # 4. Detect objects
        objects = self.detect_objects(screen, top_k=5)

        # 5. Read game state
        game_state = self.read_game_state()

        # 6. Package perception
        perception = {
            'visual_embedding': visual_embedding,
            'scene_type': scene_type,
            'scene_probs': scene_probs,
            'objects': objects,
            'game_state': game_state,
            'timestamp': timestamp,
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
