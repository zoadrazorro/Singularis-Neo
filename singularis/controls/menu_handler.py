"""MenuHandler: prevents getting stuck in menus/dialogues."""

from typing import Dict, Any, Optional
from .action_space import HighLevelAction


class MenuHandler:
    """
    Anti-stuck layer for menus and dialogues.
    
    Skyrim menus can soft-lock the AI if not handled properly.
    This ensures menus are exited quickly when not needed.
    """
    
    def __init__(self, max_menu_time: float = 3.0):
        """
        Initialize menu handler.
        
        Args:
            max_menu_time: Max seconds to stay in menu before forcing exit
        """
        self.max_menu_time = max_menu_time
        self.menu_enter_time = None
        self.last_was_menu = False
        
        self.stats = {
            'menu_exits': 0,
            'forced_exits': 0,
        }
    
    def handle(self, game_state: Dict[str, Any]) -> Optional[HighLevelAction]:
        """
        Check if menu action needed.
        
        Args:
            game_state: Current game state
            
        Returns:
            CLOSE_MENU if stuck in menu, None otherwise
        """
        import time
        
        in_menu = game_state.get('in_menu', False)
        in_dialogue = game_state.get('in_dialogue', False)
        
        # Track menu entry
        if (in_menu or in_dialogue) and not self.last_was_menu:
            self.menu_enter_time = time.time()
        
        self.last_was_menu = in_menu or in_dialogue
        
        # If in menu/dialogue, suggest closing it
        if in_menu or in_dialogue:
            # Check if we've been stuck too long
            if self.menu_enter_time is not None:
                time_in_menu = time.time() - self.menu_enter_time
                if time_in_menu > self.max_menu_time:
                    print(f"[MENU] Forcing exit after {time_in_menu:.1f}s")
                    self.stats['forced_exits'] += 1
                    self.menu_enter_time = None
                    return HighLevelAction.CLOSE_MENU
            
            # Normal menu exit
            self.stats['menu_exits'] += 1
            return HighLevelAction.CLOSE_MENU
        
        # Not in menu
        self.menu_enter_time = None
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get menu handler statistics."""
        return self.stats.copy()
