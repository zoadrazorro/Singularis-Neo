"""
Inventory and equipment management for Skyrim AGI.

Tracks approximate inventory state, recommends context-aware loadout changes,
and handles consumable usage decisions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional


class InventoryManager:
    """Provides a lightweight, heuristic-based system for managing inventory.

    This class tracks the agent's approximate inventory state, including items,
    weight, and equipped gear. It offers recommendations for equipping items,
    using consumables, and optimizing loadouts for different situations (e.g.,
    combat, stealth) without needing a perfect, real-time inventory representation.
    """

    def __init__(self, weight_limit: float = 300.0) -> None:
        """Initializes the InventoryManager.

        Args:
            weight_limit: The initial assumption for the agent's carry weight limit.
        """
        self.weight_limit = weight_limit
        self.current_weight: float = 0.0
        self.inventory: Dict[str, Dict[str, Any]] = {}
        self.equipped: Dict[str, str] = {}
        self.value_threshold: float = 50.0
        self._recent_consumables: Dict[str, int] = defaultdict(int)

    def update_from_state(self, state: Dict[str, Any]) -> None:
        """Updates the internal inventory representation from a game state dictionary.

        Args:
            state: A dictionary containing game state information, which may
                   include 'carry_weight', 'equipped_items', and 'inventory_items'.
        """
        self.current_weight = state.get("carry_weight", self.current_weight)
        self.weight_limit = state.get("max_carry_weight", self.weight_limit)
        equipment = state.get("equipped_items", {})
        if isinstance(equipment, dict):
            self.equipped.update(equipment)
        inventory_items = state.get("inventory_items", [])
        if isinstance(inventory_items, list):
            for item in inventory_items:
                if isinstance(item, dict) and "name" in item:
                    self.inventory[item["name"]] = item

    def recommend_menu_action(self, situation: str, available_actions: List[str]) -> Optional[str]:
        """Recommends a high-level menu action based on the current situation.

        Args:
            situation: A string describing the current context (e.g., "combat", "healing").
            available_actions: A list of possible menu actions.

        Returns:
            The recommended action string (e.g., "equip_item"), or None.
        """
        if situation == "combat" and "equip_item" in available_actions:
            return "equip_item"
        if situation == "healing" and "consume_item" in available_actions:
            return "consume_item"
        if situation == "overweight" and "drop_item" in available_actions:
            return "drop_item"
        return None

    def should_use_consumable(self, state: Dict[str, Any]) -> Optional[str]:
        """Determines if a consumable item should be used based on the agent's status.

        Args:
            state: A dictionary representing the current game state, including
                   'health', 'stamina', 'magicka', and 'in_combat'.

        Returns:
            The type of potion to use (e.g., "health_potion"), or None if no
            consumable is immediately needed.
        """
        health = state.get("health", 100)
        stamina = state.get("stamina", 100)
        magicka = state.get("magicka", 100)
        in_combat = state.get("in_combat", False)
        if health < 35:
            return "health_potion"
        if in_combat and stamina < 25:
            return "stamina_potion"
        if in_combat and magicka < 25:
            return "magicka_potion"
        return None

    def record_consumable_use(self, item_name: str) -> None:
        """Records that a consumable item has been used.

        Args:
            item_name: The name of the item that was used.
        """
        self._recent_consumables[item_name] += 1

    def optimize_loadout(self, situation: str) -> Dict[str, Any]:
        """Recommends an optimized loadout for a given situation.

        Args:
            situation: The current context (e.g., "combat", "stealth", "dungeon").

        Returns:
            A dictionary with recommendations, such as a "preferred_weapon"
            or "preferred_tool".
        """
        if situation == "combat":
            return {"preferred_weapon": self._find_highest_value_item("weapon")}
        if situation == "stealth":
            return {"preferred_weapon": self._find_lightest_weapon()}
        if situation == "dungeon":
            return {"preferred_tool": self._find_item_with_keyword("torch")}
        return {}

    def _find_highest_value_item(self, item_type: str) -> Optional[str]:
        """Finds the most valuable item of a given type in the inventory.

        Args:
            item_type: The type of item to search for (e.g., "weapon").

        Returns:
            The name of the best item, or None if no items of that type are found.
        """
        best_item = None
        best_value = -1
        for item in self.inventory.values():
            if item.get("type") == item_type:
                value = item.get("value", 0)
                if value > best_value:
                    best_value = value
                    best_item = item.get("name")
        return best_item

    def _find_lightest_weapon(self) -> Optional[str]:
        """Finds the lightest weapon in the inventory, suitable for stealth.

        Returns:
            The name of the lightest weapon, or None if no weapons are found.
        """
        best_item = None
        best_weight = float("inf")
        for item in self.inventory.values():
            if item.get("type") == "weapon":
                weight = item.get("weight", 0)
                if weight < best_weight:
                    best_weight = weight
                    best_item = item.get("name")
        return best_item

    def _find_item_with_keyword(self, keyword: str) -> Optional[str]:
        """Finds an item in the inventory whose name contains a given keyword.

        Args:
            keyword: The keyword to search for (case-insensitive).

        Returns:
            The name of the first matching item, or None.
        """
        keyword_lower = keyword.lower()
        for item in self.inventory.values():
            if keyword_lower in item.get("name", "").lower():
                return item.get("name")
        return None
