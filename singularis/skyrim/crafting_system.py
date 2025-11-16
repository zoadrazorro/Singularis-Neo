"""
Crafting and enchanting helper for Skyrim AGI.

Provides lightweight heuristics for determining when crafting is worthwhile
and which enchantments best suit the agent's current build.
"""

from __future__ import annotations

from typing import Dict, Optional


class CraftingSystem:
    """Provides a simple system for making crafting and enchanting decisions.

    This class uses heuristics to determine whether crafting an item is
    economically viable and to select the best enchantment for an item based
    on the agent's character build.
    """

    def __init__(self) -> None:
        """Initializes the CraftingSystem.

        Sets up predefined known recipes and enchantment preferences for
        different character builds (warrior, mage, thief, hybrid).
        """
        self.known_recipes: Dict[str, Dict[str, float]] = {
            "steel_ingot": {"value": 25.0},
            "iron_dagger": {"value": 20.0},
        }
        self.enchantment_preferences: Dict[str, Dict[str, float]] = {
            "warrior": {"Fortify One-Handed": 0.8, "Fortify Stamina": 0.6},
            "mage": {"Fortify Destruction": 0.8, "Fortify Magicka": 0.7},
            "thief": {"Fortify Sneak": 0.8, "Fortify Carry Weight": 0.5},
            "hybrid": {"Fortify Health": 0.6, "Fortify Stamina": 0.6},
        }

    def should_craft_item(self, recipe: Dict[str, any], materials: Dict[str, int]) -> bool:
        """Determines if crafting an item is worth the material cost.

        Args:
            recipe: A dictionary describing the item to craft, including its
                    materials and base value.
            materials: A dictionary of available materials and their quantities.

        Returns:
            True if the estimated value of the item is at least 30% greater
            than the cost of its materials, False otherwise.
        """
        value = self._estimate_item_value(recipe)
        cost = self._estimate_material_cost(recipe, materials)
        if cost == 0:
            return False
        return value > cost * 1.3

    def _estimate_item_value(self, recipe: Dict[str, any]) -> float:
        """Estimates the value of an item from a recipe.

        Args:
            recipe: The recipe dictionary for the item.

        Returns:
            The known value if the recipe is recognized, otherwise the base
            value from the recipe.
        """
        name = recipe.get("name")
        if name and name in self.known_recipes:
            return self.known_recipes[name].get("value", 0.0)
        return recipe.get("base_value", 0.0)

    def _estimate_material_cost(self, recipe: Dict[str, any], materials: Dict[str, int]) -> float:
        """Estimates the cost of materials for a recipe.

        If the agent does not have enough materials, the cost is infinite.

        Args:
            recipe: The recipe dictionary.
            materials: A dictionary of available materials.

        Returns:
            The total estimated cost of materials, or infinity if materials
            are insufficient.
        """
        cost = 0.0
        for material, qty in recipe.get("materials", {}).items():
            stock = materials.get(material, 0)
            if stock < qty:
                return float("inf")
            cost += qty * 5.0  # Assume a base value of 5 for all materials
        return cost

    def select_enchantment(self, item_type: str, build: str) -> Optional[str]:
        """Selects the best enchantment for an item based on the character build.

        Args:
            item_type: The type of item to be enchanted (e.g., 'weapon', 'armor').
                       (Note: This parameter is not used in the current implementation).
            build: The character build (e.g., 'warrior', 'mage').

        Returns:
            The name of the most preferred enchantment, or None if no suitable
            enchantment is found.
        """
        build_preferences = self.enchantment_preferences.get(build, {})
        best_enchantment = None
        best_score = -1.0
        for enchantment, score in build_preferences.items():
            if score > best_score:
                best_score = score
                best_enchantment = enchantment
        return best_enchantment
