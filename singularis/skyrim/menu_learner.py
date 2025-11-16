"""
Menu Interaction Learner

Learns how to navigate and interact with game menus through experience:
1. Tracks menu states and transitions
2. Learns which actions work in which menus
3. Builds a mental model of menu structure
4. Optimizes menu navigation paths

Philosophical grounding:
- Learning through interaction (enactive cognition)
- Building adequate ideas of menu affordances
- Increasing agency through understanding
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time


@dataclass
class MenuState:
    """Represents a snapshot of a specific menu screen.

    Attributes:
        menu_type: The type of menu (e.g., 'inventory', 'map').
        timestamp: The time the menu was entered.
        actions_available: A list of actions that were available in this menu.
        successful_actions: A list of actions that were successfully performed
                            from this menu state.
    """
    menu_type: str
    timestamp: float
    actions_available: List[str]
    successful_actions: List[str]


@dataclass
class MenuTransition:
    """Represents a learned transition between two menu states.

    Attributes:
        from_menu: The starting menu type.
        to_menu: The resulting menu type.
        action: The action that caused the transition.
        success: Whether the transition was successful.
        duration: The time spent in the 'from_menu' before transitioning.
    """
    from_menu: str
    to_menu: str
    action: str
    success: bool
    duration: float


class MenuLearner:
    """Learns to navigate and interact with game menus through experience.

    This class builds a model of the game's menu system by observing which
    actions lead to which menu states. It tracks the success rate of different
    actions within each menu and can use this learned graph to find optimal
    paths to achieve menu-based goals (e.g., equipping an item).
    """
    
    def __init__(self):
        """Initializes the MenuLearner."""
        # Menu state history
        self.menu_history: List[MenuState] = []
        
        # Transition graph: {from_menu: {action: to_menu}}
        self.transition_graph: Dict[str, Dict[str, str]] = defaultdict(dict)
        
        # Action success rates: {menu_type: {action: success_rate}}
        self.action_success: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        
        # Action attempt counts: {menu_type: {action: count}}
        self.action_attempts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        
        # Current menu state
        self.current_menu: Optional[str] = None
        self.menu_entry_time: float = 0.0
        
        # Learned menu structures
        self.menu_structures: Dict[str, Dict[str, Any]] = {
            'inventory': {
                'purpose': 'Manage items and equipment',
                'common_actions': ['equip', 'drop', 'use', 'exit'],
                'navigation': []
            },
            'map': {
                'purpose': 'View world and set markers',
                'common_actions': ['zoom', 'marker', 'fast_travel', 'exit'],
                'navigation': []
            },
            'skills': {
                'purpose': 'Level up and manage perks',
                'common_actions': ['select_perk', 'exit'],
                'navigation': []
            },
            'magic': {
                'purpose': 'Equip spells',
                'common_actions': ['equip_spell', 'exit'],
                'navigation': []
            }
        }
        
        print("[MENU] Menu Learner initialized")
    
    def enter_menu(self, menu_type: str, available_actions: List[str]):
        """Records that the agent has entered a menu.

        Args:
            menu_type: The type of menu that was entered (e.g., 'inventory').
            available_actions: A list of actions available in this menu.
        """
        self.current_menu = menu_type
        self.menu_entry_time = time.time()
        
        menu_state = MenuState(
            menu_type=menu_type,
            timestamp=self.menu_entry_time,
            actions_available=available_actions,
            successful_actions=[]
        )
        self.menu_history.append(menu_state)
        
        print(f"[MENU] Entered {menu_type} menu")
        print(f"[MENU] Available actions: {available_actions}")
    
    def record_action(
        self,
        action: str,
        success: bool,
        resulted_in_menu: Optional[str] = None
    ):
        """Records an action performed within a menu and updates the learned model.

        This method updates action success rates and, if the menu changes,
        updates the transition graph.

        Args:
            action: The action that was taken.
            success: Whether the action was successful.
            resulted_in_menu: The new menu type if the action caused a
                              transition, otherwise None.
        """
        if not self.current_menu:
            return
        
        # Update action statistics
        self.action_attempts[self.current_menu][action] += 1
        
        current_success = self.action_success[self.current_menu][action]
        attempts = self.action_attempts[self.current_menu][action]
        
        # Update success rate (running average)
        new_success = (current_success * (attempts - 1) + (1.0 if success else 0.0)) / attempts
        self.action_success[self.current_menu][action] = new_success
        
        # Record successful action in current menu state
        if success and self.menu_history:
            self.menu_history[-1].successful_actions.append(action)
        
        # Record transition if menu changed
        if resulted_in_menu and resulted_in_menu != self.current_menu:
            duration = time.time() - self.menu_entry_time
            
            transition = MenuTransition(
                from_menu=self.current_menu,
                to_menu=resulted_in_menu,
                action=action,
                success=success,
                duration=duration
            )
            
            # Update transition graph
            self.transition_graph[self.current_menu][action] = resulted_in_menu
            
            print(f"[MENU] Learned transition: {self.current_menu} --[{action}]--> {resulted_in_menu}")
            
            # Update current menu
            self.current_menu = resulted_in_menu
            self.menu_entry_time = time.time()
    
    def exit_menu(self):
        """Records that the agent has exited the menu system."""
        if self.current_menu:
            duration = time.time() - self.menu_entry_time
            print(f"[MENU] Exited {self.current_menu} (duration: {duration:.1f}s)")
            self.current_menu = None
    
    def get_recommended_actions(self, menu_type: str) -> List[Tuple[str, float]]:
        """Gets a list of recommended actions for a given menu type, sorted by learned
        success rate.

        Args:
            menu_type: The type of menu to get recommendations for.

        Returns:
            A list of (action, success_rate) tuples, sorted from highest to
            lowest success rate.
        """
        if menu_type not in self.action_success:
            # No experience with this menu, return common actions
            if menu_type in self.menu_structures:
                common = self.menu_structures[menu_type]['common_actions']
                return [(action, 0.5) for action in common]
            return []
        
        # Sort actions by success rate
        actions = self.action_success[menu_type]
        sorted_actions = sorted(
            actions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_actions
    
    def get_menu_path(self, from_menu: str, to_menu: str) -> Optional[List[str]]:
        """Finds the shortest sequence of actions to navigate from one menu to another.

        This uses a Breadth-First Search (BFS) on the learned transition graph.

        Args:
            from_menu: The starting menu type.
            to_menu: The target menu type.

        Returns:
            A list of action strings representing the path, or None if no path
            has been learned.
        """
        # Simple BFS to find path
        from collections import deque
        
        queue = deque([(from_menu, [])])
        visited = {from_menu}
        
        while queue:
            current, path = queue.popleft()
            
            if current == to_menu:
                return path
            
            # Explore transitions from current menu
            if current in self.transition_graph:
                for action, next_menu in self.transition_graph[current].items():
                    if next_menu not in visited:
                        visited.add(next_menu)
                        queue.append((next_menu, path + [action]))
        
        return None  # No path found
    
    def suggest_menu_action(
        self,
        menu_type: str,
        goal: str = 'explore'
    ) -> Optional[str]:
        """Suggests the best single action to take within a menu, given a high-level goal.

        Args:
            menu_type: The current menu type.
            goal: The agent's current goal (e.g., 'exit', 'explore').

        Returns:
            The name of the suggested action, or None if no suitable action is found.
        """
        # Get recommended actions
        recommendations = self.get_recommended_actions(menu_type)
        
        if not recommendations:
            return None
        
        # Filter by goal
        if goal == 'exit':
            # Prioritize exit actions
            for action, rate in recommendations:
                if 'exit' in action.lower() or 'back' in action.lower():
                    return action
        elif goal == 'explore':
            # Try actions we haven't tried much
            for action, rate in recommendations:
                attempts = self.action_attempts[menu_type][action]
                if attempts < 3:  # Explore less-tried actions
                    return action
        
        # Default: return highest success rate action
        return recommendations[0][0]
    
    def get_menu_knowledge(self, menu_type: str) -> Dict[str, Any]:
        """Retrieves all learned knowledge about a specific menu type.

        Args:
            menu_type: The menu to query.

        Returns:
            A dictionary summarizing the learned information, including visit
            count, learned actions, and transitions.
        """
        knowledge = {
            'menu_type': menu_type,
            'times_visited': len([m for m in self.menu_history if m.menu_type == menu_type]),
            'actions_learned': len(self.action_success.get(menu_type, {})),
            'successful_actions': [],
            'transitions_from': {},
            'average_success_rate': 0.0
        }
        
        # Get successful actions
        if menu_type in self.action_success:
            for action, rate in self.action_success[menu_type].items():
                if rate > 0.7:  # Consider >70% success as "learned"
                    knowledge['successful_actions'].append(action)
        
        # Get transitions
        if menu_type in self.transition_graph:
            knowledge['transitions_from'] = dict(self.transition_graph[menu_type])
        
        # Calculate average success rate
        if menu_type in self.action_success:
            rates = list(self.action_success[menu_type].values())
            if rates:
                knowledge['average_success_rate'] = sum(rates) / len(rates)
        
        return knowledge
    
    def get_stats(self) -> Dict[str, Any]:
        """Retrieves statistics about the menu learning process.

        Returns:
            A dictionary of statistics, including the number of menus explored,
            total actions taken, and transitions learned.
        """
        total_actions = sum(
            sum(attempts.values())
            for attempts in self.action_attempts.values()
        )
        
        menus_explored = len(self.action_success)
        
        return {
            'menus_explored': menus_explored,
            'total_menu_actions': total_actions,
            'menu_visits': len(self.menu_history),
            'transitions_learned': sum(
                len(transitions) for transitions in self.transition_graph.values()
            ),
            'current_menu': self.current_menu
        }
    
    def print_menu_graph(self):
        """Prints a human-readable representation of the learned menu transition graph to the console."""
        print("\n[MENU] Learned Menu Transition Graph:")
        for from_menu, transitions in self.transition_graph.items():
            for action, to_menu in transitions.items():
                success_rate = self.action_success[from_menu].get(action, 0.0)
                print(f"  {from_menu} --[{action} ({success_rate:.0%})]--> {to_menu}")
