import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from core.agent_base import BaseAgent, MessageType, Message
from core.message_bus import MessageBus
from android_env_simulator import AndroidEnv, Action, ActionType, Subgoal

class ExecutorAgent(BaseAgent):
    """Enhanced Executor Agent - Inspects UI hierarchy and executes subgoals"""
    
    def __init__(self, android_env: AndroidEnv, message_bus: MessageBus):
        super().__init__("Executor", message_bus)
        self.env = android_env
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ExecutorAgent")
        
        # UI inspection configuration
        self.ui_inspection_enabled = True
        self.max_ui_search_depth = 5
        self.element_match_threshold = 0.8

    def _setup_message_handlers(self):
        """Setup message handlers for executor"""
        self.message_bus.subscribe(MessageType.EXECUTION_REQUEST, self._handle_execution_request)
    
    async def _handle_execution_request(self, message: Message):
        """Handle execution request messages"""
        subgoal_data = message.payload.get("subgoal")
        
        if subgoal_data:
            # Convert to Subgoal object
            actions = []
            for action_data in subgoal_data.get("actions", []):
                action_type_str = action_data.get("action_type", "").upper()
                if hasattr(ActionType, action_type_str):
                    actions.append(Action(
                        action_type=ActionType[action_type_str],
                        element_id=action_data.get("element_id"),
                        coordinates=action_data.get("coordinates"),
                        text=action_data.get("text"),
                        parameters=action_data.get("parameters", {}),
                        delay=action_data.get("delay")
                    ))
            
            subgoal = Subgoal(
                description=subgoal_data["description"],
                expected_state=subgoal_data.get("expected_state"),
                actions=actions
            )
            
            success, state = await self.process(subgoal)
            
            # Send response
            message_type = MessageType.EXECUTION_RESPONSE if success else MessageType.EXECUTION_FAILED
            await self.send_message(
                message_type,
                message.sender,
                {
                    "success": success,
                    "state": state,
                    "subgoal": subgoal_data,
                    "ui_inspection_results": state.get("ui_inspection_results", {})
                },
                correlation_id=message.id
            )
    
    async def process(self, subgoal: Subgoal) -> Tuple[bool, Dict]:
        """Execute a subgoal with UI inspection"""
        await self.broadcast_status("busy", {"task": "executing", "subgoal": subgoal.description})
        self.logger.info(f"Executing subgoal: {subgoal.description}")
        
        if not subgoal.actions:
            self.logger.warning("No actions provided for subgoal")
            await self.broadcast_status("ready")
            return False, {"error": "No actions provided"}
        
        # Initial UI inspection
        initial_ui_state = await self._inspect_ui_hierarchy()
        self.logger.info(f"Initial UI inspection found {len(initial_ui_state.get('elements', []))} elements")
        
        results = []
        state = {
            "initial_ui": initial_ui_state,
            "action_results": [],
            "ui_inspection_results": {}
        }
        
        for i, action in enumerate(subgoal.actions):
            self.logger.info(f"Executing action {i+1}/{len(subgoal.actions)}: {action.action_type.value}")
            
            # Pre-action UI inspection
            if self.ui_inspection_enabled:
                pre_action_ui = await self._inspect_ui_hierarchy()
                action_context = await self._analyze_ui_for_action(action, pre_action_ui)
                
                # Log UI analysis results
                self.logger.info(f"UI Analysis: {action_context}")
                
                # Adapt action based on UI state if needed
                adapted_action = await self._adapt_action_to_ui(action, action_context)
                if adapted_action != action:
                    self.logger.info(f"Action adapted based on UI state")
            else:
                adapted_action = action
                action_context = {}
            
            # Execute the action
            success, execution_state = await self.env.execute_action(adapted_action)
            
            # Post-action UI inspection
            post_action_ui = None
            if self.ui_inspection_enabled and success:
                post_action_ui = await self._inspect_ui_hierarchy()
                ui_changes = self._detect_ui_changes(pre_action_ui, post_action_ui)
                
                self.logger.info(f"UI changes detected: {len(ui_changes)} elements changed")
            
            # Record action result
            action_result = {
                "action": action.action_type.value,
                "success": success,
                "ui_context": action_context,
                "ui_changes": ui_changes if success else None,
                "execution_state": execution_state
            }
            state["action_results"].append(action_result)
            results.append(success)
            
            if not success:
                error_msg = f"Action failed: {action.action_type.value}"
                self.logger.error(error_msg)
                
                # Try to understand why it failed from UI
                failure_analysis = await self._analyze_action_failure(
                    action, pre_action_ui, execution_state
                )
                
                await self.send_message(
                    MessageType.EXECUTION_FAILED,
                    None,  # Broadcast
                    {
                        "error": error_msg,
                        "action": action.action_type.value,
                        "subgoal": subgoal.description,
                        "state": execution_state,
                        "failure_analysis": failure_analysis
                    }
                )
                
                await self.broadcast_status("ready")
                state["error"] = error_msg
                state["failure_analysis"] = failure_analysis
                return False, state
            
            # Update current state
            state["current_ui"] = post_action_ui
            
            # Small delay between actions
            await asyncio.sleep(0.5)
        
        # Final UI inspection to verify expected state
        final_ui_state = await self._inspect_ui_hierarchy()
        state_verification = await self._verify_expected_state(
            subgoal.expected_state, final_ui_state
        )
        
        state["final_ui"] = final_ui_state
        state["state_verification"] = state_verification
        state["ui_inspection_results"] = {
            "elements_found": len(final_ui_state.get("elements", [])),
            "expected_state_achieved": state_verification.get("matched", False),
            "confidence": state_verification.get("confidence", 0.0)
        }
        
        self.log_decision({
            "subgoal": subgoal.description,
            "actions": [str(a.action_type.value) for a in subgoal.actions],
            "result": "success",
            "state": state,
            "ui_elements_tracked": len(state.get("action_results", []))
        })
        
        await self.broadcast_status("ready")
        return True, state
    
    async def _inspect_ui_hierarchy(self) -> Dict[str, Any]:
        """Inspect current UI hierarchy"""
        try:
            ui_hierarchy = await self.env.get_ui_hierarchy()
            
            # Extract key information from UI tree
            elements = []
            nodes = ui_hierarchy.get("nodes", [])
            
            for node in nodes:
                element_info = {
                    "id": node.get("resource_id", ""),
                    "text": node.get("text", ""),
                    "class": node.get("class", ""),
                    "bounds": node.get("bounds", []),
                    "clickable": node.get("clickable", False),
                    "enabled": node.get("enabled", True),
                    "visible": node.get("visible", True),
                    "checked": node.get("checked", False),
                    "focused": node.get("focused", False),
                    "scrollable": node.get("scrollable", False)
                }
                
                # Calculate element center
                if element_info["bounds"]:
                    bounds = element_info["bounds"]
                    element_info["center"] = (
                        (bounds[0] + bounds[2]) // 2,
                        (bounds[1] + bounds[3]) // 2
                    )
                
                elements.append(element_info)
            
            # Analyze UI structure
            ui_analysis = {
                "total_elements": len(elements),
                "clickable_elements": sum(1 for e in elements if e["clickable"]),
                "text_elements": sum(1 for e in elements if e["text"]),
                "input_fields": sum(1 for e in elements if "EditText" in e["class"]),
                "buttons": sum(1 for e in elements if "Button" in e["class"]),
                "current_app": self._detect_current_app(elements),
                "screen_type": self._detect_screen_type(elements)
            }
            
            return {
                "elements": elements,
                "analysis": ui_analysis,
                "raw_hierarchy": ui_hierarchy
            }
            
        except Exception as e:
            self.logger.error(f"Error inspecting UI hierarchy: {e}")
            return {"elements": [], "error": str(e)}
    
    async def _analyze_ui_for_action(self, action: Action, ui_state: Dict) -> Dict[str, Any]:
        """Analyze UI state to provide context for action execution"""
        context = {
            "action_type": action.action_type.value,
            "target_found": False,
            "alternatives": [],
            "obstacles": [],
            "recommendations": []
        }
        
        elements = ui_state.get("elements", [])
        
        if action.action_type in [ActionType.TOUCH, ActionType.TYPE]:
            if action.element_id:
                # Find target element
                target = self._find_element_by_id(elements, action.element_id)
                if target:
                    context["target_found"] = True
                    context["target_element"] = target
                    
                    # Check if element is interactable
                    if not target.get("clickable", False) and action.action_type == ActionType.TOUCH:
                        context["obstacles"].append("Target element is not clickable")
                    if not target.get("enabled", True):
                        context["obstacles"].append("Target element is disabled")
                    if not target.get("visible", True):
                        context["obstacles"].append("Target element is not visible")
                else:
                    # Find similar elements
                    similar = self._find_similar_elements(elements, action.element_id)
                    context["alternatives"] = similar
                    if similar:
                        context["recommendations"].append(
                            f"Target '{action.element_id}' not found, but found {len(similar)} similar elements"
                        )
            
            elif action.coordinates:
                # Check what's at the coordinates
                element_at_coords = self._find_element_at_coordinates(
                    elements, action.coordinates
                )
                if element_at_coords:
                    context["element_at_coordinates"] = element_at_coords
                    context["target_found"] = True
        
        # Check for potential blockers
        overlays = self._detect_overlays(elements)
        if overlays:
            context["obstacles"].append(f"Detected {len(overlays)} overlay elements")
            context["overlays"] = overlays
        
        return context
    
    async def _adapt_action_to_ui(self, action: Action, ui_context: Dict) -> Action:
        """Adapt action based on current UI state"""
        adapted = Action(
            action_type=action.action_type,
            element_id=action.element_id,
            coordinates=action.coordinates,
            text=action.text,
            parameters=action.parameters,
            delay=action.delay
        )
        
        # If target not found but alternatives exist
        if not ui_context["target_found"] and ui_context["alternatives"]:
            best_alternative = ui_context["alternatives"][0]
            self.logger.info(f"Adapting action to use alternative element: {best_alternative['id']}")
            adapted.element_id = best_alternative["id"]
            adapted.coordinates = best_alternative.get("center")
        
        # Handle overlays
        if ui_context.get("overlays"):
            self.logger.info("Detected overlay, may need to dismiss first")
            # Could add logic to dismiss overlays first
        
        # If element is not clickable but we're trying to click
        if (action.action_type == ActionType.TOUCH and 
            ui_context.get("target_element") and 
            not ui_context["target_element"].get("clickable", False)):
            # Try parent element or adjust coordinates
            self.logger.warning("Target not clickable, adjusting action")
        
        return adapted
    
    def _detect_ui_changes(self, before: Dict, after: Dict) -> List[Dict]:
        """Detect changes between two UI states"""
        changes = []
        
        before_elements = {e["id"]: e for e in before.get("elements", []) if e["id"]}
        after_elements = {e["id"]: e for e in after.get("elements", []) if e["id"]}
        
        # Find new elements
        for elem_id, elem in after_elements.items():
            if elem_id not in before_elements:
                changes.append({
                    "type": "element_added",
                    "element": elem
                })
        
        # Find removed elements
        for elem_id, elem in before_elements.items():
            if elem_id not in after_elements:
                changes.append({
                    "type": "element_removed",
                    "element": elem
                })
        
        # Find modified elements
        for elem_id in set(before_elements.keys()) & set(after_elements.keys()):
            before_elem = before_elements[elem_id]
            after_elem = after_elements[elem_id]
            
            # Check for changes
            if before_elem.get("text") != after_elem.get("text"):
                changes.append({
                    "type": "text_changed",
                    "element": elem_id,
                    "before": before_elem.get("text"),
                    "after": after_elem.get("text")
                })
            
            if before_elem.get("checked") != after_elem.get("checked"):
                changes.append({
                    "type": "checked_changed",
                    "element": elem_id,
                    "before": before_elem.get("checked"),
                    "after": after_elem.get("checked")
                })
        
        return changes
    
    async def _analyze_action_failure(self, action: Action, ui_state: Dict, 
                                    execution_state: Dict) -> Dict[str, Any]:
        """Analyze why an action failed"""
        analysis = {
            "action": action.action_type.value,
            "possible_reasons": [],
            "ui_state_issues": [],
            "recommendations": []
        }
        
        elements = ui_state.get("elements", [])
        
        if action.element_id:
            target = self._find_element_by_id(elements, action.element_id)
            if not target:
                analysis["possible_reasons"].append(f"Element '{action.element_id}' not found in UI")
                similar = self._find_similar_elements(elements, action.element_id)
                if similar:
                    analysis["recommendations"].append(
                        f"Try one of {len(similar)} similar elements: {[e['id'] for e in similar[:3]]}"
                    )
            else:
                if not target.get("enabled", True):
                    analysis["possible_reasons"].append("Target element is disabled")
                if not target.get("visible", True):
                    analysis["possible_reasons"].append("Target element is not visible")
                if action.action_type == ActionType.TOUCH and not target.get("clickable", False):
                    analysis["possible_reasons"].append("Target element is not clickable")
        
        # Check for common UI issues
        if self._detect_loading_indicators(elements):
            analysis["ui_state_issues"].append("Loading indicators detected - UI might not be ready")
            analysis["recommendations"].append("Add wait time before action")
        
        if self._detect_error_messages(elements):
            analysis["ui_state_issues"].append("Error messages detected in UI")
        
        overlays = self._detect_overlays(elements)
        if overlays:
            analysis["ui_state_issues"].append("Overlay/dialog blocking the UI")
            analysis["recommendations"].append("Dismiss overlay before proceeding")
        
        return analysis
    
    async def _verify_expected_state(self, expected_state: str, ui_state: Dict) -> Dict[str, Any]:
        """Verify if the expected state has been achieved"""
        if not expected_state:
            return {"matched": True, "confidence": 1.0, "reason": "No expected state defined"}
        
        verification = {
            "matched": False,
            "confidence": 0.0,
            "matching_elements": [],
            "reason": ""
        }
        
        elements = ui_state.get("elements", [])
        expected_lower = expected_state.lower()
        
        # Check for expected text in UI
        for element in elements:
            element_text = element.get("text", "").lower()
            if element_text and expected_lower in element_text:
                verification["matching_elements"].append(element)
                verification["confidence"] = max(verification["confidence"], 0.8)
        
        # Check screen type
        screen_type = ui_state.get("analysis", {}).get("screen_type", "")
        if screen_type and screen_type.lower() in expected_lower:
            verification["confidence"] = max(verification["confidence"], 0.7)
        
        # Set matched based on confidence threshold
        if verification["confidence"] >= 0.7:
            verification["matched"] = True
            verification["reason"] = f"Found {len(verification['matching_elements'])} matching elements"
        else:
            verification["reason"] = "Expected state not clearly detected in UI"
        
        return verification
    
    def _find_element_by_id(self, elements: List[Dict], element_id: str) -> Optional[Dict]:
        """Find element by resource ID"""
        for element in elements:
            if element.get("id") == element_id:
                return element
        return None
    
    def _find_similar_elements(self, elements: List[Dict], target_id: str) -> List[Dict]:
        """Find elements similar to the target"""
        similar = []
        target_lower = target_id.lower()
        
        for element in elements:
            elem_id = element.get("id", "").lower()
            elem_text = element.get("text", "").lower()
            
            # Check ID similarity
            if target_lower in elem_id or elem_id in target_lower:
                similar.append(element)
            # Check text similarity
            elif target_lower in elem_text or elem_text in target_lower:
                similar.append(element)
        
        return similar
    
    def _find_element_at_coordinates(self, elements: List[Dict], 
                                   coordinates: Tuple[int, int]) -> Optional[Dict]:
        """Find element at specific coordinates"""
        x, y = coordinates
        
        for element in elements:
            bounds = element.get("bounds", [])
            if len(bounds) == 4:
                if (bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]):
                    return element
        
        return None
    
    def _detect_overlays(self, elements: List[Dict]) -> List[Dict]:
        """Detect overlay elements like dialogs or popups"""
        overlays = []
        
        for element in elements:
            # Common overlay indicators
            if any(indicator in element.get("class", "").lower() 
                   for indicator in ["dialog", "popup", "modal", "overlay"]):
                overlays.append(element)
            
            # Full-screen elements that might be overlays
            bounds = element.get("bounds", [])
            if bounds and bounds[2] - bounds[0] > 1000 and bounds[3] - bounds[1] > 1500:
                if element.get("clickable") and not element.get("text"):
                    overlays.append(element)
        
        return overlays
    
    def _detect_loading_indicators(self, elements: List[Dict]) -> bool:
        """Detect if loading indicators are present"""
        for element in elements:
            class_name = element.get("class", "").lower()
            text = element.get("text", "").lower()
            
            if any(indicator in class_name for indicator in ["progress", "loading", "spinner"]):
                return True
            if any(indicator in text for indicator in ["loading", "please wait", "processing"]):
                return True
        
        return False
    
    def _detect_error_messages(self, elements: List[Dict]) -> bool:
        """Detect error messages in UI"""
        for element in elements:
            text = element.get("text", "").lower()
            if any(error in text for error in ["error", "failed", "invalid", "incorrect"]):
                return True
        return False
    
    def _detect_current_app(self, elements: List[Dict]) -> str:
        """Detect current app from UI elements"""
        for element in elements:
            resource_id = element.get("id", "")
            if "com.android.settings" in resource_id:
                return "Settings"
            elif "com.google.android.gm" in resource_id:
                return "Gmail"
            elif "com.android.deskclock" in resource_id:
                return "Clock"
            elif "com.google.android.apps.maps" in resource_id:
                return "Maps"
        return "Unknown"
    
    def _detect_screen_type(self, elements: List[Dict]) -> str:
        """Detect type of screen from UI patterns"""
        # Check for common screen patterns
        has_back_button = any("back" in e.get("id", "").lower() for e in elements)
        has_list = sum(1 for e in elements if "recycler" in e.get("class", "").lower()) > 0
        has_input = sum(1 for e in elements if "edittext" in e.get("class", "").lower()) > 0
        
        if has_input:
            return "input_form"
        elif has_list:
            return "list_view"
        elif has_back_button:
            return "detail_view"
        else:
            return "main_screen"