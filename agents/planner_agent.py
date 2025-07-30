from core.agent_base import BaseAgent, Message, MessageType
from core.message_bus import MessageBus
import json
from openai import OpenAI
from typing import Dict, List, Optional
import asyncio,logging
from android_integration.android_env_simulator import Subgoal,Action,ActionType

class PlannerAgent(BaseAgent):
    """Planner Agent with GPT-4 integration"""
    
    def __init__(self,message_bus: MessageBus):
        super().__init__("Planner",message_bus)
        self.client = OpenAI()
        self.llm_model = "gpt-4o-2024-08-06"
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MultiAgentQA")
    
    def _setup_message_handlers(self):
        """Setup message handlers for planner"""
        self.message_bus.subscribe(MessageType.PLAN_REQUEST, self._handle_plan_request)
        self.message_bus.subscribe(MessageType.PLAN_ADAPTATION_REQUEST, self._handle_adaptation_request)

    async def _handle_plan_request(self, message: Message):
        """Handle plan request messages"""
        goal = message.payload.get("goal")
        current_state = message.payload.get("current_state")
        
        if goal:
            subgoals = await self.process(goal, current_state)
            
            await self.send_message(
                MessageType.PLAN_RESPONSE,
                message.sender,
                {
                    "subgoals": [
                        {
                            "description": sg.description,
                            "expected_state": sg.expected_state,
                            "actions": [
                                {
                                    "action_type": a.action_type.value,
                                    "element_id": a.element_id,
                                    "coordinates": a.coordinates,
                                    "text": a.text,
                                    "parameters": a.parameters,
                                    "delay": a.delay
                                }
                                for a in sg.actions
                            ]
                        }
                        for sg in subgoals
                    ]
                },
                correlation_id=message.id
            )
    
    async def _handle_adaptation_request(self, message: Message):
        """Handle plan adaptation request messages"""
        original_plan = message.payload.get("original_plan", [])
        error = message.payload.get("error", "Unknown error")
        current_state = message.payload.get("current_state", {})
        
        # Convert back to Subgoal objects
        subgoals = []
        for sg_data in original_plan:
            actions = []
            for action_data in sg_data.get("actions", []):
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
            
            subgoals.append(Subgoal(
                description=sg_data["description"],
                expected_state=sg_data.get("expected_state"),
                actions=actions
            ))
        
        new_plan = await self.adapt_plan(subgoals, error, current_state)
        
        await self.send_message(
            MessageType.PLAN_ADAPTATION_RESPONSE,
            message.sender,
            {
                "adapted_plan": [
                    {
                        "description": sg.description,
                        "expected_state": sg.expected_state,
                        "actions": [
                            {
                                "action_type": a.action_type.value,
                                "element_id": a.element_id,
                                "coordinates": a.coordinates,
                                "text": a.text,
                                "parameters": a.parameters,
                                "delay": a.delay
                            }
                            for a in sg.actions
                        ]
                    }
                    for sg in new_plan
                ]
            },
            correlation_id=message.id
        )
    
    async def _call_gpt4(self, prompt: str, max_tokens: int = 500) -> str:
        """Make API call to GPT-4"""
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
                response_format={"type": "json_object"},  # Fixed: Changed to correct format
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"GPT-4 API error: {e}")
            return ""
    
    async def process(self, goal: str, current_state: Optional[Dict] = None) -> List[Subgoal]:
        """Generate subgoals from high-level goal using GPT-4"""
        await self.broadcast_status("busy", {"task": "planning", "goal": goal})
        self.logger.info(f"Planning for goal: {goal}")
        
        # Prepare prompt for GPT-4
        prompt = f"""
        You are an AI test planner for Android applications. Your task is to break down high-level 
        testing goals into specific, actionable subgoals that can be executed on a mobile device.

        Current task: {goal}
        
        Please generate a sequence of subgoals to accomplish this testing task. For each subgoal:
        1. Provide a clear description of what needs to be done
        2. Specify the expected state after completion
        3. List the specific UI actions needed (touch, type, scroll, etc.)
        
        Format your response as a JSON array where each element has:
        - "description": string describing the subgoal
        - "expected_state": string describing expected UI state
        - "actions": array of action objects with "action_type" and other needed parameters
        
        Response format: {{"subgoals": [array of subgoal objects]}}
        
        Example subgoal:
        {{
            "description": "Open Settings app",
            "expected_state": "Settings app main screen is visible",
            "actions": [
                {{
                    "action_type": "touch",
                    "element_id": "com.android.settings:id/main_content"
                }}
            ]
        }}
        """
        
        # Call GPT-4
        response = await self._call_gpt4(prompt)
        
        try:
            # Parse the response into Subgoal objects
            if response:
                data = json.loads(response)
                subgoals_data = data.get("subgoals", [])
            else:
                subgoals_data = []
            
            subgoals = []
            for sg in subgoals_data:
                actions = []
                for action_data in sg.get("actions", []):
                    action_type_str = action_data.get("action_type", "").upper()
                    if hasattr(ActionType, action_type_str):
                        action_type = ActionType[action_type_str]
                        actions.append(Action(
                            action_type=action_type,
                            element_id=action_data.get("element_id"),
                            coordinates=action_data.get("coordinates"),
                            text=action_data.get("text"),
                            parameters=action_data.get("parameters", {}),
                            delay=action_data.get("delay")
                        ))
                
                subgoals.append(Subgoal(
                    description=sg["description"],
                    expected_state=sg.get("expected_state"),
                    actions=actions
                ))
            
            self.log_decision({
                "goal": goal,
                "subgoals": [sg.description for sg in subgoals],
                "llm_response": response
            })
            await self.broadcast_status("ready")
            return subgoals if subgoals else await self._fallback_plan(goal)
            
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to parse GPT-4 response: {e}")
            return await self._fallback_plan(goal)
    
    async def _fallback_plan(self, goal: str) -> List[Subgoal]:
        """Fallback planning method when LLM fails"""
        self.logger.warning("Using fallback plan")
        if "wifi" in goal.lower():
            return await self._plan_wifi_test(goal)
        elif "alarm" in goal.lower():
            return await self._plan_alarm_test(goal)
        else:
            return await self._generic_plan(goal)
    
    async def _plan_wifi_test(self, goal: str) -> List[Subgoal]:
        """Create a plan for Wi-Fi testing"""
        return [
            Subgoal(
                description="Navigate to Settings app",
                expected_state="Settings app is open",
                actions=[
                    Action(
                        action_type=ActionType.TOUCH,
                        element_id="Settings"
                    )
                ]
            ),
            Subgoal(
                description="Open Wi-Fi settings",
                expected_state="Wi-Fi settings page is visible",
                actions=[
                    Action(
                        action_type=ActionType.TOUCH,
                        element_id="Wi-Fi Settings"
                    )
                ]
            ),
            Subgoal(
                description="Toggle Wi-Fi status",
                expected_state="Wi-Fi toggle has been interacted with",
                actions=[
                    Action(
                        action_type=ActionType.TOUCH,
                        element_id="com.android.settings:id/wifi"
                    )
                ]
            ),
            Subgoal(
                description="Return to home screen",
                expected_state="Home screen is visible",
                actions=[
                    Action(action_type=ActionType.HOME)
                ]
            )
        ]
    
    async def _plan_alarm_test(self, goal: str) -> List[Subgoal]:
        """Create a plan for alarm testing"""
        return [
            Subgoal(
                description="Navigate to Clock app",
                expected_state="Clock app is open",
                actions=[
                    Action(
                        action_type=ActionType.TOUCH,
                        element_id="Clock"
                    )
                ]
            ),
            Subgoal(
                description="Add a new alarm",
                expected_state="New alarm has been added",
                actions=[
                    Action(
                        action_type=ActionType.TYPE,
                        text="Test Alarm 7:00 AM"
                    )
                ]
            ),
            Subgoal(
                description="Return to home screen",
                expected_state="Home screen is visible",
                actions=[
                    Action(action_type=ActionType.HOME)
                ]
            )
        ]
    
    async def _generic_plan(self, goal: str) -> List[Subgoal]:
        """Create a generic test plan"""
        return [
            Subgoal(
                description="Navigate to home screen",
                expected_state="Home screen is visible",
                actions=[
                    Action(action_type=ActionType.HOME)
                ]
            ),
            Subgoal(
                description="Wait for system to stabilize",
                expected_state="System is ready",
                actions=[
                    Action(action_type=ActionType.WAIT, delay=2.0)
                ]
            )
        ]
    
    async def adapt_plan(self, original_plan: List[Subgoal], error: str, current_state: Dict) -> List[Subgoal]:
        """Adapt the plan using GPT-4 when errors occur"""
        self.logger.info(f"Adapting plan due to error: {error}")
        
        # Prepare prompt for GPT-4
        prompt = f"""
        You are an AI test planner for Android applications. During test execution, an error occurred:
        
        Error: {error}
        
        Current UI state nodes count: {len(current_state.get('ui_hierarchy', {}).get('nodes', []))}
        
        Original plan steps:
        {json.dumps([sg.description for sg in original_plan], indent=2)}
        
        Please generate a modified plan that:
        1. Includes recovery steps to handle the error
        2. Continues with the original test plan where possible
        3. Provides clear action steps
        
        Format your response as: {{"recovery_plan": [array of subgoal objects]}}
        """
        
        # Call GPT-4
        response = await self._call_gpt4(prompt)
        
        try:
            # Parse the response into Subgoal objects
            if response:
                data = json.loads(response)
                new_subgoals_data = data.get("recovery_plan", [])
            else:
                new_subgoals_data = []
            
            new_subgoals = []
            for sg in new_subgoals_data:
                actions = []
                for action_data in sg.get("actions", []):
                    action_type_str = action_data.get("action_type", "").upper()
                    if hasattr(ActionType, action_type_str):
                        action_type = ActionType[action_type_str]
                        actions.append(Action(
                            action_type=action_type,
                            element_id=action_data.get("element_id"),
                            coordinates=action_data.get("coordinates"),
                            text=action_data.get("text"),
                            parameters=action_data.get("parameters", {}),
                            delay=action_data.get("delay")
                        ))
                
                new_subgoals.append(Subgoal(
                    description=sg["description"],
                    expected_state=sg.get("expected_state"),
                    actions=actions
                ))
            
            self.log_decision({
                "original_plan": [sg.description for sg in original_plan],
                "error": error,
                "new_plan": [sg.description for sg in new_subgoals],
                "llm_response": response,
                "action": "Adapted plan using GPT-4"
            })
            await self.broadcast_status("ready")
            return new_subgoals if new_subgoals else self._simple_recovery_plan() + original_plan
            
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to parse GPT-4 adaptation response: {e}")
            # Fallback to simple recovery
            return self._simple_recovery_plan() + original_plan
    
    def _simple_recovery_plan(self) -> List[Subgoal]:
        """Simple recovery plan when adaptation fails"""
        return [
            Subgoal(
                description="Recovery: Navigate to home screen",
                expected_state="Home screen is visible",
                actions=[
                    Action(action_type=ActionType.BACK),
                    Action(action_type=ActionType.HOME)
                ]
            )
        ]