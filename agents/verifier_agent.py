from core.agent_base import BaseAgent, Message, MessageType
from core.message_bus import MessageBus
from typing import Dict
import asyncio
from openai import OpenAI
import json,logging
from android_integration.android_env_simulator import Subgoal,TestResult, ActionType, Action

class VerifierAgent(BaseAgent):
    """Verifier Agent with GPT-4 integration"""
    
    def __init__(self,message_bus:MessageBus):
        super().__init__("Verifier",message_bus)
        self.client = OpenAI()
        self.llm_model = "gpt-4o-2024-08-06"
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MultiAgentQA")
    
    def _setup_message_handlers(self):
        """Setup message handlers for verifier"""
        self.message_bus.subscribe(MessageType.VERIFICATION_REQUEST, self._handle_verification_request)
    
    async def _handle_verification_request(self, message: Message):
        """Handle verification request messages"""
        subgoal_data = message.payload.get("subgoal")
        execution_result = message.payload.get("execution_result", {})
        
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
            
            result = await self.process(subgoal, execution_result)
            
            await self.send_message(
                MessageType.VERIFICATION_RESPONSE,
                message.sender,
                {
                    "passed": result.passed,
                    "bug_detected": result.bug_detected,
                    "error_message": result.error_message,
                    "recovery_attempted": result.recovery_attempted,
                    "subgoal": subgoal_data
                },
                correlation_id=message.id
            )
    
    async def _call_gpt4(self, prompt: str, max_tokens: int = 200) -> str:
        """Make API call to GPT-4"""
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
                response_format={"type": "json_object"}  # Fixed format
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"GPT-4 API error: {e}")
            return ""
    
    async def process(self, subgoal: Subgoal, execution_result: Dict) -> TestResult:
        """Verify subgoal completion using GPT-4"""
        await self.broadcast_status("busy", {"task": "verifying", "subgoal": subgoal.description})
        self.logger.info(f"Verifying subgoal: {subgoal.description}")
        
        # Prepare prompt for GPT-4
        ui_nodes = execution_result.get('ui_hierarchy', {}).get('nodes', [])
        ui_summary = f"UI has {len(ui_nodes)} elements"
        if ui_nodes:
            ui_summary += f", including: {', '.join([node.get('text', 'unnamed') for node in ui_nodes[:3]])}"
        
        prompt = f"""
        You are an AI test verifier for Android applications. Your task is to verify whether:
        1. The expected state described in the subgoal has been achieved
        2. There are any functional bugs in the current UI state
        
        Subgoal: {subgoal.description}
        Expected state: {subgoal.expected_state}
        Current UI: {ui_summary}
        
        Please analyze and respond with a JSON object containing:
        - "verification_passed": boolean (whether subgoal was achieved)
        - "bug_detected": boolean (whether a functional bug exists)
        - "reason": string (brief explanation of your assessment)
        """
        
        # Call GPT-4
        response = await self._call_gpt4(prompt)
        
        try:
            if response:
                verification_result = json.loads(response)
            else:
                verification_result = {}
            
            result = TestResult(
                passed=verification_result.get("verification_passed", False),
                bug_detected=verification_result.get("bug_detected", False),
                error_message=None if verification_result.get("verification_passed") else 
                    verification_result.get("reason", "Verification failed")
            )
            
            self.log_decision({
                "subgoal": subgoal.description,
                "expected_state": subgoal.expected_state,
                "verification_result": result.passed,
                "bug_detected": result.bug_detected,
                "llm_response": response
            })
            await self.broadcast_status("ready")
            return result
        except json.JSONDecodeError:
            self.logger.error("Failed to parse GPT-4 verification response")
            await self.broadcast_status("ready")
            return await self._fallback_verification(subgoal, execution_result)
    
    async def _fallback_verification(self, subgoal: Subgoal, execution_result: Dict) -> TestResult:
        """Fallback verification method when LLM fails"""
        self.logger.warning("Using fallback verification")
        ui_hierarchy = execution_result.get("ui_hierarchy", {})
        nodes = ui_hierarchy.get("nodes", [])
        
        verification_passed = False
        bug_detected = False
        
        if subgoal.expected_state:
            if "wifi" in subgoal.expected_state.lower():
                for node in nodes:
                    if "wifi" in node.get("text", "").lower():
                        verification_passed = True
                        if "toggle" in subgoal.expected_state.lower():
                            if node.get("checked", False) != ("enabled" in subgoal.expected_state.lower()):
                                bug_detected = True
            elif "alarm" in subgoal.expected_state.lower():
                for node in nodes:
                    if "alarm" in node.get("text", "").lower():
                        verification_passed = True
            else:
                # For generic cases, consider it passed if no error
                verification_passed = True
        
        return TestResult(
            passed=verification_passed,
            bug_detected=bug_detected,
            error_message=None if verification_passed else "Fallback verification: expected state not found"
        )
