from typing import Dict
import asyncio
import logging
from typing import Dict, List, Optional, Union
from agents.planner_agent import PlannerAgent
from agents.verifier_agent import VerifierAgent
from agents.supervisor_agent import SupervisorAgent
from agents.executor_agent import ExecutorAgent
from android_env_simulator import AndroidEnv, Action, ActionType, Subgoal
import time,uuid
from core.agent_base import Message, MessageType
from core.message_bus import MessageBus
from dotenv import load_dotenv
import os
import openai
            
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MultiAgentQA")
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

class MultiAgentQASystem:
    """Orchestrates the complete multi-agent QA system"""
    
    def __init__(self, task_name: str = "settings_wifi", config: Optional[Dict] = None):
        self.task_name = task_name
        self.config = config or {}
        self.message_bus = MessageBus()
        # Initialize Android environment
        android_env_config = {
            "task_name": task_name,
            "max_retries": self.config.get("max_retries", 3),
            "action_delay": self.config.get("action_delay", 0.5)
        }
        self.android_env = AndroidEnv(android_env_config)
        available_tasks = self.android_env.get_available_tasks()
        if task_name not in available_tasks:
            raise ValueError(
                f"Invalid task name: '{task_name}'. "
                f"Available tasks are: {', '.join(available_tasks)}"
            )
        
        self.task_name = task_name
        self.planner = PlannerAgent(self.message_bus)
        self.executor = ExecutorAgent(self.android_env,self.message_bus)
        self.verifier = VerifierAgent(self.message_bus)
        self.supervisor = SupervisorAgent(self.message_bus)
        
        # Test episode tracking
        self.current_episode = {
            "task_name": task_name,
            "start_time": time.time(),
            "steps": [],
            "logs": [],
            "passed": False,
            "bug_detected": False,
            "config": android_env_config
        }
        self._setup_system_handlers()
    
    def _setup_system_handlers(self):
        self.message_bus.subscribe(MessageType.AGENT_ERROR, self._handle_agent_error)
        self.message_bus.subscribe(MessageType.EXECUTION_FAILED, self._handle_execution_failed)
    
    async def _handle_agent_error(self, message: Message):
        logger.error(f"Agent error from {message.sender}: {message.payload}")
        self.current_episode["logs"].append(f"Agent error: {message.sender} - {message.payload.get('error', 'Unknown error')}")
    
    async def _handle_execution_failed(self, message: Message):
        logger.error(f"Execution failed: {message.payload}")
        self.current_episode["logs"].append(f"Execution failed: {message.payload.get('error', 'Unknown error')}")
    
    async def initialize(self) -> bool:
        try:
            logger.info("Initializing multi-agent QA system...")
            await self.message_bus.start()
            if not await self.android_env.initialize():
                logger.error("Failed to initialize Android environment")
                return False
            await self.message_bus.publish(Message(
                type=MessageType.SYSTEM_STATUS,
                sender="system",
                payload={"status": "initialized"}
            ))
            logger.info("Multi-agent QA system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    async def run_test(self, goal: str) -> Dict:
        """Run a complete test for the given goal"""
        try:
            logger.info(f"Starting test for goal: {goal}")
            await self.message_bus.publish(Message(
                type=MessageType.TEST_STARTED,
                sender="system",
                payload={"goal": goal, "task_name": self.task_name}
            ))
            self.current_episode = {
                "task_name": self.task_name,
                "goal": goal,
                "start_time": time.time(),
                "steps": [],
                "logs": [],
                "passed": False,
                "bug_detected": False,
                "config": self.android_env.config
            }
            
            # Step 1: Planning
            plan_request_id = str(uuid.uuid4())
            await self.message_bus.publish(Message(
                id=plan_request_id,
                type=MessageType.PLAN_REQUEST,
                sender="system",
                recipient="Planner",
                payload={"goal": goal, "current_state": None}
            ))
            
            # Wait for plan response
            plan_response = await self._wait_for_response(
                MessageType.PLAN_RESPONSE, 
                correlation_id=plan_request_id,
                timeout=30
            )
            
            if not plan_response:
                raise Exception("Failed to get plan from Planner")
            
            subgoals_data = plan_response.payload.get("subgoals", [])
            
            # Convert subgoals data to Subgoal objects
            subgoals = []
            for sg_data in subgoals_data:
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
            
            self.current_episode["initial_plan"] = [sg.description for sg in subgoals]
            
            # Step 2: Execute and verify each subgoal
            all_passed = True
            bug_detected = False
            
            for i, subgoal in enumerate(subgoals):
                logger.info(f"Processing subgoal {i+1}/{len(subgoals)}: {subgoal.description}")
                
                # Convert Subgoal to dictionary for message
                subgoal_data = {
                    "description": subgoal.description,
                    "expected_state": subgoal.expected_state,
                    "actions": [
                        {
                            "action_type": a.action_type.value,
                            "element_id": a.element_id,
                            "coordinates": a.coordinates,
                            "text": a.text,
                            "parameters": a.parameters,
                            "delay": a.delay
                        }
                        for a in subgoal.actions
                    ]
                }
                
                # Request execution through message bus
                exec_request_id = str(uuid.uuid4())
                await self.message_bus.publish(Message(
                    id=exec_request_id,
                    type=MessageType.EXECUTION_REQUEST,
                    sender="system",
                    recipient="Executor",
                    payload={"subgoal": subgoal_data},
                    priority=5
                ))
                
                # Wait for execution response
                exec_response = await self._wait_for_response(
                    [MessageType.EXECUTION_RESPONSE, MessageType.EXECUTION_FAILED],
                    correlation_id=exec_request_id,
                    timeout=60
                )
                
                if not exec_response:
                    raise Exception(f"Execution timeout for subgoal: {subgoal_data['description']}")
                
                execution_success = exec_response.type == MessageType.EXECUTION_RESPONSE
                execution_result = exec_response.payload.get("state", {})
                
                # Create step data
                step_data = {
                    "subgoal": subgoal.description,
                    "execution_result": execution_result,
                    "timestamp": time.time(),
                    "execution_success": execution_success
                }
                
                if not execution_success:
                    logger.error(f"Execution failed for subgoal: {subgoal.description}")
                    error_msg = exec_response.payload.get("error", "Unknown execution error")
                    
                    # Request plan adaptation through message bus
                    adapt_request_id = str(uuid.uuid4())
                    
                    # Convert remaining subgoals to data format for message
                    remaining_plan = []
                    for sg in subgoals[i:]:
                        remaining_plan.append({
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
                        })
                    
                    await self.message_bus.publish(Message(
                        id=adapt_request_id,
                        type=MessageType.PLAN_ADAPTATION_REQUEST,
                        sender="system",
                        recipient="Planner",
                        payload={
                            "original_plan": remaining_plan,
                            "error": error_msg,
                            "current_state": execution_result
                        },
                        priority=8
                    ))
                    
                    # Wait for adapted plan
                    adapt_response = await self._wait_for_response(
                        MessageType.PLAN_ADAPTATION_RESPONSE,
                        correlation_id=adapt_request_id,
                        timeout=30
                    )
                    
                    if adapt_response:
                        # Update remaining subgoals with adapted plan
                        adapted_plan_data = adapt_response.payload.get("adapted_plan", [])
                        
                        # Convert adapted plan data back to Subgoal objects
                        adapted_subgoals = []
                        for sg_data in adapted_plan_data:
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
                            
                            adapted_subgoals.append(Subgoal(
                                description=sg_data["description"],
                                expected_state=sg_data.get("expected_state"),
                                actions=actions
                            ))
                        
                        subgoals = subgoals[:i] + adapted_subgoals
                        self.current_episode["plan_adapted"] = True
                        self.current_episode["logs"].append(f"Plan adapted at step {i}")
                        
                        # Retry current subgoal
                        continue
                    else:
                        all_passed = False
                        step_data["recovery_failed"] = True
                        self.current_episode["steps"].append(step_data)
                        break
                
                # Request verification through message bus
                verify_request_id = str(uuid.uuid4())
                await self.message_bus.publish(Message(
                    id=verify_request_id,
                    type=MessageType.VERIFICATION_REQUEST,
                    sender="system",
                    recipient="Verifier",
                    payload={
                        "subgoal": subgoal_data,
                        "execution_result": execution_result
                    }
                ))
                
                # Wait for verification response
                verify_response = await self._wait_for_response(
                    MessageType.VERIFICATION_RESPONSE,
                    correlation_id=verify_request_id,
                    timeout=30
                )
                
                if verify_response:
                    verification_result = verify_response.payload
                    step_data["verification_result"] = {
                        "passed": verification_result.get("passed", False),
                        "bug_detected": verification_result.get("bug_detected", False),
                        "error_message": verification_result.get("error_message")
                    }
                    
                    if not verification_result.get("passed", False):
                        all_passed = False
                        
                    if verification_result.get("bug_detected", False):
                        bug_detected = True
                        self.current_episode["logs"].append(f"Bug detected at step {i}: {subgoal.description}")
                else:
                    step_data["verification_result"] = {
                        "passed": False,
                        "error_message": "Verification timeout"
                    }
                    all_passed = False
                
                self.current_episode["steps"].append(step_data)
                
                # Small delay between subgoals
                await asyncio.sleep(1)
            
            # Step 3: Finalize episode
            self.current_episode["passed"] = all_passed
            self.current_episode["bug_detected"] = bug_detected
            self.current_episode["end_time"] = time.time()
            self.current_episode["duration"] = self.current_episode["end_time"] - self.current_episode["start_time"]
            
            # Get message history for analysis
            self.current_episode["message_history"] = [
                {
                    "type": msg.type.value,
                    "sender": msg.sender,
                    "recipient": msg.recipient,
                    "timestamp": msg.timestamp
                }
                for msg in self.message_bus.get_message_history(limit=50)
            ]
            
            # Request supervision through message bus
            supervision_request_id = str(uuid.uuid4())
            await self.message_bus.publish(Message(
                id=supervision_request_id,
                type=MessageType.SUPERVISION_REQUEST,
                sender="system",
                recipient="Supervisor",
                payload={"test_episode": self.current_episode}
            ))
            
            # Wait for supervision response
            supervision_response = await self._wait_for_response(
                MessageType.SUPERVISION_RESPONSE,
                correlation_id=supervision_request_id,
                timeout=30
            )
            
            if supervision_response:
                self.current_episode["supervisor_feedback"] = supervision_response.payload.get("feedback", {})
            
            # Save episode log
            log_path = await self.android_env.save_episode_log(self.current_episode)
            self.current_episode["log_path"] = log_path
            
            # Broadcast test completion
            await self.message_bus.publish(Message(
                type=MessageType.TEST_COMPLETED,
                sender="system",
                payload={
                    "task_name": self.task_name,
                    "goal": goal,
                    "passed": all_passed,
                    "bug_detected": bug_detected,
                    "duration": self.current_episode["duration"]
                }
            ))
            
            logger.info(f"Test completed. Result: {'PASSED' if all_passed else 'FAILED'}")
            logger.info(f"Detailed log saved to: {log_path}")
            
            return self.current_episode
            
        except Exception as e:
            logger.error(f"Error running test: {e}")
            await self.message_bus.publish(Message(
                type=MessageType.SYSTEM_ERROR,
                sender="system",
                payload={"error": str(e), "task_name": self.task_name, "goal": goal}
            ))
            self.current_episode["error"] = str(e)
            return self.current_episode
            
    async def _wait_for_response(self, message_types: Union[MessageType, List[MessageType]], 
                                correlation_id: str, timeout: float = 30) -> Optional[Message]:
        """Wait for a specific message response"""
        if isinstance(message_types, MessageType):
            message_types = [message_types]
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            history = self.message_bus.get_message_history(limit=10)
            for msg in reversed(history):
                if msg.correlation_id == correlation_id and msg.type in message_types:
                    return msg
            
            await asyncio.sleep(0.1)
        
        return None
    
    async def cleanup(self):
        """Cleanup all system resources"""
        try:
            await self.message_bus.stop()
            await self.android_env.cleanup()
            logger.info("Multi-agent QA system cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    

async def main():
    """Example usage of the multi-agent QA system"""

    qa_system = MultiAgentQASystem(
        task_name="settings_wifi",
        config={
            "max_retries": 3,
            "action_delay": 0.5
        }
    )
    
    if not await qa_system.initialize():
        print("Initialization failed")
        return
    
    try:
        # Run a test
        test_goal = "Test turning Wi-Fi on and off"
        test_result = await qa_system.run_test(test_goal)
        
        print("\nTest Results:")
        print(f"Goal: {test_goal}")
        print(f"Status: {'PASSED' if test_result['passed'] else 'FAILED'}")
        print(f"Bug detected: {'YES' if test_result['bug_detected'] else 'NO'}")
        print(f"Duration: {test_result['duration']:.2f} seconds")
        print(f"Log file: {test_result.get('log_path', 'Not saved')}")
        print("\nGenerating evaluation report...")
        report_path = await qa_system.supervisor.generate_report([test_result])
        print(f"Evaluation report saved to: {report_path}")
        
        
        
    finally:
        await qa_system.cleanup()

if __name__ == "__main__":
    asyncio.run(main())