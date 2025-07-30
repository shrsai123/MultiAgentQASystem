from core.agent_base import BaseAgent, MessageType, Message
from core.message_bus import MessageBus
from typing import Dict
import asyncio
from openai import OpenAI
import json
from typing import Dict, List
import os,logging

OUTPUT_DIR = "outputs"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

class SupervisorAgent(BaseAgent):
    """Supervisor Agent with GPT-4 integration"""
    
    def __init__(self,message_bus:MessageBus):
        super().__init__("Supervisor",message_bus)
        self.client = OpenAI()
        self.llm_model = "gpt-4o-2024-08-06"
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MultiAgentQA")

    def _setup_message_handlers(self):
        """Setup message handlers for supervisor"""
        self.message_bus.subscribe(MessageType.SUPERVISION_REQUEST, self._handle_supervision_request)
        self.message_bus.subscribe(MessageType.TEST_COMPLETED, self._handle_test_completed)
        self.message_bus.subscribe(MessageType.SYSTEM_ERROR, self._handle_system_error)

    async def _handle_supervision_request(self, message: Message):
        """Handle supervision request messages"""
        test_episode = message.payload.get("test_episode")
        
        if test_episode:
            feedback = await self.process(test_episode)
            
            await self.send_message(
                MessageType.SUPERVISION_RESPONSE,
                message.sender,
                {"feedback": feedback},
                correlation_id=message.id
            )
    
    async def _handle_test_completed(self, message: Message):
        """Handle test completion notifications"""
        self.logger.info(f"Test completed notification from {message.sender}")
        # Could trigger automatic report generation or other actions
    
    async def _handle_system_error(self, message: Message):
        """Handle system error notifications"""
        error = message.payload.get("error", "Unknown error")
        self.logger.error(f"System error reported: {error}")
    
    async def _call_gpt4(self, prompt: str, max_tokens: int = 500) -> str:
        """Make API call to GPT-4"""
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"GPT-4 API error: {e}")
            return ""
    
    async def process(self, test_episode: Dict) -> Dict:
        """Review test episode using GPT-4"""
        await self.broadcast_status("busy", {"task": "supervising"})
        self.logger.info("Supervising test episode with GPT-4")
        
        # Prepare prompt for GPT-4
        steps_summary = []
        for step in test_episode.get('steps', []):
            steps_summary.append({
                "subgoal": step.get("subgoal", "Unknown"),
                "success": step.get("execution_success", False),
                "verification_passed": step.get("verification_result", {}).get("passed", False) if isinstance(step.get("verification_result"), dict) else False
            })
        
        prompt = f"""
        You are an AI test supervisor for Android applications. Your task is to review a completed 
        test episode and provide constructive feedback to improve future testing.
        
        Test Episode Summary:
        - Task: {test_episode.get('task_name', 'Unknown')}
        - Goal: {test_episode.get('goal', 'Unknown')}
        - Status: {'PASSED' if test_episode.get('passed') else 'FAILED'}
        - Bug detected: {'YES' if test_episode.get('bug_detected') else 'NO'}
        - Total steps: {len(test_episode.get('steps', []))}
        
        Please provide detailed feedback in the following JSON format:
        {{
            "prompt_improvements": ["list of suggestions to improve test prompts"],
            "plan_issues": ["list of any issues found in the test plan"],
            "coverage_suggestions": ["suggestions for additional test coverage"],
            "execution_issues": ["any issues with test execution"],
            "verification_issues": ["any issues with verification"],
            "overall_rating": "excellent/good/needs_improvement",
            "summary": "brief overall assessment"
        }}
        """
        
        # Call GPT-4
        response = await self._call_gpt4(prompt)
        
        try:
            if response:
                feedback = json.loads(response)
            else:
                feedback = self._default_feedback()
            
            self.log_decision({
                "test_episode": test_episode.get("task_name", "unknown"),
                "feedback": feedback,
                "llm_response": response
            })
            await self.broadcast_status("ready")
            return feedback
        except json.JSONDecodeError:
            self.logger.error("Failed to parse GPT-4 supervision response")
            await self.broadcast_status("ready")
            return self._default_feedback()
    
    def _default_feedback(self) -> Dict:
        """Default feedback when GPT-4 fails"""
        return {
            "prompt_improvements": ["Consider more specific test goals"],
            "plan_issues": ["Plan execution completed"],
            "coverage_suggestions": ["Add edge case testing"],
            "execution_issues": [],
            "verification_issues": [],
            "overall_rating": "good",
            "summary": "Test completed with standard execution"
        }
    
    async def generate_report(self, test_episodes: List[Dict]) -> str:
        """Generate evaluation report using GPT-4"""
        await self.broadcast_status("busy", {"task": "generating_report"})
        self.logger.info("Generating evaluation report with GPT-4")
        total_steps = 0
        steps_with_bugs = 0
        for episode in test_episodes:
            for step in episode.get('steps', []):
                total_steps += 1
                verification_result = step.get('verification_result', {})
                if verification_result.get('bug_detected', False):
                    steps_with_bugs += 1
        # Prepare prompt for GPT-4
        summary_data = {
            "total_tests": len(test_episodes),
            "passed": sum(1 for e in test_episodes if e.get('passed', False)),
            "failed": sum(1 for e in test_episodes if not e.get('passed', True)),
            "bugs_detected": sum(1 for e in test_episodes if e.get('bug_detected', False)),
            "total_steps": total_steps,
            "steps_with_bugs": steps_with_bugs,
            "bug_detection_rate": (steps_with_bugs / total_steps * 100) if total_steps > 0 else 0

        }
        
        prompt = f"""
        You are an AI test analyst for Android applications. Your task is to generate a comprehensive 
        evaluation report based on multiple test episodes.
        
        Test Episodes Summary:
        - Total tests: {summary_data['total_tests']}
        - Passed: {summary_data['passed']}
        - Failed: {summary_data['failed']}
        - Bugs detected in episodes: {summary_data['bugs_detected']}
        - Total test steps: {summary_data['total_steps']}
        - Steps with bugs detected: {summary_data['steps_with_bugs']}
        - Bug detection rate: {summary_data['bug_detection_rate']:.1f}%
        
        Please analyze these test episodes and provide a detailed report in JSON format with:
        {{
            "summary": "overall statistics and high-level assessment",
            "bug_detection_accuracy": {{
                "bugs_found": {summary_data['bugs_detected']},
                "detection_rate": "{summary_data['bug_detection_rate']:.1f}%",
                "steps_analyzed": {summary_data['total_steps']},
                "steps_with_bugs": {summary_data['steps_with_bugs']},
                "assessment": "Provide assessment of bug detection effectiveness"
            }},
            "agent_performance": {{"planner": "assessment", "executor": "assessment", "verifier": "assessment"}},
            "key_findings": ["notable observations from the tests"],
            "recommendations": ["actionable suggestions for improvement"],
            "next_steps": ["suggested future work"]
        }}
        """
        
        # Call GPT-4
        response = await self._call_gpt4(prompt, max_tokens=1000)
        
        try:
            if response:
                report = json.loads(response)
            else:
                report = await self._fallback_report(test_episodes)
            
            # Save report to file
            report_path = os.path.join(LOG_DIR, "evaluation_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            
            self.log_decision({
                "action": "generated_evaluation_report",
                "report_path": report_path,
                "llm_response": response
            })
            await self.broadcast_status("ready")
            return report_path
        except json.JSONDecodeError:
            self.logger.error("Failed to parse GPT-4 report response")
            await self.broadcast_status("ready")
            return await self._fallback_report(test_episodes)
    
    async def _fallback_report(self, test_episodes: List[Dict]) -> str:
        """Fallback report generation when LLM fails"""
        self.logger.warning("Using fallback report generation")
        report = {
            "summary": {
                "total_tests": len(test_episodes),
                "passed": sum(1 for e in test_episodes if e.get("passed", False)),
                "failed": sum(1 for e in test_episodes if not e.get("passed", True)),
                "bugs_detected": sum(1 for e in test_episodes if e.get("bug_detected", False))
            },
            "agent_performance": {
                "planner": {
                    "adaptations_required": sum(1 for e in test_episodes if e.get("plan_adapted", False))
                },
                "executor": {
                    "success_rate": sum(1 for e in test_episodes if e.get("execution_success", False)) / len(test_episodes) if test_episodes else 0
                },
                "verifier": {
                    "accuracy": sum(1 for e in test_episodes if e.get("verification_correct", True)) / len(test_episodes) if test_episodes else 1
                }
            },
            "recommendations": ["Review failed test cases", "Improve error recovery mechanisms"]
        }
        
        report_path = os.path.join(LOG_DIR, "evaluation_report_fallback.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        return report_path