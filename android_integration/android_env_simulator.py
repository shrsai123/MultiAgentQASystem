import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import base64
from PIL import Image
import io
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MultiAgentQA")

# Constants
OUTPUT_DIR = "outputs"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
SCREENSHOTS_DIR = os.path.join(OUTPUT_DIR, "screenshots")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# Data models
class ActionType(Enum):
    TOUCH = "touch"
    TYPE = "type"
    SCROLL = "scroll"
    BACK = "back"
    HOME = "home"
    WAIT = "wait"

@dataclass
class Action:
    action_type: ActionType
    element_id: Optional[str] = None
    coordinates: Optional[Tuple[int, int]] = None
    text: Optional[str] = None
    parameters: Dict[str, Any] = None
    delay: Optional[float] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

@dataclass
class Subgoal:
    description: str
    expected_state: Optional[str] = None
    actions: List[Action] = None

    def __post_init__(self):
        if self.actions is None:
            self.actions = []

@dataclass
class TestResult:
    passed: bool
    bug_detected: bool
    error_message: Optional[str] = None
    recovery_attempted: bool = False



class MockAndroidEnv:
    """Mock implementation of Android environment"""
    
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.current_screen = "home"
        self.wifi_enabled = True
        self.alarms = []
        self.mock_state = self._create_initial_state()
        
    def reset(self):
        """Reset the mock environment"""
        self.current_screen = "home"
        self.wifi_enabled = True
        self.alarms = []
        self.mock_state = self._create_initial_state()
        return self.mock_state
        
    def step(self, action_dict: Dict):
        """Execute an action in the mock environment"""
        action_type = action_dict.get("action_type")
        
        if action_type == "touch":
            return self._handle_touch(action_dict)
        elif action_type == "type":
            return self._handle_type(action_dict)
        elif action_type == "scroll":
            return self._handle_scroll()
        elif action_type == "key":
            return self._handle_key(action_dict)
        else:
            return self.mock_state
            
    def render(self, mode="rgb_array"):
        """Render a mock screenshot"""
        img = Image.new('RGB', (400, 800), color=(73, 109, 137))
        d = ImageDraw.Draw(img)
        
        # Draw basic UI elements based on current screen
        if self.current_screen == "home":
            d.rectangle([50, 100, 350, 150], fill=(200, 200, 200))
            d.text((60, 110), "Settings", fill=(0, 0, 0))
            
            d.rectangle([50, 200, 350, 250], fill=(200, 200, 200))
            d.text((60, 210), "Clock", fill=(0, 0, 0))
            
        elif self.current_screen == "settings":
            d.rectangle([50, 100, 350, 150], fill=(200, 200, 200))
            d.text((60, 110), "Wi-Fi Settings", fill=(0, 0, 0))
            
        elif self.current_screen == "wifi":
            d.rectangle([50, 100, 350, 150], fill=(200, 200, 200))
            status = "ON" if self.wifi_enabled else "OFF"
            d.text((60, 110), f"Wi-Fi: {status}", fill=(0, 0, 0))
            
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
        
    def _handle_touch(self, action_dict: Dict):
        """Handle touch actions"""
        if self.current_screen == "home":
            if 100 <= action_dict.get("coordinate", [0,0])[1] <= 150:
                self.current_screen = "settings"
            elif 200 <= action_dict.get("coordinate", [0,0])[1] <= 250:
                self.current_screen = "clock"
                
        elif self.current_screen == "settings":
            if 100 <= action_dict.get("coordinate", [0,0])[1] <= 150:
                self.current_screen = "wifi"
                
        elif self.current_screen == "wifi":
            self.wifi_enabled = not self.wifi_enabled
            
        self.mock_state = self._create_state()
        return self.mock_state
        
    def _handle_type(self, action_dict: Dict):
        """Handle text input actions"""
        if self.current_screen == "clock":
            if "alarm" in action_dict.get("text", "").lower():
                self.alarms.append(action_dict["text"])
                
        self.mock_state = self._create_state()
        return self.mock_state
        
    def _handle_scroll(self):
        """Handle scroll actions"""
        # Just return current state
        return self.mock_state
        
    def _handle_key(self, action_dict: Dict):
        """Handle key presses"""
        if action_dict.get("key") == "BACK":
            if self.current_screen in ["wifi", "clock"]:
                self.current_screen = "home"
            elif self.current_screen == "settings":
                self.current_screen = "home"
                
        elif action_dict.get("key") == "HOME":
            self.current_screen = "home"
            
        self.mock_state = self._create_state()
        return self.mock_state
        
    def _create_initial_state(self):
        """Create initial mock state"""
        return {
            "ui_tree": {
                "nodes": [
                    {
                        "resource_id": "com.android.launcher:id/home",
                        "text": "Home",
                        "bounds": [0, 0, 400, 800],
                        "clickable": True,
                        "class": "android.widget.FrameLayout"
                    }
                ]
            },
            "screenshot": self.render()
        }
        
    def _create_state(self):
        """Create current mock state"""
        nodes = []
        
        if self.current_screen == "home":
            nodes = [
                {
                    "resource_id": "com.android.launcher:id/home",
                    "text": "Home",
                    "bounds": [0, 0, 400, 800],
                    "clickable": True,
                    "class": "android.widget.FrameLayout"
                },
                {
                    "resource_id": "com.android.launcher:id/settings_button",
                    "text": "Settings",
                    "bounds": [50, 100, 350, 150],
                    "clickable": True,
                    "class": "android.widget.Button"
                },
                {
                    "resource_id": "com.android.launcher:id/clock_button",
                    "text": "Clock",
                    "bounds": [50, 200, 350, 250],
                    "clickable": True,
                    "class": "android.widget.Button"
                }
            ]
            
        elif self.current_screen == "settings":
            nodes = [
                {
                    "resource_id": "com.android.settings:id/main",
                    "text": "Settings",
                    "bounds": [0, 0, 400, 800],
                    "clickable": True,
                    "class": "android.widget.FrameLayout"
                },
                {
                    "resource_id": "com.android.settings:id/wifi_settings",
                    "text": "Wi-Fi Settings",
                    "bounds": [50, 100, 350, 150],
                    "clickable": True,
                    "class": "android.widget.Button"
                }
            ]
            
        elif self.current_screen == "wifi":
            nodes = [
                {
                    "resource_id": "com.android.settings:id/wifi",
                    "text": f"Wi-Fi: {'ON' if self.wifi_enabled else 'OFF'}",
                    "bounds": [50, 100, 350, 150],
                    "clickable": True,
                    "class": "android.widget.Switch",
                    "checked": self.wifi_enabled
                }
            ]
            
        elif self.current_screen == "clock":
            nodes = [
                {
                    "resource_id": "com.android.deskclock:id/main",
                    "text": "Clock",
                    "bounds": [0, 0, 400, 800],
                    "clickable": True,
                    "class": "android.widget.FrameLayout"
                }
            ]
            for i, alarm in enumerate(self.alarms):
                nodes.append({
                    "resource_id": f"com.android.deskclock:id/alarm_{i}",
                    "text": alarm,
                    "bounds": [50, 100 + i*60, 350, 150 + i*60],
                    "clickable": True,
                    "class": "android.widget.TextView"
                })
        
        return {
            "ui_tree": {"nodes": nodes},
            "screenshot": self.render()
        }

class AndroidEnv:
    """
    Android Environment Wrapper with Mock Implementation
    Provides the same interface as the real AndroidEnvWrapper but simulates behavior
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.task_name = config.get("task_name", "settings_wifi")
        self.max_retries = config.get("max_retries", 3)
        self.action_delay = config.get("action_delay", 0.5)
        
        # Setup logging
        self.logger = logging.getLogger("AndroidEnv")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - AndroidEnv - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.env = MockAndroidEnv(self.task_name)
        self.current_observation = None
        self.screenshots_dir = "outputs/screenshots"
        self.logs_dir = "outputs/logs"
        self.current_step = 0
        self.episode_start_time = None
        
        # Ensure output directories exist
        os.makedirs(self.screenshots_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.logger.info(f"Mock Android environment initialized for task: {self.task_name}")
    
    async def initialize(self) -> bool:
        """Initialize the mock Android environment"""
        try:
            self.logger.info("Initializing mock Android environment...")
            self.episode_start_time = time.time()
            self.current_observation = await self._reset_environment()
            self.logger.info("Mock Android environment initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize mock Android environment: {e}")
            return False
    
    async def _reset_environment(self) -> Dict[str, Any]:
        """Reset the environment to initial state"""
        try:
            self.current_step = 0
            self.episode_start_time = time.time()
            observation = self.env.reset()
            return self._convert_observation(observation)
        except Exception as e:
            self.logger.error(f"Error resetting environment: {e}")
            return self._create_mock_observation()
    
    def _convert_observation(self, observation: Any) -> Dict[str, Any]:
        """Convert observation to our standardized format"""
        try:
            ui_tree = observation.get("ui_tree", {})
            screenshot = observation.get("screenshot")
            
            # Convert screenshot to base64 if available
            screenshot_base64 = base64.b64encode(screenshot).decode('utf-8') if screenshot else None
            
            return {
                "ui_tree": ui_tree,
                "screenshot": screenshot_base64,
                "timestamp": time.time(),
                "step": self.current_step,
                "task": self.task_name
            }
        except Exception as e:
            self.logger.error(f"Error converting observation: {e}")
            return self._create_mock_observation()
    
    def _create_mock_observation(self) -> Dict[str, Any]:
        """Create mock observation for error recovery"""
        return {
            "ui_tree": {
                "nodes": [
                    {
                        "resource_id": "com.android.settings:id/main_content",
                        "text": "Settings",
                        "bounds": [0, 0, 400, 800],
                        "clickable": True,
                        "class": "android.widget.FrameLayout"
                    }
                ]
            },
            "screenshot": None,
            "timestamp": time.time(),
            "step": self.current_step,
            "task": self.task_name,
            "is_mock": True
        }
    
    async def execute_action(self, action: Action) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute an Android action with retry logic
       
        """
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Executing action (attempt {attempt + 1}/{self.max_retries}): {action}")
                
                result = False
                if action.action_type == ActionType.TOUCH:
                    result = await self._execute_touch(action)
                elif action.action_type == ActionType.TYPE:
                    result = await self._execute_type(action)
                elif action.action_type == ActionType.SCROLL:
                    result = await self._execute_scroll(action)
                elif action.action_type == ActionType.BACK:
                    result = await self._execute_back()
                elif action.action_type == ActionType.HOME:
                    result = await self._execute_home()
                elif action.action_type == ActionType.WAIT:
                    await asyncio.sleep(action.delay or 1.0)
                    result = True
                else:
                    self.logger.error(f"Unknown action type: {action.action_type}")
                    return False, self.current_observation
                
                if result:
                    # Add delay between actions if specified
                    if action.delay:
                        await asyncio.sleep(action.delay)
                    elif self.action_delay > 0:
                        await asyncio.sleep(self.action_delay)
                        
                    # Get updated observation
                    current_state = await self.get_current_state()
                    return True, current_state
                
            except Exception as e:
                self.logger.error(f"Action execution failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    return False, self.current_observation
                
                # Wait before retry
                await asyncio.sleep(1.0)
        
        return False, self.current_observation
    
    async def _execute_touch(self, action: Action) -> bool:
        """Execute touch action with coordinate calculation"""
        try:
            if action.coordinates:
                x, y = action.coordinates
                self.logger.info(f"Touching at coordinates: ({x}, {y})")
            elif action.element_id:
                self.logger.info(f"Touching element: {action.element_id}")
                coords = await self._find_element_coordinates(action.element_id)
                if not coords:
                    self.logger.error(f"Element not found: {action.element_id}")
                    return False
                x, y = coords
            else:
                self.logger.error("Touch action requires coordinates or element_id")
                return False
            
            # Execute the touch in mock environment
            action_dict = {
                "action_type": "touch",
                "coordinate": [x, y]
            }
            observation = self.env.step(action_dict)
            self.current_observation = self._convert_observation(observation)
            self.current_step += 1
            return True
                
        except Exception as e:
            self.logger.error(f"Touch execution failed: {e}")
            return False
    
    async def _execute_type(self, action: Action) -> bool:
        """Execute text input action"""
        try:
            if not action.text:
                self.logger.error("Type action requires text")
                return False
            
            self.logger.info(f"Typing text: {action.text}")
            
            # Execute the type in mock environment
            action_dict = {
                "action_type": "type",
                "text": action.text
            }
            observation = self.env.step(action_dict)
            self.current_observation = self._convert_observation(observation)
            self.current_step += 1
            return True
                
        except Exception as e:
            self.logger.error(f"Type execution failed: {e}")
            return False
    
    async def _execute_scroll(self, action: Action) -> bool:
        """Execute scroll action with direction handling"""
        try:
            direction = action.parameters.get("direction", "down") if action.parameters else "down"
            self.logger.info(f"Scrolling {direction}")
            action_dict = {
                "action_type": "scroll"
            }
            observation = self.env.step(action_dict)
            self.current_observation = self._convert_observation(observation)
            self.current_step += 1
            return True
                
        except Exception as e:
            self.logger.error(f"Scroll execution failed: {e}")
            return False
    
    async def _execute_back(self) -> bool:
        """Execute back button press"""
        try:
            self.logger.info("Pressing back button")
            
            # Execute the back in mock environment
            action_dict = {
                "action_type": "key",
                "key": "BACK"
            }
            observation = self.env.step(action_dict)
            self.current_observation = self._convert_observation(observation)
            self.current_step += 1
            return True
                
        except Exception as e:
            self.logger.error(f"Back execution failed: {e}")
            return False
    
    async def _execute_home(self) -> bool:
        """Execute home button press"""
        try:
            self.logger.info("Pressing home button")
            
            # Execute the home in mock environment
            action_dict = {
                "action_type": "key",
                "key": "HOME"
            }
            observation = self.env.step(action_dict)
            self.current_observation = self._convert_observation(observation)
            self.current_step += 1
            return True
                
        except Exception as e:
            self.logger.error(f"Home execution failed: {e}")
            return False
    
    async def _find_element_coordinates(self, element_id: str) -> Optional[Tuple[int, int]]:
        """Find coordinates of UI element by ID with enhanced search"""
        try:
            if not self.current_observation or "ui_tree" not in self.current_observation:
                self.logger.warning("No current observation or UI tree available")
                return None
            
            ui_tree = self.current_observation["ui_tree"]
            nodes = ui_tree.get("nodes", [])
            
            # First try exact resource ID match
            for node in nodes:
                if node.get("resource_id") == element_id:
                    return self._calculate_element_center(node)
            
            # Fallback to partial ID match
            for node in nodes:
                if element_id in node.get("resource_id", ""):
                    return self._calculate_element_center(node)
            
            # Fallback to text content match
            for node in nodes:
                if element_id.lower() in node.get("text", "").lower():
                    return self._calculate_element_center(node)
            
            self.logger.warning(f"Element not found in UI tree: {element_id}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding element coordinates: {e}")
            return None
    
    def _calculate_element_center(self, node: Dict) -> Tuple[int, int]:
        """Calculate center coordinates of a UI element"""
        bounds = node.get("bounds", [0, 0, 100, 100])
        x = (bounds[0] + bounds[2]) // 2
        y = (bounds[1] + bounds[3]) // 2
        return (x, y)
    
    async def get_ui_hierarchy(self) -> Dict[str, Any]:
        """Get current UI hierarchy with enhanced error handling"""
        try:
            if self.current_observation and "ui_tree" in self.current_observation:
                return self.current_observation["ui_tree"]
            observation = self.env.step({"action_type": "none"})  
            self.current_observation = self._convert_observation(observation)
            return self.current_observation.get("ui_tree", {})
                    
        except Exception as e:
            self.logger.error(f"Error getting UI hierarchy: {e}")
            return {}
    
    async def render_visual_frame(self) -> Optional[bytes]:
        """Render visual frame using required rgb_array mode"""
        try:
            frame = self.env.render(mode="rgb_array")
            return frame
        except Exception as e:
            self.logger.error(f"Error rendering visual frame: {e}")
            return None
    
    async def take_screenshot(self) -> str:
        """Take screenshot and save to file with timestamp"""
        try:
            timestamp = int(time.time() * 1000)
            screenshot_path = os.path.join(
                self.screenshots_dir, 
                f"{self.task_name}_step{self.current_step}_{timestamp}.png"
            )
            frame = await self.render_visual_frame()
            if frame is not None:
                with open(screenshot_path, 'wb') as f:
                    f.write(frame)
                self.logger.info(f"Screenshot saved: {screenshot_path}")
                return screenshot_path
            self._create_mock_screenshot(screenshot_path)
            return screenshot_path
                
        except Exception as e:
            self.logger.error(f"Error taking screenshot: {e}")
            return ""
    
    def _create_mock_screenshot(self, file_path: str):
        """Create a mock screenshot with task and step information"""
        try:
            img = Image.new('RGB', (400, 200), color=(73, 109, 137))
            d = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.load_default()
                d.text((10,10), f"Mock Screenshot", font=font, fill=(255,255,0))
                d.text((10,30), f"Task: {self.task_name}", font=font, fill=(255,255,0))
                d.text((10,50), f"Step: {self.current_step}", font=font, fill=(255,255,0))
                d.text((10,70), f"Time: {time.ctime()}", font=font, fill=(255,255,0))
            except:
                pass
            
            img.save(file_path)
            self.logger.info(f"Created mock screenshot: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating mock screenshot: {e}")
    
    async def get_current_state(self) -> Dict[str, Any]:
        """Get complete current state including UI and visual"""
        try:
            ui_hierarchy = await self.get_ui_hierarchy()
            screenshot_path = await self.take_screenshot()
            
            return {
                "ui_hierarchy": ui_hierarchy,
                "screenshot_path": screenshot_path,
                "current_step": self.current_step,
                "task_name": self.task_name,
                "timestamp": time.time(),
                "episode_time": time.time() - self.episode_start_time
            }
            
        except Exception as e:
            self.logger.error(f"Error getting current state: {e}")
            return {
                "error": str(e),
                "timestamp": time.time(),
                "is_error_state": True
            }
    
    async def reset(self) -> Dict[str, Any]:
        """Reset environment and return initial state"""
        try:
            self.logger.info("Resetting environment...")
            state = await self._reset_environment()
            self.logger.info("Environment reset complete")
            return state
        except Exception as e:
            self.logger.error(f"Error resetting environment: {e}")
            return self._create_mock_observation()
    
    async def cleanup(self):
        """Cleanup resources and close environment"""
        try:
            self.logger.info("Mock Android environment cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def get_available_tasks(self) -> List[str]:
        """Get list of available tasks from registry"""
        return ["settings_wifi", "clock_alarm", "email_search"]
    
    def get_task_info(self, task_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific task"""
        tasks = {
            "settings_wifi": {
                "description": "Test Wi-Fi settings functionality",
                "app": "Settings",
                "complexity": "medium",
                "required_permissions": [],
                "average_duration": 30
            },
            "clock_alarm": {
                "description": "Test alarm functionality in Clock app",
                "app": "Clock",
                "complexity": "easy",
                "required_permissions": [],
                "average_duration": 20
            },
            "email_search": {
                "description": "Test email search functionality",
                "app": "Email",
                "complexity": "hard",
                "required_permissions": [],
                "average_duration": 45
            }
        }
        
        if task_name in tasks:
            return {
                "name": task_name,
                **tasks[task_name]
            }
        else:
            return {
                "error": f"Task not found: {task_name}",
                "available_tasks": self.get_available_tasks()
            }
    
    async def save_episode_log(self, episode_data: Dict) -> str:
        """Save complete episode log to file"""
        try:
            timestamp = int(time.time() * 1000)
            log_path = os.path.join(
                self.logs_dir,
                f"episode_{self.task_name}_{timestamp}.json"
            )
            
            with open(log_path, 'w') as f:
                json.dump(episode_data, f, indent=2)
            
            self.logger.info(f"Episode log saved: {log_path}")
            return log_path
        except Exception as e:
            self.logger.error(f"Error saving episode log: {e}")
            return ""
    