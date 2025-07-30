"""
Android-in-the-Wild Dataset Integration for Multi-Agent QA System
Incorporates real user sessions for training, evaluation, and robustness testing
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import base64
from PIL import Image
import io
import numpy as np
from datetime import datetime
import hashlib

# Optional imports for Android-in-the-Wild
try:
    import tensorflow as tf
except ImportError:
    tf = None
    print("TensorFlow not available. Install with: pip install tensorflow")

try:
    # Add google-research to path if cloned
    import sys
    if os.path.exists('./google-research'):
        sys.path.append('./google-research')
    from android_in_the_wild import visualization_utils
except ImportError:
    visualization_utils = None
    print("Android-in-the-Wild visualization utils not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AndroidWildIntegration")

@dataclass
class VideoFrame:
    """Represents a single frame from the video trace"""
    timestamp: float
    image: np.ndarray
    ui_elements: List[Dict[str, Any]]
    
@dataclass
class UserAction:
    """Represents a user action extracted from the video"""
    action_type: str
    coordinates: Optional[Tuple[int, int]]
    element: Optional[str]
    timestamp: float
    frame_index: int
    confidence: float

@dataclass
class VideoTrace:
    """Complete video trace from Android-in-the-Wild"""
    video_id: str
    app_name: str
    duration: float
    frames: List[VideoFrame]
    actions: List[UserAction]
    ui_traces: List[Dict[str, Any]]
    metadata: Dict[str, Any]

@dataclass
class TaskHypothesis:
    """Hypothesis about what task the user was trying to complete"""
    task_description: str
    confidence: float
    key_actions: List[str]
    expected_outcome: str
    app_context: str

@dataclass
class ComparisonResult:
    """Result of comparing agent execution with ground truth"""
    video_id: str
    task: str
    accuracy_score: float
    robustness_score: float
    generalization_score: float
    matched_actions: int
    total_actions: int
    divergence_points: List[Dict[str, Any]]
    execution_time_ratio: float
    ui_coverage: float

class AndroidWildDatasetLoader:
    """Loads and processes Android-in-the-Wild dataset from Google Cloud Storage"""
    
    def __init__(self, dataset_name: str = "general"):
        """
        Initialize dataset loader
        Args:
            dataset_name: One of ["general", "google_apps", "install", "single", "web_shopping"]
        """
        self.dataset_name = dataset_name
        self.logger = logging.getLogger("DatasetLoader")
        
        # Dataset directories on Google Cloud Storage
        self.dataset_directories = {
            'general': 'gs://gresearch/android-in-the-wild/general/*',
            'google_apps': 'gs://gresearch/android-in-the-wild/google_apps/*',
            'install': 'gs://gresearch/android-in-the-wild/install/*',
            'single': 'gs://gresearch/android-in-the-wild/single/*',
            'web_shopping': 'gs://gresearch/android-in-the-wild/web_shopping/*',
        }
        
        # Try to import required libraries
        self.tf_available = False
        self.viz_available = False
        try:
            import tensorflow as tf
            self.tf = tf
            self.tf_available = True
            self.logger.info("TensorFlow imported successfully")
        except ImportError:
            self.logger.warning("TensorFlow not available. Install with: pip install tensorflow")
        
        try:
            from android_in_the_wild import visualization_utils
            self.visualization_utils = visualization_utils
            self.viz_available = True
            self.logger.info("Android-in-the-Wild visualization utils imported")
        except ImportError:
            self.logger.warning("Android-in-the-Wild visualization utils not available")
        
        self._dataset_cache = {}
        
    def _get_episode_from_dataset(self, dataset_iterator):
        """Extract a complete episode from the dataset"""
        episode = []
        episode_id = None
        
        for data in dataset_iterator:
            ex = self.tf.train.Example()
            ex.ParseFromString(data)
            
            # Get episode ID
            ep_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
            
            if episode_id is None:
                episode_id = ep_id
                episode.append(ex)
            elif ep_id == episode_id:
                episode.append(ex)
            else:
                # We've reached the next episode
                break
                
        return episode, episode_id
    
    def _extract_action_from_example(self, example) -> UserAction:
        """Extract user action from TensorFlow example"""
        features = example.features.feature
        
        # Extract action type
        action_type = features['action_type'].int64_list.value[0] if 'action_type' in features else 0
        action_type_str = self._map_action_type(action_type)
        
        # Extract coordinates if available
        coordinates = None
        if 'touch_x' in features and 'touch_y' in features:
            x = features['touch_x'].int64_list.value[0]
            y = features['touch_y'].int64_list.value[0]
            coordinates = (x, y)
        
        # Extract text if it's a type action
        text = None
        if 'type_text' in features:
            text = features['type_text'].bytes_list.value[0].decode('utf-8')
        
        # Extract timestamp
        timestamp = features['timestamp'].float_list.value[0] if 'timestamp' in features else 0.0
        
        # Extract UI element if available
        element = None
        if 'ui_element_resource_id' in features:
            element = features['ui_element_resource_id'].bytes_list.value[0].decode('utf-8')
        
        return UserAction(
            action_type=action_type_str,
            coordinates=coordinates,
            element=element,
            timestamp=timestamp,
            frame_index=0,  # Will be set later
            confidence=0.95  # Default confidence
        )
    
    def _extract_ui_hierarchy(self, example) -> Dict[str, Any]:
        """Extract UI hierarchy from example"""
        features = example.features.feature
        
        ui_elements = []
        if 'ui_element_list' in features:
            # Parse UI elements from the example
            # This would depend on the actual format in the dataset
            pass
        
        return {
            "nodes": ui_elements,
            "screen_info": {
                "width": features['screen_width'].int64_list.value[0] if 'screen_width' in features else 1080,
                "height": features['screen_height'].int64_list.value[0] if 'screen_height' in features else 2340
            }
        }
    
    def _map_action_type(self, action_type_int: int) -> str:
        """Map integer action type to string"""
        action_map = {
            0: "tap",
            1: "type", 
            2: "scroll",
            3: "swipe",
            4: "back",
            5: "home",
            6: "long_press"
        }
        return action_map.get(action_type_int, "unknown")
    
    async def load_video_trace(self, episode_id: str) -> VideoTrace:
        """Load a video trace from the actual dataset or use mock data"""
        self.logger.info(f"Loading video trace: {episode_id}")
        
        # Try to load from real dataset if available
        if self.tf_available:
            try:
                return await self._load_real_trace(episode_id)
            except Exception as e:
                self.logger.warning(f"Failed to load real trace: {e}. Using mock data.")
        
        # Fall back to mock data
        mock_traces = {
            "gmail_compose": self._create_gmail_compose_trace(),
            "settings_notification": self._create_settings_notification_trace(),
            "maps_search": self._create_maps_search_trace(),
            "whatsapp_message": self._create_whatsapp_message_trace(),
            "calendar_event": self._create_calendar_event_trace()
        }
        
        return mock_traces.get(episode_id, self._create_default_trace(episode_id))
    
    async def _load_real_trace(self, target_episode_id: str) -> VideoTrace:
        """Load actual trace from Android-in-the-Wild dataset"""
        # Get dataset files
        dataset_path = self.dataset_directories.get(self.dataset_name)
        if not dataset_path:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        # Load TFRecord files
        filenames = self.tf.io.gfile.glob(dataset_path)
        if not filenames:
            raise ValueError(f"No files found for dataset: {self.dataset_name}")
        
        # Create dataset
        raw_dataset = self.tf.data.TFRecordDataset(
            filenames, 
            compression_type='GZIP'
        ).as_numpy_iterator()
        
        # Find the requested episode or get the first one
        episode = None
        episode_id = target_episode_id
        
        if target_episode_id == "first" or not target_episode_id:
            # Get the first complete episode
            episode, episode_id = self._get_episode_from_dataset(raw_dataset)
        else:
            # Search for specific episode
            current_episode = []
            for data in raw_dataset:
                ex = self.tf.train.Example()
                ex.ParseFromString(data)
                ep_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
                
                if ep_id == target_episode_id:
                    current_episode.append(ex)
                elif current_episode:
                    # We've passed the target episode
                    break
            
            if current_episode:
                episode = current_episode
            else:
                self.logger.warning(f"Episode {target_episode_id} not found, getting first episode")
                raw_dataset = self.tf.data.TFRecordDataset(filenames, compression_type='GZIP').as_numpy_iterator()
                episode, episode_id = self._get_episode_from_dataset(raw_dataset)
        
        if not episode:
            raise ValueError("No episodes found in dataset")
        
        # Convert episode to VideoTrace
        return self._convert_episode_to_trace(episode, episode_id)
    
    def _convert_episode_to_trace(self, episode: List, episode_id: str) -> VideoTrace:
        """Convert TensorFlow episode to VideoTrace format"""
        actions = []
        ui_traces = []
        frames = []
        
        # Extract metadata from first example
        first_ex = episode[0]
        features = first_ex.features.feature
        
        app_name = features['app_name'].bytes_list.value[0].decode('utf-8') if 'app_name' in features else "Unknown"
        
        # Process each example in the episode
        for i, example in enumerate(episode):
            # Extract action
            action = self._extract_action_from_example(example)
            action.frame_index = i
            actions.append(action)
            
            # Extract UI state
            ui_hierarchy = self._extract_ui_hierarchy(example)
            ui_traces.append({
                "timestamp": action.timestamp,
                "screen": f"screen_{i}",
                "ui_hierarchy": ui_hierarchy
            })
            
            # Extract frame if available
            if 'screenshot' in example.features.feature:
                screenshot_bytes = example.features.feature['screenshot'].bytes_list.value[0]
                # Convert to numpy array or PIL Image
                frames.append(VideoFrame(
                    timestamp=action.timestamp,
                    image=None,  # Would convert screenshot_bytes to image
                    ui_elements=ui_hierarchy.get("nodes", [])
                ))
        
        # Calculate duration
        duration = actions[-1].timestamp - actions[0].timestamp if actions else 0.0
        
        # Extract additional metadata
        metadata = {
            "episode_id": episode_id,
            "dataset": self.dataset_name,
            "num_actions": len(actions),
            "num_frames": len(frames),
            "app": app_name
        }
        
        return VideoTrace(
            video_id=episode_id,
            app_name=app_name,
            duration=duration,
            frames=frames,
            actions=actions,
            ui_traces=ui_traces,
            metadata=metadata
        )
    
    async def get_random_episodes(self, count: int = 5) -> List[str]:
        """Get random episode IDs from the dataset"""
        if not self.tf_available:
            # Return mock episode IDs
            return ["gmail_compose", "settings_notification", "maps_search", 
                   "whatsapp_message", "calendar_event"][:count]
        
        try:
            # Get all episode IDs from dataset
            dataset_path = self.dataset_directories.get(self.dataset_name)
            filenames = self.tf.io.gfile.glob(dataset_path)
            
            if not filenames:
                return []
            
            # Sample first file to get episode IDs
            raw_dataset = self.tf.data.TFRecordDataset(
                filenames[:1], 
                compression_type='GZIP'
            ).as_numpy_iterator()
            
            episode_ids = set()
            for data in raw_dataset:
                if len(episode_ids) >= count * 2:  # Get extra to allow for selection
                    break
                    
                ex = self.tf.train.Example()
                ex.ParseFromString(data)
                ep_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
                episode_ids.add(ep_id)
            
            # Randomly select requested count
            episode_list = list(episode_ids)
            import random
            random.shuffle(episode_list)
            return episode_list[:count]
            
        except Exception as e:
            self.logger.error(f"Error getting random episodes: {e}")
            return ["gmail_compose", "settings_notification", "maps_search"][:count]
    
    def visualize_episode(self, episode: List, show_annotations: bool = True, 
                         show_actions: bool = True):
        """Visualize an episode using Android-in-the-Wild utilities"""
        if self.viz_available:
            self.visualization_utils.plot_episode(
                episode, 
                show_annotations=show_annotations,
                show_actions=show_actions
            )
        else:
            self.logger.warning("Visualization utilities not available")
    
    def _create_gmail_compose_trace(self) -> VideoTrace:
        """Create mock trace for Gmail compose email task"""
        return VideoTrace(
            video_id="gmail_compose",
            app_name="Gmail",
            duration=45.2,
            frames=[],  # Would contain actual frames
            actions=[
                UserAction("tap", (100, 200), "compose_button", 2.1, 60, 0.95),
                UserAction("tap", (200, 300), "to_field", 5.3, 150, 0.92),
                UserAction("type", None, "john.doe@example.com", 8.5, 255, 0.88),
                UserAction("tap", (200, 400), "subject_field", 12.1, 363, 0.90),
                UserAction("type", None, "Project Update", 15.7, 471, 0.87),
                UserAction("tap", (200, 500), "body_field", 19.2, 576, 0.91),
                UserAction("type", None, "Please find the attached report", 25.8, 774, 0.85),
                UserAction("tap", (350, 100), "send_button", 30.5, 915, 0.93)
            ],
            ui_traces=[
                {"timestamp": 0, "screen": "inbox"},
                {"timestamp": 2.1, "screen": "compose"},
                {"timestamp": 30.5, "screen": "inbox", "toast": "Message sent"}
            ],
            metadata={
                "resolution": "1080x2340",
                "android_version": "12",
                "dark_mode": False,
                "notifications_present": True
            }
        )
    
    def _create_settings_notification_trace(self) -> VideoTrace:
        """Create mock trace for settings notification management"""
        return VideoTrace(
            video_id="settings_notification",
            app_name="Settings",
            duration=38.7,
            frames=[],
            actions=[
                UserAction("tap", (100, 300), "settings_app", 1.5, 45, 0.94),
                UserAction("scroll", (540, 1000), None, 3.2, 96, 0.89),
                UserAction("tap", (200, 800), "notifications_menu", 5.8, 174, 0.91),
                UserAction("tap", (150, 400), "app_notifications", 8.3, 249, 0.88),
                UserAction("scroll", (540, 1200), None, 10.5, 315, 0.87),
                UserAction("tap", (300, 600), "gmail_app", 13.2, 396, 0.90),
                UserAction("tap", (900, 400), "toggle_switch", 15.7, 471, 0.92),
                UserAction("tap", (50, 100), "back_button", 18.1, 543, 0.93)
            ],
            ui_traces=[
                {"timestamp": 0, "screen": "home"},
                {"timestamp": 1.5, "screen": "settings_main"},
                {"timestamp": 5.8, "screen": "notifications_settings"},
                {"timestamp": 15.7, "notification_state": "gmail_disabled"}
            ],
            metadata={
                "resolution": "1080x2400",
                "android_version": "13",
                "dark_mode": True,
                "system_ui": "OneUI"
            }
        )
    
    def _create_maps_search_trace(self) -> VideoTrace:
        """Create mock trace for Maps location search"""
        return VideoTrace(
            video_id="maps_search",
            app_name="Google Maps",
            duration=52.3,
            frames=[],
            actions=[
                UserAction("tap", (200, 150), "search_bar", 2.3, 69, 0.96),
                UserAction("type", None, "coffee near me", 5.8, 174, 0.89),
                UserAction("tap", (1000, 150), "search_button", 9.2, 276, 0.91),
                UserAction("scroll", (540, 1500), None, 12.5, 375, 0.85),
                UserAction("tap", (300, 600), "location_card", 15.8, 474, 0.88),
                UserAction("tap", (500, 1800), "directions_button", 19.3, 579, 0.90)
            ],
            ui_traces=[
                {"timestamp": 0, "screen": "map_view"},
                {"timestamp": 9.2, "screen": "search_results"},
                {"timestamp": 15.8, "screen": "place_details"},
                {"timestamp": 19.3, "screen": "navigation"}
            ],
            metadata={
                "resolution": "1440x3200",
                "android_version": "14",
                "location_enabled": True,
                "network_type": "5G"
            }
        )
    
    def _create_whatsapp_message_trace(self) -> VideoTrace:
        """Create mock trace for WhatsApp messaging"""
        return VideoTrace(
            video_id="whatsapp_message",
            app_name="WhatsApp",
            duration=28.9,
            frames=[],
            actions=[
                UserAction("tap", (100, 500), "chat_contact", 2.1, 63, 0.93),
                UserAction("tap", (500, 1900), "message_input", 4.5, 135, 0.91),
                UserAction("type", None, "Hey, are you free for lunch?", 8.2, 246, 0.87),
                UserAction("tap", (950, 1900), "send_button", 12.7, 381, 0.94),
                UserAction("tap", (900, 1900), "attachment_button", 15.3, 459, 0.89),
                UserAction("tap", (200, 600), "gallery_option", 17.8, 534, 0.88),
                UserAction("tap", (300, 800), "image_thumbnail", 20.2, 606, 0.90),
                UserAction("tap", (950, 1900), "send_button", 23.5, 705, 0.92)
            ],
            ui_traces=[
                {"timestamp": 0, "screen": "chat_list"},
                {"timestamp": 2.1, "screen": "chat_conversation"},
                {"timestamp": 15.3, "screen": "attachment_menu"},
                {"timestamp": 17.8, "screen": "gallery_picker"}
            ],
            metadata={
                "resolution": "1080x2340",
                "android_version": "11",
                "keyboard_type": "gboard",
                "media_shared": True
            }
        )
    
    def _create_calendar_event_trace(self) -> VideoTrace:
        """Create mock trace for Calendar event creation"""
        return VideoTrace(
            video_id="calendar_event",
            app_name="Google Calendar",
            duration=48.6,
            frames=[],
            actions=[
                UserAction("tap", (950, 1800), "fab_add", 2.5, 75, 0.95),
                UserAction("tap", (200, 400), "title_field", 5.1, 153, 0.91),
                UserAction("type", None, "Team Meeting", 8.7, 261, 0.88),
                UserAction("tap", (200, 600), "date_picker", 12.3, 369, 0.90),
                UserAction("tap", (600, 1000), "date_selection", 15.8, 474, 0.87),
                UserAction("tap", (200, 800), "time_picker", 19.2, 576, 0.89),
                UserAction("scroll", (800, 1000), None, 22.5, 675, 0.85),
                UserAction("tap", (950, 200), "save_button", 26.1, 783, 0.93)
            ],
            ui_traces=[
                {"timestamp": 0, "screen": "calendar_month_view"},
                {"timestamp": 2.5, "screen": "event_creation"},
                {"timestamp": 12.3, "screen": "date_picker_dialog"},
                {"timestamp": 26.1, "screen": "calendar_month_view", "event_added": True}
            ],
            metadata={
                "resolution": "1170x2532",
                "android_version": "12",
                "calendar_view": "month",
                "account_type": "google"
            }
        )
    
    def _create_default_trace(self, video_id: str) -> VideoTrace:
        """Create a default trace for unknown video IDs"""
        return VideoTrace(
            video_id=video_id,
            app_name="Unknown",
            duration=30.0,
            frames=[],
            actions=[],
            ui_traces=[],
            metadata={}
        )

class TaskPromptGenerator:
    """Generates task prompts from video traces using LLM"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.logger = logging.getLogger("TaskPromptGenerator")
        
    async def generate_task_prompt(self, video_trace: VideoTrace) -> TaskHypothesis:
        """Generate task prompt from video trace"""
        self.logger.info(f"Generating task prompt for video: {video_trace.video_id}")
        
        # Analyze action sequence
        action_summary = self._summarize_actions(video_trace.actions)
        ui_flow = self._analyze_ui_flow(video_trace.ui_traces)
        
        # Use LLM to generate task hypothesis
        prompt = f"""
        Analyze this user interaction sequence and determine what task they were trying to complete:
        
        App: {video_trace.app_name}
        Duration: {video_trace.duration}s
        
        Actions performed:
        {action_summary}
        
        UI Flow:
        {ui_flow}
        
        Based on this sequence, provide:
        1. A clear task description (what the user was trying to accomplish)
        2. The expected outcome
        3. Key actions that define this task
        4. Confidence level (0-1)
        
        Format as JSON with fields: task_description, expected_outcome, key_actions, confidence
        """
        
        # For demonstration, return pre-defined hypotheses
        hypotheses = {
            "gmail_compose": TaskHypothesis(
                task_description="Compose and send an email to john.doe@example.com with subject 'Project Update'",
                confidence=0.92,
                key_actions=["open_compose", "enter_recipient", "enter_subject", "enter_body", "send"],
                expected_outcome="Email successfully sent",
                app_context="Gmail app in inbox view"
            ),
            "settings_notification": TaskHypothesis(
                task_description="Disable Gmail notifications in system settings",
                confidence=0.88,
                key_actions=["open_settings", "navigate_notifications", "find_gmail", "toggle_off"],
                expected_outcome="Gmail notifications disabled",
                app_context="Android Settings app"
            ),
            "maps_search": TaskHypothesis(
                task_description="Search for nearby coffee shops and get directions",
                confidence=0.90,
                key_actions=["search_location", "view_results", "select_place", "get_directions"],
                expected_outcome="Navigation started to selected coffee shop",
                app_context="Google Maps with location enabled"
            ),
            "whatsapp_message": TaskHypothesis(
                task_description="Send a message with an image attachment to a contact",
                confidence=0.91,
                key_actions=["open_chat", "type_message", "send_message", "attach_image", "send_image"],
                expected_outcome="Message and image sent successfully",
                app_context="WhatsApp conversation"
            ),
            "calendar_event": TaskHypothesis(
                task_description="Create a new calendar event titled 'Team Meeting'",
                confidence=0.89,
                key_actions=["create_event", "set_title", "set_date", "set_time", "save_event"],
                expected_outcome="Event added to calendar",
                app_context="Google Calendar month view"
            )
        }
        
        return hypotheses.get(video_trace.video_id, TaskHypothesis(
            task_description="Unknown task",
            confidence=0.5,
            key_actions=[],
            expected_outcome="Unknown",
            app_context="Unknown app"
        ))
    
    def _summarize_actions(self, actions: List[UserAction]) -> str:
        """Summarize the action sequence"""
        summary = []
        for i, action in enumerate(actions):
            if action.element:
                summary.append(f"{i+1}. {action.action_type} on {action.element} at {action.timestamp:.1f}s")
            elif action.action_type == "type":
                summary.append(f"{i+1}. typed text at {action.timestamp:.1f}s")
            else:
                summary.append(f"{i+1}. {action.action_type} at {action.timestamp:.1f}s")
        return "\n".join(summary)
    
    def _analyze_ui_flow(self, ui_traces: List[Dict]) -> str:
        """Analyze UI state transitions"""
        flow = []
        for trace in ui_traces:
            screen = trace.get("screen", "unknown")
            timestamp = trace.get("timestamp", 0)
            extras = [f"{k}: {v}" for k, v in trace.items() if k not in ["screen", "timestamp"]]
            flow_item = f"[{timestamp:.1f}s] {screen}"
            if extras:
                flow_item += f" ({', '.join(extras)})"
            flow.append(flow_item)
        return "\n".join(flow)

class TraceComparator:
    """Compares agent execution traces with ground truth"""
    
    def __init__(self):
        self.logger = logging.getLogger("TraceComparator")
        
    async def compare_traces(self, 
                           ground_truth: VideoTrace, 
                           agent_trace: Dict[str, Any],
                           task: TaskHypothesis) -> ComparisonResult:
        """Compare agent execution with ground truth video trace"""
        self.logger.info(f"Comparing traces for task: {task.task_description}")
        
        # Extract agent actions from execution log
        agent_actions = self._extract_agent_actions(agent_trace)
        
        # Compute various metrics
        accuracy = self._compute_accuracy(ground_truth.actions, agent_actions)
        robustness = self._compute_robustness(agent_trace)
        generalization = self._compute_generalization(ground_truth, agent_trace)
        ui_coverage = self._compute_ui_coverage(ground_truth.ui_traces, agent_trace)
        
        # Find divergence points
        divergences = self._find_divergences(ground_truth.actions, agent_actions)
        
        # Calculate execution time ratio
        agent_duration = agent_trace.get("duration", 0)
        time_ratio = agent_duration / ground_truth.duration if ground_truth.duration > 0 else 0
        
        return ComparisonResult(
            video_id=ground_truth.video_id,
            task=task.task_description,
            accuracy_score=accuracy,
            robustness_score=robustness,
            generalization_score=generalization,
            matched_actions=len([a for a in agent_actions if a.get("matched", False)]),
            total_actions=len(ground_truth.actions),
            divergence_points=divergences,
            execution_time_ratio=time_ratio,
            ui_coverage=ui_coverage
        )
    
    def _extract_agent_actions(self, agent_trace: Dict) -> List[Dict]:
        """Extract actions from agent execution trace"""
        actions = []
        for step in agent_trace.get("steps", []):
            for action in step.get("actions", []):
                actions.append({
                    "type": action.get("action_type"),
                    "element": action.get("element_id"),
                    "timestamp": step.get("timestamp", 0),
                    "success": step.get("execution_success", False)
                })
        return actions
    
    def _compute_accuracy(self, ground_truth: List[UserAction], agent_actions: List[Dict]) -> float:
        """Compute action matching accuracy"""
        if not ground_truth:
            return 0.0
            
        matched = 0
        for gt_action in ground_truth:
            for agent_action in agent_actions:
                if self._actions_match(gt_action, agent_action):
                    matched += 1
                    agent_action["matched"] = True
                    break
                    
        return matched / len(ground_truth)
    
    def _actions_match(self, gt_action: UserAction, agent_action: Dict) -> bool:
        """Check if two actions match"""
        # Type must match
        if gt_action.action_type != agent_action.get("type"):
            return False
            
        # For UI interactions, check element similarity
        if gt_action.element and agent_action.get("element"):
            return self._elements_similar(gt_action.element, agent_action["element"])
            
        return True
    
    def _elements_similar(self, elem1: str, elem2: str) -> bool:
        """Check if two UI elements are similar"""
        # Simple similarity check - in practice would use more sophisticated matching
        elem1_lower = elem1.lower()
        elem2_lower = elem2.lower()
        
        # Exact match
        if elem1_lower == elem2_lower:
            return True
            
        # Partial match
        if elem1_lower in elem2_lower or elem2_lower in elem1_lower:
            return True
            
        # Common UI element mappings
        mappings = {
            "compose_button": ["fab_add", "compose", "new_email"],
            "send_button": ["send", "submit", "send_message"],
            "back_button": ["back", "navigate_up", "close"]
        }
        
        for key, values in mappings.items():
            if elem1_lower in values and elem2_lower in values:
                return True
                
        return False
    
    def _compute_robustness(self, agent_trace: Dict) -> float:
        """Compute robustness score based on error recovery"""
        total_steps = len(agent_trace.get("steps", []))
        if total_steps == 0:
            return 0.0
            
        failed_steps = sum(1 for step in agent_trace["steps"] 
                          if not step.get("execution_success", False))
        recovered_steps = sum(1 for step in agent_trace["steps"] 
                             if step.get("recovery_attempted", False))
        
        # Robustness = ability to complete despite failures
        if failed_steps == 0:
            return 1.0
        else:
            recovery_rate = recovered_steps / failed_steps
            completion_rate = (total_steps - failed_steps) / total_steps
            return (recovery_rate + completion_rate) / 2
    
    def _compute_generalization(self, ground_truth: VideoTrace, agent_trace: Dict) -> float:
        """Compute generalization score"""
        # Check if agent handled variations not in ground truth
        factors = []
        
        # UI variation handling
        if agent_trace.get("handled_dark_mode", False):
            factors.append(1.0)
        else:
            factors.append(0.5)
            
        # Notification/dialog handling
        if agent_trace.get("handled_interruptions", False):
            factors.append(1.0)
        else:
            factors.append(0.7)
            
        # Different screen resolutions
        if agent_trace.get("adapted_to_layout", False):
            factors.append(1.0)
        else:
            factors.append(0.6)
            
        return sum(factors) / len(factors) if factors else 0.5
    
    def _compute_ui_coverage(self, gt_ui_traces: List[Dict], agent_trace: Dict) -> float:
        """Compute UI state coverage"""
        gt_screens = set(trace.get("screen", "") for trace in gt_ui_traces)
        agent_screens = set()
        
        for step in agent_trace.get("steps", []):
            ui_state = step.get("execution_result", {}).get("ui_hierarchy", {})
            # Extract screen from UI state (simplified)
            if ui_state:
                agent_screens.add(step.get("subgoal", "").split()[0].lower())
                
        if not gt_screens:
            return 0.0
            
        covered = len(gt_screens.intersection(agent_screens))
        return covered / len(gt_screens)
    
    def _find_divergences(self, ground_truth: List[UserAction], agent_actions: List[Dict]) -> List[Dict]:
        """Find points where agent diverged from ground truth"""
        divergences = []
        
        for i, gt_action in enumerate(ground_truth):
            if i < len(agent_actions):
                agent_action = agent_actions[i]
                if not self._actions_match(gt_action, agent_action):
                    divergences.append({
                        "index": i,
                        "expected": f"{gt_action.action_type} on {gt_action.element}",
                        "actual": f"{agent_action.get('type')} on {agent_action.get('element')}",
                        "timestamp": gt_action.timestamp
                    })
            else:
                divergences.append({
                    "index": i,
                    "expected": f"{gt_action.action_type} on {gt_action.element}",
                    "actual": "No action",
                    "timestamp": gt_action.timestamp
                })
                
        return divergences

class AndroidWildEvaluator:
    """Main evaluator for Android-in-the-Wild integration"""
    
    def __init__(self, qa_system, dataset_loader, llm_client):
        self.qa_system = qa_system
        self.dataset_loader = dataset_loader
        self.task_generator = TaskPromptGenerator(llm_client)
        self.comparator = TraceComparator()
        self.logger = logging.getLogger("AndroidWildEvaluator")
        
    async def evaluate_on_videos(self, video_ids: List[str]) -> Dict[str, Any]:
        """Evaluate the multi-agent system on selected videos"""
        self.logger.info(f"Starting evaluation on {len(video_ids)} videos")
        
        results = []
        
        for video_id in video_ids:
            self.logger.info(f"\nProcessing video: {video_id}")
            
            # 1. Load video trace
            video_trace = await self.dataset_loader.load_video_trace(video_id)
            
            # 2. Generate task prompt
            task_hypothesis = await self.task_generator.generate_task_prompt(video_trace)
            self.logger.info(f"Generated task: {task_hypothesis.task_description}")
            
            # 3. Execute task with multi-agent system
            agent_trace = await self._execute_with_qa_system(task_hypothesis)
            
            # 4. Compare traces and compute metrics
            comparison = await self.comparator.compare_traces(
                video_trace, agent_trace, task_hypothesis
            )
            
            results.append({
                "video_id": video_id,
                "task": task_hypothesis,
                "comparison": comparison,
                "agent_trace": agent_trace
            })
            
            # Log intermediate results
            self._log_comparison_summary(comparison)
            
        # Generate comprehensive report
        report = self._generate_evaluation_report(results)
        
        return {
            "individual_results": results,
            "summary_report": report
        }
    
    async def _execute_with_qa_system(self, task: TaskHypothesis) -> Dict[str, Any]:
        """Execute task using the multi-agent QA system"""
        # Map task to appropriate mock environment task
        task_mapping = {
            "gmail": "email_search",  # Closest available
            "settings": "settings_wifi",
            "maps": "clock_alarm",  # No maps, use another app
            "whatsapp": "email_search",  # Similar messaging
            "calendar": "clock_alarm"  # Similar time-based
        }
        
        # Determine mock task based on app
        mock_task = "settings_wifi"  # Default
        for app, mapped_task in task_mapping.items():
            if app in task.app_context.lower():
                mock_task = mapped_task
                break
                
        # Initialize QA system with mapped task
        qa_system = MultiAgentQASystem(
            task_name=mock_task,
            config={"max_retries": 3, "action_delay": 0.5}
        )
        
        if not await qa_system.initialize():
            return {"error": "Failed to initialize QA system"}
            
        try:
            # Run test with generated task description
            result = await qa_system.run_test(task.task_description)
            return result
        finally:
            await qa_system.cleanup()
    
    def _log_comparison_summary(self, comparison: ComparisonResult):
        """Log summary of comparison results"""
        self.logger.info(f"\n--- Comparison Results for {comparison.video_id} ---")
        self.logger.info(f"Task: {comparison.task}")
        self.logger.info(f"Accuracy: {comparison.accuracy_score:.2%}")
        self.logger.info(f"Robustness: {comparison.robustness_score:.2%}")
        self.logger.info(f"Generalization: {comparison.generalization_score:.2%}")
        self.logger.info(f"Actions matched: {comparison.matched_actions}/{comparison.total_actions}")
        self.logger.info(f"UI Coverage: {comparison.ui_coverage:.2%}")
        self.logger.info(f"Time ratio: {comparison.execution_time_ratio:.2f}x")
        
        if comparison.divergence_points:
            self.logger.info(f"Divergences found: {len(comparison.divergence_points)}")
            for div in comparison.divergence_points[:3]:  # Show first 3
                self.logger.info(f"  - At index {div['index']}: "
                               f"Expected '{div['expected']}', got '{div['actual']}'")
    
    def _generate_evaluation_report(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        total_videos = len(results)
        
        # Aggregate metrics
        avg_accuracy = np.mean([r["comparison"].accuracy_score for r in results])
        avg_robustness = np.mean([r["comparison"].robustness_score for r in results])
        avg_generalization = np.mean([r["comparison"].generalization_score for r in results])
        avg_ui_coverage = np.mean([r["comparison"].ui_coverage for r in results])
        
        # Task success analysis
        successful_tasks = sum(1 for r in results 
                              if r["agent_trace"].get("passed", False))
        
        # Divergence analysis
        total_divergences = sum(len(r["comparison"].divergence_points) for r in results)
        
        # Performance analysis
        avg_time_ratio = np.mean([r["comparison"].execution_time_ratio for r in results])
        
        report = {
            "evaluation_summary": {
                "total_videos_evaluated": total_videos,
                "evaluation_date": datetime.now().isoformat(),
                "dataset": "Android-in-the-Wild"
            },
            "aggregate_metrics": {
                "average_accuracy": f"{avg_accuracy:.2%}",
                "average_robustness": f"{avg_robustness:.2%}",
                "average_generalization": f"{avg_generalization:.2%}",
                "average_ui_coverage": f"{avg_ui_coverage:.2%}",
                "task_success_rate": f"{successful_tasks/total_videos:.2%}"
            },
            "performance_analysis": {
                "average_execution_time_ratio": f"{avg_time_ratio:.2f}x",
                "total_divergences": total_divergences,
                "divergences_per_video": f"{total_divergences/total_videos:.1f}"
            },
            "per_video_results": [
                {
                    "video_id": r["video_id"],
                    "task": r["task"].task_description,
                    "accuracy": f"{r['comparison'].accuracy_score:.2%}",
                    "robustness": f"{r['comparison'].robustness_score:.2%}",
                    "generalization": f"{r['comparison'].generalization_score:.2%}",
                    "success": r["agent_trace"].get("passed", False)
                }
                for r in results
            ],
            "insights": self._generate_insights(results),
            "recommendations": self._generate_recommendations(results)
        }
        
        # Save report
        report_path = os.path.join("outputs/logs", f"android_wild_evaluation_{int(time.time())}.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
            
        self.logger.info(f"\nEvaluation report saved to: {report_path}")
        
        return report
    
    def _generate_insights(self, results: List[Dict]) -> List[str]:
        """Generate insights from evaluation results"""
        insights = []
        
        # Accuracy insights
        high_accuracy = [r for r in results if r["comparison"].accuracy_score > 0.8]
        if high_accuracy:
            insights.append(f"High accuracy (>80%) achieved on {len(high_accuracy)} videos, "
                          f"particularly for structured tasks like {high_accuracy[0]['task'].task_description}")
        
        # Robustness insights
        low_robustness = [r for r in results if r["comparison"].robustness_score < 0.6]
        if low_robustness:
            insights.append(f"Robustness challenges identified in {len(low_robustness)} videos, "
                          f"suggesting need for better error recovery mechanisms")
        
        # UI coverage insights
        avg_coverage = np.mean([r["comparison"].ui_coverage for r in results])
        if avg_coverage < 0.7:
            insights.append(f"UI state coverage averaging {avg_coverage:.1%} indicates "
                          f"potential gaps in screen navigation logic")
        
        # Time efficiency
        fast_execution = [r for r in results if r["comparison"].execution_time_ratio < 1.5]
        if fast_execution:
            insights.append(f"Efficient execution observed in {len(fast_execution)} cases, "
                          f"completing tasks faster than human users")
        
        return insights
    
    def _generate_recommendations(self, results: List[Dict]) -> List[str]:
        """Generate recommendations based on evaluation"""
        recommendations = []
        
        # Check for common failure patterns
        failed_tasks = [r for r in results if not r["agent_trace"].get("passed", False)]
        if failed_tasks:
            recommendations.append(
                f"Focus on improving task completion rate (currently {len(failed_tasks)}/{len(results)} failures)"
            )
        
        # Check for divergence patterns
        high_divergence = [r for r in results if len(r["comparison"].divergence_points) > 3]
        if high_divergence:
            recommendations.append(
                "Implement adaptive action matching to handle UI variations better"
            )
        
        # Generalization improvements
        low_gen = [r for r in results if r["comparison"].generalization_score < 0.7]
        if low_gen:
            recommendations.append(
                "Enhance training with diverse UI states (dark mode, different layouts, interruptions)"
            )
        
        # Performance optimization
        slow_execution = [r for r in results if r["comparison"].execution_time_ratio > 2.0]
        if slow_execution:
            recommendations.append(
                "Optimize action execution pipeline to reduce latency"
            )
        
        recommendations.append(
            "Consider incorporating real Android-in-the-Wild data for training to improve real-world performance"
        )
        
        return recommendations


# Import the multi-agent system components
from main import (
    MultiAgentQASystem
)
from openai import OpenAI

async def main():
    """Demonstrate Android-in-the-Wild integration with real dataset access"""
    
    print("="*60)
    print("Android-in-the-Wild Dataset Integration")
    print("="*60)
    
    # Check for required dependencies
    try:
        import tensorflow as tf
        print("✓ TensorFlow is available")
    except ImportError:
        print("✗ TensorFlow not found. Install with: pip install tensorflow")
        print("  Continuing with mock data...")
    
    # Initialize components
    dataset_name = "google_apps"  # Can be: general, google_apps, install, single, web_shopping
    dataset_loader = AndroidWildDatasetLoader(dataset_name=dataset_name)
    
    # Try to initialize LLM client
    try:
        llm_client = OpenAI()
        print("✓ OpenAI client initialized")
    except Exception as e:
        print(f"✗ OpenAI client initialization failed: {e}")
        llm_client = None
    
    # Create evaluator
    evaluator = AndroidWildEvaluator(
        qa_system=None,  # Will be created per task
        dataset_loader=dataset_loader,
        llm_client=llm_client
    )
    
    # Get episodes to evaluate
    print(f"\nLoading episodes from '{dataset_name}' dataset...")
    
    # Option 1: Get random episodes from the dataset
    use_random_episodes = True
    
    if use_random_episodes:
        try:
            episode_ids = await dataset_loader.get_random_episodes(count=3)
            print(f"Selected {len(episode_ids)} random episodes: {episode_ids}")
        except Exception as e:
            print(f"Failed to get random episodes: {e}")
            # Fall back to predefined episode IDs
            episode_ids = ["gmail_compose", "settings_notification", "maps_search"]
    else:
        # Option 2: Use specific episode IDs
        episode_ids = ["first"]  # Get the first episode from the dataset
    
    print(f"\nEvaluating on {len(episode_ids)} episodes")
    print("="*60)
    
    # Run evaluation
    evaluation_results = await evaluator.evaluate_on_videos(episode_ids)
    
    # Display summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    summary = evaluation_results["summary_report"]
    
    print(f"\nDataset: {dataset_name}")
    print(f"Episodes evaluated: {summary['evaluation_summary']['total_videos_evaluated']}")
    
    print(f"\nAggregate Metrics:")
    for metric, value in summary["aggregate_metrics"].items():
        print(f"  - {metric}: {value}")
    
    print(f"\nPerformance Analysis:")
    for metric, value in summary["performance_analysis"].items():
        print(f"  - {metric}: {value}")
    
    print(f"\nPer-Episode Results:")
    for result in summary["per_video_results"]:
        print(f"\n  Episode: {result['video_id']}")
        print(f"  Task: {result['task']}")
        print(f"  Accuracy: {result['accuracy']}")
        print(f"  Success: {'✓' if result['success'] else '✗'}")
    
    print(f"\nKey Insights:")
    for insight in summary["insights"]:
        print(f"  • {insight}")
    
    print(f"\nRecommendations:")
    for rec in summary["recommendations"]:
        print(f"  → {rec}")
    
    print(f"\nFull report saved to: outputs/logs/android_wild_evaluation_*.json")
    
    # Optional: Visualize an episode if visualization utils are available
    if dataset_loader.viz_available and evaluation_results["individual_results"]:
        print("\n" + "="*60)
        print("EPISODE VISUALIZATION")
        print("="*60)
        
        first_result = evaluation_results["individual_results"][0]
        video_id = first_result["video_id"]
        
        try:
            # Load the episode for visualization
            if dataset_loader.tf_available:
                print(f"\nVisualizing episode: {video_id}")
                # This would display the episode with annotations
                # dataset_loader.visualize_episode(episode_data)
                print("(Visualization would appear in notebook environment)")
        except Exception as e:
            print(f"Visualization failed: {e}")

# Additional utility function to setup the environment
def setup_android_in_wild_environment():
    """Setup script to prepare the environment for Android-in-the-Wild"""
    
    print("Setting up Android-in-the-Wild environment...")
    
    # Clone the repository if needed
    import subprocess
    import os
    
    if not os.path.exists("google-research"):
        print("Cloning google-research repository...")
        subprocess.run(["git", "clone", "https://github.com/google-research/google-research.git"])
    
    # Add to Python path
    import sys
    sys.path.append('./google-research')
    
    # Install required packages
    print("Installing required packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow"])
    
    print("Setup complete!")

if __name__ == "__main__":
    # Uncomment to setup environment first time
    # setup_android_in_wild_environment()
    
    asyncio.run(main())