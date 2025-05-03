"""
Main application class for the GrimaceGuide application
"""

import os
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.metrics import dp
from kivy.core.window import Window
from kivy.clock import Clock

# Import app configuration and components from local modules
from ..config import COLORS, WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, SCORE_CATEGORIES
from ..database import DatabaseManager
from ..api import send_image_for_processing
from .widgets import BorderedBox, BackgroundLabel, StyledButton, ScoreRowLayout, ImageContainer
from .popups import FileChooserPopup, MessagePopup, TutorialPopup, StartupChoicePopup

class GrimaceGuideApp(App):
    """Main application class"""
    
    def build(self):
        # Set window properties
        self.title = WINDOW_TITLE
        Window.size = (WINDOW_WIDTH, WINDOW_HEIGHT)
        Window.clearcolor = COLORS['light_gray']  # Set background color for the window
        
        # Initialize database to store images, scores and processing results
        self.db_manager = DatabaseManager()
        
        # Create the main layout - horizontal split for image panel and results panel
        main_layout = BoxLayout(orientation='horizontal', padding=dp(10), spacing=dp(10))
        
        # Left side - Image and controls
        self.left_panel = BorderedBox(
            orientation='vertical', 
            size_hint=(0.6, 1),  # Takes 60% of the width
            spacing=dp(10),
            padding=dp(10)
        )
        
        # Filename display - shows currently loaded image name
        self.filename_label = BackgroundLabel(
            text="No file selected", 
            size_hint_y=None, 
            height=dp(40),
            bg_color=(0.9, 0.9, 0.95, 1),
            color=COLORS['black'],
            bold=True
        )
        
        # Image display area with border for better visual separation
        self.image_container_border = BorderedBox(
            orientation='vertical', 
            border_color=(0.4, 0.4, 0.4, 1),
            padding=dp(2)
        )
        
        # Custom container to handle image aspect ratio properly
        self.image_container = ImageContainer()
        self.upload_button = StyledButton(text='Upload Image', size_hint=(1, 1))
        self.upload_button.bind(on_release=self.show_file_chooser)
        
        # Image widget that will display the selected image
        self.image_display = Image(
            source='', 
            allow_stretch=True, 
            keep_ratio=True,  # Maintains aspect ratio
            size_hint=(None, None)  # Will be positioned by the container
        )
        
        # Store reference for aspect ratio calculation
        self.image_container.image = self.image_display
        self.image_display.opacity = 0  # Initially hidden, will show after image is loaded
        
        # Put both in container but only one will be visible at a time
        # Toggle between upload button and actual image
        self.image_container.add_widget(self.upload_button)
        self.image_container.add_widget(self.image_display)
        
        self.image_container_border.add_widget(self.image_container)
        
        # Control buttons row for actions after image is loaded
        control_buttons = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(50), spacing=dp(10))
        
        # Upload Another Image button - allows changing image after processing
        self.upload_another_button = StyledButton(
            text='Upload Another Image', 
            disabled=True,  # Initially disabled until first image is loaded
            bg_color=COLORS['secondary']
        )
        self.upload_another_button.bind(on_release=self.show_file_chooser)
        
        # Only one processing option: API
        self.api_button = StyledButton(
            text='Predict FGS Score', 
            disabled=True,  # Disabled until image is loaded
            bg_color=COLORS['success']
        )
        
        # Connect buttons to their handler functions
        self.api_button.bind(on_release=self.process_with_api)
        
        # Add buttons to the layout
        control_buttons.add_widget(self.upload_another_button)
        control_buttons.add_widget(self.api_button)
        
        # Assemble left panel components in order (top to bottom)
        self.left_panel.add_widget(self.filename_label)
        self.left_panel.add_widget(self.image_container_border)
        self.left_panel.add_widget(control_buttons)
        
        # Right side - Results display for FGS scores
        self.right_panel = BorderedBox(
            orientation='vertical', 
            size_hint=(0.4, 1),  # Takes 40% of the width
            spacing=dp(15),
            padding=dp(10),
            bg_color=COLORS['background']
        )
        
        # Title for results section
        results_title = BackgroundLabel(
            text="RESULTS", 
            size_hint_y=None, 
            height=dp(40), 
            font_size=dp(20),
            bold=True,
            bg_color=COLORS['primary'],
            color=COLORS['white']
        )
        
        # Container for all score rows
        scores_container = BoxLayout(orientation='vertical', spacing=dp(8))
        
        # Create individual score rows for each FGS category
        self.score_rows = {}
        
        for category in SCORE_CATEGORIES:
            row = ScoreRowLayout(category)
            scores_container.add_widget(row)
            self.score_rows[category] = row  # Store reference to update later
        
        # Total score display at bottom of results panel
        total_score_layout = BorderedBox(
            orientation='horizontal', 
            size_hint_y=None, 
            height=dp(60),
            border_color=(0.4, 0.4, 0.4, 1),
            bg_color=(0.9, 0.9, 0.9, 1),
            padding=dp(10)
        )
        
        # Label for total score
        total_label = BackgroundLabel(
            text="TOTAL FGS SCORE:", 
            font_size=dp(18), 
            bold=True, 
            size_hint_x=0.7,
            color=COLORS['black'],
            bg_color=(0.9, 0.9, 0.9, 1)
        )
        
        # Value for total score - updated after processing
        self.total_score_value = BackgroundLabel(
            text="-", 
            font_size=dp(24), 
            bold=True, 
            size_hint_x=0.3,
            color=COLORS['black'],
            bg_color=(0.9, 0.9, 0.9, 1)
        )
        
        total_score_layout.add_widget(total_label)
        total_score_layout.add_widget(self.total_score_value)
        
        # Assemble right panel in order (top to bottom)
        self.right_panel.add_widget(results_title)
        self.right_panel.add_widget(scores_container)
        self.right_panel.add_widget(total_score_layout)
        self.right_panel.add_widget(Widget(size_hint_y=0.1))  # Empty spacer at bottom
        
        # Combine both panels into main layout
        main_layout.add_widget(self.left_panel)
        main_layout.add_widget(self.right_panel)
        
        # Initialize tracking variables for current image
        self.current_image_path = None
        self.current_image_id = None
        
        # Schedule canvas updates after layout is complete
        # This ensures proper rendering of all custom widgets
        Clock.schedule_once(self.update_all_canvases, 0.1)
        Clock.schedule_once(self.update_all_canvases, 0.5)  # Second update to catch any missed updates
        
        return main_layout

    def on_start(self):
        """Runs tutorial on startup"""
        Clock.schedule_once(self.show_tutorial_then_startup, 0.5)

    def show_tutorial_then_startup(self, dt):
        """Tutorial popup"""
        tutorial = TutorialPopup()
        tutorial.bind(on_dismiss=lambda *args: self.show_startup_choice())
        tutorial.open()

    def show_startup_choice(self):
        """Popup for user choice of importing or camera image"""
        from .popups import StartupChoicePopup
        popup = StartupChoicePopup(
            upload_callback=self.show_file_chooser,
            camera_callback=self.open_camera_capture
        )
        popup.open()

    def open_camera_capture(self, instance):
        from grimaceguide.ui.camera_cv import capture_image_with_overlay
        image_path = capture_image_with_overlay()
        if not image_path:
            print("Camera capture was canceled or failed.")
            return
        self.load_image(image_path) #Loaded images must have path
 
    def update_all_canvases(self, dt):
        """Force update of all canvases after layout is complete"""
        # Update left panel canvases for proper borders and backgrounds
        if hasattr(self, 'left_panel'):
            self.left_panel.update_canvas()
            self.filename_label.update_canvas()
            self.image_container_border.update_canvas()
        
        # Update right panel canvases for proper borders and backgrounds
        if hasattr(self, 'right_panel'):
            self.right_panel.update_canvas()
            for row in self.score_rows.values():
                row.update_canvas()
    
    def show_file_chooser(self, instance):
        """Show file chooser dialog to select an image"""
        # Create and open the file chooser popup with callback to load_image
        file_chooser = FileChooserPopup(load=self.load_image)
        file_chooser.open()
    
    def load_image(self, file_path):
        """Load an image from file path"""
        try:
            # Update the image path for processing
            self.current_image_path = file_path
            
            # Extract and display filename
            filename = os.path.basename(file_path)
            self.filename_label.text = filename
            
            # Update the image widget with selected file
            self.image_display.source = file_path
            self.image_display.reload()
            self.image_display.opacity = 1  # Show the image
            self.upload_button.opacity = 0  # Hide the upload button
            
            # Recalculate image sizing and position
            self.image_container._update_rect(None, None)
            
            # Store the image in the database for tracking
            self.current_image_id = self.db_manager.store_image(filename, file_path)
            
            if self.current_image_id:
                print(f"Image stored in database with ID: {self.current_image_id}")
            else:
                print("Failed to store image in database")
            
            # Enable all action buttons now that an image is loaded
            self.api_button.disabled = False
            self.upload_another_button.disabled = False
            
        except Exception as e:
            print(f"Error loading image: {e}")
            # Show error in popup for user feedback
            popup = MessagePopup(
                title="Error",
                message=f"Error loading image: {str(e)}"
            )
            popup.open()
    
    def process_with_api(self, instance):
        """Process the current image using the API"""
        if not self.current_image_path or not self.current_image_id:
            return
            
        # Show loading message during API processing
        # API calls may take time, so inform the user
        popup = MessagePopup(
            title="Processing",
            message="Processing image with API...\nThis may take a moment."
        )
        popup.open()
        
        # Use Clock to avoid freezing UI during processing
        Clock.schedule_once(lambda dt: self._do_api_processing(popup), 0.1)
    
    def _do_api_processing(self, loading_popup):
        """Handle API processing in a separate callback to avoid blocking UI"""
        try:
            # Call the API processing function from our api module
            result = send_image_for_processing(self.current_image_path)
            
            # Close loading popup when processing is done
            loading_popup.dismiss()
            
            # Handle API errors
            if not result['success']:
                popup = MessagePopup(
                    title="API Error",
                    message=f"Error processing image: {result.get('error', 'Unknown error')}"
                )
                popup.open()
                return
            
            # If API returned a processed image, display it
            if 'processed_image' in result:
                processed_path = result['processed_image']
                
                # Update database with processed image location
                self.db_manager.update_processed_image(self.current_image_id, processed_path)
                
                # Display the processed image with landmarks/markers
                self.image_display.source = processed_path
                self.image_display.reload()
                # Ensure image is sized correctly
                self.image_container._update_rect(None, None)
            
            # Store landmark data in database if available
            # The API result now contains 'labeled_landmarks' which might be more suitable
            # for direct use or further processing, but we still store the raw response.
            if 'api_response' in result:
                # Pass the raw api_response which contains the list of landmark dicts
                self.db_manager.store_landmarks(self.current_image_id, result['api_response']) 
            
            # Update UI with score results and store in database
            if 'scores' in result:
                # Store the calculated scores (which might be -1 if calculation failed)
                self.db_manager.store_scores(self.current_image_id, result['scores'], "API")
                self.update_scores(result['scores'])
            
        except Exception as e:
            print(f"Error in API processing: {e}")
            # Show error in popup for user feedback
            popup = MessagePopup(
                title="Processing Error",
                message=f"Error during processing: {str(e)}"
            )
            popup.open()
    
    def update_scores(self, scores):
        """Update the UI with new scores"""
        # Update each category score in the UI
        for category, score in scores.items():
            if category in self.score_rows:
                self.score_rows[category].score_value.text = str(score)
        
        # Update total score - either use provided total or calculate it
        if 'total_score' in scores:
            self.total_score_value.text = str(scores['total_score'])
        else:
            # Calculate total if not provided by summing all categories
            total = sum(scores.get(cat, 0) for cat in SCORE_CATEGORIES)
            self.total_score_value.text = str(total)
    
    def on_stop(self):
        """Close database connection when app closes"""
        # Clean up resources when application exits
        if hasattr(self, 'db_manager'):
            self.db_manager.close()
