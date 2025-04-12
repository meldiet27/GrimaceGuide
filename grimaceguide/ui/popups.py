"""
Custom popup dialogs for the GrimaceGuide application
"""

import os
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserListView
from kivy.properties import ObjectProperty, StringProperty
from kivy.metrics import dp
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle

from .widgets import StyledButton
from ..config import COLORS, HELP_IMAGES

class ImageHelpPopup(Popup):
    """Popup for displaying a help image with explanations."""
    def __init__(self, title, image_path, **kwargs):
        # Default popup styling
        kwargs.setdefault('title', title)
        kwargs.setdefault('size_hint', (0.8, 0.8))  # Larger size for image viewing
        kwargs.setdefault('title_color', COLORS['white'])
        kwargs.setdefault('separator_color', COLORS['dark_gray'])
        kwargs.setdefault('background_color', (0.15, 0.15, 0.15, 1))  # Dark background
        
        super(ImageHelpPopup, self).__init__(**kwargs)
        
        # Create content layout
        content_layout = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(10))
        
        # Image display
        image = Image(
            source=image_path,
            allow_stretch=True,
            keep_ratio=True,
            size_hint=(1, 1)
        )
        
        # Close button
        close_button = Button(
            text="Close",
            size_hint_y=None,
            height=dp(40),
            background_normal='',  # Remove default button background
            background_color=COLORS['dark_gray'],
            color=COLORS['white']
        )
        close_button.bind(on_release=self.dismiss)
        
        # Add widgets to layout
        content_layout.add_widget(image)
        content_layout.add_widget(close_button)
        
        self.content = content_layout
        
        # Add custom background
        with self.content.canvas.before:
            Color(0.18, 0.18, 0.18, 1)  # Slightly lighter than the popup background
            self.rect = Rectangle(size=self.content.size, pos=self.content.pos)
            
        # Update background when layout resizes
        self.content.bind(size=self._update_rect, pos=self._update_rect)
    
    def _update_rect(self, instance, value):
        """Keep the background rectangle in sync with the content size"""
        self.rect.pos = instance.pos
        self.rect.size = instance.size

class MessagePopup(Popup):
    """Popup for displaying a simple message to the user."""
    def __init__(self, title, message, **kwargs):
        # Default values can be overridden by kwargs
        kwargs.setdefault('title', title)
        kwargs.setdefault('size_hint', (0.6, 0.4))  # Standard size for text messages
        super(MessagePopup, self).__init__(**kwargs)
        
        # Vertical layout for message and button
        content_layout = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(10))
        
        # Message label with word wrapping
        msg_label = Label(
            text=message,
            text_size=(Window.width * 0.5, None),  # Wrap text at 50% of window width
            halign='center',
            valign='middle',
            markup=True  # Allow markup for formatting text
        )
        
        # Use our themed button
        close_button = StyledButton(
            text="Close",
            size_hint_y=None,
            height=dp(40)
        )
        close_button.bind(on_release=self.dismiss)
        
        content_layout.add_widget(msg_label)
        content_layout.add_widget(close_button)
        
        self.content = content_layout

class FileChooserPopup(Popup):
    """Popup dialog for selecting an image file."""
    # Callback function after file selection
    load = ObjectProperty()
    
    def __init__(self, **kwargs):
        super(FileChooserPopup, self).__init__(**kwargs)
        self.title = "Select an Image"
        self.size_hint = (0.9, 0.9)  # Almost full screen for better browsing
        
        # Main container
        content = BoxLayout(orientation='vertical')
        
        # File chooser - limited to image types
        self.file_chooser = FileChooserListView(filters=['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif'])
        
        # Bottom buttons: Cancel and Select
        buttons = BoxLayout(size_hint_y=None, height=dp(50))
        cancel_button = StyledButton(text='Cancel', bg_color=COLORS['danger'])
        select_button = StyledButton(text='Select')
        
        buttons.add_widget(cancel_button)
        buttons.add_widget(select_button)
        
        content.add_widget(self.file_chooser)
        content.add_widget(buttons)
        
        self.content = content
        
        # Connect button events
        cancel_button.bind(on_release=self.dismiss)
        select_button.bind(on_release=self._select)
        
    def _select(self, instance):
        """Process the selected file and call the callback function."""
        try:
            if self.file_chooser.selection:
                selection = self.file_chooser.selection[0]
                self.load(selection) # Call the external callback
                self.dismiss()
        except Exception as e:
            print(f"Error selecting file: {e}")

class ScoreHelpButton(StyledButton):
    """Small help button that shows scoring instructions for a given category."""
    # Scoring category (e.g., "ears", "eyes")
    category = StringProperty('')
    
    def __init__(self, category, **kwargs):
        # Use our dark gray theme for the help button
        super(ScoreHelpButton, self).__init__(bg_color=COLORS['dark_gray'], **kwargs)
        self.category = category
        self.text = "?"  # Question mark as visual indicator
        self.size_hint = (None, None)
        self.size = (dp(25), dp(25))  # Small, circular-ish button
        self.color = COLORS['black']
        self.bind(on_release=self.show_help)
    
    def show_help(self, instance):
        """Open a popup showing the help image for this category."""
        # Get the help image path from our config
        image_path = HELP_IMAGES.get(self.category.lower())
        if image_path and os.path.exists(image_path):
            # Create and show the help popup with the appropriate image
            popup = ImageHelpPopup(title=f"{self.category.upper()} Score Guide", image_path=image_path)
            popup.open()
        else:
            # Fallback if image not found - could enhance with a default image
            print(f"Error: Image not found at path {image_path}")