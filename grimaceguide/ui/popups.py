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
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition

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

class TutorialPopup(Popup):
    """A fancier multi-page tutorial popup."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = "Welcome to GrimaceGuide!"
        self.size_hint = (0.85, 0.8)
        self.auto_dismiss = False

        self.page_index = 0  # Track the current page index

        layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(10))

        # Create a ScreenManager to switch between tutorial pages
        self.screen_manager = ScreenManager(transition=SlideTransition())

        # Define and add each tutorial page
        self.screen_manager.add_widget(self.create_screen(
            "Welcome to the meowgical world of feline facial analysis!\n\nPress 'Next' to continue.", 
            image_path="imagesGUI/tutorial-1.png" 
        ))
        self.screen_manager.add_widget(self.create_screen(
            "Paw-tograph please!\n\nUpload a photo of your feline friend to begin.",
            image_path="imagesGUI/tutorial-2.png" 
        ))
        self.screen_manager.add_widget(self.create_screen(
            "Results Panel:\n\nSee scores for Ears, Eyes, Muzzle, Whiskers, and Head — all under paw-servation!",
            image_path="imagesGUI/tutorial-3.png" 
        ))
        self.screen_manager.add_widget(self.create_screen(
            "Prediction:\n\nUse the Predict FGS Score button for accurate FGS score prediction based on facial landmarks.", # Updated text
            image_path="imagesGUI/tutorial-4.png" 
        ))
        self.screen_manager.add_widget(self.create_screen(
            "Ready to get started?\n\nLet's paw-sess those scores! 🐾",
            image_path="imagesGUI/tutorial-5.png" ,
            final=True  # Mark the last page
        ))

        # Create navigation buttons (Back and Next)
        self.button_layout = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(10))
        self.back_button = Button(text="Back", disabled=True, size_hint_x=0.4)
        self.next_button = Button(text="Next", size_hint_x=0.6)

        # Bind button actions
        self.back_button.bind(on_release=self.go_back)
        self.next_button.bind(on_release=self.go_next)

        self.button_layout.add_widget(self.back_button)
        self.button_layout.add_widget(self.next_button)

        # Add everything into the popup layout
        layout.add_widget(self.screen_manager)
        layout.add_widget(self.button_layout)

        self.content = layout

    def create_screen(self, text, image_path=None, final=False):
        """Create an individual tutorial page (screen) with optional image."""
        # Create a new screen with a unique name
        screen = Screen(name=f"screen{len(self.screen_manager.screens)}")

        screen_layout = BoxLayout(orientation='vertical', spacing=dp(10))

        # Add an image if provided
        if image_path:
            img = Image(source=image_path, size_hint=(1, 0.5), allow_stretch=True, keep_ratio=True)
            screen_layout.add_widget(img)

        # Add tutorial text
        lbl = Label(
            text=text,
            halign="center",
            valign="middle",
            markup=True,
            text_size=(dp(500), None),
            size_hint_y=None
        )
        lbl.bind(texture_size=lambda instance, value: setattr(lbl, 'height', value[1]))
        screen_layout.add_widget(lbl)

        # Add the layout to the screen
        screen.add_widget(screen_layout)

        # Mark if this is the final page
        screen.final = final
        return screen

    def go_back(self, instance):
        """Go to the previous tutorial page."""
        if self.page_index > 0:
            self.page_index -= 1
            self.screen_manager.transition.direction = 'right'
            self.screen_manager.current = self.screen_manager.screens[self.page_index].name
        self.update_buttons()

    def go_next(self, instance):
        """Go to the next tutorial page or dismiss if on the last page."""
        if self.page_index < len(self.screen_manager.screens) - 1:
            self.page_index += 1
            self.screen_manager.transition.direction = 'left'
            self.screen_manager.current = self.screen_manager.screens[self.page_index].name
        else:
            self.dismiss()
        self.update_buttons()

    def update_buttons(self):
        """Enable/disable navigation buttons based on the current page."""
        self.back_button.disabled = self.page_index == 0
        if self.page_index == len(self.screen_manager.screens) - 1:
            self.next_button.text = "Finish"
        else:
            self.next_button.text = "Next"

