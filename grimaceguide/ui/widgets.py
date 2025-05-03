"""
Custom widgets for the GrimaceGuide application
"""

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.graphics import Color, Rectangle, Line
from kivy.metrics import dp
from kivy.clock import Clock

from ..config import COLORS

class BorderedBox(BoxLayout):
    """BoxLayout with a border and background color"""
    def __init__(self, border_color=COLORS['dark_gray'], bg_color=COLORS['white'], **kwargs):
        super(BorderedBox, self).__init__(**kwargs)
        # Store colors for drawing operations
        self.border_color = border_color
        self.bg_color = bg_color
        # Update canvas when widget position or size changes
        self.bind(pos=self.update_canvas, size=self.update_canvas)
        # Schedule initial drawing after widget is fully initialized
        Clock.schedule_once(self.update_canvas, 0)
        
    def update_canvas(self, *args):
        # Clear previous drawings to prevent overlap
        self.canvas.before.clear()
        with self.canvas.before:
            # Background
            Color(*self.bg_color)
            Rectangle(pos=self.pos, size=self.size)
            # Border
            Color(*self.border_color)
            # Draw border lines - slightly thicker than default for better visibility
            Line(rectangle=(self.pos[0], self.pos[1], self.size[0], self.size[1]), width=1.2)

class BackgroundLabel(Label):
    """Label with a colored background"""
    def __init__(self, bg_color=COLORS['light_gray'], **kwargs):
        super(BackgroundLabel, self).__init__(**kwargs)
        self.bg_color = bg_color
        # Update canvas when widget position or size changes
        self.bind(size=self.update_canvas, pos=self.update_canvas)
        # Schedule initial drawing after widget is fully initialized
        Clock.schedule_once(self.update_canvas, 0)
        
    def update_canvas(self, *args):
        # Clear previous drawings to prevent overlap
        self.canvas.before.clear()
        with self.canvas.before:
            # Draw background rectangle with specified color
            Color(*self.bg_color)
            Rectangle(pos=self.pos, size=self.size)

class StyledButton(Button):
    """Button with custom background color and white text"""
    def __init__(self, bg_color=COLORS['primary'], **kwargs):
        super(StyledButton, self).__init__(**kwargs)
        self.bg_color = bg_color
        # Remove default button background texture
        self.background_normal = ''
        # Apply our custom background color
        self.background_color = bg_color
        # White text for better contrast against colored backgrounds
        self.color = COLORS['white']

class ScoreRowLayout(BoxLayout):
    """Horizontal row showing score category, value, and help button"""
    def __init__(self, category, **kwargs):
        super(ScoreRowLayout, self).__init__(**kwargs)
        # Horizontal layout for category name, score value, and help button
        self.orientation = 'horizontal'
        self.padding = dp(5)
        self.spacing = dp(5)
        # Fixed height for consistent row sizing
        self.size_hint_y = None
        self.height = dp(40)
        
        # Draw background and border
        with self.canvas.before:
            # Background color for the row
            Color(*COLORS['light_gray'])
            self.bg_rect = Rectangle(pos=self.pos, size=self.size)
            # Border color with medium gray
            Color(0.7, 0.7, 0.7, 1)
            # Draw border around the entire row
            self.border = Line(rectangle=(self.pos[0], self.pos[1], self.size[0], self.size[1]), width=1)
        
        # Update canvas when widget position or size changes
        self.bind(pos=self.update_canvas, size=self.update_canvas)
        # Schedule initial drawing after widget is fully initialized
        Clock.schedule_once(self.update_canvas, 0)
        
        # Label for category name (left side)
        category_label = Label(
            text=category.upper(), 
            size_hint_x=0.5,
            bold=True,
            color=COLORS['black']
        )
        
       # Score value (center)
        self.score_value = Label(
            text="-", 
            size_hint_x=0.3,
            color=COLORS['black']
        )
        
        # Help button (right side)
        from .popups import ScoreHelpButton
        
        # Help button takes 20% of the row width
        help_button = ScoreHelpButton(category=category, size_hint_x=0.2)
        
        # Add widgets to layout in order: label, score, help button
        self.add_widget(category_label)
        self.add_widget(self.score_value)
        self.add_widget(help_button)
    
    def update_canvas(self, *args):
        # Clear previous drawings to prevent overlap
        self.canvas.before.clear()
        with self.canvas.before:
            # Redraw background and border when size/position changes
            Color(*COLORS['light_gray'])
            self.bg_rect = Rectangle(pos=self.pos, size=self.size)
            Color(0.7, 0.7, 0.7, 1)
            self.border = Line(rectangle=(self.pos[0], self.pos[1], self.size[0], self.size[1]), width=1)

class ImageContainer(BoxLayout):
    """Container for displaying an image with correct aspect ratio"""
    def __init__(self, **kwargs):
        super(ImageContainer, self).__init__(orientation='vertical', **kwargs)
        # Update image layout when container size changes
        self.bind(size=self._update_rect, pos=self._update_rect)

    # Prevents calculations from being made when the container hasn't been established yet
    # Height=0 or Width=0
    def _update_rect(self, instance, value):
    # Ensure image exists and has texture loaded
    if hasattr(self, 'image') and self.image:
        if self.height == 0 or self.width == 0:
            # Prevent division by zero if layout hasn't been established
            return

        if self.image.texture:
            # Calculate aspect ratios to determine sizing strategy
            image_ratio = self.image.texture.width / self.image.texture.height
            container_ratio = self.width / self.height

            # Different strategies based on whether image or container is wider
            if image_ratio > container_ratio:  # Image is wider than container
                # Fit to width
                self.image.height = self.width / image_ratio
                self.image.width = self.width
                # Center vertically
                self.image.center_y = self.center_y
            else:  # Image is taller than container
                # Set width proportionally based on height constraint
                self.image.width = self.height * image_ratio
                self.image.height = self.height
                # Center horizontally
                self.image.center_x = self.center_x

