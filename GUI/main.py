from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.app import App
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.textinput import TextInput


class FGSApp(BoxLayout):
    # Initialize the parent BoxLayout with vertical orientation
    def __init__(self, **kwargs):
        BoxLayout.__init__(self, orientation='horizontal', **kwargs)

         # === Left Panel ===
        left_panel = BoxLayout(orientation='vertical', spacing=10, padding=10)

        # File name display
        self.file_name = TextInput(text='Name of the file', size_hint=(1, 0.1), readonly=True)
        left_panel.add_widget(self.file_name)

        # Image preview
        self.img = Image(size_hint=(1, 0.6))
        left_panel.add_widget(self.img)

        # Prediction buttons
        button_box = BoxLayout(size_hint=(1, 0.2), spacing=10)
        api_btn = Button(text='Use API to Predict')
        model_btn = Button(text='Use model to Predict')
        button_box.add_widget(api_btn)
        button_box.add_widget(model_btn)
        left_panel.add_widget(button_box)

        # Upload Button
        upload_btn = Button(text='Upload Image', size_hint=(1, 0.1), background_normal='', background_color=(0.2, 0.6, 1, 1), color=(1, 1, 1, 1))
        upload_btn.bind(on_press=self.choose_file)
        left_panel.add_widget(upload_btn)

        # === Right Panel ===
        right_panel = BoxLayout(orientation='vertical', spacing=10, padding=10, size_hint=(0.5, 1))

        # Action Unit Label
        right_panel.add_widget(Label(text='ACTION UNITS', bold=True, size_hint=(1, 0.1)))

        # Action Unit Selectors
        self.au_values = {}
        for au in ['EARS', 'EYES', 'MUZZLE', 'WHISKERS', 'HEAD']:
            box = BoxLayout(size_hint=(1, 0.15))
            box.add_widget(Label(text=f'{au}', size_hint=(0.3, 1)))
            toggle_group = BoxLayout(size_hint=(0.7, 1))
            self.au_values[au] = []
            for i in range(3):
                btn = ToggleButton(text=str(i), group=au, allow_no_selection=False)
                self.au_values[au].append(btn)
                toggle_group.add_widget(btn)
            box.add_widget(toggle_group)
            right_panel.add_widget(box)

        # FGS Score Display
        self.result = Label(text='TOTAL FGS SCORE:\n0 / 10', size_hint=(1, 0.2))
        right_panel.add_widget(self.result)

        # Add both panels to main layout
        self.add_widget(left_panel)
        self.add_widget(right_panel)

    # Function called when the button is pressed
    def choose_file(self, instance):
        # Create a vertical layout for the popup content
        content = BoxLayout(orientation='vertical')

        # Create a file chooser to select image files
        filechooser = FileChooserIconView(filters=['*.png', '*.jpg', '*.jpeg'], size_hint=(1, 0.9))

        # Create a "Select" button
        select_btn = Button(text='Select', size_hint=(1, 0.1))

        # Add the file chooser and button to the layout
        content.add_widget(filechooser)
        content.add_widget(select_btn)

        # Create a popup window with the layout
        popup = Popup(title="Select an Image", content=content, size_hint=(0.9, 0.9))

        # Define what happens when "Select" button is pressed
        def select_callback(instance):
            selected = filechooser.selection   # Get selected files
            if selected:
                # Create and display a popup window showing a "Loading..." message
                # Used to indicate that a process (e.g., image upload or API call) is in progress
                loading_popup = Popup(
                    title="Loading",                     # Title of the popup
                    content=Label(text="Loading..."),   # Text to display in the popup
                    size_hint=(0.5, 0.3)                 # Size of the popup relative to the window
                )
                loading_popup.open()  # Show the popup on the screen

                def load_image(dt):
                    self.img.source = selected[0]  # Set image source
                    self.img.reload()              # Reload the image
                    self.file_name.text = selected[0].split('/')[-1] # Extract the file name from the full path and display it in the text field
                    popup.dismiss()                # Close the popup
                    loading_popup.dismiss()        # Close the loading popup
                
                # Schedule the load_image function to run after 0.5 seconds
                Clock.schedule_once(load_image, 0.5)
  
        # Bind the select button to the callback
        select_btn.bind(on_press=select_callback)

        # Open the popup window
        popup.open()
    
     # Function to process the image (not yet implemented)
    def process_image(self, path, popup):
        pass


# Define the main App class
class AppMain(App):
    # Build and return the root widget of the app
    def build(self):
        return FGSApp()

if __name__ == '__main__':
    AppMain().run()