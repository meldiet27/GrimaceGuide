from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.app import App
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup


class FGSApp(BoxLayout):
    # Initialize the parent BoxLayout with vertical orientation
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)

        # Create an Image widget to display the uploaded cat image
        self.img = Image(size_hint=(1, 0.6))
        self.add_widget(self.img)

        # Create a Label to upload an image
        self.label = Label(text='Upload a cat image', size_hint=(1, 0.1))
        self.add_widget(self.label)

        # Create a Button that opens the file chooser when pressed
        self.btn = Button(
                    text='Upload Image',
                    size_hint=(1, 0.15),
                    background_normal='',
                    background_color=(0.2, 0.6, 1, 1),  # blue button
                    color=(1, 1, 1, 1)  # white text
                )
        self.btn.bind(on_press=self.choose_file)
        self.add_widget(self.btn)

        # Create a Label to display the predicted FGS score
        self.result = Label(text='', size_hint=(1, 0.15))
        self.add_widget(self.result)

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
                self.img.source = selected[0]  # Set image source
                self.img.reload()              # Reload the image
                popup.dismiss()                # Close the popup

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