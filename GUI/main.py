from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.app import App


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
        self.btn = Button(text='Upload Image', size_hint=(1, 0.15))
        self.btn.bind(on_press=self.choose_file)
        self.add_widget(self.btn)

        # Create a Label to display the predicted FGS score
        self.result = Label(text='', size_hint=(1, 0.15))
        self.add_widget(self.result)

    # Function called when the button is pressed
    def choose_file(self, instance):
        pass
    
     # Function to process the image (not yet implemented)
    def process_image(self, path, popup):
        pass


class AppMain(App):
    def build(self):
        return FGSApp()

if __name__ == '__main__':
    AppMain().run()