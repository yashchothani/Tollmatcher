#NEW

import board
import busio
import adafruit_ssd1306
from PIL import Image, ImageDraw, ImageFont

class LanguageDisplay:
    def __init__(self, width=128, height=32):
        self.width = width
        self.height = height

        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.oled = adafruit_ssd1306.SSD1306_I2C(width, height, self.i2c)

        self.image = Image.new("1", (self.width, self.height))
        self.draw = ImageDraw.Draw(self.image)
        self.font = ImageFont.load_default()

        self.clear()

    def clear(self):
        self.oled.fill(0)
        self.oled.show()

    def _render(self, lines):
        self.draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)

        if self.height == 32:
            y_positions = [0, 16]
        else:
            y_positions = [0, 16, 32, 48]

        for i, line in enumerate(lines[:len(y_positions)]):
            self.draw.text((0, y_positions[i]), line, font=self.font, fill=255)

        self.oled.image(self.image)
        self.oled.show()

    def show_languages(self, source_lang, target_lang):
        source_lang = (source_lang or "--").upper()
        target_lang = (target_lang or "--").upper()

        if self.height == 32:
            self._render([
                f"G: {source_lang}",
                f"U: {target_lang}",
            ])
        else:
            self._render([
                "Language Set",
                f"Guest: {source_lang}",
                f"User : {target_lang}",
            ])

    def show_status(self, status):
        self._render([status])

    def show_error(self, msg="Error"):
        self._render(["System", msg[:16]])
